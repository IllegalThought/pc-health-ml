from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "thermal_features.csv"
MODEL_PATH = ROOT / "models" / "thermal_bundle.joblib"

df = pd.read_csv(DATA)
df["minute"] = pd.to_datetime(df["minute"])

# ---------------------------------------------------------
# IMPORTANT:
# Each workload was recorded as one 20-minute healthy block.
# We split chronologically INSIDE every workload so all
# operating regimes appear in train/validation/test.
# ---------------------------------------------------------

train_parts = []
val_parts = []
test_parts = []

for workload, group in df.groupby("workload"):
    group = group.sort_values("minute").reset_index(drop=True)

    n = len(group)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_parts.append(group.iloc[:train_end])
    val_parts.append(group.iloc[train_end:val_end])
    test_parts.append(group.iloc[val_end:])

train = pd.concat(train_parts, ignore_index=True)
validation = pd.concat(val_parts, ignore_index=True)
test = pd.concat(test_parts, ignore_index=True)

print("\n=== SPLIT ===")
print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))

# ---------------------------------------------------------
# MODEL 1:
# Predict the CPU temperature expected for the current
# workload / power / cooling / case-heat conditions.
# ---------------------------------------------------------

REGRESSION_FEATURES = [
    "cpu_load_mean",
    "cpu_power_mean",
    "cpu_fan_mean",
    "pump_fan_mean",
    "cpu_p_clock_mean",
    "cpu_e_clock_mean",
    "gpu_load_mean",
    "gpu_temp_mean",
    "gpu_power_mean",
    "motherboard_system_temp_mean",
    "motherboard_vrm_temp_mean",
    "motherboard_pch_temp_mean",
    "ssd_temp_mean",
]

temperature_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=600,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

temperature_model.fit(
    train[REGRESSION_FEATURES],
    train["cpu_temp_mean"],
)

def add_expected_temperature(frame):
    frame = frame.copy()

    expected = temperature_model.predict(
        frame[REGRESSION_FEATURES]
    )

    frame["expected_cpu_temp"] = expected

    frame["cpu_temp_residual"] = (
        frame["cpu_temp_mean"] - frame["expected_cpu_temp"]
    )

    return frame

train = add_expected_temperature(train)
validation = add_expected_temperature(validation)
test = add_expected_temperature(test)

for name, part in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test),
]:
    mae = mean_absolute_error(
        part["cpu_temp_mean"],
        part["expected_cpu_temp"],
    )

    r2 = r2_score(
        part["cpu_temp_mean"],
        part["expected_cpu_temp"],
    )

    print(
        f"{name}: MAE={mae:.3f} C | R2={r2:.3f}"
    )

# ---------------------------------------------------------
# MODEL 2:
# Multivariate anomaly detector.
# ---------------------------------------------------------

ANOMALY_FEATURES = [
    "cpu_load_mean",
    "cpu_temp_mean",
    "cpu_power_mean",
    "cpu_p_clock_mean",
    "cpu_e_clock_mean",
    "cpu_fan_mean",
    "pump_fan_mean",
    "cpu_temp_delta",
    "cpu_temp_per_load",
    "cpu_temp_per_power",
    "cpu_temp_residual",
    "motherboard_vrm_temp_mean",
    "cpu_socket_temp_mean",
]

anomaly_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    (
        "isolation_forest",
        IsolationForest(
            n_estimators=500,
            max_samples="auto",
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

anomaly_model.fit(train[ANOMALY_FEATURES])

validation_scores = -anomaly_model.decision_function(
    validation[ANOMALY_FEATURES]
)

# Provisional threshold because the first dataset is small.
if_threshold = float(
    np.quantile(validation_scores, 0.95)
)

# A normal gaming session showed thermal soak.
# Use a conservative temperature residual threshold for V0.1.
learned_residual_q99 = float(
    np.quantile(
        validation["cpu_temp_residual"],
        0.99,
    )
)

residual_threshold = max(
    6.0,
    learned_residual_q99,
)

print("\n=== PROVISIONAL THRESHOLDS ===")
print(
    "Isolation Forest threshold:",
    round(if_threshold, 6),
)
print(
    "CPU temperature residual threshold:",
    round(residual_threshold, 3),
    "C",
)

# Evaluate false alarms on held-out healthy data.
test_scores = -anomaly_model.decision_function(
    test[ANOMALY_FEATURES]
)

test_if_flag = test_scores > if_threshold

test_residual_flag = (
    test["cpu_temp_residual"] > residual_threshold
)

# V0.1 validator:
# A severe residual is enough for warning.
# A weaker multivariate anomaly needs at least some positive
# thermal residual evidence.
test_warning = (
    test_residual_flag
    |
    (
        test_if_flag
        &
        (test["cpu_temp_residual"] > 3.0)
    )
)

print("\n=== HEALTHY HOLD-OUT CHECK ===")
print(
    "Held-out healthy warning rate:",
    round(float(test_warning.mean()), 4),
)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

bundle = {
    "temperature_model": temperature_model,
    "anomaly_model": anomaly_model,
    "regression_features": REGRESSION_FEATURES,
    "anomaly_features": ANOMALY_FEATURES,
    "if_threshold": if_threshold,
    "residual_threshold": residual_threshold,
}

joblib.dump(bundle, MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)
