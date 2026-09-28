from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "processed" / "thermal_features.csv"
MODEL = ROOT / "models" / "thermal_bundle.joblib"

df = pd.read_csv(DATA)

bundle = joblib.load(MODEL)

temperature_model = bundle["temperature_model"]
anomaly_model = bundle["anomaly_model"]
reg_features = bundle["regression_features"]
anomaly_features = bundle["anomaly_features"]
if_threshold = bundle["if_threshold"]
residual_threshold = bundle["residual_threshold"]


def prepare(frame):
    frame = frame.copy()

    frame["expected_cpu_temp"] = (
        temperature_model.predict(
            frame[reg_features]
        )
    )

    frame["cpu_temp_residual"] = (
        frame["cpu_temp_mean"]
        - frame["expected_cpu_temp"]
    )

    return frame


def score(frame):
    frame = prepare(frame)

    if_score = float(
        -anomaly_model.decision_function(
            frame[anomaly_features]
        )[0]
    )

    residual = float(
        frame["cpu_temp_residual"].iloc[0]
    )

    severe_residual = (
        residual > residual_threshold
    )

    multivariate_abnormal = (
        if_score > if_threshold
    )

    warning = (
        severe_residual
        or (
            multivariate_abnormal
            and residual > 3.0
        )
    )

    return {
        "actual_cpu_temp": round(
            float(frame["cpu_temp_mean"].iloc[0]),
            2,
        ),
        "expected_cpu_temp": round(
            float(frame["expected_cpu_temp"].iloc[0]),
            2,
        ),
        "temperature_residual": round(
            residual,
            2,
        ),
        "isolation_score": round(
            if_score,
            5,
        ),
        "warning": bool(warning),
    }


sample = df.iloc[-1:].copy()

print("\n=== NORMAL RECORDED SAMPLE ===")
print(score(sample))

# Pipeline demonstration only.
# This does NOT represent a real hardware failure sample.
synthetic = sample.copy()

synthetic["cpu_temp_mean"] += 10.0
synthetic["cpu_temp_max"] += 10.0

synthetic["cpu_temp_per_load"] = (
    synthetic["cpu_temp_mean"]
    / (synthetic["cpu_load_mean"] + 1.0)
)

synthetic["cpu_temp_per_power"] = (
    synthetic["cpu_temp_mean"]
    / (synthetic["cpu_power_mean"] + 1.0)
)

print("\n=== SYNTHETIC THERMAL-DEGRADATION TEST ===")
print(score(synthetic))

print(
    "\nNOTE: The synthetic sample is only for validating "
    "the software pipeline, not for claiming real failure "
    "prediction accuracy."
)
