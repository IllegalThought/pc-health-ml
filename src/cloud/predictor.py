from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "models"
    / "thermal_bundle.joblib"
)


print("Loading ML model from:")
print(MODEL_PATH)


bundle = joblib.load(
    MODEL_PATH
)


temperature_model = bundle[
    "temperature_model"
]

anomaly_model = bundle[
    "anomaly_model"
]

regression_features = bundle[
    "regression_features"
]

anomaly_features = bundle[
    "anomaly_features"
]

if_threshold = bundle[
    "if_threshold"
]

residual_threshold = bundle[
    "residual_threshold"
]


def predict_thermal(data: dict):

    row = pd.DataFrame(
        [data]
    )

    # -----------------------------------
    # Derived features
    # -----------------------------------

    row["cpu_temp_per_load"] = (
        row["cpu_temp_mean"]
        /
        (
            row["cpu_load_mean"]
            + 1.0
        )
    )

    row["cpu_temp_per_power"] = (
        row["cpu_temp_mean"]
        /
        (
            row["cpu_power_mean"]
            + 1.0
        )
    )

    # -----------------------------------
    # Expected CPU temperature
    # -----------------------------------

    expected_temp = float(

        temperature_model.predict(

            row[
                regression_features
            ]

        )[0]
    )


    actual_temp = float(
        row["cpu_temp_mean"].iloc[0]
    )


    residual = (
        actual_temp
        - expected_temp
    )


    row[
        "cpu_temp_residual"
    ] = residual


    # -----------------------------------
    # Isolation Forest
    # -----------------------------------

    isolation_score = float(

        -anomaly_model
        .decision_function(

            row[
                anomaly_features
            ]

        )[0]
    )


    isolation_abnormal = (
        isolation_score
        >
        if_threshold
    )


    residual_abnormal = (
        residual
        >
        residual_threshold
    )


    # Same validation logic used
    # during local testing.

    warning = (

        residual_abnormal

        or

        (
            isolation_abnormal
            and residual > 3.0
        )
    )


    # -----------------------------------
    # Human-readable explanation
    # -----------------------------------

    reasons = []


    if residual_abnormal:

        reasons.append(
            "CPU temperature is significantly "
            "higher than expected for the "
            "current hardware workload."
        )


    if isolation_abnormal:

        reasons.append(
            "The combined hardware telemetry "
            "pattern differs from the learned "
            "healthy baseline."
        )


    if (
        data["cpu_fan_mean"] > 0
        and residual > 5
    ):

        reasons.append(
            "Elevated CPU temperature persists "
            "despite active cooling."
        )


    if not warning:

        reasons.append(
            "Thermal behaviour is consistent "
            "with the learned healthy baseline."
        )


    status = (
        "WARNING"
        if warning
        else "NORMAL"
    )


    return {

        "status":
            status,

        "actual_cpu_temperature":
            round(
                actual_temp,
                2
            ),

        "expected_cpu_temperature":
            round(
                expected_temp,
                2
            ),

        "temperature_residual":
            round(
                residual,
                2
            ),

        "isolation_score":
            round(
                isolation_score,
                5
            ),

        "isolation_threshold":
            round(
                float(if_threshold),
                5
            ),

        "residual_threshold":
            round(
                float(residual_threshold),
                2
            ),

        "isolation_abnormal":
            bool(
                isolation_abnormal
            ),

        "thermal_residual_abnormal":
            bool(
                residual_abnormal
            ),

        "warning":
            bool(
                warning
            ),

        "reasons":
            reasons
    }