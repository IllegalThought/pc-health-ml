from pathlib import Path
import os

import pandas as pd
import requests


ROOT = Path(
    __file__
).resolve().parents[2]


DATA = (
    ROOT
    / "data"
    / "processed"
    / "thermal_features.csv"
)


API_URL = os.getenv(

    "PC_HEALTH_URL",

    "http://127.0.0.1:8000"
)


API_KEY = os.getenv(

    "PC_HEALTH_API_KEY",

    "development-key"
)


df = pd.read_csv(DATA)


sample = df.iloc[-1]


payload = {

    "machine_id":
        str(
            sample["machine_id"]
        ),

    "cpu_load_mean":
        float(
            sample["cpu_load_mean"]
        ),

    "cpu_temp_mean":
        float(
            sample["cpu_temp_mean"]
        ),

    "cpu_power_mean":
        float(
            sample["cpu_power_mean"]
        ),

    "cpu_p_clock_mean":
        float(
            sample[
                "cpu_p_clock_mean"
            ]
        ),

    "cpu_e_clock_mean":
        float(
            sample[
                "cpu_e_clock_mean"
            ]
        ),

    "cpu_fan_mean":
        float(
            sample["cpu_fan_mean"]
        ),

    "pump_fan_mean":
        float(
            sample["pump_fan_mean"]
        ),

    "gpu_load_mean":
        float(
            sample["gpu_load_mean"]
        ),

    "gpu_temp_mean":
        float(
            sample["gpu_temp_mean"]
        ),

    "gpu_power_mean":
        float(
            sample["gpu_power_mean"]
        ),

    "ssd_temp_mean":
        float(
            sample["ssd_temp_mean"]
        ),

    "motherboard_system_temp_mean":
        float(
            sample[
                "motherboard_system_temp_mean"
            ]
        ),

    "motherboard_vrm_temp_mean":
        float(
            sample[
                "motherboard_vrm_temp_mean"
            ]
        ),

    "motherboard_pch_temp_mean":
        float(
            sample[
                "motherboard_pch_temp_mean"
            ]
        ),

    "cpu_socket_temp_mean":
        float(
            sample[
                "cpu_socket_temp_mean"
            ]
        ),

    "cpu_temp_delta":
        float(
            sample["cpu_temp_delta"]
        )
}


response = requests.post(

    API_URL
    + "/predict/thermal",

    json=payload,

    headers={

        "x-api-key":
            API_KEY
    },

    timeout=90
)


print(
    "HTTP:",
    response.status_code
)

print()

print(
    response.json()
)