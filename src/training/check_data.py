from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "pc_telemetry.csv"

df = pd.read_csv(DATA)
df["timestamp"] = pd.to_datetime(df["timestamp"])

print("\n=== DATASET SHAPE ===")
print(df.shape)

print("\n=== TIME RANGE ===")
print(df["timestamp"].min())
print(df["timestamp"].max())
print(
    "Duration (minutes):",
    round(
        (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 60,
        2
    )
)

print("\n=== WORKLOAD COUNTS ===")
print(df["workload"].value_counts())

print("\n=== MISSING VALUES ===")
missing = df.isna().sum()
print(missing[missing > 0] if (missing > 0).any() else "No missing values")

print("\n=== SELECTED SENSOR SUMMARY ===")
cols = [
    "cpu_load",
    "cpu_temp",
    "cpu_power",
    "cpu_p_clock",
    "cpu_e_clock",
    "cpu_fan_rpm",
    "pump_fan_rpm",
    "gpu_load",
    "gpu_temp",
    "gpu_power",
    "ssd_temp",
    "motherboard_system_temp",
    "motherboard_vrm_temp",
    "motherboard_pch_temp",
    "cpu_socket_temp",
]
print(df[cols].describe().T.round(2))
