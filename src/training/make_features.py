from pathlib import Path
import pandas as pd

from features import build_features

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "pc_telemetry.csv"
OUTPUT = ROOT / "data" / "processed" / "thermal_features.csv"

df = pd.read_csv(RAW)

features = build_features(df)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
features.to_csv(OUTPUT, index=False)

print("\n=== FEATURE DATASET CREATED ===")
print("Rows:", len(features))
print("\nWindows per workload:")
print(features["workload"].value_counts())
print("\nSaved to:")
print(OUTPUT)
