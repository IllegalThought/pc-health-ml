import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # The one-minute "test" run was only for collector verification.
    df = df[df["workload"] != "test"].copy()

    df = df.sort_values(["machine_id", "timestamp"])

    df["minute"] = df["timestamp"].dt.floor("1min")

    grouped = df.groupby(
        ["machine_id", "workload", "minute"],
        as_index=False,
    )

    features = grouped.agg(
        cpu_load_mean=("cpu_load", "mean"),
        cpu_load_max=("cpu_load", "max"),

        cpu_temp_mean=("cpu_temp", "mean"),
        cpu_temp_max=("cpu_temp", "max"),
        cpu_temp_std=("cpu_temp", "std"),
        cpu_temp_first=("cpu_temp", "first"),
        cpu_temp_last=("cpu_temp", "last"),

        cpu_power_mean=("cpu_power", "mean"),
        cpu_power_max=("cpu_power", "max"),

        cpu_p_clock_mean=("cpu_p_clock", "mean"),
        cpu_p_clock_min=("cpu_p_clock", "min"),
        cpu_e_clock_mean=("cpu_e_clock", "mean"),

        cpu_fan_mean=("cpu_fan_rpm", "mean"),
        cpu_fan_max=("cpu_fan_rpm", "max"),
        pump_fan_mean=("pump_fan_rpm", "mean"),

        gpu_load_mean=("gpu_load", "mean"),
        gpu_temp_mean=("gpu_temp", "mean"),
        gpu_power_mean=("gpu_power", "mean"),

        ssd_temp_mean=("ssd_temp", "mean"),

        motherboard_system_temp_mean=("motherboard_system_temp", "mean"),
        motherboard_vrm_temp_mean=("motherboard_vrm_temp", "mean"),
        motherboard_pch_temp_mean=("motherboard_pch_temp", "mean"),
        cpu_socket_temp_mean=("cpu_socket_temp", "mean"),

        motherboard_vcore_mean=("motherboard_vcore", "mean"),
        motherboard_vcore_std=("motherboard_vcore", "std"),
    )

    features["cpu_temp_delta"] = (
        features["cpu_temp_last"] - features["cpu_temp_first"]
    )

    features["cpu_temp_per_load"] = (
        features["cpu_temp_mean"] / (features["cpu_load_mean"] + 1.0)
    )

    features["cpu_temp_per_power"] = (
        features["cpu_temp_mean"] / (features["cpu_power_mean"] + 1.0)
    )

    return features
