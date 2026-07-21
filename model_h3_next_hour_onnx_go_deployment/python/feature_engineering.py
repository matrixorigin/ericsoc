"""Feature engineering for the H3 next-hour demand model. / H3 下一小时客流模型的特征工程。"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRAIN_END = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2025-01-01")

CALENDAR_FEATURES = [
    "cell_code", "hour", "dow_mon0", "month", "day_of_year", "is_weekend",
    "is_holiday", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
]

DEMAND_FEATURES = [
    "pickup_now", "pickup_lag_1h", "pickup_lag_2h", "pickup_lag_3h", "pickup_lag_6h",
    "pickup_lag_24h", "pickup_lag_168h", "pickup_roll_mean_3h", "pickup_roll_mean_6h",
    "pickup_roll_mean_24h", "pickup_roll_mean_168h", "pickup_roll_std_3h",
    "pickup_roll_std_6h", "pickup_roll_std_24h", "pickup_roll_std_168h",
    "expected_pickup", "std_pickup", "pickup_z", "abs_z", "delta_z_1h",
    "z_velocity_3h", "z_acceleration", "pickup_pct_change_1h", "peak_age_hours",
]

FLOW_FEATURES = [
    "dropoff_now", "dropoff_lag_1h", "dropoff_lag_2h", "dropoff_lag_3h", "dropoff_lag_6h",
    "dropoff_lag_24h", "dropoff_lag_168h", "dropoff_z", "net_flow",
    "dropoff_to_pickup_ratio", "net_flow_roll_3h", "net_flow_roll_6h", "net_flow_roll_24h",
]

SPATIAL_FEATURES = [
    "neighbor_pickup_count", "active_neighbor_cells", "expected_neighbor", "std_neighbor",
    "neighbor_z", "neighbor_z_change_1h", "neighbor_pickup_lag_1h",
    "neighbor_pickup_lag_2h", "neighbor_pickup_lag_3h", "neighbor_z_lag_1h",
    "neighbor_z_lag_2h",
]

FEATURES = CALENDAR_FEATURES + DEMAND_FEATURES + FLOW_FEATURES + SPATIAL_FEATURES


def _safe_std(series: pd.Series) -> pd.Series:
    """Replace unusable standard deviations with one. / 将不可用标准差替换为 1。"""
    return series.replace(0, np.nan).fillna(1.0).clip(lower=1.0)


def build_model_frame(hourly: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Create leakage-safe h+1 features from a complete H3-hour grid. / 从完整 H3 小时网格构造无泄漏的 h+1 特征。"""
    frame = hourly.copy()
    frame["d"] = pd.to_datetime(frame["d"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["h3", "timestamp"]).reset_index(drop=True)

    train = frame[frame["d"] < TRAIN_END]
    baseline = (
        train.groupby(["h3", "dow_sun0", "hour"])
        .agg(
            expected_pickup=("pickup_count", "mean"),
            std_pickup=("pickup_count", "std"),
            expected_dropoff=("dropoff_count", "mean"),
            std_dropoff=("dropoff_count", "std"),
            expected_neighbor=("neighbor_pickup_count", "mean"),
            std_neighbor=("neighbor_pickup_count", "std"),
        )
        .reset_index()
    )
    for column in ["std_pickup", "std_dropoff", "std_neighbor"]:
        baseline[column] = _safe_std(baseline[column])

    frame = frame.merge(baseline, on=["h3", "dow_sun0", "hour"], how="left")
    frame = frame.sort_values(["h3", "timestamp"]).reset_index(drop=True)
    grouped = frame.groupby("h3", sort=False)

    frame["pickup_z"] = (frame["pickup_count"] - frame["expected_pickup"]) / frame["std_pickup"]
    frame["dropoff_z"] = (frame["dropoff_count"] - frame["expected_dropoff"]) / frame["std_dropoff"]
    frame["neighbor_z"] = (
        (frame["neighbor_pickup_count"] - frame["expected_neighbor"]) / frame["std_neighbor"]
    )
    frame["abs_z"] = frame["pickup_z"].abs()
    frame["delta_z_1h"] = grouped["pickup_z"].diff()
    frame["z_velocity_3h"] = frame.groupby("h3", sort=False)["delta_z_1h"].transform(
        lambda series: series.rolling(3, min_periods=1).mean()
    )
    frame["z_acceleration"] = frame.groupby("h3", sort=False)["delta_z_1h"].diff()
    frame["pickup_pct_change_1h"] = grouped["pickup_count"].pct_change(fill_method=None)
    frame["pickup_pct_change_1h"] = (
        frame["pickup_pct_change_1h"].replace([np.inf, -np.inf], np.nan).clip(-5, 5)
    )

    is_high = ((frame["pickup_z"] >= 3.0) & (frame["pickup_count"] >= 100)).astype("int8")
    breaks = (
        (is_high != is_high.groupby(frame["h3"]).shift(1))
        | (frame["timestamp"] != grouped["timestamp"].shift(1) + pd.Timedelta(hours=1))
    ).astype("int32")
    run_id = breaks.groupby(frame["h3"]).cumsum()
    frame["peak_age_hours"] = 0
    high_mask = is_high == 1
    frame.loc[high_mask, "peak_age_hours"] = (
        frame.loc[high_mask].groupby(["h3", run_id[high_mask]]).cumcount().add(1).astype("int16")
    )

    cells = sorted(frame["h3"].astype(str).unique().tolist())
    cell_codes = {cell: index for index, cell in enumerate(cells)}
    frame["cell_code"] = frame["h3"].map(cell_codes).astype("int16")
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["dow_mon0"] / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["dow_mon0"] / 7)
    frame["doy_sin"] = np.sin(2 * np.pi * frame["day_of_year"] / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * frame["day_of_year"] / 365.25)

    frame["pickup_now"] = frame["pickup_count"]
    frame["dropoff_now"] = frame["dropoff_count"]
    for lag in [1, 2, 3, 6, 24, 168]:
        frame[f"pickup_lag_{lag}h"] = grouped["pickup_count"].shift(lag)
        frame[f"dropoff_lag_{lag}h"] = grouped["dropoff_count"].shift(lag)

    for window in [3, 6, 24, 168]:
        minimum_periods = max(2, window // 3)
        frame[f"pickup_roll_mean_{window}h"] = frame.groupby("h3")["pickup_count"].transform(
            lambda series, size=window, minimum=minimum_periods: series.shift(1)
            .rolling(size, min_periods=minimum)
            .mean()
        )
        frame[f"pickup_roll_std_{window}h"] = frame.groupby("h3")["pickup_count"].transform(
            lambda series, size=window, minimum=minimum_periods: series.shift(1)
            .rolling(size, min_periods=minimum)
            .std()
        )

    frame["net_flow"] = frame["dropoff_count"] - frame["pickup_count"]
    frame["dropoff_to_pickup_ratio"] = (
        frame["dropoff_count"] / frame["pickup_count"].replace(0, np.nan)
    )
    for window in [3, 6, 24]:
        frame[f"net_flow_roll_{window}h"] = frame.groupby("h3")["net_flow"].transform(
            lambda series, size=window: series.shift(1).rolling(size, min_periods=2).sum()
        )

    frame["neighbor_z_change_1h"] = frame.groupby("h3")["neighbor_z"].diff()
    for lag in [1, 2, 3]:
        frame[f"neighbor_pickup_lag_{lag}h"] = grouped["neighbor_pickup_count"].shift(lag)
    for lag in [1, 2]:
        frame[f"neighbor_z_lag_{lag}h"] = frame.groupby("h3")["neighbor_z"].shift(lag)

    frame["target_pickup_h1"] = grouped["pickup_count"].shift(-1)
    frame["target_timestamp_h1"] = frame["timestamp"] + pd.Timedelta(hours=1)
    frame["target_dst_h1"] = grouped["is_dst_missing_hour"].shift(-1)
    frame["naive_current"] = frame["pickup_count"]
    frame["naive_weekly_h1"] = grouped["pickup_count"].shift(167)
    return frame, cell_codes
