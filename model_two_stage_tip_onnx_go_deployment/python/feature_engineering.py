"""Build leakage-safe recorded-tip features. / 构造避免信息泄漏的记录小费特征。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import holidays
import numpy as np
import pandas as pd


TRAIN_END = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2025-01-01")
MAX_TIP = 100.0
MAX_FARE = 500.0
MAX_MILES = 100.0
MAX_SECONDS = 6 * 60 * 60

FEATURES = [
    "pickup_hour",
    "pickup_dow",
    "pickup_month",
    "pickup_is_weekend",
    "pickup_is_holiday",
    "pickup_h3_code",
    "dropoff_h3_code",
    "same_h3",
    "log_route_frequency",
    "dropoff_hour",
    "dropoff_dow",
    "crosses_date",
    "timestamp_duration_minutes",
    "log_fare",
    "log_miles",
    "log_seconds",
    "speed_mph",
]


@dataclass
class PreparedData:
    """Hold transformed train/test data and preprocessing state. / 保存训练测试数据与预处理状态。"""

    train: pd.DataFrame
    test: pd.DataFrame
    tip_cap: float
    pickup_h3_codes: dict[str, int]
    dropoff_h3_codes: dict[str, int]
    route_train_frequency: dict[str, int]
    holiday_dates: list[str]


def _ordered_codes(values: pd.Series) -> dict[str, int]:
    """Create deterministic first-seen category codes. / 按首次出现顺序生成稳定类别编码。"""
    return {str(value): index for index, value in enumerate(pd.unique(values.astype(str)))}


def _illinois_holidays(years: list[int]) -> tuple[set[date], list[str]]:
    """Return Illinois holiday dates in set and JSON forms. / 返回集合与 JSON 形式的伊利诺伊节假日。"""
    holiday_calendar = holidays.US(subdiv="IL", years=years)
    holiday_set = set(holiday_calendar.keys())
    return holiday_set, sorted(item.isoformat() for item in holiday_set)


def clean_raw_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's basic data-quality rules. / 应用 Notebook 中的基础质量规则。"""
    required = {
        "trip_start_timestamp",
        "trip_end_timestamp",
        "pickup_h3",
        "dropoff_h3",
        "trip_seconds",
        "trip_miles",
        "fare",
        "tip",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing columns / 缺少字段: {missing}")

    frame = raw.copy()
    frame["trip_start_timestamp"] = pd.to_datetime(
        frame["trip_start_timestamp"], errors="coerce"
    )
    frame["trip_end_timestamp"] = pd.to_datetime(
        frame["trip_end_timestamp"], errors="coerce"
    )
    for column in ["trip_seconds", "trip_miles", "fare", "tip"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    valid = (
        frame["trip_start_timestamp"].notna()
        & frame["trip_end_timestamp"].notna()
        & frame["pickup_h3"].notna()
        & frame["dropoff_h3"].notna()
        & frame["trip_seconds"].gt(0)
        & frame["trip_seconds"].le(MAX_SECONDS)
        & frame["trip_miles"].ge(0)
        & frame["trip_miles"].le(MAX_MILES)
        & frame["fare"].ge(0)
        & frame["fare"].le(MAX_FARE)
        & frame["tip"].ge(0)
        & frame["tip"].le(MAX_TIP)
        & frame["trip_start_timestamp"].lt(TEST_END)
    )
    frame = frame.loc[valid].copy()
    frame["pickup_h3"] = frame["pickup_h3"].astype(str)
    frame["dropoff_h3"] = frame["dropoff_h3"].astype(str)
    frame["route"] = frame["pickup_h3"] + " -> " + frame["dropoff_h3"]
    frame["has_tip"] = frame["tip"].gt(0).astype("int8")
    return frame


def _transform(
    frame: pd.DataFrame,
    pickup_h3_codes: dict[str, int],
    dropoff_h3_codes: dict[str, int],
    route_train_frequency: dict[str, int],
    holiday_dates: set[date],
) -> pd.DataFrame:
    """Transform cleaned trips into the fixed model feature table. / 将清理后行程转换为固定模型特征表。"""
    result = frame.copy()
    start = result["trip_start_timestamp"]
    end = result["trip_end_timestamp"]

    result["pickup_hour"] = start.dt.hour
    result["pickup_dow"] = start.dt.dayofweek
    result["pickup_month"] = start.dt.month
    result["pickup_is_weekend"] = result["pickup_dow"].ge(5).astype("int8")
    result["pickup_is_holiday"] = start.dt.date.map(
        lambda item: int(item in holiday_dates)
    )
    result["dropoff_hour"] = end.dt.hour
    result["dropoff_dow"] = end.dt.dayofweek
    result["crosses_date"] = (start.dt.date != end.dt.date).astype("int8")
    result["same_h3"] = result["pickup_h3"].eq(result["dropoff_h3"]).astype("int8")

    result["pickup_h3_code"] = (
        result["pickup_h3"].map(pickup_h3_codes).fillna(-1).astype("int32")
    )
    result["dropoff_h3_code"] = (
        result["dropoff_h3"].map(dropoff_h3_codes).fillna(-1).astype("int32")
    )
    route_frequency = result["route"].map(route_train_frequency).fillna(0)
    result["log_route_frequency"] = np.log1p(route_frequency.astype(float))

    result["timestamp_duration_minutes"] = (
        (end - start).dt.total_seconds() / 60.0
    ).clip(lower=0, upper=MAX_SECONDS / 60.0)
    result["speed_mph"] = np.where(
        result["trip_seconds"].gt(0),
        result["trip_miles"] / (result["trip_seconds"] / 3600.0),
        0.0,
    )
    result["speed_mph"] = (
        pd.Series(result["speed_mph"], index=result.index)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(0, 100)
    )
    result["log_fare"] = np.log1p(result["fare"].clip(lower=0))
    result["log_miles"] = np.log1p(result["trip_miles"].clip(lower=0))
    result["log_seconds"] = np.log1p(result["trip_seconds"].clip(lower=0))
    return result


def prepare_training_data(raw: pd.DataFrame) -> PreparedData:
    """Fit preprocessing on 2022-2023 and transform 2024. / 在 2022-2023 拟合预处理并转换 2024。"""
    cleaned = clean_raw_rows(raw)
    train = cleaned.loc[cleaned["trip_start_timestamp"].lt(TRAIN_END)].copy()
    test = cleaned.loc[
        cleaned["trip_start_timestamp"].ge(TRAIN_END)
        & cleaned["trip_start_timestamp"].lt(TEST_END)
    ].copy()

    if train.empty or test.empty:
        raise ValueError("Training or test split is empty. / 训练集或测试集为空。")

    tip_cap = min(MAX_TIP, float(train["tip"].quantile(0.995)))
    train = train.loc[train["tip"].le(tip_cap)].copy()
    test = test.loc[test["tip"].le(tip_cap)].copy()

    pickup_codes = _ordered_codes(train["pickup_h3"])
    dropoff_codes = _ordered_codes(train["dropoff_h3"])
    route_frequency = {
        str(key): int(value) for key, value in train["route"].value_counts().items()
    }
    holiday_set, holiday_strings = _illinois_holidays([2022, 2023, 2024])

    train = _transform(
        train, pickup_codes, dropoff_codes, route_frequency, holiday_set
    )
    test = _transform(
        test, pickup_codes, dropoff_codes, route_frequency, holiday_set
    )
    return PreparedData(
        train=train,
        test=test,
        tip_cap=tip_cap,
        pickup_h3_codes=pickup_codes,
        dropoff_h3_codes=dropoff_codes,
        route_train_frequency=route_frequency,
        holiday_dates=holiday_strings,
    )


def preprocessing_json(data: PreparedData) -> dict[str, Any]:
    """Create serializable preprocessing metadata. / 创建可序列化的预处理元数据。"""
    return {
        "pickup_h3_codes": data.pickup_h3_codes,
        "dropoff_h3_codes": data.dropoff_h3_codes,
        "route_train_frequency": data.route_train_frequency,
        "holiday_dates": data.holiday_dates,
        "unknown_h3_code": -1,
        "unknown_route_frequency": 0,
    }
