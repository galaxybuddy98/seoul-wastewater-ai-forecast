"""처리장·유입수 관측 계통별 표준 데이터셋 구성."""

from __future__ import annotations

import hashlib

import pandas as pd

from src.preprocessing.flow import add_treatment_total_hypothesis


class DatasetError(ValueError):
    """명시적 규칙 없이 데이터가 손실·증식될 위험이 있을 때 발생한다."""


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join("" if pd.isna(part) else str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def build_observation_table(quality: pd.DataFrame) -> pd.DataFrame:
    """유입수 원본 행을 관측 계통 단위로 보존한다."""
    required = {"date", "facility", "plant", "stream_type", "bod_mg_l", "tn_mg_l", "tp_mg_l"}
    missing = sorted(required.difference(quality.columns))
    if missing:
        raise DatasetError(f"수질 필수 열 누락: {missing}")

    result = quality.loc[quality["stream_type"].eq("유입수")].copy()
    result["date"] = pd.to_datetime(result["date"])
    key = ["date", "facility", "plant", "stream_type"]
    duplicates = result.duplicated(key, keep=False)
    if duplicates.any():
        raise DatasetError(f"유입 관측 복합키 중복 {int(duplicates.sum())}행: 평균·임의 선택 금지")

    result["source_plant"] = result["plant"]
    result["center_id"] = result["facility"].map(lambda value: _stable_id("center", value))
    result["series_id"] = result.apply(
        lambda row: _stable_id("series", row["facility"], row["plant"], row["stream_type"]),
        axis=1,
    )
    result["observation_scope"] = "원본 유입 관측 계통(대표성 미확인)"
    return result.sort_values(["series_id", "date"]).reset_index(drop=True)


def join_exact_flow(observations: pd.DataFrame, flow: pd.DataFrame) -> pd.DataFrame:
    """날짜·센터·처리장이 정확히 같은 처리량만 연결한다."""
    key = ["date", "facility", "plant"]
    if flow.duplicated(key, keep=False).any():
        raise DatasetError("처리량 복합키 중복으로 안전한 연결이 불가능합니다")

    prepared = flow.copy()
    if "total_treatment_volume_hypothesis" not in prepared.columns:
        prepared = add_treatment_total_hypothesis(prepared)
    value_columns = [
        "primary_treatment_volume",
        "secondary_treatment_volume",
        "total_treatment_volume_hypothesis",
    ]
    before = observations[["bod_mg_l", "tn_mg_l", "tp_mg_l"]].reset_index(drop=True)
    result = observations.merge(
        prepared[key + value_columns],
        on=key,
        how="left",
        validate="many_to_one",
        indicator="_flow_merge",
    )
    if len(result) != len(observations):
        raise DatasetError("처리량 결합으로 행 수가 증가했습니다")
    after = result[["bod_mg_l", "tn_mg_l", "tp_mg_l"]].reset_index(drop=True)
    if not before.equals(after):
        raise DatasetError("처리량 결합 중 원본 수질 값이 변경됐습니다")
    result["flow_link_status"] = result["_flow_merge"].map(
        {"both": "정확키 연결", "left_only": "대응 불명확"}
    )
    return result.drop(columns="_flow_merge")


def join_weather(observations: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """센터·날짜가 같은 기상만 many-to-one으로 연결한다."""
    key = ["date", "facility"]
    if weather.duplicated(key, keep=False).any():
        raise DatasetError("기상 복합키 중복으로 안전한 연결이 불가능합니다")
    value_columns = [
        column for column in weather.columns
        if column not in key
    ]
    result = observations.merge(
        weather[key + value_columns], on=key, how="left", validate="many_to_one", indicator="_weather_merge"
    )
    if len(result) != len(observations):
        raise DatasetError("기상 결합으로 행 수가 증가했습니다")
    result["weather_link_status"] = result["_weather_merge"].map(
        {"both": "연결", "left_only": "기상 미수집·결측"}
    )
    return result.drop(columns="_weather_merge")


def add_weather_history(weather: pd.DataFrame) -> pd.DataFrame:
    """센터별 강수 과거값과 t일까지의 누적값을 만들며 결측을 보존한다."""
    result = weather.sort_values(["facility", "date"]).copy()
    grouped = result.groupby("facility", sort=False)["precipitation_mm"]
    for lag in (1, 2, 3):
        result[f"precipitation_lag_{lag}d"] = grouped.shift(lag)
    result["precipitation_sum_3d"] = grouped.transform(
        lambda values: values.rolling(3, min_periods=3).sum()
    )
    result["precipitation_sum_7d"] = grouped.transform(
        lambda values: values.rolling(7, min_periods=7).sum()
    )
    return result.reset_index(drop=True)


def assign_target_split(target_dates: pd.Series) -> pd.Series:
    """예측 대상일 기준 고정 시간 분할."""
    dates = pd.to_datetime(target_dates)
    return pd.Series(
        pd.cut(
            dates,
            bins=[
                pd.Timestamp.min,
                pd.Timestamp("2022-12-31 23:59:59"),
                pd.Timestamp("2024-12-31 23:59:59"),
                pd.Timestamp("2026-09-30 23:59:59"),
            ],
            labels=["train", "validation", "test"],
        ),
        index=target_dates.index,
        dtype="object",
    )
