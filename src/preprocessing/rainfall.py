"""ASOS 일강수량 원본을 보존하며 보수적으로 전처리한다."""

from __future__ import annotations

import pandas as pd


def select_hourly_intervals_for_days(
    hourly: pd.DataFrame, start_date: str, end_date: str
) -> pd.DataFrame:
    """일계 00:01~24:00에 대응하는 (시작일 00시, 종료 다음날 00시] 구간을 고른다."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    timestamp = pd.to_datetime(hourly["timestamp"])
    return hourly.loc[timestamp.gt(start) & timestamp.le(end)].copy()


def classify_daily_rainfall(frame: pd.DataFrame) -> pd.DataFrame:
    """수치 관측만 확정값으로 사용하고 의미가 미확인인 공란은 NaN으로 둔다."""
    result = frame.copy()
    raw = pd.to_numeric(result["precipitation_mm"], errors="coerce")
    blank = raw.isna()
    result["precipitation_raw_mm"] = raw
    result["precipitation_raw_blank"] = blank
    result["precipitation_processed_mm"] = raw
    result["precipitation_missing"] = blank.astype("int8")
    result["precipitation_processing_reason"] = "수치_관측값_보존"
    result.loc[blank, "precipitation_processing_reason"] = "공란_의미_미확인"
    result["precipitation_processing_evidence"] = "ASOS_API_sumRn"
    result.loc[blank, "precipitation_processing_evidence"] = (
        "공식_공란규칙_미확인_및_지속시간_양수_반례"
    )
    return result


def add_rainfall_features(frame: pd.DataFrame) -> pd.DataFrame:
    """미확인 결측을 0으로 간주하지 않는 과거 강수 파생변수를 만든다."""
    result = frame.sort_values("date").copy()
    rain = result["precipitation_processed_mm"]
    for lag in (1, 2, 3):
        result[f"precipitation_lag_{lag}d"] = rain.shift(lag)
    result["precipitation_sum_3d"] = rain.rolling(3, min_periods=3).sum()
    result["precipitation_sum_7d"] = rain.rolling(7, min_periods=7).sum()
    if "precipitation_missing" not in result:
        result["precipitation_missing"] = rain.isna().astype("int8")
    return result


def reconstruct_daily_rainfall(
    daily: pd.DataFrame, hourly: pd.DataFrame
) -> pd.DataFrame:
    """일자료를 우선하고 시간자료가 직접 입증하는 범위에서만 공란을 복원한다."""
    source = daily[["date", "precipitation_mm"]].copy()
    source["date"] = pd.to_datetime(source["date"]).dt.normalize()
    source = source.rename(columns={"precipitation_mm": "original_daily_precipitation_mm"})

    values = hourly.copy()
    values["date"] = pd.to_datetime(values["date"]).dt.normalize()
    values["rainfall_qc_flag"] = values["rainfall_qc_flag"].fillna("").astype(str)
    values["_numeric"] = values["hourly_precipitation_mm"].notna()
    values["_positive"] = values["hourly_precipitation_mm"].gt(0)
    values["_zero"] = values["hourly_precipitation_mm"].eq(0)
    values["_qc_error"] = values["rainfall_qc_flag"].eq("1")
    values["_qc_missing"] = values["rainfall_qc_flag"].eq("9")
    summary = values.groupby("date", as_index=False).agg(
        hourly_row_count=("timestamp", "size"),
        hourly_unique_timestamp_count=("timestamp", "nunique"),
        hourly_numeric_count=("_numeric", "sum"),
        hourly_positive_count=("_positive", "sum"),
        hourly_zero_count=("_zero", "sum"),
        hourly_qc_error_count=("_qc_error", "sum"),
        hourly_qc_missing_count=("_qc_missing", "sum"),
        hourly_precipitation_sum_mm=("hourly_precipitation_mm", lambda x: x.sum(min_count=1)),
    )
    result = source.merge(summary, on="date", how="left", validate="one_to_one")
    count_columns = [column for column in result if column.endswith("_count")]
    result[count_columns] = result[count_columns].fillna(0).astype(int)
    result["hourly_duplicate_timestamp_count"] = (
        result["hourly_row_count"] - result["hourly_unique_timestamp_count"]
    )

    known_daily = result["original_daily_precipitation_mm"].notna()
    no_error = result["hourly_qc_error_count"].eq(0) & result[
        "hourly_duplicate_timestamp_count"
    ].eq(0)
    recovered_positive = (~known_daily) & no_error & result["hourly_positive_count"].gt(0)
    confirmed_zero = (
        (~known_daily)
        & no_error
        & result["hourly_row_count"].gt(0)
        & result["hourly_numeric_count"].eq(result["hourly_row_count"])
        & result["hourly_zero_count"].eq(result["hourly_row_count"])
    )
    observation_error = (~known_daily) & (
        result["hourly_qc_error_count"].gt(0)
        | result["hourly_duplicate_timestamp_count"].gt(0)
        | result["hourly_row_count"].eq(0)
    )

    result["validated_precipitation_mm"] = result["original_daily_precipitation_mm"]
    result.loc[recovered_positive, "validated_precipitation_mm"] = result.loc[
        recovered_positive, "hourly_precipitation_sum_mm"
    ]
    result.loc[confirmed_zero, "validated_precipitation_mm"] = 0.0
    result["rainfall_status"] = "C_미량강수_또는_미확인"
    result.loc[known_daily, "rainfall_status"] = "A_일자료_확인"
    result.loc[recovered_positive, "rainfall_status"] = "A_시간자료_복원"
    result.loc[confirmed_zero, "rainfall_status"] = "B_시간자료_무강수_확인"
    result.loc[observation_error, "rainfall_status"] = "D_관측오류_또는_누락"
    result["rainfall_decision_reason"] = "시간 rn 공란의 의미가 미확인되어 NaN 유지"
    result.loc[known_daily, "rainfall_decision_reason"] = "공식 일자료 sumRn 수치 보존"
    result.loc[recovered_positive, "rainfall_decision_reason"] = "시간 rn 양수 관측 합계로 복원"
    result.loc[confirmed_zero, "rainfall_decision_reason"] = "반환된 모든 시간 rn이 QC 정상 0.0"
    result.loc[observation_error, "rainfall_decision_reason"] = "QC 오류·중복·시간자료 부재"
    return result
