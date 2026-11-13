"""원본 전처리를 독립적으로 표본 재계산하는 순수 함수."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def classify_split(target_date: pd.Timestamp) -> str:
    date = pd.Timestamp(target_date).normalize()
    if date <= pd.Timestamp("2022-12-31"):
        return "train"
    if date <= pd.Timestamp("2024-12-31"):
        return "validation"
    if date <= pd.Timestamp("2026-09-30"):
        return "test"
    return "outside"


def aggregate_station_day(frame: pd.DataFrame) -> dict[str, object]:
    """10분 구간값 144개를 달력일 기준으로 독립 집계한다."""
    required = {"timestamp", "rainfall_mm"}
    if missing := required - set(frame.columns):
        raise ValueError(f"필수 열 누락: {sorted(missing)}")
    work = frame[["timestamp", "rainfall_mm"]].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work["rainfall_mm"] = pd.to_numeric(work["rainfall_mm"], errors="coerce")
    valid_time = work.dropna(subset=["timestamp"])
    grouped = valid_time.groupby("timestamp", sort=True)["rainfall_mm"]
    counts = grouped.size()
    nunique = grouped.nunique(dropna=False)
    duplicate_same = int(((counts - 1).where((counts > 1) & nunique.eq(1), 0)).sum())
    duplicate_conflict = int(((counts - 1).where((counts > 1) & nunique.gt(1), 0)).sum())
    unique = grouped.first().sort_index()
    complete = len(unique) == 144 and unique.notna().all() and duplicate_conflict == 0
    status = "conflict" if duplicate_conflict else ("confirmed" if complete else "incomplete")
    daily = float(unique.sum()) if status == "confirmed" else np.nan
    maximum_hour = float(unique.rolling(6, min_periods=6).sum().max()) if status == "confirmed" else np.nan
    return {
        "raw_row_count": int(len(frame)),
        "unique_interval_count": int(len(unique)),
        "numeric_interval_count": int(unique.notna().sum()),
        "duplicate_same_count": duplicate_same,
        "duplicate_conflict_count": duplicate_conflict,
        "negative_count": int(unique.lt(0).sum()),
        "independent_daily_rainfall_mm": daily,
        "independent_max_1h_mm": maximum_hour,
        "audit_status": status,
    }


def reconstruct_lags(
    observations: pd.DataFrame,
    series_id: str,
    target_date: pd.Timestamp,
    value_column: str,
    window_days: int,
) -> list[float]:
    subset = observations.loc[observations["series_id"].eq(series_id), ["date", value_column]].copy()
    subset["date"] = pd.to_datetime(subset["date"])
    if subset.date.duplicated().any():
        raise ValueError("동일 계통 날짜 중복이 있습니다")
    values = subset.set_index("date")[value_column]
    target = pd.Timestamp(target_date).normalize()
    return [values.get(target - pd.Timedelta(days=lag + 1), np.nan) for lag in range(window_days)]


def join_rainfall_reference(enriched: pd.DataFrame, rainfall: pd.DataFrame) -> pd.DataFrame:
    """ASOS station_id를 보존하며 서울시 강우 관측소 ID를 별도 이름으로 결합한다."""
    reference = rainfall.rename(columns={"station_id": "source_rainfall_station_id"})
    return enriched.merge(reference, on=["date", "facility"], how="left", validate="many_to_one")
