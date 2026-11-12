"""서울시 10분 강우량 CSV의 공통 로더와 보수적 일집계."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CsvLayout:
    encoding: str
    skiprows: int
    rainfall_column: str
    time_column: str


def _decode_sample(path: Path, size: int = 65536) -> tuple[str, str]:
    raw = path.read_bytes()[:size]
    candidates = []
    for priority, encoding in enumerate(("utf-8-sig", "cp949", "euc-kr")):
        text = raw.decode(encoding, errors="replace")
        header_score = sum(
            text.count(token)
            for token in ("강우량계명", "시간", "10분우량", "10분강우량")
        )
        candidates.append((header_score, -text.count("�"), -priority, encoding, text))
    _, _, _, encoding, text = max(candidates)
    return encoding, text


def detect_csv_layout(path: str | Path) -> CsvLayout:
    """인코딩과 실제 헤더 행을 대표 샘플에서 판별한다."""
    source = Path(path)
    encoding, text = _decode_sample(source)
    lines = text.splitlines()[:10]
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "강우량계명" in line and ("시간" in line or "자료수집 시각" in line)
            and ("10분우량" in line or "10분강우량" in line)
        ),
        -1,
    )
    if header_index < 0:
        raise ValueError(f"지원하지 않는 서울시 강우 CSV 헤더: {source.name}")
    rainfall_column = "10분강우량" if "10분강우량" in lines[header_index] else "10분우량"
    time_column = "자료수집 시각" if "자료수집 시각" in lines[header_index] else "시간"
    return CsvLayout(
        encoding=encoding,
        skiprows=header_index,
        rainfall_column=rainfall_column,
        time_column=time_column,
    )


def _station_id(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    return f"SEOUL_NAME_{digest}"


def load_normalized_rainfall_file(path: str | Path) -> pd.DataFrame:
    """한 파일을 원본 추적 가능한 서울시 10분 강우 표준 구조로 변환한다."""
    source = Path(path)
    layout = detect_csv_layout(source)
    frame = pd.read_csv(
        source,
        encoding=layout.encoding,
        encoding_errors="replace",
        skiprows=layout.skiprows,
        dtype=str,
        low_memory=False,
    )
    frame.columns = [str(column).replace("\ufeff", "").strip() for column in frame.columns]
    required = {"강우량계명", layout.time_column, layout.rainfall_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"필수 열 누락: {source.name}, {sorted(required - set(frame.columns))}")
    station_name = frame["강우량계명"].fillna("").astype(str).str.strip()
    raw_time = frame[layout.time_column].fillna("").astype(str).str.strip()
    raw_rain = frame[layout.rainfall_column].fillna("").astype(str).str.strip()
    timestamp = pd.to_datetime(raw_time, format="mixed", errors="coerce")
    timestamp_kst = timestamp.dt.tz_localize("Asia/Seoul", ambiguous="NaT", nonexistent="NaT")
    rainfall = pd.to_numeric(raw_rain, errors="coerce")
    invalid_time = timestamp_kst.isna()
    missing_rain = rainfall.isna()
    status = pd.Series("valid_positive", index=frame.index, dtype="object")
    status.loc[rainfall.eq(0)] = "valid_zero"
    status.loc[rainfall.lt(0)] = "negative"
    status.loc[missing_rain] = "missing_rainfall"
    status.loc[invalid_time] = "invalid_timestamp"
    status.loc[invalid_time & missing_rain] = "invalid_timestamp_and_missing"
    result = pd.DataFrame(
        {
            "timestamp_kst": timestamp_kst,
            "station_id": station_name.map(_station_id),
            "station_name": station_name,
            "station_code": frame.get("강우량계 코드", pd.Series(pd.NA, index=frame.index)).astype("string").str.strip(),
            "district_code": frame.get("구청 코드", pd.Series(pd.NA, index=frame.index)).astype("string").str.strip(),
            "district_name": frame.get("구청명", pd.Series(pd.NA, index=frame.index)).astype("string").str.strip(),
            "rainfall_10min_mm": rainfall,
            "rainfall_unit": "mm/10min",
            "source_file": str(source),
            "source_system": "서울시 강우량 관측망",
            "quality_status": status,
            "raw_timestamp": raw_time,
            "raw_rainfall": raw_rain,
        }
    )
    return result


def resolve_timestamp_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """동일 관측소·시각의 동일값은 한 번만 쓰고 값 충돌은 NaN으로 보류한다."""
    keys = ["station_id", "timestamp_kst"]
    ordered = frame.sort_values(keys, na_position="last").reset_index(drop=True)
    grouped = ordered.groupby(keys, dropna=False, sort=False)
    counts = grouped.size().rename("duplicate_record_count")
    distinct = grouped["rainfall_10min_mm"].nunique(dropna=True).rename("distinct_value_count")
    source_counts = grouped["source_file"].nunique(dropna=True).rename("duplicate_source_file_count")
    result = ordered.drop_duplicates(keys, keep="first").set_index(keys)
    result = result.join([counts, distinct, source_counts]).reset_index()
    result["duplicate_status"] = "unique"
    identical = result["duplicate_record_count"].gt(1) & result["distinct_value_count"].le(1)
    conflicting = result["distinct_value_count"].gt(1)
    result.loc[identical, "duplicate_status"] = "identical_duplicate"
    result.loc[conflicting, "duplicate_status"] = "conflicting_duplicate"
    result.loc[conflicting, "rainfall_10min_mm"] = np.nan
    result.loc[conflicting, "quality_status"] = "conflicting_duplicate"
    return result.drop(columns="distinct_value_count").sort_values(keys).reset_index(drop=True)


def aggregate_station_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """144개 10분 구간이 모두 수치일 때만 일강우와 최대 1시간 강우를 확정한다."""
    valid_time = frame.loc[frame["timestamp_kst"].notna()].copy()
    valid_time["date"] = valid_time["timestamp_kst"].dt.tz_localize(None).dt.normalize()
    rows = []
    for (station_id, date), group in valid_time.groupby(["station_id", "date"], sort=True):
        ordered = group.sort_values("timestamp_kst")
        interval_count = int(ordered["timestamp_kst"].nunique())
        numeric_count = int(ordered["rainfall_10min_mm"].notna().sum())
        conflict_count = int(ordered.get("duplicate_status", "unique").eq("conflicting_duplicate").sum())
        negative_count = int(ordered["rainfall_10min_mm"].lt(0).sum())
        complete = (
            interval_count == 144 and numeric_count == 144
            and conflict_count == 0 and negative_count == 0
        )
        daily_rain = float(ordered["rainfall_10min_mm"].sum()) if complete else np.nan
        max_1h = (
            float(ordered["rainfall_10min_mm"].rolling(6, min_periods=6).sum().max())
            if complete else np.nan
        )
        if conflict_count:
            quality = "conflicting_duplicate"
        elif negative_count:
            quality = "negative_value"
        elif not complete:
            quality = "incomplete_intervals"
        else:
            quality = "complete"
        rows.append(
            {
                "date": date,
                "station_id": station_id,
                "station_name": ordered["station_name"].iloc[0],
                "daily_rainfall_mm": daily_rain,
                "max_1h_rainfall_mm": max_1h,
                "rain_day": int(daily_rain > 0) if pd.notna(daily_rain) else pd.NA,
                "interval_count": interval_count,
                "numeric_interval_count": numeric_count,
                "expected_interval_count": 144,
                "duplicate_conflict_count": conflict_count,
                "negative_count": negative_count,
                "daily_quality_status": quality,
                "source_files": "|".join(sorted(ordered["source_file"].astype(str).unique())),
                "source_system": "서울시 강우량 관측망",
            }
        )
    columns = [
        "date", "station_id", "station_name", "daily_rainfall_mm",
        "max_1h_rainfall_mm", "rain_day", "interval_count",
        "numeric_interval_count", "expected_interval_count",
        "duplicate_conflict_count", "negative_count", "daily_quality_status",
        "source_files", "source_system",
    ]
    return pd.DataFrame(rows, columns=columns)


def add_grouped_rainfall_features(daily: pd.DataFrame) -> pd.DataFrame:
    """관측소별 날짜 순서로 과거 강우 파생변수를 만들며 미확정값을 전파한다."""
    required = {"date", "station_id", "daily_rainfall_mm"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"강우 파생변수 필수 열 누락: {sorted(missing)}")
    result = daily.copy().sort_values(["station_id", "date"]).reset_index(drop=True)
    grouped = result.groupby("station_id", sort=False)["daily_rainfall_mm"]
    for lag in (1, 2, 3):
        result[f"rainfall_lag_{lag}d_mm"] = grouped.shift(lag)
    result["rainfall_sum_3d_mm"] = grouped.transform(
        lambda values: values.rolling(3, min_periods=3).sum()
    )
    result["rainfall_sum_7d_mm"] = grouped.transform(
        lambda values: values.rolling(7, min_periods=7).sum()
    )
    result["rainfall_missing"] = result["daily_rainfall_mm"].isna().astype("int8")
    return result


def build_center_rainfall(daily: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """검토된 센터-관측소 대응표만 사용해 센터별 강우 테이블을 구성한다."""
    required_mapping = {"facility", "station_id", "selection_status"}
    missing = required_mapping - set(mapping.columns)
    if missing:
        raise ValueError(f"센터 대응표 필수 열 누락: {sorted(missing)}")
    if mapping["facility"].duplicated().any():
        raise ValueError("센터당 관측소는 하나만 지정해야 합니다.")
    audit_mapping = mapping.rename(columns={"station_name": "mapping_station_name"})
    result = audit_mapping.merge(daily, on="station_id", how="left", validate="one_to_many")
    if "mapping_station_name" in result:
        mismatch = result["mapping_station_name"].notna() & result["station_name"].notna() & result["mapping_station_name"].ne(result["station_name"])
        if mismatch.any():
            raise ValueError("센터 대응표 관측소명과 일자료 관측소명이 일치하지 않습니다.")
    return add_grouped_rainfall_features(result)


def join_center_rainfall(observations: pd.DataFrame, center_daily: pd.DataFrame) -> pd.DataFrame:
    """관측계통 행 수를 보존하며 같은 센터·예측시점 t의 강우만 결합한다."""
    keys = ["date", "facility"]
    if center_daily.duplicated(keys).any():
        raise ValueError("센터 일강우 키가 중복되어 결합할 수 없습니다.")
    rename = {
        "station_id": "seoul_rainfall_station_id",
        "station_name": "seoul_rainfall_station_name",
        "daily_rainfall_mm": "seoul_rainfall_mm",
        "max_1h_rainfall_mm": "seoul_rainfall_max_1h_mm",
        "rain_day": "seoul_rain_day",
        "daily_quality_status": "seoul_rainfall_quality_status",
        "rainfall_lag_1d_mm": "seoul_rainfall_lag_1d_mm",
        "rainfall_lag_2d_mm": "seoul_rainfall_lag_2d_mm",
        "rainfall_lag_3d_mm": "seoul_rainfall_lag_3d_mm",
        "rainfall_sum_3d_mm": "seoul_rainfall_sum_3d_mm",
        "rainfall_sum_7d_mm": "seoul_rainfall_sum_7d_mm",
        "rainfall_missing": "seoul_rainfall_missing",
    }
    available = keys + [column for column in rename if column in center_daily.columns]
    right = center_daily[available].rename(columns=rename)
    before = len(observations)
    result = observations.merge(right, on=keys, how="left", validate="many_to_one")
    if len(result) != before:
        raise AssertionError("강우 결합으로 관측계통 행 수가 변경되었습니다.")
    return result


def summarize_center_coverage(center_daily: pd.DataFrame) -> pd.DataFrame:
    """센터·연도별 일강우와 누적강우 가용 일수·비율을 계산한다."""
    frame = center_daily.copy()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    summary = (
        frame.groupby(["facility", "year"], dropna=False)
        .agg(
            calendar_days=("date", "size"),
            daily_available_days=("daily_rainfall_mm", "count"),
            max_1h_available_days=("max_1h_rainfall_mm", "count"),
            sum_3d_available_days=("rainfall_sum_3d_mm", "count"),
            sum_7d_available_days=("rainfall_sum_7d_mm", "count"),
        )
        .reset_index()
    )
    for stem in ("daily", "max_1h", "sum_3d", "sum_7d"):
        summary[f"{stem}_available_rate"] = (
            summary[f"{stem}_available_days"] / summary["calendar_days"]
        )
    return summary
