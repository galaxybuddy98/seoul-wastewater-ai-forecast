"""서울시 원본 열을 의미 손실 없이 표준 열로 변환한다."""

from __future__ import annotations

import pandas as pd


class SchemaError(ValueError):
    """필수 원본 열이 없거나 값의 형식이 잘못된 경우."""


QUALITY_COLUMNS = {
    "측정일자": "date",
    "물재생센터명칭": "facility",
    "처리장구분": "plant",
    "계통구분": "stream_type",
    "BOD": "bod_mg_l",
    "TOC(2021년 1월 이전자료는 COD)": "toc_or_cod_value",
    "SS": "ss_mg_l",
    "T-N": "tn_mg_l",
    "TP": "tp_mg_l",
    "PH": "ph",
    "총대장군균": "total_coliform",
}

FLOW_COLUMNS = {
    "측정일자": "date",
    "물재생센터명칭": "facility",
    "처리장구분": "plant",
    "1차하수처리량": "primary_treatment_volume",
    "2차하수처리량": "secondary_treatment_volume",
}


def _require_columns(frame: pd.DataFrame, mapping: dict[str, str]) -> None:
    missing = [column for column in mapping if column not in frame.columns]
    if missing:
        raise SchemaError(f"필수 원본 열 누락: {', '.join(missing)}")


def _parse_date(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        bad_count = int(parsed.isna().sum())
        raise SchemaError(f"측정일자 변환 실패: {bad_count}건")
    return parsed


def normalize_quality_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """수질 원본을 표준 열로 복사하고 COD/TOC의 시기 의미를 보존한다."""
    _require_columns(frame, QUALITY_COLUMNS)
    normalized = frame.loc[:, list(QUALITY_COLUMNS)].rename(columns=QUALITY_COLUMNS).copy()
    normalized["date"] = _parse_date(normalized["date"])
    numeric_columns = ["bod_mg_l", "toc_or_cod_value", "ss_mg_l", "tn_mg_l", "tp_mg_l", "ph", "total_coliform"]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["toc_or_cod_label"] = normalized["date"].map(
        lambda value: "COD" if value < pd.Timestamp("2021-01-01") else "TOC"
    )
    normalized["source_tp_column"] = "TP"
    return normalized


def normalize_flow_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """1차·2차 처리량을 서로 다른 변수로 유지해 정규화한다."""
    _require_columns(frame, FLOW_COLUMNS)
    normalized = frame.loc[:, list(FLOW_COLUMNS)].rename(columns=FLOW_COLUMNS).copy()
    normalized["date"] = _parse_date(normalized["date"])
    for column in ("primary_treatment_volume", "secondary_treatment_volume"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized
