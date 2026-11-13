"""동일 계통의 과거 SS를 기존 예측 윈도에 안전하게 결합하는 도구."""

from __future__ import annotations

import pandas as pd


def ss_feature_columns(window_days: int) -> list[str]:
    if window_days < 1:
        raise ValueError("window_days는 1 이상이어야 합니다")
    return [f"ss_mg_l_lag_{lag}" for lag in range(window_days)]


def attach_ss_lags(
    base: pd.DataFrame,
    observations: pd.DataFrame,
    window_days: int,
) -> pd.DataFrame:
    """기존 평가행을 보존하며 target_date 이전 SS lag를 동일 계통에서만 붙인다."""
    required_base = {"series_id", "target_date"}
    required_observations = {"series_id", "date", "ss_mg_l"}
    if missing := required_base - set(base.columns):
        raise ValueError(f"기준 데이터 열이 없습니다: {sorted(missing)}")
    if missing := required_observations - set(observations.columns):
        raise ValueError(f"SS 관측 열이 없습니다: {sorted(missing)}")
    if observations.duplicated(["series_id", "date"]).any():
        raise ValueError("SS 관측에 series_id × date 중복 키가 있습니다")

    result = base.copy()
    result["target_date"] = pd.to_datetime(result["target_date"])
    lookup = observations[["series_id", "date", "ss_mg_l"]].copy()
    lookup["date"] = pd.to_datetime(lookup["date"])
    lookup = lookup.set_index(["series_id", "date"])["ss_mg_l"]
    series_values = result["series_id"].to_numpy()
    for lag, column in enumerate(ss_feature_columns(window_days)):
        dates = result["target_date"] - pd.to_timedelta(lag + 1, unit="D")
        keys = pd.MultiIndex.from_arrays([series_values, dates])
        result[column] = lookup.reindex(keys).to_numpy(dtype=float)
    return result
