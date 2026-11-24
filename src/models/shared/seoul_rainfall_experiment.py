"""서울시 강우량 LightGBM 실험의 피처·평가행·시점 계약."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


BASE_PREFIXES = (
    "bod_mg_l_",
    "tn_mg_l_",
    "tp_mg_l_",
    "mean_temperature_c_",
)
RAINFALL_PREFIXES = {
    "C": ("seoul_rainfall_mm_",),
    "D": (
        "seoul_rainfall_mm_",
        "seoul_rainfall_lag_1d_mm_",
        "seoul_rainfall_lag_2d_mm_",
        "seoul_rainfall_lag_3d_mm_",
        "seoul_rainfall_sum_3d_mm_",
        "seoul_rainfall_sum_7d_mm_",
    ),
    "E": (
        "seoul_rainfall_mm_",
        "seoul_rainfall_lag_1d_mm_",
        "seoul_rainfall_lag_2d_mm_",
        "seoul_rainfall_lag_3d_mm_",
        "seoul_rainfall_sum_3d_mm_",
        "seoul_rainfall_sum_7d_mm_",
        "seoul_rainfall_max_1h_mm_",
    ),
}
EVALUATION_KEYS = ["series_id", "target_date", "target", "split"]


def load_model_checkpoint(path: Path, expected_features: list[str]) -> Any:
    """저장 모델의 피처 계약이 같을 때만 체크포인트 모델을 반환한다."""
    payload = joblib.load(path)
    if payload.get("features") != expected_features:
        raise ValueError(f"저장 모델의 피처 계약이 현재 실험과 다릅니다: {path}")
    return payload["model"]


def select_seoul_rainfall_features(frame: pd.DataFrame, variant: str) -> list[str]:
    """요청된 C·D·E 피처만 선택하고 ASOS 강수·결측표시는 제외한다."""
    variant = variant.upper()
    if variant not in RAINFALL_PREFIXES:
        raise ValueError(f"지원하지 않는 서울시 강우 모델: {variant}")
    prefixes = (*BASE_PREFIXES, *RAINFALL_PREFIXES[variant])
    return [
        column
        for column in frame.columns
        if re.search(r"_lag_\d+$", column) and column.startswith(prefixes)
    ]


def common_prediction_rows(
    predictions: pd.DataFrame,
    variants: Sequence[str],
) -> pd.DataFrame:
    """모든 입력 구성에 공통인 평가 키만 남기고 실제값 일관성을 검증한다."""
    selected = predictions.loc[predictions["variant"].isin(variants)].copy()
    duplicate = selected.duplicated([*EVALUATION_KEYS, "variant"], keep=False)
    if duplicate.any():
        raise ValueError("모델별 평가 키에 중복이 있습니다.")
    if set(selected["variant"].unique()) != set(variants):
        raise ValueError("비교에 필요한 모델 입력 구성이 모두 존재하지 않습니다.")

    counts = selected.groupby(EVALUATION_KEYS, dropna=False)["variant"].nunique()
    common_keys = counts.loc[counts.eq(len(variants))].index.to_frame(index=False)
    result = selected.merge(common_keys, on=EVALUATION_KEYS, how="inner", validate="many_to_one")
    actual_counts = result.groupby(EVALUATION_KEYS, dropna=False)["actual"].nunique(dropna=False)
    if actual_counts.gt(1).any():
        raise ValueError("동일 평가 키의 실제값이 모델 간 일치하지 않습니다.")
    return result.sort_values([*EVALUATION_KEYS, "variant"]).reset_index(drop=True)


def attach_input_day_rainfall(
    predictions: pd.DataFrame,
    rainfall: pd.DataFrame,
) -> pd.DataFrame:
    """t+1 목표 예측에 입력 시점 t의 센터별 강우 정보를 결합한다."""
    if rainfall.duplicated(["facility", "date"]).any():
        raise ValueError("센터별 일강우량 키에 중복이 있습니다.")
    result = predictions.copy()
    result["target_date"] = pd.to_datetime(result["target_date"])
    result["input_date"] = result["target_date"] - pd.Timedelta(days=1)
    rain = rainfall.copy().rename(columns={"date": "input_date"})
    rain["input_date"] = pd.to_datetime(rain["input_date"])
    before = len(result)
    result = result.merge(
        rain,
        on=["facility", "input_date"],
        how="left",
        validate="many_to_one",
    )
    if len(result) != before:
        raise RuntimeError("강우량 결합으로 평가행 수가 변경되었습니다.")
    return result
