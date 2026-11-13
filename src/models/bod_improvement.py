"""목표일 달력 피처와 Train-only 고농도 가중 실험 계약."""

from __future__ import annotations

import numpy as np
import pandas as pd


EVALUATION_KEYS = ["split", "series_id", "target_date"]


def add_target_calendar_features(
    frame: pd.DataFrame,
    *,
    include_season: bool,
    include_weekday: bool,
) -> pd.DataFrame:
    """사전에 아는 목표일에서 비중복 주기·요일 피처를 생성한다."""
    result = frame.copy()
    target_date = pd.to_datetime(result["target_date"])
    if include_season:
        angle = 2.0 * np.pi * (target_date.dt.dayofyear - 1) / 365.2425
        result["target_doy_sin"] = np.sin(angle).astype("float32")
        result["target_doy_cos"] = np.cos(angle).astype("float32")
    if include_weekday:
        weekday = target_date.dt.dayofweek
        for day in range(7):
            result[f"target_dow_{day}"] = weekday.eq(day).astype("float32")
    return result


def training_sample_weights(
    train: pd.DataFrame,
    target_column: str,
    *,
    threshold: float,
    high_weight: float,
) -> np.ndarray:
    """Train 목표값만으로 고농도 표본 가중치를 만든다."""
    if "split" not in train or not train["split"].eq("train").all():
        raise ValueError("Train 행만 가중치 계산에 사용할 수 있습니다")
    return np.where(train[target_column].to_numpy() >= threshold, high_weight, 1.0)


def assert_common_evaluation_rows(baseline: pd.DataFrame, candidate: pd.DataFrame) -> None:
    """두 예측의 평가키와 실제값이 완전히 같은지 검증한다."""
    for name, frame in (("baseline", baseline), ("candidate", candidate)):
        if frame.duplicated(EVALUATION_KEYS).any():
            raise ValueError(f"{name} 평가키 중복")
    left = baseline[EVALUATION_KEYS + ["actual"]].sort_values(EVALUATION_KEYS).reset_index(drop=True)
    right = candidate[EVALUATION_KEYS + ["actual"]].sort_values(EVALUATION_KEYS).reset_index(drop=True)
    if not left[EVALUATION_KEYS].equals(right[EVALUATION_KEYS]):
        raise ValueError("평가키 불일치")
    if not np.allclose(left["actual"], right["actual"], equal_nan=True):
        raise ValueError("모델 간 실제값 불일치")


def meets_replacement_goal(
    baseline_overall_mae: float,
    baseline_high_mae: float,
    candidate_overall_mae: float,
    candidate_high_mae: float,
    *,
    max_overall_degradation: float = 0.02,
    min_high_improvement: float = 0.10,
) -> bool:
    """사전 선언한 전체·고농도 Validation 목표를 동시에 판정한다."""
    epsilon = 1e-12
    return bool(
        candidate_overall_mae <= baseline_overall_mae * (1 + max_overall_degradation) + epsilon
        and candidate_high_mae <= baseline_high_mae * (1 - min_high_improvement) + epsilon
    )
