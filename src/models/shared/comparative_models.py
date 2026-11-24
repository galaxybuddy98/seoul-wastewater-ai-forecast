"""XGBoost·LSTM 공정 비교를 위한 전처리와 평가행 계약."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.models.lightgbm.lightgbm_baseline import regression_metrics

EVALUATION_KEYS = ["series_id", "target_date", "target", "split"]


@dataclass(frozen=True)
class SequencePreprocessor:
    """Train 통계만 보존하는 시퀀스 결측 대치·표준화 변환기."""

    medians: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    missing_indicator_indices: tuple[int, ...]

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64).copy()
        if array.ndim != 3 or array.shape[2] != len(self.medians):
            raise ValueError("시퀀스의 피처 차원이 Train 전처리 계약과 다릅니다.")
        indicators = [np.isnan(array[:, :, index]).astype(np.float32) for index in self.missing_indicator_indices]
        for index, median in enumerate(self.medians):
            missing = np.isnan(array[:, :, index])
            array[:, :, index][missing] = median
        array = ((array - self.means) / self.stds).astype(np.float32)
        if indicators:
            array = np.concatenate([array, np.stack(indicators, axis=2)], axis=2)
        return array


def fit_sequence_preprocessor(
    train_values: np.ndarray,
    missing_indicator_indices: Sequence[int] = (),
) -> SequencePreprocessor:
    """Train 시퀀스에만 적합한 결측 대치·표준화 통계를 계산한다."""
    train = np.asarray(train_values, dtype=np.float64)
    if train.ndim != 3:
        raise ValueError("시퀀스 입력은 [표본, 시간, 피처] 3차원이어야 합니다.")
    indices = tuple(int(index) for index in missing_indicator_indices)
    if any(index < 0 or index >= train.shape[2] for index in indices):
        raise ValueError("결측 표시 피처 인덱스가 범위를 벗어났습니다.")
    medians = np.nanmedian(train, axis=(0, 1))
    if np.isnan(medians).any():
        raise ValueError("Train 전체가 결측인 피처는 대치 통계를 계산할 수 없습니다.")
    imputed = train.copy()
    for index, median in enumerate(medians):
        missing = np.isnan(imputed[:, :, index])
        imputed[:, :, index][missing] = median
    means = imputed.mean(axis=(0, 1))
    stds = imputed.std(axis=(0, 1))
    stds[stds == 0] = 1.0
    return SequencePreprocessor(medians, means, stds, indices)


def encode_series_ids(
    train_ids: np.ndarray,
    *other_ids: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Train에서 확정한 계통 범주만 정수 인덱스로 변환한다."""
    categories = sorted(pd.unique(np.asarray(train_ids)).tolist())
    mapping = {category: index for index, category in enumerate(categories)}
    result = []
    for values in (train_ids, *other_ids):
        unknown = sorted(set(pd.unique(np.asarray(values))) - set(categories))
        if unknown:
            raise ValueError(f"Train에 없는 계통이 평가 구간에 있습니다: {unknown}")
        result.append(np.asarray([mapping[value] for value in values], dtype=np.int64))
    return tuple(result)


def tabular_to_sequence(
    frame: pd.DataFrame,
    feature_stems: Sequence[str],
    window_days: int,
) -> np.ndarray:
    """lag_0이 최신인 표형 피처를 오래된 날→최신 날 시퀀스로 바꾼다."""
    ordered_columns = [
        [f"{stem}_lag_{lag}" for stem in feature_stems]
        for lag in range(window_days - 1, -1, -1)
    ]
    missing = [column for columns in ordered_columns for column in columns if column not in frame]
    if missing:
        raise ValueError(f"시퀀스 변환에 필요한 lag 열이 없습니다: {missing[:5]}")
    steps = [frame[columns].to_numpy(dtype=np.float64) for columns in ordered_columns]
    return np.stack(steps, axis=1)


def common_algorithm_rows(
    predictions: pd.DataFrame,
    algorithms: Sequence[str],
    require_identical_keys: bool = False,
) -> pd.DataFrame:
    """모든 알고리즘에 공통인 평가 키만 남기고 실제값 일관성을 확인한다."""
    selected = predictions.loc[predictions["algorithm"].isin(algorithms)].copy()
    duplicate = selected.duplicated([*EVALUATION_KEYS, "algorithm"], keep=False)
    if duplicate.any():
        raise ValueError("알고리즘별 평가 키에 중복이 있습니다.")
    if set(selected["algorithm"].unique()) != set(algorithms):
        raise ValueError("비교 대상 알고리즘의 예측이 모두 존재하지 않습니다.")
    if require_identical_keys:
        reference = None
        for algorithm in algorithms:
            keys = pd.MultiIndex.from_frame(
                selected.loc[selected["algorithm"].eq(algorithm), EVALUATION_KEYS]
                .sort_values(EVALUATION_KEYS)
                .reset_index(drop=True)
            )
            if reference is None:
                reference = keys
            elif not reference.equals(keys):
                raise ValueError("동일 조건 모델의 평가 키 집합이 일치하지 않습니다.")
    counts = selected.groupby(EVALUATION_KEYS, dropna=False)["algorithm"].nunique()
    keys = counts.loc[counts.eq(len(algorithms))].index.to_frame(index=False)
    result = selected.merge(keys, on=EVALUATION_KEYS, how="inner", validate="many_to_one")
    actual_counts = result.groupby(EVALUATION_KEYS, dropna=False)["actual"].nunique(dropna=False)
    if actual_counts.gt(1).any():
        raise ValueError("동일 평가 키의 실제값이 알고리즘 간 일치하지 않습니다.")
    return result.sort_values([*EVALUATION_KEYS, "algorithm"]).reset_index(drop=True)


def high_concentration_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """동일 기준 이상의 고농도 표본 오차·과소예측·편향을 계산한다."""
    actual_array = np.asarray(actual, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    mask = actual_array >= threshold
    if not mask.any():
        return {
            "rows": 0, "mae": float("nan"), "rmse": float("nan"),
            "r2": float("nan"), "underprediction_rate": float("nan"),
            "mean_bias": float("nan"),
        }
    selected_actual = actual_array[mask]
    selected_prediction = prediction_array[mask]
    error = selected_prediction - selected_actual
    return {
        "rows": int(mask.sum()),
        **regression_metrics(selected_actual, selected_prediction),
        "underprediction_rate": float(np.mean(error < 0)),
        "mean_bias": float(np.mean(error)),
    }


def summarize_prediction_metrics(
    predictions: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """그룹별 회귀 지표와 같은 평가행 Persistence 대비 MAE 개선율을 계산한다."""
    rows = []
    for keys, group in predictions.groupby(list(group_columns), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "eval_rows": len(group),
                **regression_metrics(group["actual"].to_numpy(), group["prediction"].to_numpy()),
            }
        )
    result = pd.DataFrame(rows)
    base_columns = [column for column in group_columns if column != "algorithm"]
    persistence = result.loc[result["algorithm"].eq("persistence")].copy()
    if base_columns:
        persistence = persistence[base_columns + ["mae"]].rename(
            columns={"mae": "persistence_mae"}
        )
        result = result.merge(persistence, on=base_columns, how="left", validate="many_to_one")
    else:
        persistence_mae = float(persistence.iloc[0]["mae"])
        result["persistence_mae"] = persistence_mae
    result["persistence_mae_improvement_pct"] = (
        (result["persistence_mae"] - result["mae"]) / result["persistence_mae"] * 100
    )
    return result
