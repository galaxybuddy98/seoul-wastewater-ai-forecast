"""고정 설정 LightGBM 기준 모델과 공정 비교 유틸리티."""

from __future__ import annotations

import re

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_SEED = 42
FLOW_PREFIXES = (
    "primary_treatment_volume_",
    "secondary_treatment_volume_",
    "total_treatment_volume_hypothesis_",
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """동일 평가행의 MAE·RMSE·R²를 계산한다."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
    }


def select_lag_features(frame: pd.DataFrame, include_flow: bool = True) -> list[str]:
    """명시적인 과거 lag 열만 선택해 목표·날짜 누수를 차단한다."""
    features = [
        column for column in frame.columns
        if re.search(r"_lag_\d+$", column)
    ]
    if not include_flow:
        features = [
            column for column in features
            if not column.startswith(FLOW_PREFIXES)
        ]
    return features


def add_series_code(
    train: pd.DataFrame,
    *others: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    """Train에서 확정한 계통 범주만 정수 코드로 변환한다."""
    categories = sorted(train["series_id"].dropna().unique())
    result = []
    for frame in (train, *others):
        converted = frame.copy()
        converted["series_code"] = pd.Categorical(
            converted["series_id"], categories=categories
        ).codes
        result.append(converted)
    return tuple(result)


def fit_lightgbm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
) -> lgb.LGBMRegressor:
    """Validation 조기종료만 사용하는 고정 설정 회귀 모델."""
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        train[feature_columns],
        train[target_column],
        eval_X=validation[feature_columns],
        eval_y=validation[target_column],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model
