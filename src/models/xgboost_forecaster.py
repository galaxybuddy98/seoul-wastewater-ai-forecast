"""고정 설정 XGBoost 회귀 모델."""

from __future__ import annotations

import pandas as pd
import numpy as np
from xgboost import XGBRegressor

RANDOM_SEED = 42


def fit_xgboost(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    sample_weight: np.ndarray | pd.Series | None = None,
) -> XGBRegressor:
    """Validation MAE 조기종료만 사용하는 CPU XGBoost 모델을 적합한다."""
    model = XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        tree_method="hist",
        eval_metric="mae",
        early_stopping_rounds=50,
        n_jobs=4,
    )
    fit_kwargs = {
        "eval_set": [(validation[feature_columns], validation[target_column])],
        "verbose": False,
    }
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(train[feature_columns], train[target_column], **fit_kwargs)
    return model
