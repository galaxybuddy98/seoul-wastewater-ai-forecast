import numpy as np
import pandas as pd

import src.models.xgboost_forecaster as xgb_module
from src.models.xgboost_forecaster import RANDOM_SEED, fit_xgboost


def test_fit_xgboost_uses_validation_early_stopping_and_fixed_seed():
    x = np.arange(120, dtype=float)
    frame = pd.DataFrame({"x_lag_0": x, "target": x * 0.5 + 2.0})
    train = frame.iloc[:80]
    validation = frame.iloc[80:]

    model = fit_xgboost(train, validation, "target", ["x_lag_0"])
    predictions = model.predict(validation[["x_lag_0"]])

    assert model.get_params()["random_state"] == RANDOM_SEED
    assert model.get_params()["eval_metric"] == "mae"
    assert model.best_iteration >= 0
    assert np.isfinite(predictions).all()


def test_fit_xgboost_passes_sample_weight_to_training_only(monkeypatch):
    captured = {}

    class FakeRegressor:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def fit(self, x, y, **kwargs):
            captured["train_rows"] = len(x)
            captured["fit"] = kwargs
            return self

    monkeypatch.setattr(xgb_module, "XGBRegressor", FakeRegressor)
    train = pd.DataFrame({"x": [1.0, 2.0], "target": [10.0, 20.0]})
    validation = pd.DataFrame({"x": [3.0], "target": [30.0]})
    weights = np.array([1.0, 2.0])
    fit_xgboost(train, validation, "target", ["x"], sample_weight=weights)
    np.testing.assert_array_equal(captured["fit"]["sample_weight"], weights)
    assert captured["train_rows"] == 2
    assert "sample_weight_eval_set" not in captured["fit"]
    assert len(captured["fit"]["eval_set"][0][0]) == 1
