import numpy as np
import pandas as pd
import pytest

from src.models.xgboost.bod_improvement import (
    add_target_calendar_features,
    assert_common_evaluation_rows,
    meets_replacement_goal,
    training_sample_weights,
)
from scripts.training.xgboost.train_calendar_weighted_xgboost import _validation_selection


def test_target_calendar_features_use_target_date_without_redundant_calendar_columns():
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2024-01-01", "2024-01-07"]),
            "input_end_date": pd.to_datetime(["2023-12-31", "2024-01-06"]),
        }
    )
    result = add_target_calendar_features(frame, include_season=True, include_weekday=True)
    assert result.loc[0, "target_doy_sin"] == pytest.approx(0.0)
    assert result.loc[0, "target_doy_cos"] == pytest.approx(1.0)
    assert result.loc[0, "target_dow_0"] == 1.0
    assert result.loc[1, "target_dow_6"] == 1.0
    assert result.filter(regex=r"^target_dow_").sum(axis=1).eq(1.0).all()
    assert not any("month" in column or "weekend" in column or "season" in column for column in result)


def test_training_weights_use_train_targets_and_include_180_boundary():
    train = pd.DataFrame(
        {"split": ["train", "train", "train"], "target_bod_mg_l": [179.9, 180.0, 250.0]}
    )
    weights = training_sample_weights(train, "target_bod_mg_l", threshold=180.0, high_weight=2.0)
    np.testing.assert_array_equal(weights, np.array([1.0, 2.0, 2.0]))


def test_training_weights_reject_validation_or_test_rows():
    not_train = pd.DataFrame({"split": ["train", "validation"], "target_bod_mg_l": [100.0, 200.0]})
    with pytest.raises(ValueError, match="Train"):
        training_sample_weights(not_train, "target_bod_mg_l", threshold=180.0, high_weight=2.0)


def test_common_evaluation_rows_reject_actual_value_mismatch():
    baseline = pd.DataFrame(
        {"series_id": ["s1"], "target_date": pd.to_datetime(["2024-01-01"]), "split": ["validation"], "actual": [10.0]}
    )
    candidate = baseline.assign(actual=11.0)
    with pytest.raises(ValueError, match="실제값"):
        assert_common_evaluation_rows(baseline, candidate)


@pytest.mark.parametrize(
    ("overall_mae", "high_mae", "expected"),
    [(23.4625 * 1.02, 60.0 * 0.90, True), (23.4625 * 1.0201, 54.0, False), (23.0, 60.0 * 0.901, False)],
)
def test_replacement_goal_uses_predeclared_two_percent_and_ten_percent_boundaries(
    overall_mae: float, high_mae: float, expected: bool
):
    assert meets_replacement_goal(23.4625, 60.0, overall_mae, high_mae) is expected


def test_validation_selection_never_recommends_persistence_comparator():
    rows = []
    for target in ("bod_mg_l", "tn_mg_l", "tp_mg_l"):
        for model_name, mae in (
            ("baseline", 2.0),
            ("persistence", 0.5),
            ("season", 1.9),
            ("weekday", 1.8),
            ("season_weekday", 1.7),
        ):
            rows.append(
                {
                    "split": "validation",
                    "target": target,
                    "model_name": model_name,
                    "mae": mae,
                    "rmse": mae + 1.0,
                    "high_mae": 10.0,
                }
            )
    rows.append(
        {
            "split": "validation",
            "target": "bod_mg_l",
            "model_name": "base_weighted2",
            "mae": 2.0,
            "rmse": 3.0,
            "high_mae": 10.0,
        }
    )

    selected = _validation_selection(pd.DataFrame(rows))

    assert selected["recommended_model"].eq("season_weekday").all()
