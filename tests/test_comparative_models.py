import numpy as np
import pandas as pd
import pytest

from src.models.comparative_models import (
    common_algorithm_rows,
    encode_series_ids,
    fit_sequence_preprocessor,
    high_concentration_metrics,
    summarize_prediction_metrics,
    tabular_to_sequence,
)


def test_sequence_preprocessor_fits_statistics_on_train_only():
    train = np.array([[[1.0], [3.0]], [[3.0], [1.0]]])
    validation = np.array([[[100.0], [100.0]]])

    preprocessor = fit_sequence_preprocessor(train)
    transformed = preprocessor.transform(validation)

    assert preprocessor.medians.tolist() == [2.0]
    assert preprocessor.means.tolist() == [2.0]
    assert preprocessor.stds.tolist() == [1.0]
    assert transformed[0, 0, 0] == 98.0


def test_rainfall_missing_indicator_distinguishes_nan_from_real_zero():
    train = np.array(
        [
            [[10.0, 0.0], [11.0, np.nan]],
            [[12.0, 2.0], [13.0, 4.0]],
        ]
    )
    evaluation = np.array([[[20.0, 0.0], [21.0, np.nan]]])

    transformed = fit_sequence_preprocessor(
        train, missing_indicator_indices=[1]
    ).transform(evaluation)

    assert transformed.shape == (1, 2, 3)
    assert transformed[0, 0, 2] == 0.0
    assert transformed[0, 1, 2] == 1.0
    assert np.isfinite(transformed).all()


def test_series_encoding_uses_train_categories_and_rejects_unseen_series():
    train_codes, validation_codes = encode_series_ids(
        np.array(["s2", "s1", "s2"]), np.array(["s1", "s2"])
    )

    assert train_codes.tolist() == [1, 0, 1]
    assert validation_codes.tolist() == [0, 1]
    with pytest.raises(ValueError, match="Train에 없는 계통"):
        encode_series_ids(np.array(["s1"]), np.array(["new"]))


def test_tabular_to_sequence_reverses_lags_to_oldest_first():
    frame = pd.DataFrame(
        {
            "quality_lag_0": [30.0],
            "quality_lag_1": [20.0],
            "quality_lag_2": [10.0],
            "rain_lag_0": [3.0],
            "rain_lag_1": [2.0],
            "rain_lag_2": [1.0],
        }
    )

    result = tabular_to_sequence(frame, ["quality", "rain"], window_days=3)

    assert result.tolist() == [[[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]]]


def test_common_algorithm_rows_uses_intersection_and_rejects_duplicates():
    frame = pd.DataFrame(
        {
            "series_id": ["s1", "s2", "s1", "s3"],
            "target_date": pd.to_datetime(["2025-01-01"] * 4),
            "target": ["bod_mg_l"] * 4,
            "split": ["test"] * 4,
            "algorithm": ["lightgbm", "lightgbm", "xgboost", "xgboost"],
            "actual": [10.0, 20.0, 10.0, 30.0],
            "prediction": [9.0, 19.0, 11.0, 29.0],
        }
    )

    common = common_algorithm_rows(frame, ["lightgbm", "xgboost"])
    assert common[["series_id", "algorithm"]].to_dict("records") == [
        {"series_id": "s1", "algorithm": "lightgbm"},
        {"series_id": "s1", "algorithm": "xgboost"},
    ]

    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="중복"):
        common_algorithm_rows(duplicated, ["lightgbm", "xgboost"])


def test_common_algorithm_rows_strict_mode_rejects_missing_evaluation_keys():
    frame = pd.DataFrame(
        {
            "series_id": ["s1", "s2", "s1", "s3"],
            "target_date": pd.to_datetime(["2025-01-01"] * 4),
            "target": ["bod_mg_l"] * 4,
            "split": ["test"] * 4,
            "algorithm": ["lightgbm", "lightgbm", "xgboost", "xgboost"],
            "actual": [10.0, 20.0, 10.0, 30.0],
            "prediction": [9.0, 19.0, 11.0, 29.0],
        }
    )

    with pytest.raises(ValueError, match="평가 키 집합"):
        common_algorithm_rows(
            frame,
            ["lightgbm", "xgboost"],
            require_identical_keys=True,
        )


def test_high_concentration_metrics_reports_underprediction_and_bias():
    result = high_concentration_metrics(
        np.array([100.0, 200.0]), np.array([90.0, 150.0]), threshold=150.0
    )

    assert result["rows"] == 1
    assert result["mae"] == 50.0
    assert result["rmse"] == 50.0
    assert result["underprediction_rate"] == 1.0
    assert result["mean_bias"] == -50.0


def test_prediction_summary_uses_persistence_mae_from_same_group():
    frame = pd.DataFrame(
        {
            "algorithm": ["persistence", "persistence", "model", "model"],
            "actual": [2.0, 4.0, 2.0, 4.0],
            "prediction": [1.0, 2.0, 2.0, 3.0],
        }
    )

    result = summarize_prediction_metrics(frame, group_columns=["algorithm"])
    model = result.loc[result["algorithm"].eq("model")].iloc[0]

    assert model["mae"] == 0.5
    assert model["persistence_mae"] == 1.5
    assert np.isclose(model["persistence_mae_improvement_pct"], 100 * (1.5 - 0.5) / 1.5)
