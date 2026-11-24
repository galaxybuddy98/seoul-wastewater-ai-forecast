import pandas as pd
import pytest
import joblib

from src.models.shared.seoul_rainfall_experiment import (
    attach_input_day_rainfall,
    common_prediction_rows,
    load_model_checkpoint,
    select_seoul_rainfall_features,
)


def test_variant_features_include_only_requested_seoul_rainfall_groups():
    frame = pd.DataFrame(
        columns=[
            "bod_mg_l_lag_0",
            "mean_temperature_c_lag_0",
            "seoul_rainfall_mm_lag_0",
            "seoul_rainfall_lag_1d_mm_lag_0",
            "seoul_rainfall_lag_2d_mm_lag_0",
            "seoul_rainfall_lag_3d_mm_lag_0",
            "seoul_rainfall_sum_3d_mm_lag_0",
            "seoul_rainfall_sum_7d_mm_lag_0",
            "seoul_rainfall_max_1h_mm_lag_0",
            "seoul_rainfall_missing_lag_0",
            "precipitation_mm_lag_0",
            "target_bod_mg_l",
        ]
    )

    assert select_seoul_rainfall_features(frame, "C") == [
        "bod_mg_l_lag_0",
        "mean_temperature_c_lag_0",
        "seoul_rainfall_mm_lag_0",
    ]
    assert select_seoul_rainfall_features(frame, "D") == [
        "bod_mg_l_lag_0",
        "mean_temperature_c_lag_0",
        "seoul_rainfall_mm_lag_0",
        "seoul_rainfall_lag_1d_mm_lag_0",
        "seoul_rainfall_lag_2d_mm_lag_0",
        "seoul_rainfall_lag_3d_mm_lag_0",
        "seoul_rainfall_sum_3d_mm_lag_0",
        "seoul_rainfall_sum_7d_mm_lag_0",
    ]
    assert select_seoul_rainfall_features(frame, "E")[-1] == (
        "seoul_rainfall_max_1h_mm_lag_0"
    )


def test_common_prediction_rows_rejects_duplicate_evaluation_keys():
    predictions = pd.DataFrame(
        {
            "series_id": ["s1", "s1"],
            "target_date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "target": ["bod_mg_l", "bod_mg_l"],
            "split": ["test", "test"],
            "variant": ["C", "C"],
            "actual": [10.0, 10.0],
            "prediction": [9.0, 9.5],
        }
    )

    with pytest.raises(ValueError, match="중복"):
        common_prediction_rows(predictions, ["C"])


def test_common_prediction_rows_uses_intersection_and_checks_actual_values():
    predictions = pd.DataFrame(
        {
            "series_id": ["s1", "s2", "s1", "s3"],
            "target_date": pd.to_datetime(["2025-01-02"] * 4),
            "target": ["tn_mg_l"] * 4,
            "split": ["test"] * 4,
            "variant": ["A", "A", "C", "C"],
            "actual": [4.0, 5.0, 4.0, 6.0],
            "prediction": [3.5, 5.5, 3.8, 6.2],
        }
    )

    result = common_prediction_rows(predictions, ["A", "C"])

    assert result[["series_id", "variant"]].to_dict("records") == [
        {"series_id": "s1", "variant": "A"},
        {"series_id": "s1", "variant": "C"},
    ]


def test_attach_input_day_rainfall_uses_day_before_target_not_future_day():
    predictions = pd.DataFrame(
        {
            "facility": ["중랑물재생센터"],
            "target_date": pd.to_datetime(["2025-01-02"]),
            "prediction": [9.0],
        }
    )
    rainfall = pd.DataFrame(
        {
            "facility": ["중랑물재생센터", "중랑물재생센터"],
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "daily_rainfall_mm": [2.0, 99.0],
            "rainfall_lag_1d_mm": [0.0, 2.0],
        }
    )

    result = attach_input_day_rainfall(predictions, rainfall)

    assert result.loc[0, "input_date"] == pd.Timestamp("2025-01-01")
    assert result.loc[0, "daily_rainfall_mm"] == 2.0


def test_model_checkpoint_is_reused_only_when_feature_contract_matches(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump({"model": "saved-model", "features": ["x_lag_0"]}, path)

    assert load_model_checkpoint(path, ["x_lag_0"]) == "saved-model"
    with pytest.raises(ValueError, match="피처 계약"):
        load_model_checkpoint(path, ["different_lag_0"])
