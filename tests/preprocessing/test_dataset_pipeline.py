import numpy as np
import pandas as pd
import pytest

from src.preprocessing.dataset import (
    DatasetError,
    add_weather_history,
    assign_target_split,
    build_observation_table,
    join_exact_flow,
)
from src.preprocessing.windows import build_sequence_windows, build_tabular_windows


def _quality_frame():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-12-29", "2022-12-30", "2022-12-31", "2023-01-01", "2023-01-02", "2023-01-01"]),
            "facility": ["중랑물재생센터"] * 6,
            "plant": ["제1처리장"] * 5 + ["제1처리장"],
            "stream_type": ["유입수"] * 5 + ["방류"],
            "bod_mg_l": [1, 2, 3, 4, 5, 99],
            "tn_mg_l": [10, 11, 12, 13, 14, 99],
            "tp_mg_l": [2, 2, 3, 3, 4, 99],
        }
    )


def test_observation_table_keeps_only_influent_and_preserves_source_values():
    result = build_observation_table(_quality_frame())

    assert len(result) == 5
    assert set(result["stream_type"]) == {"유입수"}
    assert result["source_plant"].eq("제1처리장").all()
    assert result["series_id"].nunique() == 1
    assert result["bod_mg_l"].tolist() == [1, 2, 3, 4, 5]


def test_observation_table_rejects_duplicate_series_dates_without_averaging():
    frame = _quality_frame().iloc[:5]
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(DatasetError, match="중복"):
        build_observation_table(duplicated)


def test_exact_flow_join_does_not_assign_other_plant_or_multiply_rows():
    observations = build_observation_table(_quality_frame())
    flow = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-12-29", "2022-12-29"]),
            "facility": ["중랑물재생센터", "중랑물재생센터"],
            "plant": ["제1처리장", "제2처리장"],
            "primary_treatment_volume": [1.0, 9.0],
            "secondary_treatment_volume": [10.0, 90.0],
            "total_treatment_volume_hypothesis": [11.0, 99.0],
        }
    )

    result = join_exact_flow(observations, flow)

    assert len(result) == len(observations)
    assert result.loc[result["date"].eq(pd.Timestamp("2022-12-29")), "secondary_treatment_volume"].item() == 10.0
    assert result["bod_mg_l"].tolist() == observations["bod_mg_l"].tolist()


def test_exact_flow_join_rejects_duplicate_flow_keys():
    observations = build_observation_table(_quality_frame())
    flow = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-12-29", "2022-12-29"]),
            "facility": ["중랑물재생센터"] * 2,
            "plant": ["제1처리장"] * 2,
            "primary_treatment_volume": [1.0, 2.0],
            "secondary_treatment_volume": [10.0, 20.0],
        }
    )

    with pytest.raises(DatasetError, match="처리량.*중복"):
        join_exact_flow(observations, flow)


def test_weather_history_preserves_missing_rain_and_uses_past_dates_only():
    weather = pd.DataFrame(
        {
            "facility": ["A"] * 4,
            "date": pd.date_range("2024-01-01", periods=4),
            "mean_temperature_c": [1.0, 2.0, 3.0, 4.0],
            "precipitation_mm": [1.0, np.nan, 3.0, 4.0],
        }
    )

    result = add_weather_history(weather)

    assert np.isnan(result.loc[1, "precipitation_mm"])
    assert result.loc[3, "precipitation_lag_1d"] == 3.0
    assert np.isnan(result.loc[3, "precipitation_sum_3d"])


def test_tabular_windows_keep_series_separate_and_split_by_target_date():
    observations = build_observation_table(_quality_frame())

    result = build_tabular_windows(
        observations,
        window_days=2,
        feature_columns=["bod_mg_l", "tn_mg_l", "tp_mg_l"],
        target_columns=["bod_mg_l"],
    )

    assert len(result) == 3
    assert result["series_id"].nunique() == 1
    assert result.loc[result["target_date"].eq(pd.Timestamp("2023-01-01")), "split"].item() == "validation"
    assert result.loc[0, "bod_mg_l_lag_0"] == 2
    assert result.loc[0, "target_bod_mg_l"] == 3


def test_tabular_windows_exclude_non_contiguous_dates():
    observations = build_observation_table(_quality_frame()).drop(index=[1]).reset_index(drop=True)

    result = build_tabular_windows(
        observations,
        window_days=2,
        feature_columns=["bod_mg_l"],
        target_columns=["bod_mg_l"],
    )

    assert result["target_date"].tolist() == [pd.Timestamp("2023-01-02")]


def test_sequence_windows_match_tabular_sample_metadata():
    observations = build_observation_table(_quality_frame())

    result = build_sequence_windows(
        observations,
        window_days=2,
        feature_columns=["bod_mg_l", "tn_mg_l"],
        target_columns=["bod_mg_l"],
    )

    assert result["X"].shape == (3, 2, 2)
    assert result["y"].shape == (3, 1)
    assert result["metadata"]["target_date"].tolist()[0] == pd.Timestamp("2022-12-31")


from src.data.validation import compare_snapshots


def test_snapshot_comparison_reports_missing_and_changed_rows():
    csv_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "facility": ["A", "A"],
            "plant": ["P", "P"],
            "stream_type": ["유입수", "유입수"],
            "bod_mg_l": [1.0, 2.0],
        }
    )
    api_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "facility": ["A", "A"],
            "plant": ["P", "P"],
            "stream_type": ["유입수", "유입수"],
            "bod_mg_l": [9.0, 3.0],
        }
    )

    result = compare_snapshots(
        csv_frame,
        api_frame,
        key_columns=["date", "facility", "plant", "stream_type"],
        value_columns=["bod_mg_l"],
    )

    assert result["summary"] == {
        "csv_rows": 2,
        "api_rows": 2,
        "csv_only_rows": 1,
        "api_only_rows": 1,
        "changed_rows": 1,
    }


def test_fast_target_counts_match_window_contract():
    from src.preprocessing.windows import count_target_window_samples

    observations = build_observation_table(_quality_frame())
    result = count_target_window_samples(
        observations,
        window_days=2,
        feature_columns=["bod_mg_l", "tn_mg_l", "tp_mg_l"],
        target_columns=["bod_mg_l"],
    )

    assert result["sample_count"].sum() == 3
    assert set(result["split"]) == {"train", "validation"}


def test_tabular_windows_allow_optional_rain_missing_without_losing_quality_sample():
    observations = build_observation_table(_quality_frame())
    observations["mean_temperature_c"] = 10.0
    observations["precipitation_processed_mm"] = [0.0, np.nan, 1.0, 0.0, 0.0]

    result = build_tabular_windows(
        observations,
        window_days=2,
        feature_columns=["bod_mg_l", "mean_temperature_c", "precipitation_processed_mm"],
        required_feature_columns=["bod_mg_l", "mean_temperature_c"],
        target_columns=["bod_mg_l"],
    )

    assert len(result) == 3
    assert result.filter(like="precipitation_processed_mm_lag_").isna().any(axis=None)


def test_tabular_windows_still_reject_missing_required_quality():
    observations = build_observation_table(_quality_frame())
    observations.loc[observations.index[1], "bod_mg_l"] = np.nan
    observations["precipitation_processed_mm"] = np.nan

    result = build_tabular_windows(
        observations,
        window_days=2,
        feature_columns=["bod_mg_l", "precipitation_processed_mm"],
        required_feature_columns=["bod_mg_l"],
        target_columns=["bod_mg_l"],
    )

    assert len(result) == 1
