import numpy as np
import pandas as pd
import pytest

from scripts.validation.run_preprocessing_audit import _read_rain_source, _sample_window_rows

from src.audit.preprocessing_audit import (
    aggregate_station_day,
    classify_split,
    join_rainfall_reference,
    reconstruct_lags,
)


def _complete_day(date: str, value: float = 0.0) -> pd.DataFrame:
    # 원본 시각 규칙(:09, :19, ..., :59)을 모사한다.
    timestamps = pd.date_range(f"{date} 00:09", periods=144, freq="10min")
    return pd.DataFrame({"timestamp": timestamps, "rainfall_mm": value})


def test_aggregate_station_day_sums_complete_intervals_and_maximum_hour():
    frame = _complete_day("2024-07-01")
    frame.loc[60:65, "rainfall_mm"] = [1, 2, 3, 4, 5, 6]

    result = aggregate_station_day(frame)

    assert result["unique_interval_count"] == 144
    assert result["duplicate_same_count"] == 0
    assert result["duplicate_conflict_count"] == 0
    assert result["independent_daily_rainfall_mm"] == 21.0
    assert result["independent_max_1h_mm"] == 21.0
    assert result["audit_status"] == "confirmed"


def test_aggregate_station_day_does_not_convert_incomplete_dry_records_to_zero():
    result = aggregate_station_day(_complete_day("2024-01-01").iloc[:-1])

    assert np.isnan(result["independent_daily_rainfall_mm"])
    assert result["audit_status"] == "incomplete"


def test_aggregate_station_day_distinguishes_same_duplicates_from_conflicts():
    base = _complete_day("2024-07-01")
    same = pd.concat([base, base.iloc[[0]]], ignore_index=True)
    conflict_row = base.iloc[[1]].copy()
    conflict_row["rainfall_mm"] = 9.0
    conflict = pd.concat([base, conflict_row], ignore_index=True)

    same_result = aggregate_station_day(same)
    conflict_result = aggregate_station_day(conflict)

    assert same_result["duplicate_same_count"] == 1
    assert same_result["audit_status"] == "confirmed"
    assert conflict_result["duplicate_conflict_count"] == 1
    assert conflict_result["audit_status"] == "conflict"
    assert np.isnan(conflict_result["independent_daily_rainfall_mm"])


def test_reconstruct_lags_uses_only_same_series_days_before_target():
    dates = pd.date_range("2023-12-01", "2024-01-01", freq="D")
    frame = pd.DataFrame(
        {
            "series_id": ["a"] * len(dates) + ["b"] * len(dates),
            "date": list(dates) * 2,
            "value": list(range(len(dates))) + [999.0] * len(dates),
        }
    )

    actual = reconstruct_lags(frame, "a", pd.Timestamp("2024-01-01"), "value", 30)

    assert actual[0] == frame.loc[(frame.series_id == "a") & (frame.date == "2023-12-31"), "value"].iloc[0]
    assert actual[29] == frame.loc[(frame.series_id == "a") & (frame.date == "2023-12-02"), "value"].iloc[0]
    assert 999.0 not in actual


@pytest.mark.parametrize(
    ("date", "expected"),
    [("2022-12-31", "train"), ("2023-01-01", "validation"),
     ("2025-01-01", "test"), ("2026-10-01", "outside")],
)
def test_classify_split_uses_target_date_boundaries(date, expected):
    assert classify_split(pd.Timestamp(date)) == expected


def test_read_rain_source_detects_single_title_row(tmp_path):
    source = tmp_path / "rain.csv"
    source.write_text(
        "2025년,,\n강우량계명,시간,10분우량\n공항동P,2025-01-01 00:09,1.5\n",
        encoding="cp949",
    )

    actual = _read_rain_source(source)

    assert actual.loc[0, "station_name"] == "공항동P"
    assert actual.loc[0, "rainfall_mm"] == 1.5


def test_join_rainfall_reference_keeps_asos_and_rainfall_station_ids_distinct():
    enriched = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-01"]), "facility": ["중랑"],
         "station_id": ["108"], "seoul_rainfall_station_id": ["301"]}
    )
    rainfall = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-01"]), "facility": ["중랑"],
         "station_id": ["301"], "daily_rainfall_mm": [2.0]}
    )

    actual = join_rainfall_reference(enriched, rainfall)

    assert actual.loc[0, "station_id"] == "108"
    assert actual.loc[0, "source_rainfall_station_id"] == "301"


def test_read_rain_source_parses_mixed_minute_and_second_precision(tmp_path):
    source = tmp_path / "mixed.csv"
    source.write_text(
        "강우량계명,시간,10분우량\n공항동P,2023-07-23 00:09,1\n"
        "공항동P,2023-07-23 00:19:59,2\n",
        encoding="cp949",
    )

    actual = _read_rain_source(source)

    assert actual.timestamp.notna().all()
    assert actual.timestamp.iloc[1] == pd.Timestamp("2023-07-23 00:19:59")


def test_sample_window_rows_always_includes_split_and_year_boundaries():
    dates = pd.to_datetime(["2022-12-31", "2023-01-01", "2024-12-31", "2025-01-01", "2026-01-01"])
    frame = pd.DataFrame(
        {"series_id": ["a"] * len(dates), "target_date": dates,
         "split": [classify_split(date) for date in dates]}
    )

    actual = _sample_window_rows(frame)

    assert set(dates).issubset(set(actual.target_date))
