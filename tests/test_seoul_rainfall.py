from pathlib import Path

import numpy as np
import pandas as pd

from src.data.seoul_rainfall import (
    add_grouped_rainfall_features,
    aggregate_station_daily,
    build_center_rainfall,
    detect_csv_layout,
    join_center_rainfall,
    load_normalized_rainfall_file,
    resolve_timestamp_duplicates,
    summarize_center_coverage,
)
from src.data.seoul_rainfall_incremental import (
    find_incremental_sources,
    merge_incremental_station_daily,
)


def test_loader_handles_cp949_standard_header_and_marks_zero_positive(tmp_path: Path):
    path = tmp_path / "station2016.csv"
    path.write_text(
        "강우량계명,시간,10분우량\n가산2P,2016-01-01 00:00,0\n가산2P,2016-01-01 00:10,0.5\n",
        encoding="cp949",
    )

    layout = detect_csv_layout(path)
    result = load_normalized_rainfall_file(path)

    assert layout.encoding == "cp949"
    assert layout.skiprows == 0
    assert result["quality_status"].tolist() == ["valid_zero", "valid_positive"]
    assert str(result["timestamp_kst"].dt.tz) == "Asia/Seoul"
    assert result["rainfall_unit"].eq("mm/10min").all()


def test_loader_skips_year_metadata_row_and_preserves_bad_values(tmp_path: Path):
    path = tmp_path / "station2025.csv"
    path.write_text(
        "2025년\n강우량계명,시간,10분우량\n가산2P,잘못된시각,\n가산2P,2025-01-01 00:10,-0.5\n",
        encoding="utf-8-sig",
    )

    layout = detect_csv_layout(path)
    result = load_normalized_rainfall_file(path)

    assert layout.skiprows == 1
    assert result["quality_status"].tolist() == ["invalid_timestamp_and_missing", "negative"]


def test_loader_handles_2026_official_code_schema(tmp_path: Path):
    path = tmp_path / "seoul_202601.csv"
    path.write_text(
        '"강우량계 코드","강우량계명","구청 코드","구청명","10분우량","자료수집 시각"\n'
        "1501,마포구청,115,마포구,0.5,2026-01-01 00:19\n",
        encoding="cp949",
    )

    layout = detect_csv_layout(path)
    result = load_normalized_rainfall_file(path)

    assert layout.time_column == "자료수집 시각"
    assert result.loc[0, "station_code"] == "1501"
    assert result.loc[0, "district_code"] == "115"
    assert result.loc[0, "district_name"] == "마포구"
    assert result.loc[0, "rainfall_10min_mm"] == 0.5


def test_duplicate_resolution_keeps_identical_once_and_flags_conflict():
    frame = pd.DataFrame(
        {
            "station_id": ["S1", "S1", "S1", "S1"],
            "station_name": ["A"] * 4,
            "timestamp_kst": pd.to_datetime([
                "2024-01-01 00:00", "2024-01-01 00:00",
                "2024-01-01 00:10", "2024-01-01 00:10",
            ]).tz_localize("Asia/Seoul"),
            "rainfall_10min_mm": [0.0, 0.0, 0.5, 1.0],
            "quality_status": ["valid_zero", "valid_zero", "valid_positive", "valid_positive"],
            "source_file": ["a", "b", "a", "b"],
        }
    )

    result = resolve_timestamp_duplicates(frame)

    assert len(result) == 2
    assert result.loc[0, "rainfall_10min_mm"] == 0.0
    assert result.loc[0, "duplicate_status"] == "identical_duplicate"
    assert pd.isna(result.loc[1, "rainfall_10min_mm"])
    assert result.loc[1, "duplicate_status"] == "conflicting_duplicate"


def test_daily_sum_requires_complete_expected_intervals():
    timestamps = pd.date_range("2024-01-01", periods=144, freq="10min", tz="Asia/Seoul")
    complete = pd.DataFrame(
        {
            "station_id": "S1", "station_name": "A", "timestamp_kst": timestamps,
            "rainfall_10min_mm": [0.0] * 138 + [1.0] * 6,
            "quality_status": "valid_zero", "duplicate_status": "unique",
            "source_file": "a.csv",
        }
    )
    incomplete = complete.iloc[:-1].copy()
    incomplete["station_id"] = "S2"
    incomplete["station_name"] = "B"

    result = aggregate_station_daily(pd.concat([complete, incomplete], ignore_index=True))
    complete_day = result.loc[result["station_id"].eq("S1")].iloc[0]
    incomplete_day = result.loc[result["station_id"].eq("S2")].iloc[0]

    assert complete_day["daily_rainfall_mm"] == 6.0
    assert complete_day["max_1h_rainfall_mm"] == 6.0
    assert complete_day["daily_quality_status"] == "complete"
    assert pd.isna(incomplete_day["daily_rainfall_mm"])
    assert incomplete_day["daily_quality_status"] == "incomplete_intervals"


def test_daily_aggregation_returns_schema_for_empty_target_period():
    empty = pd.DataFrame(
        columns=[
            "station_id", "station_name", "timestamp_kst", "rainfall_10min_mm",
            "quality_status", "duplicate_status", "source_file",
        ]
    )
    empty["timestamp_kst"] = pd.to_datetime(empty["timestamp_kst"], utc=True).dt.tz_convert("Asia/Seoul")

    result = aggregate_station_daily(empty)

    assert result.empty
    assert {"date", "station_id", "daily_rainfall_mm"}.issubset(result.columns)


def test_grouped_features_do_not_cross_station_and_propagate_unknown():
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
            "station_id": ["A"] * 3 + ["B"] * 3,
            "daily_rainfall_mm": [1.0, np.nan, 3.0, 10.0, 20.0, 30.0],
        }
    )

    result = add_grouped_rainfall_features(daily)
    station_a = result[result["station_id"] == "A"].reset_index(drop=True)
    station_b = result[result["station_id"] == "B"].reset_index(drop=True)

    assert station_a.loc[1, "rainfall_lag_1d_mm"] == 1.0
    assert pd.isna(station_a.loc[2, "rainfall_lag_1d_mm"])
    assert pd.isna(station_a.loc[2, "rainfall_sum_3d_mm"])
    assert station_b.loc[1, "rainfall_lag_1d_mm"] == 10.0


def test_center_rainfall_join_preserves_rows_and_uses_same_day_t_features():
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "station_id": ["A", "A"],
            "station_name": ["관측소A", "관측소A"],
            "daily_rainfall_mm": [1.0, 2.0],
            "daily_quality_status": ["complete", "complete"],
        }
    )
    mapping = pd.DataFrame(
        {
            "facility": ["중랑물재생센터"],
            "station_id": ["A"],
            "station_name": ["관측소A"],
            "selection_status": ["provisional"],
        }
    )
    center = build_center_rainfall(daily, mapping)
    assert center["station_name"].eq("관측소A").all()
    observations = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "facility": ["중랑물재생센터", "중랑물재생센터"],
            "series_id": ["s1", "s1"],
        }
    )

    joined = join_center_rainfall(observations, center)

    assert len(joined) == len(observations)
    assert joined["seoul_rainfall_mm"].tolist() == [1.0, 2.0]
    assert joined["seoul_rainfall_station_id"].tolist() == ["A", "A"]


def test_incremental_discovery_uses_inventory_path_and_size(tmp_path: Path):
    unchanged = tmp_path / "old.csv"
    changed = tmp_path / "changed.csv"
    new = tmp_path / "new.csv"
    unchanged.write_bytes(b"123")
    changed.write_bytes(b"12345")
    new.write_bytes(b"12")
    inventory = pd.DataFrame(
        {
            "source_file": ["old.csv", "changed.csv"],
            "size_bytes": [3, 4],
        }
    )

    result = find_incremental_sources(tmp_path, inventory)

    assert result.set_index("source_file")["change_status"].to_dict() == {
        "changed.csv": "changed",
        "new.csv": "new",
        "old.csv": "unchanged",
    }
    assert result.loc[result["change_status"].eq("unchanged"), "sha256"].isna().all()
    assert result.loc[result["change_status"].ne("unchanged"), "sha256"].notna().all()


def test_incremental_daily_merge_recomputes_year_boundary_features():
    existing = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-12-30", "2025-12-31", "2026-01-01"]),
            "station_id": ["A"] * 3,
            "station_name": ["관측소A"] * 3,
            "daily_rainfall_mm": [1.0, 2.0, np.nan],
            "daily_quality_status": ["complete", "complete", "no_records"],
            "source_system": ["서울시 강우량 관측망"] * 3,
        }
    )
    incremental = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "station_id": ["A"],
            "station_name": ["관측소A"],
            "daily_rainfall_mm": [3.0],
            "daily_quality_status": ["complete"],
            "source_system": ["서울시 강우량 관측망"],
        }
    )

    result = merge_incremental_station_daily(existing, incremental)
    boundary = result.loc[result["date"].eq(pd.Timestamp("2026-01-01"))].iloc[0]

    assert boundary["rainfall_lag_1d_mm"] == 2.0
    assert boundary["rainfall_lag_2d_mm"] == 1.0
    assert boundary["rainfall_sum_3d_mm"] == 6.0


def test_center_coverage_reports_confirmed_and_accumulation_rates():
    center = pd.DataFrame(
        {
            "facility": ["중랑물재생센터"] * 3,
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "daily_rainfall_mm": [0.0, 1.0, np.nan],
            "max_1h_rainfall_mm": [0.0, 1.0, np.nan],
            "rainfall_sum_3d_mm": [np.nan, np.nan, 1.0],
            "rainfall_sum_7d_mm": [np.nan, np.nan, np.nan],
        }
    )

    result = summarize_center_coverage(center)

    assert result.loc[0, "daily_available_days"] == 2
    assert result.loc[0, "daily_available_rate"] == 2 / 3
    assert result.loc[0, "sum_3d_available_days"] == 1
    assert result.loc[0, "sum_7d_available_rate"] == 0.0
