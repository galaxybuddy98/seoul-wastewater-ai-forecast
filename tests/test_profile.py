import pandas as pd

from scripts.analyze_external_csv import _outliers
from src.preprocessing.profile import find_calendar_gaps, find_missing_runs


def test_find_calendar_gaps_reports_internal_missing_dates_per_series():
    frame = pd.DataFrame({
        "center": ["A", "A", "B"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-01"]),
    })

    result = find_calendar_gaps(frame, ["center"], "date")

    row = result.loc[result["center"].eq("A")].iloc[0]
    assert row["missing_date_count"] == 1
    assert row["longest_gap_days"] == 1
    assert row["first_missing_date"] == pd.Timestamp("2024-01-02")


def test_find_missing_runs_reports_consecutive_missing_measurements():
    frame = pd.DataFrame({
        "center": ["A"] * 5,
        "date": pd.date_range("2024-01-01", periods=5),
        "bod": [1.0, None, None, 2.0, None],
    })

    result = find_missing_runs(frame, ["center"], "date", ["bod"])

    longest = result.sort_values("run_length", ascending=False).iloc[0]
    assert longest["variable"] == "bod"
    assert longest["start_date"] == pd.Timestamp("2024-01-02")
    assert longest["end_date"] == pd.Timestamp("2024-01-03")
    assert longest["run_length"] == 2


def test_target_outliers_exclude_discharge_rows():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"] * 10),
        "facility": ["A"] * 10,
        "plant": ["P"] * 10,
        "stream_type": ["유입수"] * 5 + ["방류"] * 5,
        "bod_mg_l": [1.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1000.0],
        "tn_mg_l": [1.0] * 10,
        "tp_mg_l": [1.0] * 10,
    })

    result = _outliers(frame)

    assert set(result["stream_type"]) == {"유입수"}
    assert 1000.0 not in set(result["original_value"])


def test_count_next_day_window_samples_requires_contiguous_history_and_target():
    from src.preprocessing.profile import count_next_day_window_samples

    frame = pd.DataFrame(
        {
            "series": ["A"] * 5,
            "date": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-05", "2022-01-06"]),
            "bod_mg_l": [1.0, 2.0, 3.0, 5.0, 6.0],
        }
    )
    result = count_next_day_window_samples(
        frame, group_columns=["series"], date_column="date",
        target_columns=["bod_mg_l"], windows=[2],
    )
    assert result.loc[0, "candidate_count"] == 1


def test_count_next_day_window_samples_assigns_split_from_target_date():
    from src.preprocessing.profile import count_next_day_window_samples

    frame = pd.DataFrame(
        {
            "series": ["A"] * 4,
            "date": pd.date_range("2022-12-29", periods=4),
            "bod_mg_l": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = count_next_day_window_samples(
        frame, group_columns=["series"], date_column="date",
        target_columns=["bod_mg_l"], windows=[2],
        validation_start="2023-01-01", test_start="2025-01-01",
    )
    counts = dict(zip(result["split"], result["candidate_count"]))
    assert counts == {"train": 1, "validation": 1}
