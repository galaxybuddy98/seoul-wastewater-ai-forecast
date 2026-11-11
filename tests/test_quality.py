import pandas as pd

from src.preprocessing.quality import build_quality_summary, detect_outlier_candidates


def test_quality_summary_distinguishes_zero_missing_negative_and_duplicates():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"]),
            "facility": ["중랑"] * 4,
            "plant": ["제1처리장"] * 4,
            "bod_mg_l": [0.0, None, None, -1.0],
        }
    )

    summary = build_quality_summary(
        frame,
        group_columns=["facility", "plant"],
        value_columns=["bod_mg_l"],
        key_columns=["date", "facility", "plant"],
    )

    row = summary.iloc[0]
    assert row["observation_count"] == 4
    assert row["missing_count"] == 2
    assert row["zero_count"] == 1
    assert row["negative_count"] == 1
    assert row["duplicate_record_count"] == 2


def test_outlier_detection_marks_candidates_without_removing_rows():
    frame = pd.DataFrame({"bod_mg_l": [10.0, 11.0, 9.0, 10.0, 1000.0]})

    candidates = detect_outlier_candidates(frame, ["bod_mg_l"])

    assert len(frame) == 5
    assert candidates.loc[candidates["original_value"] == 1000.0, "iqr_candidate"].item()
