import pandas as pd
import pytest

from src.evaluation.portfolio_visualization import (
    SIMPLE_LINE_COLORS,
    assert_prediction_alignment,
    consecutive_segments,
    metric_row,
    select_dynamic_validation_window,
    two_line_series,
)


def test_alignment_rejects_actual_mismatch():
    a = pd.DataFrame({"split": ["validation"], "target": ["bod_mg_l"], "series_id": ["s"], "target_date": ["2023-01-01"], "actual": [1.0]})
    b = a.assign(actual=2.0)
    with pytest.raises(ValueError, match="실제값"):
        assert_prediction_alignment(a, b)


def test_consecutive_segments_never_bridges_gap():
    dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-05"])
    parts = consecutive_segments(pd.DataFrame({"target_date": dates}))
    assert [len(x) for x in parts] == [2, 1]


def test_metric_residual_is_prediction_minus_actual():
    row = metric_row(pd.Series([1.0, 3.0]), pd.Series([2.0, 1.0]))
    assert row["bias"] == pytest.approx(-0.5)
    assert row["mae"] == pytest.approx(1.5)


def test_dynamic_window_uses_validation_actual_only():
    dates = pd.date_range("2023-01-01", periods=8)
    frame = pd.DataFrame({"split": "validation", "target_date": dates, "series_id": "s", "actual": [0, 0, 0, 10, 0, 0, 0, 0]})
    start, end = select_dynamic_validation_window(frame, days=3)
    assert start <= pd.Timestamp("2023-01-04") <= end
    assert (end - start).days == 2


def test_two_line_series_keeps_only_actual_and_final_prediction():
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "actual": [1.0, 2.0],
            "prediction": [1.2, 1.8],
            "model": ["xgboost", "xgboost"],
        }
    )
    result = two_line_series(frame)
    assert list(result.columns) == ["target_date", "actual", "xgboost"]
    assert result["xgboost"].tolist() == [1.2, 1.8]
    assert SIMPLE_LINE_COLORS == {"actual": "#0072B2", "xgboost": "#E69F00"}


def test_two_line_series_rejects_non_xgboost_rows():
    frame = pd.DataFrame(
        {"target_date": ["2023-01-01"], "actual": [1.0], "prediction": [1.0], "model": ["persistence"]}
    )
    with pytest.raises(ValueError, match="XGBoost"):
        two_line_series(frame)
