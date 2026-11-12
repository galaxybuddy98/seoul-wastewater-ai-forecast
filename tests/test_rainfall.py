import numpy as np
import pandas as pd

from src.preprocessing.rainfall import add_rainfall_features, classify_daily_rainfall


def test_unconfirmed_blank_rainfall_stays_nan_with_reason_and_raw_value():
    raw = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "precipitation_mm": [np.nan, 0.0, 2.5],
        }
    )

    result = classify_daily_rainfall(raw)

    assert np.isnan(result.loc[0, "precipitation_processed_mm"])
    assert result.loc[0, "precipitation_raw_blank"]
    assert result.loc[0, "precipitation_processing_reason"] == "공란_의미_미확인"
    assert result.loc[1, "precipitation_processed_mm"] == 0.0
    assert result.loc[2, "precipitation_processed_mm"] == 2.5


def test_rainfall_rolling_sum_requires_every_day_to_be_known():
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8),
            "precipitation_processed_mm": [1.0, 0.0, np.nan, 2.0, 3.0, 0.0, 1.0, 4.0],
        }
    )

    result = add_rainfall_features(daily)

    assert np.isnan(result.loc[4, "precipitation_sum_3d"])
    assert result.loc[7, "precipitation_sum_3d"] == 5.0
    assert np.isnan(result.loc[7, "precipitation_sum_7d"])
    assert result.loc[3, "precipitation_lag_1d"] != result.loc[3, "precipitation_lag_1d"]
    assert result.loc[2, "precipitation_missing"] == 1
