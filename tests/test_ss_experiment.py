import numpy as np
import pandas as pd

from src.models.ss_experiment import attach_ss_lags, ss_feature_columns


def test_attach_ss_lags_uses_only_same_series_past_dates_without_dropping_rows():
    base = pd.DataFrame(
        {
            "series_id": ["a", "b"],
            "target_date": pd.to_datetime(["2024-02-01", "2024-02-01"]),
            "marker": [1, 2],
        }
    )
    observations = pd.DataFrame(
        {
            "series_id": ["a", "a", "a", "b", "b", "b"],
            "date": pd.to_datetime(
                ["2024-01-30", "2024-01-31", "2024-02-01"] * 2
            ),
            "ss_mg_l": [10.0, 11.0, 999.0, 20.0, np.nan, 888.0],
        }
    )

    actual = attach_ss_lags(base, observations, window_days=2)

    assert len(actual) == 2
    assert actual.loc[0, "ss_mg_l_lag_0"] == 11.0
    assert actual.loc[0, "ss_mg_l_lag_1"] == 10.0
    assert np.isnan(actual.loc[1, "ss_mg_l_lag_0"])
    assert actual.loc[1, "ss_mg_l_lag_1"] == 20.0
    assert 999.0 not in actual.filter(like="ss_mg_l_lag_").to_numpy()
    assert 888.0 not in actual.filter(like="ss_mg_l_lag_").to_numpy()


def test_attach_ss_lags_rejects_duplicate_observation_keys():
    base = pd.DataFrame({"series_id": ["a"], "target_date": pd.to_datetime(["2024-02-01"])})
    observations = pd.DataFrame(
        {
            "series_id": ["a", "a"],
            "date": pd.to_datetime(["2024-01-31", "2024-01-31"]),
            "ss_mg_l": [10.0, 11.0],
        }
    )

    try:
        attach_ss_lags(base, observations, window_days=1)
    except ValueError as error:
        assert "중복" in str(error)
    else:
        raise AssertionError("중복 SS 키를 허용했습니다")


def test_ss_feature_columns_are_exactly_the_requested_window():
    assert ss_feature_columns(3) == [
        "ss_mg_l_lag_0",
        "ss_mg_l_lag_1",
        "ss_mg_l_lag_2",
    ]
