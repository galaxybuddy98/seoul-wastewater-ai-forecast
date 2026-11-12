import numpy as np
import pandas as pd

from src.models.lightgbm_baseline import (
    regression_metrics,
    select_lag_features,
)


def test_regression_metrics_match_hand_calculated_values():
    result = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))

    assert result["mae"] == 2.0 / 3.0
    assert np.isclose(result["rmse"], np.sqrt(4.0 / 3.0))
    assert result["r2"] == -1.0


def test_select_lag_features_never_uses_targets_or_dates():
    frame = pd.DataFrame(
        columns=[
            "series_id",
            "target_date",
            "bod_mg_l_lag_0",
            "mean_temperature_c_lag_0",
            "target_bod_mg_l",
        ]
    )

    assert select_lag_features(frame) == [
        "bod_mg_l_lag_0",
        "mean_temperature_c_lag_0",
    ]


def test_select_lag_features_can_exclude_flow_for_fair_reference():
    frame = pd.DataFrame(
        columns=[
            "bod_mg_l_lag_0",
            "primary_treatment_volume_lag_0",
            "secondary_treatment_volume_lag_0",
        ]
    )

    assert select_lag_features(frame, include_flow=False) == ["bod_mg_l_lag_0"]
