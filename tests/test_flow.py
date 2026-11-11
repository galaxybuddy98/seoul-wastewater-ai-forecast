import numpy as np
import pandas as pd

from src.preprocessing.flow import add_treatment_total_hypothesis


def test_treatment_total_hypothesis_sums_components_and_preserves_sources():
    frame = pd.DataFrame(
        {
            "primary_treatment_volume": [10.0],
            "secondary_treatment_volume": [90.0],
        }
    )

    result = add_treatment_total_hypothesis(frame)

    assert result.loc[0, "total_treatment_volume_hypothesis"] == 100.0
    assert result.loc[0, "primary_treatment_volume"] == 10.0
    assert result.loc[0, "secondary_treatment_volume"] == 90.0


def test_treatment_total_hypothesis_stays_missing_when_a_component_is_missing():
    frame = pd.DataFrame(
        {
            "primary_treatment_volume": [np.nan, 0.0],
            "secondary_treatment_volume": [90.0, np.nan],
        }
    )

    result = add_treatment_total_hypothesis(frame)

    assert result["total_treatment_volume_hypothesis"].isna().all()
