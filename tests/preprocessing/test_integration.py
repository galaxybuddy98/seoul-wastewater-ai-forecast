import pandas as pd
import pytest

from src.preprocessing.integration import IntegrationRuleError, aggregate_center_daily


def test_pending_integration_rule_is_rejected():
    frame = pd.DataFrame({"facility": ["중랑"], "date": ["2024-01-01"], "bod_mg_l": [100.0]})

    with pytest.raises(IntegrationRuleError, match="보류"):
        aggregate_center_daily(frame, {"status": "보류", "method": None})


def test_unverified_arithmetic_mean_is_rejected():
    frame = pd.DataFrame({"facility": ["중랑"], "date": ["2024-01-01"], "bod_mg_l": [100.0]})

    with pytest.raises(IntegrationRuleError, match="산술평균"):
        aggregate_center_daily(frame, {"status": "검증완료", "method": "산술평균"})
