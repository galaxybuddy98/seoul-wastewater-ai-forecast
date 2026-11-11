"""검증되지 않은 센터 통합을 방지하는 안전장치."""

from __future__ import annotations

from typing import Any

import pandas as pd


class IntegrationRuleError(ValueError):
    """통합 근거가 없거나 허용되지 않은 집계인 경우."""


def aggregate_center_daily(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    """현재 단계에서는 미검증·산술평균 규칙을 명시적으로 거부한다."""
    status = rule.get("status")
    method = rule.get("method")
    if status != "검증완료":
        raise IntegrationRuleError(f"센터 통합 규칙이 {status or '미지정'} 상태입니다")
    if method == "산술평균":
        raise IntegrationRuleError("근거 없는 산술평균은 허용되지 않습니다")
    raise IntegrationRuleError(f"구현 전 공식 근거가 필요한 통합 방식입니다: {method or '미지정'}")
