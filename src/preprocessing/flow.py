"""처리량 합산 가설용 보수적 변환."""

from __future__ import annotations

import pandas as pd


_REQUIRED_COLUMNS = (
    "primary_treatment_volume",
    "secondary_treatment_volume",
)


def add_treatment_total_hypothesis(frame: pd.DataFrame) -> pd.DataFrame:
    """두 처리량 성분이 모두 관측된 행에만 합산 가설을 추가한다."""
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"처리량 합산에 필요한 열이 없습니다: {missing}")

    result = frame.copy()
    result["total_treatment_volume_hypothesis"] = result.loc[:, _REQUIRED_COLUMNS].sum(
        axis=1, min_count=2
    )
    return result
