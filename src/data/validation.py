"""API 정규화 결과와 CSV 스냅샷의 비파괴 대조."""

from __future__ import annotations

import pandas as pd


def compare_snapshots(
    csv_frame: pd.DataFrame,
    api_frame: pd.DataFrame,
    key_columns: list[str],
    value_columns: list[str],
) -> dict[str, object]:
    """복합키 누락과 공통행 측정값 차이를 분리해 반환한다."""
    for label, frame in (("CSV", csv_frame), ("API", api_frame)):
        if frame.duplicated(key_columns, keep=False).any():
            raise ValueError(f"{label} 복합키에 중복이 있어 일대일 대조가 불가능합니다")

    merged = csv_frame[key_columns + value_columns].merge(
        api_frame[key_columns + value_columns],
        on=key_columns,
        how="outer",
        suffixes=("_csv", "_api"),
        indicator=True,
        validate="one_to_one",
    )
    common = merged["_merge"].eq("both")
    changed = pd.Series(False, index=merged.index)
    for column in value_columns:
        left = merged[f"{column}_csv"]
        right = merged[f"{column}_api"]
        equal = left.eq(right) | (left.isna() & right.isna())
        changed |= common & ~equal

    differences = merged.loc[~common | changed].copy()
    differences["difference_type"] = "value_changed"
    differences.loc[merged["_merge"].eq("left_only"), "difference_type"] = "csv_only"
    differences.loc[merged["_merge"].eq("right_only"), "difference_type"] = "api_only"
    summary = {
        "csv_rows": len(csv_frame),
        "api_rows": len(api_frame),
        "csv_only_rows": int(merged["_merge"].eq("left_only").sum()),
        "api_only_rows": int(merged["_merge"].eq("right_only").sum()),
        "changed_rows": int(changed.sum()),
    }
    return {"summary": summary, "differences": differences.drop(columns="_merge")}
