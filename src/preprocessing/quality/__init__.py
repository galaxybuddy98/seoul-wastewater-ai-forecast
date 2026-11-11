"""원본을 변경하지 않는 데이터 품질 분석 공개 인터페이스."""

from pathlib import Path

import numpy as np
import pandas as pd


def read_source_csv(path, encoding: str = "cp949") -> pd.DataFrame:
    return pd.read_csv(path, encoding=encoding, low_memory=False)


def build_quality_summary(frame, group_columns, value_columns, key_columns):
    duplicate_mask = frame.duplicated(key_columns, keep=False)
    rows = []
    grouper = group_columns[0] if len(group_columns) == 1 else group_columns
    for group_key, group in frame.groupby(grouper, dropna=False, sort=True):
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        group_values = dict(zip(group_columns, keys))
        for column in value_columns:
            numeric = pd.to_numeric(group[column], errors="coerce")
            row = {
                **group_values,
                "variable": column,
                "observation_count": int(len(group)),
                "missing_count": int(numeric.isna().sum()),
                "missing_rate": float(numeric.isna().mean()),
                "zero_count": int(numeric.eq(0).sum()),
                "zero_rate": float(numeric.eq(0).mean()),
                "negative_count": int(numeric.lt(0).sum()),
                "duplicate_record_count": int(duplicate_mask.loc[group.index].sum()),
            }
            if "date" in group:
                dates = pd.to_datetime(group["date"], errors="coerce")
                row["start_date"], row["end_date"] = dates.min(), dates.max()
            rows.append(row)
    return pd.DataFrame(rows)


def detect_outlier_candidates(frame, value_columns):
    records = []
    for column in value_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.dropna()
        q1, q3 = valid.quantile([0.25, 0.75]) if not valid.empty else (np.nan, np.nan)
        iqr, median = q3 - q1, valid.median()
        mad = (valid - median).abs().median()
        for index, value in values.items():
            if pd.isna(value):
                continue
            iqr_candidate = bool(iqr > 0 and (value < q1 - 1.5 * iqr or value > q3 + 1.5 * iqr))
            mad_score = 0.0 if not mad or pd.isna(mad) else 0.6745 * abs(value - median) / mad
            mad_candidate = bool(mad_score > 3.5)
            if iqr_candidate or mad_candidate:
                records.append({
                    "row_index": index,
                    "variable": column,
                    "original_value": float(value),
                    "iqr_candidate": iqr_candidate,
                    "mad_candidate": mad_candidate,
                    "mad_score": float(mad_score),
                    "decision": "검토 필요",
                    "action": "유지",
                })
    return pd.DataFrame(records)


def write_quality_outputs(summary, candidates, output_dir, source_label: str):
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_csv": target / "quality_summary.csv",
        "outliers_csv": target / "outlier_candidates.csv",
        "report_md": target / "data_quality_report.md",
    }
    summary.to_csv(paths["summary_csv"], index=False, encoding="utf-8-sig")
    candidates.to_csv(paths["outliers_csv"], index=False, encoding="utf-8-sig")
    paths["report_md"].write_text(
        f"# {source_label} 데이터 품질 보고서\n\n"
        "이 보고서는 실제 원자료 기반 자동 요약입니다.\n\n"
        f"- 품질 요약 행 수: {len(summary):,}\n"
        f"- 이상치 후보 행 수: {len(candidates):,}\n"
        "- 이상치 후보는 자동 삭제하거나 대체하지 않았습니다.\n",
        encoding="utf-8",
    )
    return paths
