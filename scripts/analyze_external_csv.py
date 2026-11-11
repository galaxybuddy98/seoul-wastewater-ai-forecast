"""외부 CSV 원본을 변경하지 않고 품질 통계 파일을 생성한다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.flow import add_treatment_total_hypothesis
from src.preprocessing.profile import (
    count_next_day_window_samples,
    find_calendar_gaps,
    find_missing_runs,
)
from src.preprocessing.schema import normalize_flow_columns, normalize_quality_columns

EXTERNAL = Path("data/external")
OUTPUT = Path("reports/data_quality")
QUALITY_VALUES = ["bod_mg_l", "toc_or_cod_value", "ss_mg_l", "tn_mg_l", "tp_mg_l", "ph", "total_coliform"]
FLOW_VALUES = ["primary_treatment_volume", "secondary_treatment_volume"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUTPUT / name, index=False, encoding="utf-8-sig")


def _outliers(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.loc[frame["stream_type"].eq("유입수")].copy()
    records = []
    group_columns = ["facility", "stream_type"]
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        for variable in ["bod_mg_l", "tn_mg_l", "tp_mg_l"]:
            values = group[variable].dropna()
            if values.empty:
                continue
            q1, q3 = values.quantile([0.25, 0.75])
            median = values.median()
            iqr = q3 - q1
            mad = (values - median).abs().median()
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            score = pd.Series(0.0, index=group.index)
            if mad > 0:
                score = 0.6745 * (group[variable] - median).abs() / mad
            mask_iqr = group[variable].lt(lower) | group[variable].gt(upper)
            mask_mad = score.gt(3.5)
            for index in group.index[mask_iqr | mask_mad]:
                records.append({
                    "date": group.at[index, "date"],
                    "facility": keys[0],
                    "plant": group.at[index, "plant"],
                    "stream_type": keys[1],
                    "variable": variable,
                    "original_value": group.at[index, variable],
                    "iqr_lower": lower,
                    "iqr_upper": upper,
                    "mad_score": score.at[index],
                    "iqr_candidate": bool(mask_iqr.at[index]),
                    "mad_candidate": bool(mask_mad.at[index]),
                    "action": "원본 유지·검토 필요",
                })
    return pd.DataFrame(records)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = sorted(EXTERNAL.glob("*.csv"))
    raw = [(path, pd.read_csv(path, encoding="cp949", low_memory=False)) for path in sources]
    quality_path, quality_raw = next(item for item in raw if len(item[1].columns) == 11)
    flow_path, flow_raw = next(item for item in raw if len(item[1].columns) == 5)
    quality = normalize_quality_columns(quality_raw)
    flow = add_treatment_total_hypothesis(normalize_flow_columns(flow_raw))
    quality["year"] = quality["date"].dt.year
    flow["year"] = flow["date"].dt.year

    overview = pd.DataFrame([
        {"dataset": "수질", "file": quality_path.name, "encoding": "cp949",
         "rows": len(quality_raw), "columns": len(quality_raw.columns),
         "start_date": quality["date"].min(), "end_date": quality["date"].max(),
         "invalid_date_count": int(quality["date"].isna().sum()), "sha256": _sha256(quality_path)},
        {"dataset": "처리량", "file": flow_path.name, "encoding": "cp949",
         "rows": len(flow_raw), "columns": len(flow_raw.columns),
         "start_date": flow["date"].min(), "end_date": flow["date"].max(),
         "invalid_date_count": int(flow["date"].isna().sum()), "sha256": _sha256(flow_path)},
    ])
    _write(overview, "dataset_overview.csv")

    quality_groups = quality.groupby(["facility", "plant", "stream_type"], dropna=False).agg(
        rows=("date", "size"), unique_dates=("date", "nunique"),
        start_date=("date", "min"), end_date=("date", "max")).reset_index()
    flow_groups = flow.groupby(["facility", "plant"], dropna=False).agg(
        rows=("date", "size"), unique_dates=("date", "nunique"),
        start_date=("date", "min"), end_date=("date", "max")).reset_index()
    _write(quality_groups, "quality_group_counts.csv")
    _write(flow_groups, "flow_group_counts.csv")

    influent = quality.loc[quality["stream_type"].eq("유입수")].copy()
    daily_counts = influent.groupby(["facility", "date"]).size().rename("rows_per_date").reset_index()
    daily_structure = daily_counts.groupby(["facility", "rows_per_date"]).size().rename("date_count").reset_index()
    _write(daily_structure, "influent_daily_multiplicity.csv")
    target_summary = influent.groupby("facility").agg(
        rows=("date", "size"), unique_dates=("date", "nunique"),
        start_date=("date", "min"), end_date=("date", "max"),
        bod_missing=("bod_mg_l", lambda x: int(x.isna().sum())),
        tn_missing=("tn_mg_l", lambda x: int(x.isna().sum())),
        tp_missing=("tp_mg_l", lambda x: int(x.isna().sum())),
        bod_zero=("bod_mg_l", lambda x: int(x.eq(0).sum())),
        tn_zero=("tn_mg_l", lambda x: int(x.eq(0).sum())),
        tp_zero=("tp_mg_l", lambda x: int(x.eq(0).sum())),
    ).reset_index()
    _write(target_summary, "influent_target_summary.csv")
    coverage = influent.merge(
        flow[["date", "facility", "plant"]],
        on=["date", "facility", "plant"], how="left", indicator=True
    ).groupby(["facility", "plant", "_merge"], observed=True).size().rename("row_count").reset_index()
    _write(coverage, "quality_flow_key_coverage.csv")

    missing_rows = []
    for dataset, frame, values in [("수질", quality, QUALITY_VALUES), ("처리량", flow, FLOW_VALUES)]:
        for (facility, year), group in frame.groupby(["facility", "year"], sort=True):
            for variable in values:
                values_numeric = group[variable]
                missing_rows.append({
                    "dataset": dataset, "facility": facility, "year": year, "variable": variable,
                    "row_count": len(group), "missing_count": int(values_numeric.isna().sum()),
                    "missing_rate": float(values_numeric.isna().mean()),
                    "zero_count": int(values_numeric.eq(0).sum()),
                    "zero_rate": float(values_numeric.eq(0).mean()),
                    "negative_count": int(values_numeric.lt(0).sum()),
                })
    _write(pd.DataFrame(missing_rows), "missing_zero_negative_by_center_year.csv")

    quality_series = ["facility", "plant", "stream_type"]
    flow_series = ["facility", "plant"]
    q_gaps = find_calendar_gaps(quality, quality_series, "date").assign(dataset="수질")
    f_gaps = find_calendar_gaps(flow, flow_series, "date").assign(dataset="처리량", stream_type=pd.NA)
    _write(pd.concat([q_gaps, f_gaps], ignore_index=True, sort=False), "calendar_gaps.csv")
    q_runs = find_missing_runs(quality, quality_series, "date", QUALITY_VALUES).assign(dataset="수질")
    f_runs = find_missing_runs(flow, flow_series, "date", FLOW_VALUES).assign(dataset="처리량", stream_type=pd.NA)
    _write(pd.concat([q_runs, f_runs], ignore_index=True, sort=False), "missing_runs.csv")

    q_keys = ["date", "facility", "plant", "stream_type"]
    f_keys = ["date", "facility", "plant"]
    duplicate_summary = pd.DataFrame([
        {"dataset": "수질", "exact_duplicate_rows": int(quality_raw.duplicated(keep=False).sum()),
         "duplicate_key_rows": int(quality.duplicated(q_keys, keep=False).sum())},
        {"dataset": "처리량", "exact_duplicate_rows": int(flow_raw.duplicated(keep=False).sum()),
         "duplicate_key_rows": int(flow.duplicated(f_keys, keep=False).sum())},
    ])
    _write(duplicate_summary, "duplicate_summary.csv")
    outliers = _outliers(quality)
    _write(outliers, "influent_target_outlier_candidates.csv")
    outlier_distribution = (
        outliers.assign(year=pd.to_datetime(outliers["date"]).dt.year)
        .groupby(["facility", "year", "plant", "variable"], dropna=False)
        .agg(
            candidate_count=("date", "size"),
            iqr_candidate_count=("iqr_candidate", "sum"),
            mad_candidate_count=("mad_candidate", "sum"),
        )
        .reset_index()
    )
    _write(outlier_distribution, "outlier_distribution.csv")

    quality_zeros = (
        influent.loc[influent["bod_mg_l"].eq(0), ["date", "facility", "plant", "stream_type"]]
        .assign(variable="bod_mg_l", value=0.0, dataset="수질")
    )
    flow_zero_rows = []
    for variable in FLOW_VALUES:
        part = flow.loc[flow[variable].eq(0), ["date", "facility", "plant", variable]].copy()
        part = part.rename(columns={variable: "value"})
        part["variable"] = variable
        part["dataset"] = "처리량"
        part["stream_type"] = pd.NA
        flow_zero_rows.append(part)
    zero_events = pd.concat([quality_zeros, *flow_zero_rows], ignore_index=True, sort=False)
    _write(zero_events, "zero_value_events.csv")

    candidate_series = influent.loc[
        (
            influent["plant"].eq("처리장 구분없음")
            & influent["facility"].isin(["난지물재생센터", "탄천물재생센터", "서남물재생센터"])
        )
        | influent["facility"].eq("중랑물재생센터")
    ].copy()
    candidate_series["series_basis"] = np.where(
        candidate_series["plant"].eq("처리장 구분없음"),
        "센터 대표 후보",
        "처리장별 대안",
    )
    sample_parts = []
    for target in ["bod_mg_l", "tn_mg_l", "tp_mg_l"]:
        sample = count_next_day_window_samples(
            candidate_series,
            group_columns=["facility", "plant", "series_basis"],
            date_column="date",
            target_columns=[target],
            windows=[7, 14, 30],
        )
        sample["target"] = target
        sample_parts.append(sample)
    _write(pd.concat(sample_parts, ignore_index=True), "window_sample_availability.csv")

    flow_relation = flow.groupby(["facility", "plant"], dropna=False).agg(
        rows=("date", "size"),
        primary_missing=("primary_treatment_volume", lambda x: int(x.isna().sum())),
        secondary_missing=("secondary_treatment_volume", lambda x: int(x.isna().sum())),
        primary_zero=("primary_treatment_volume", lambda x: int(x.eq(0).sum())),
        secondary_zero=("secondary_treatment_volume", lambda x: int(x.eq(0).sum())),
        primary_sum=("primary_treatment_volume", "sum"),
        secondary_sum=("secondary_treatment_volume", "sum"),
    ).reset_index()
    _write(flow_relation, "flow_component_summary.csv")

    flow_hypothesis = flow.assign(
        primary_status=np.where(
            flow["primary_treatment_volume"].isna(),
            "결측",
            np.where(flow["primary_treatment_volume"].eq(0), "0", "양수"),
        )
    )
    hypothesis_summary = (
        flow_hypothesis.groupby(["facility", "plant", "primary_status"], dropna=False)
        .agg(
            rows=("date", "size"),
            total_observed_count=("total_treatment_volume_hypothesis", "count"),
            total_missing_count=("total_treatment_volume_hypothesis", lambda x: int(x.isna().sum())),
            primary_sum=("primary_treatment_volume", "sum"),
            secondary_sum=("secondary_treatment_volume", "sum"),
            total_hypothesis_sum=("total_treatment_volume_hypothesis", "sum"),
            secondary_median=("secondary_treatment_volume", "median"),
        )
        .reset_index()
    )
    hypothesis_summary["status"] = "계산 가설·공식 총량 및 실제 유입유량 아님"
    _write(hypothesis_summary, "treatment_total_hypothesis_summary.csv")

    manifest = {
        "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "source_files": [path.name for path in sources],
        "policy": "기술 통계 전용. 보간·삭제·모델 학습 없음. 이상치 경계는 모델 전처리 파라미터가 아님.",
        "outputs": sorted(path.name for path in OUTPUT.glob("*.csv")),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
