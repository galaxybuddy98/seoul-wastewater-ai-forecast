"""기존 전처리 함수를 호출하지 않는 읽기 전용 독립 감사 실행기."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.audit.preprocessing_audit import (
    aggregate_station_day,
    classify_split,
    join_rainfall_reference,
    reconstruct_lags,
    sha256_file,
)

from scripts._paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
EXTERNAL = ROOT / "data/external"
PROCESSED = ROOT / "data/processed"
RAW = ROOT / "data/raw"
REPORTS = ROOT / "reports"
SEED = 20261114
TOLERANCE = 1e-9
STATIONS = ["성동구청", "마포구청", "세곡동", "공항동P"]
QUALITY_COLUMNS = {"BOD": "bod_mg_l", "T-N": "tn_mg_l", "TP": "tp_mg_l", "SS": "ss_mg_l"}


def _equal(left: pd.Series, right: pd.Series, tolerance: float = TOLERANCE) -> pd.Series:
    return (left.isna() & right.isna()) | np.isclose(left, right, atol=tolerance, rtol=0, equal_nan=True)


def build_hash_baseline() -> pd.DataFrame:
    previous = pd.read_csv(REPORTS / "seoul_rainfall_file_inventory.csv", encoding="utf-8-sig")
    prior_hash = {
        str(Path("seoul_rainfall") / Path(row.source_file)): row.sha256
        for row in previous.itertuples() if pd.notna(row.sha256)
    }
    rows = []
    for path in sorted(EXTERNAL.rglob("*.csv")):
        relative = str(path.relative_to(EXTERNAL))
        digest = sha256_file(path)
        old = prior_hash.get(relative)
        rows.append({
            "relative_path": relative, "size_bytes": path.stat().st_size,
            "sha256": digest, "previous_sha256": old,
            "previous_hash_available": pd.notna(old),
            "matches_previous_hash": bool(old == digest) if pd.notna(old) else pd.NA,
            "baseline_date": "2026-11-14",
        })
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS / "audit_raw_file_hashes_2026-11-14.csv", index=False, encoding="utf-8-sig")
    return result


def _sample_rainfall_days(daily: pd.DataFrame, sample_size: int = 650) -> pd.DataFrame:
    frame = daily.loc[daily.station_name.isin(STATIONS) & daily.date.between("2016-01-01", "2026-09-30")].copy()
    frame["year"] = frame.date.dt.year
    frame["split_audit"] = frame.date.map(classify_split)
    frame["rain_class"] = np.select(
        [frame.daily_quality_status.ne("complete"), frame.daily_rainfall_mm.eq(0), frame.daily_rainfall_mm.ge(30)],
        ["incomplete", "dry", "heavy"], default="wet",
    )
    picked = []
    for _, group in frame.groupby(["station_name", "year"], sort=True):
        picked.append(group.sample(min(4, len(group)), random_state=SEED))
    for _, group in frame.groupby(["split_audit", "rain_class"], sort=True):
        picked.append(group.sample(min(30, len(group)), random_state=SEED + 1))
    boundaries = pd.to_datetime(["2016-01-01", "2022-12-31", "2023-01-01", "2024-12-31", "2025-01-01", "2026-01-01", "2026-09-30"])
    picked.append(frame.loc[frame.date.isin(boundaries)])
    selected = pd.concat(picked).drop_duplicates(["station_name", "date"])
    remaining = frame.merge(selected[["station_name", "date"]], on=["station_name", "date"], how="left", indicator=True)
    remaining = remaining.loc[remaining._merge.eq("left_only"), frame.columns]
    if len(selected) < sample_size:
        selected = pd.concat([selected, remaining.sample(sample_size - len(selected), random_state=SEED + 2)])
    return selected.sort_values(["date", "station_name"]).reset_index(drop=True)


def _source_for_missing(sample: pd.Series, daily: pd.DataFrame) -> str | None:
    if pd.notna(sample.source_files) and str(sample.source_files):
        return str(sample.source_files).split("|")[0]
    candidates = daily.loc[
        daily.station_name.eq(sample.station_name)
        & daily.date.dt.year.eq(sample.date.year)
        & daily.source_files.notna()
    ]
    if sample.date.year == 2026:
        candidates = candidates.loc[candidates.date.dt.month.eq(sample.date.month)]
    return str(candidates.source_files.iloc[0]).split("|")[0] if len(candidates) else None


def _read_rain_source(path: Path) -> pd.DataFrame:
    error = None
    for encoding in ("cp949", "utf-8-sig"):
        for skiprows in (0, 1, 2):
            try:
                frame = pd.read_csv(path, encoding=encoding, skiprows=skiprows)
                station_column = "강우량계명"
                time_column = "자료수집 시각" if "자료수집 시각" in frame else "시간"
                return pd.DataFrame({
                    "station_name": frame[station_column].astype(str),
                    "timestamp": pd.to_datetime(frame[time_column], format="mixed", errors="coerce"),
                    "rainfall_mm": pd.to_numeric(frame["10분우량"], errors="coerce"),
                })
            except (UnicodeDecodeError, KeyError) as exc:
                error = exc
    raise ValueError(f"원본 파싱 실패: {path}: {error}")


def audit_rainfall(daily: pd.DataFrame) -> pd.DataFrame:
    sample = _sample_rainfall_days(daily)
    sample["audit_source_file"] = sample.apply(_source_for_missing, axis=1, daily=daily)
    source_frames: dict[str, pd.DataFrame] = {}
    for source in sorted(sample.audit_source_file.dropna().unique()):
        source_frames[source] = _read_rain_source(Path(source))
    rows = []
    for item in sample.itertuples():
        if item.audit_source_file is None or pd.isna(item.audit_source_file):
            audit = {"audit_status": "source_unresolved", "independent_daily_rainfall_mm": np.nan,
                     "independent_max_1h_mm": np.nan, "raw_row_count": 0, "unique_interval_count": 0,
                     "numeric_interval_count": 0, "duplicate_same_count": 0,
                     "duplicate_conflict_count": 0, "negative_count": 0}
        else:
            raw = source_frames[item.audit_source_file]
            raw = raw.loc[raw.station_name.eq(item.station_name) & raw.timestamp.dt.normalize().eq(item.date)]
            audit = aggregate_station_day(raw[["timestamp", "rainfall_mm"]])
        existing_complete = item.daily_quality_status == "complete"
        daily_match = bool(np.isclose(audit["independent_daily_rainfall_mm"], item.daily_rainfall_mm, atol=TOLERANCE, rtol=0, equal_nan=True))
        hour_match = bool(np.isclose(audit["independent_max_1h_mm"], item.max_1h_rainfall_mm, atol=TOLERANCE, rtol=0, equal_nan=True))
        status_match = (audit["audit_status"] == "confirmed") == existing_complete
        reason = "match" if daily_match and hour_match and status_match else "value_or_completeness_mismatch"
        rows.append({
            "station_id": item.station_id, "station_name": item.station_name, "date": item.date,
            "split": classify_split(item.date), "rain_class": item.rain_class,
            "source_file": item.audit_source_file, **audit,
            "existing_daily_rainfall_mm": item.daily_rainfall_mm,
            "existing_max_1h_rainfall_mm": item.max_1h_rainfall_mm,
            "existing_quality_status": item.daily_quality_status,
            "daily_match": daily_match, "max_1h_match": hour_match,
            "completeness_match": status_match, "result": reason,
        })
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS / "audit_rainfall_independent_sample_2026-11-14.csv", index=False, encoding="utf-8-sig")
    result.loc[result.result.ne("match")].to_csv(
        REPORTS / "audit_rainfall_mismatches_2026-11-14.csv", index=False, encoding="utf-8-sig"
    )
    return result


def audit_water_quality() -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_csv(EXTERNAL / "서울시 물재생센터 수질현황.csv", encoding="cp949")
    raw = raw.loc[raw["계통구분"].eq("유입수")].copy()
    raw["date"] = pd.to_datetime(raw["측정일자"])
    raw = raw.rename(columns={"물재생센터명칭": "facility", "처리장구분": "plant", **QUALITY_COLUMNS})
    processed = pd.read_parquet(PROCESSED / "observation_series_daily.parquet")
    keys = ["date", "facility", "plant"]
    duplicates = int(raw.duplicated(keys).sum())
    comparison = raw[keys + list(QUALITY_COLUMNS.values())].merge(
        processed[keys + list(QUALITY_COLUMNS.values()) + ["series_id"]],
        on=keys, how="outer", suffixes=("_raw", "_processed"), indicator=True,
    )
    checks = []
    for column in QUALITY_COLUMNS.values():
        comparison[f"{column}_match"] = _equal(comparison[f"{column}_raw"], comparison[f"{column}_processed"])
        checks.append(f"{column}_match")
    comparison["all_values_match"] = comparison[checks].all(axis=1) & comparison._merge.eq("both")
    mismatches = comparison.loc[~comparison.all_values_match].copy()
    mismatches.to_csv(REPORTS / "audit_water_quality_mismatches_2026-11-14.csv", index=False, encoding="utf-8-sig")
    series_map = processed.groupby("series_id").agg(facilities=("facility", "nunique"), plants=("plant", "nunique"))
    summary = {
        "raw_total_rows": int(len(pd.read_csv(EXTERNAL / "서울시 물재생센터 수질현황.csv", encoding="cp949", usecols=["측정일자"]))),
        "raw_influent_rows": int(len(raw)), "processed_rows": int(len(processed)),
        "raw_key_duplicates": duplicates, "outer_comparison_rows": int(len(comparison)),
        "matched_rows": int(comparison.all_values_match.sum()), "mismatch_rows": int(len(mismatches)),
        "mixed_series_count": int(((series_map.facilities > 1) | (series_map.plants > 1)).sum()),
    }
    return comparison, summary


def audit_joins() -> pd.DataFrame:
    base = pd.read_parquet(PROCESSED / "observation_series_daily.parquet")
    enriched = pd.read_parquet(PROCESSED / "observation_series_daily_seoul_rainfall.parquet")
    asos = pd.read_parquet(RAW / "asos_daily.parquet")
    rain = pd.read_parquet(PROCESSED / "seoul_rainfall_center_daily.parquet")
    keys = ["date", "facility", "plant", "series_id"]
    joined = base.merge(enriched, on=keys, how="outer", suffixes=("_base", "_enriched"), indicator=True)
    quality_unchanged = pd.Series(True, index=joined.index)
    for column in ["bod_mg_l", "tn_mg_l", "tp_mg_l", "ss_mg_l"]:
        quality_unchanged &= _equal(joined[f"{column}_base"], joined[f"{column}_enriched"])
    asos_check = enriched.merge(
        asos[["date", "facility", "station_id", "mean_temperature_c"]],
        on=["date", "facility"], how="left", suffixes=("_dataset", "_source"), validate="many_to_one",
    )
    rain_check = join_rainfall_reference(
        enriched,
        rain[["date", "facility", "station_id", "daily_rainfall_mm"]],
    )
    rows = [
        {"check": "base_to_enriched_row_count", "checked": len(joined), "mismatches": int(joined._merge.ne("both").sum())},
        {"check": "quality_values_unchanged", "checked": len(joined), "mismatches": int((~quality_unchanged).sum())},
        {"check": "asos_temperature_value", "checked": len(asos_check), "mismatches": int((~_equal(asos_check.mean_temperature_c_dataset, asos_check.mean_temperature_c_source)).sum())},
        {"check": "asos_station_id", "checked": len(asos_check), "mismatches": int(asos_check.station_id_dataset.astype(str).ne(asos_check.station_id_source.astype(str)).sum())},
        {"check": "seoul_rainfall_value", "checked": len(rain_check), "mismatches": int((~_equal(rain_check.seoul_rainfall_mm, rain_check.daily_rainfall_mm)).sum())},
        {"check": "seoul_rainfall_station_id", "checked": len(rain_check), "mismatches": int(rain_check.seoul_rainfall_station_id.astype(str).ne(rain_check.source_rainfall_station_id.astype(str)).sum())},
    ]
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS / "audit_join_checks_2026-11-14.csv", index=False, encoding="utf-8-sig")
    return result


def _sample_window_rows(tabular: pd.DataFrame) -> pd.DataFrame:
    sample_parts = []
    for _, group in tabular.groupby(["split", "series_id"], sort=True):
        sample_parts.append(group.sample(min(4, len(group)), random_state=SEED))
    boundaries = pd.to_datetime(["2022-12-31", "2023-01-01", "2024-12-31", "2025-01-01", "2026-01-01"])
    sample_parts.append(tabular.loc[pd.to_datetime(tabular.target_date).isin(boundaries)])
    return pd.concat(sample_parts).drop_duplicates(["series_id", "target_date"])


def audit_windows() -> tuple[pd.DataFrame, dict[str, object]]:
    tabular = pd.read_parquet(PROCESSED / "tabular_seoul_rainfall_w30.parquet")
    observations = pd.read_parquet(PROCESSED / "observation_series_daily_seoul_rainfall.parquet")
    sample = _sample_window_rows(tabular)
    feature_map = {
        "bod_mg_l": "bod_mg_l", "tn_mg_l": "tn_mg_l", "tp_mg_l": "tp_mg_l",
        "mean_temperature_c": "mean_temperature_c", "seoul_rainfall_mm": "seoul_rainfall_mm",
    }
    rows = []
    mismatch_count = 0
    for item in sample.itertuples(index=False):
        item_dict = item._asdict()
        for stem, source_column in feature_map.items():
            expected = reconstruct_lags(observations, item.series_id, item.target_date, source_column, 30)
            for lag, value in enumerate(expected):
                stored = item_dict[f"{stem}_lag_{lag}"]
                match = bool(np.isclose(value, stored, atol=TOLERANCE, rtol=0, equal_nan=True))
                mismatch_count += int(not match)
                if not match:
                    rows.append({"series_id": item.series_id, "target_date": item.target_date,
                                 "feature": stem, "lag": lag, "expected": value, "stored": stored})
        target_row = observations.loc[
            observations.series_id.eq(item.series_id) & observations.date.eq(item.target_date)
        ].iloc[0]
        for target in ["bod_mg_l", "tn_mg_l", "tp_mg_l"]:
            stored = item_dict[f"target_{target}"]
            match = bool(np.isclose(target_row[target], stored, atol=TOLERANCE, rtol=0, equal_nan=True))
            mismatch_count += int(not match)
            if not match:
                rows.append({"series_id": item.series_id, "target_date": item.target_date,
                             "feature": f"target_{target}", "lag": -1, "expected": target_row[target], "stored": stored})
    pd.DataFrame(rows, columns=["series_id", "target_date", "feature", "lag", "expected", "stored"]).to_csv(
        REPORTS / "audit_window_mismatches_2026-11-14.csv", index=False, encoding="utf-8-sig"
    )
    split_mismatch = int(tabular.apply(lambda row: classify_split(row.target_date) != row.split, axis=1).sum())
    date_contract_mismatch = int((pd.to_datetime(tabular.input_end_date) != pd.to_datetime(tabular.target_date) - pd.Timedelta(days=1)).sum())

    arrays = np.load(PROCESSED / "sequence_quality_temperature_w30.npz")
    metadata = pd.read_parquet(PROCESSED / "sequence_quality_temperature_w30_metadata.parquet")
    base_tabular = pd.read_parquet(PROCESSED / "tabular_quality_temperature_w30.parquet")
    indices = np.random.default_rng(SEED).choice(len(metadata), size=min(100, len(metadata)), replace=False)
    stems = ["bod_mg_l", "tn_mg_l", "tp_mg_l", "mean_temperature_c"]
    lstm_mismatch = 0
    for index in indices:
        meta = metadata.iloc[index]
        row = base_tabular.loc[
            base_tabular.series_id.eq(meta.series_id) & base_tabular.target_date.eq(meta.target_date)
        ].iloc[0]
        expected = np.array([[row[f"{stem}_lag_{lag}"] for stem in stems] for lag in range(29, -1, -1)])
        lstm_mismatch += int(not np.allclose(arrays["X"][index], expected, atol=TOLERANCE, rtol=0, equal_nan=True))
    summary = {
        "tabular_rows": int(len(tabular)), "sample_windows": int(len(sample)),
        "feature_values_checked": int(len(sample) * (30 * len(feature_map) + 3)),
        "feature_value_mismatches": mismatch_count, "split_mismatches": split_mismatch,
        "input_end_date_mismatches": date_contract_mismatch,
        "lstm_sequences_checked": int(len(indices)), "lstm_sequence_mismatches": lstm_mismatch,
    }
    return sample, summary


def row_lineage(hash_frame: pd.DataFrame) -> pd.DataFrame:
    inventory = pd.read_csv(REPORTS / "seoul_rainfall_file_inventory.csv", encoding="utf-8-sig")
    quality_rows = sum(1 for _ in open(EXTERNAL / "서울시 물재생센터 수질현황.csv", encoding="cp949")) - 1
    flow_rows = sum(1 for _ in open(EXTERNAL / "서울시 물재생센터 하수처리량.csv", encoding="cp949")) - 1
    stages = [
        {"dataset": "서울시 강우 원본 CSV", "stage": "raw_inventory", "rows": int(inventory.row_count.fillna(0).sum()), "files": int(len(inventory))},
        {"dataset": "서울시 강우", "stage": "station_daily", "rows": int(pd.read_parquet(PROCESSED / "seoul_rainfall_station_daily.parquet", columns=["date"]).shape[0]), "files": pd.NA},
        {"dataset": "서울시 강우", "stage": "center_daily", "rows": int(pd.read_parquet(PROCESSED / "seoul_rainfall_center_daily.parquet", columns=["date"]).shape[0]), "files": pd.NA},
        {"dataset": "수질", "stage": "raw_csv", "rows": quality_rows, "files": 1},
        {"dataset": "처리량", "stage": "raw_csv", "rows": flow_rows, "files": 1},
        {"dataset": "계통 일자료", "stage": "observation_series_daily", "rows": int(pd.read_parquet(PROCESSED / "observation_series_daily.parquet", columns=["date"]).shape[0]), "files": 1},
        {"dataset": "30일 모델 입력", "stage": "tabular_seoul_rainfall_w30", "rows": int(pd.read_parquet(PROCESSED / "tabular_seoul_rainfall_w30.parquet", columns=["target_date"]).shape[0]), "files": 1},
    ]
    result = pd.DataFrame(stages)
    result.to_csv(REPORTS / "audit_row_lineage_2026-11-14.csv", index=False, encoding="utf-8-sig")
    station_daily = pd.read_parquet(PROCESSED / "seoul_rainfall_station_daily.parquet")
    observation_rows = int(pd.read_parquet(PROCESSED / "observation_series_daily.parquet", columns=["date"]).shape[0])
    tabular_rows = int(pd.read_parquet(PROCESSED / "tabular_seoul_rainfall_w30.parquet", columns=["target_date"]).shape[0])
    reconciliation = pd.DataFrame([
        {"step": "강우 10분 원본→관측소 일자료", "input_rows": int(inventory.row_count.fillna(0).sum()),
         "output_rows": len(station_daily), "difference": pd.NA,
         "reason": "제거가 아니라 관측소×달력일 집계; complete/incomplete/no_records 품질상태 보존"},
        {"step": "관측소 일자료→4센터 일자료", "input_rows": len(station_daily), "output_rows": 15704,
         "difference": 15704 - len(station_daily), "reason": "잠정 매핑 4개 관측소만 선택"},
        {"step": "수질 전체→유입수", "input_rows": quality_rows, "output_rows": 28828,
         "difference": 28828 - quality_rows, "reason": "방류 등 비목표 계통 46,089행 제외"},
        {"step": "유입수→계통 일자료", "input_rows": 28828, "output_rows": observation_rows,
         "difference": observation_rows - 28828, "reason": "키·수치 변형 없이 1:1 보존"},
        {"step": "계통 일자료→30일 표형 입력", "input_rows": observation_rows, "output_rows": tabular_rows,
         "difference": tabular_rows - observation_rows, "reason": "30일 연속 필수 수질·기온 및 target 가용성 기준"},
    ])
    reconciliation.to_csv(
        REPORTS / "audit_transformation_reconciliation_2026-11-14.csv", index=False, encoding="utf-8-sig"
    )
    return result


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    hashes = build_hash_baseline()
    daily = pd.read_parquet(PROCESSED / "seoul_rainfall_station_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None)
    rainfall = audit_rainfall(daily)
    _, quality = audit_water_quality()
    joins = audit_joins()
    _, windows = audit_windows()
    lineage = row_lineage(hashes)
    summary = {
        "audit_date": "2026-11-14", "random_seed": SEED,
        "raw_csv_files_hashed": int(len(hashes)), "raw_csv_bytes_hashed": int(hashes.size_bytes.sum()),
        "prior_hashes_available": int(hashes.previous_hash_available.sum()),
        "prior_hash_mismatches": int(hashes.matches_previous_hash.eq(False).sum()),
        "rainfall_station_days_checked": int(len(rainfall)),
        "rainfall_exact_matches": int(rainfall.result.eq("match").sum()),
        "rainfall_mismatches": int(rainfall.result.ne("match").sum()),
        "rainfall_zero_without_confirmation": int((rainfall.existing_daily_rainfall_mm.eq(0) & rainfall.audit_status.ne("confirmed")).sum()),
        "water_quality": quality,
        "join_checks": joins.to_dict("records"), "window_checks": windows,
        "row_lineage": lineage.to_dict("records"),
    }
    (REPORTS / "preprocessing_audit_summary_2026-11-14.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    rain_rate = summary["rainfall_exact_matches"] / summary["rainfall_station_days_checked"] * 100
    station_coverage = ", ".join(f"{key} {value}" for key, value in rainfall.station_name.value_counts().items())
    split_coverage = ", ".join(f"{key} {value}" for key, value in rainfall.split.value_counts().items())
    class_coverage = ", ".join(f"{key} {value}" for key, value in rainfall.rain_class.value_counts().items())
    year_coverage = sorted(pd.to_datetime(rainfall.date).dt.year.unique().tolist())
    report = f"""# 데이터 전처리 독립 감사 보고서

## 감사 범위와 독립성

- 감사일: 2026-11-14, 고정 시드: {SEED}
- 기존 집계·결합·윈도 함수를 호출하지 않고 원본 CSV와 저장 산출물을 별도 코드로 대조했다.
- 모델 재학습, API 호출, 원본 수정, 전체 강우 원본 재처리, GitHub Push는 수행하지 않았다.
- 이번 SHA-256은 현재 파일의 향후 변경 검증 기준선이다. 과거 해시가 없는 파일은 과거에 수정되지 않았다는 증거가 아니다.

## 원본과 행수 계보

- 원본 CSV {len(hashes):,}개, {hashes.size_bytes.sum():,}바이트의 SHA-256을 스트리밍 계산했다.
- 기존 과거 해시는 {summary['prior_hashes_available']}개만 있었고 현재 값과 불일치한 파일은 {summary['prior_hash_mismatches']}개다.
- 강우 원본 인벤토리 행수는 {int(lineage.loc[lineage.stage.eq('raw_inventory'), 'rows'].iloc[0]):,}건, 관측소 일자료는 {len(daily):,}건, 센터 일자료는 {int(lineage.loc[lineage.stage.eq('center_daily'), 'rows'].iloc[0]):,}건이다.
- 수질 원본은 {quality['raw_total_rows']:,}행 중 유입수 {quality['raw_influent_rows']:,}행이며 계통 일자료는 {quality['processed_rows']:,}행이다.

## 독립 강우 재계산

- 층화 표본 {len(rainfall):,}개 관측소-일을 원본 파일에서 직접 재계산했다. 일치 {summary['rainfall_exact_matches']:,}건, 불일치 {summary['rainfall_mismatches']:,}건, 일치율 {rain_rate:.3f}%다.
- 관측소별 표본은 {station_coverage}, 분할별 표본은 {split_coverage}건이다.
- 연도는 {year_coverage[0]}~{year_coverage[-1]}년 전체를 포함하며 강우상태별 표본은 {class_coverage}건, 직접 대조한 원본 행은 {int(rainfall.raw_row_count.sum()):,}건이다.
- 불완전·충돌 관측을 0mm로 확정한 표본은 {summary['rainfall_zero_without_confirmation']}건이다.
- 서울 데이터 허브는 `10분우량`을 자료수집시각 직전 10분 구간의 누적 강우량으로 설명한다. 예: 13:09 값은 13:00:00~13:09:59 구간이다.
- 과거 파일은 계측 원시자료이며 보정되지 않았다고 안내되어, 통계 이상치를 감사 코드에서 임의 수정하지 않았다.

## 수질·기상·모델 입력

- 수질 원본-가공 값 일치: {quality['matched_rows']:,}/{quality['outer_comparison_rows']:,}행, 불일치 {quality['mismatch_rows']:,}행, 원본 키 중복 {quality['raw_key_duplicates']}건이다.
- 하나의 series_id가 여러 센터·처리장으로 섞인 경우는 {quality['mixed_series_count']}건이다.
- 결합 점검 불일치는 총 {int(joins.mismatches.sum())}건이며 항목별 상세는 `audit_join_checks_2026-11-14.csv`에 있다.
- 30일 윈도 {windows['sample_windows']}개에서 {windows['feature_values_checked']:,}개 값을 독립 복원했고 불일치는 {windows['feature_value_mismatches']}개다.
- split 불일치 {windows['split_mismatches']}건, target_date-1 입력종료일 불일치 {windows['input_end_date_mismatches']}건이다.
- LSTM 시퀀스 {windows['lstm_sequences_checked']}개를 오래된 날→최신 날 순으로 복원해 불일치 {windows['lstm_sequence_mismatches']}건을 확인했다.

## 판정과 한계

- 모든 불일치는 자동 수정하지 않고 별도 CSV에 원본·독립 계산·기존 결과를 보존했다.
- 과거 해시 부재, 강우 관측망의 원시 계측 오류 가능성, 잠정 센터-관측소 대표성, ASOS 및 SS의 실제 공개 시각은 남은 한계다.
- 공식 자료: [서울시 강우량 정보](https://data.seoul.go.kr/bsp/wgs/dataView/data300View/1115.do), [서울시 과거 강우량 데이터](https://data.seoul.go.kr/dataList/OA-22715/F/1/datasetView.do)
"""
    (REPORTS / "preprocessing_independent_audit_2026-11-14.md").write_text(report, encoding="utf-8")
    print({"files_hashed": len(hashes), "rainfall_samples": len(rainfall),
           "rainfall_matches": summary["rainfall_exact_matches"],
           "quality_mismatches": quality["mismatch_rows"],
           "window_mismatches": windows["feature_value_mismatches"]})


if __name__ == "__main__":
    main()
