"""서울시 10분 강우 CSV를 조사하고 보수적인 관측소별 일자료로 변환한다."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.seoul_rainfall import (  # noqa: E402
    add_grouped_rainfall_features,
    aggregate_station_daily,
    detect_csv_layout,
    load_normalized_rainfall_file,
    resolve_timestamp_duplicates,
)

START_DATE = pd.Timestamp("2016-01-01")
END_DATE = pd.Timestamp("2026-09-30")


def _year_group(path: Path) -> str:
    text = str(path.parent)
    if "2006~2010" in text:
        return "2006~2010"
    if "2011~2020" in text:
        return "2011~2020"
    if "2021~2024" in text:
        return "2021~2024"
    if "2025" in text:
        return "2025"
    return "기타"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def normalize_sources(source_root: Path, parts_root: Path) -> tuple[pd.DataFrame, dict[str, list[Path]]]:
    files = sorted(source_root.rglob("*.csv"))
    if parts_root.exists():
        shutil.rmtree(parts_root)
    parts_root.mkdir(parents=True)
    inventory: list[dict[str, object]] = []
    station_parts: dict[str, list[Path]] = defaultdict(list)
    for index, path in enumerate(files):
        relative = path.relative_to(source_root)
        row: dict[str, object] = {
            "source_file": str(relative),
            "size_bytes": path.stat().st_size,
            "file_group": _year_group(path),
            "processing_status": "ok",
        }
        try:
            layout = detect_csv_layout(path)
            normalized = load_normalized_rainfall_file(path)
            row.update(
                {
                    "encoding": layout.encoding,
                    "header_skiprows": layout.skiprows,
                    "rainfall_column": layout.rainfall_column,
                    "row_count": len(normalized),
                    "station_count": normalized["station_id"].nunique(),
                    "station_names": "|".join(sorted(normalized["station_name"].dropna().unique())),
                    "start_timestamp": normalized["timestamp_kst"].min(),
                    "end_timestamp": normalized["timestamp_kst"].max(),
                    "missing_count": normalized["rainfall_10min_mm"].isna().sum(),
                    "zero_count": normalized["rainfall_10min_mm"].eq(0).sum(),
                    "positive_count": normalized["rainfall_10min_mm"].gt(0).sum(),
                    "negative_count": normalized["rainfall_10min_mm"].lt(0).sum(),
                }
            )
            for station_id, station_frame in normalized.groupby("station_id", sort=False):
                station_dir = parts_root / station_id
                station_dir.mkdir(exist_ok=True)
                part = station_dir / f"part-{index:04d}.parquet"
                station_frame.to_parquet(part, index=False)
                station_parts[station_id].append(part)
        except ValueError as exc:
            row.update({"processing_status": "excluded_non_observation", "error": str(exc)})
        except Exception as exc:  # 파일별 오류를 전체 처리 실패와 분리해 목록화한다.
            row.update({"processing_status": "error", "error": f"{type(exc).__name__}: {exc}"})
        inventory.append(row)
    return pd.DataFrame(inventory), station_parts


def build_daily_station_data(station_parts: dict[str, list[Path]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_frames: list[pd.DataFrame] = []
    duplicate_rows: list[dict[str, object]] = []
    all_dates = pd.date_range(START_DATE, END_DATE, freq="D")
    for station_id, paths in sorted(station_parts.items()):
        raw = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
        raw = raw.loc[raw["timestamp_kst"].notna()].copy()
        raw = raw.loc[
            raw["timestamp_kst"].dt.tz_localize(None).between(START_DATE, END_DATE + pd.Timedelta(days=1), inclusive="left")
        ]
        station_name = raw["station_name"].mode().iloc[0] if not raw.empty else ""
        resolved = resolve_timestamp_duplicates(raw)
        duplicate_rows.append(
            {
                "station_id": station_id,
                "station_name": station_name,
                "raw_records": len(raw),
                "unique_timestamps": len(resolved),
                "identical_duplicate_timestamps": resolved["duplicate_status"].eq("identical_duplicate").sum(),
                "conflicting_duplicate_timestamps": resolved["duplicate_status"].eq("conflicting_duplicate").sum(),
            }
        )
        observed = aggregate_station_daily(resolved)
        panel = pd.DataFrame({"date": all_dates})
        panel["station_id"] = station_id
        panel["station_name"] = station_name
        panel = panel.merge(
            observed.drop(columns=["station_name"], errors="ignore"),
            on=["date", "station_id"],
            how="left",
            validate="one_to_one",
        )
        panel["daily_quality_status"] = panel["daily_quality_status"].fillna("no_records")
        panel["source_system"] = panel["source_system"].fillna("서울시 강우량 관측망")
        daily_frames.append(panel)
    daily = pd.concat(daily_frames, ignore_index=True)
    return add_grouped_rainfall_features(daily), pd.DataFrame(duplicate_rows)


def summarize_daily(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    station = (
        daily.groupby(["station_id", "station_name"], dropna=False)
        .agg(
            target_days=("date", "size"),
            confirmed_days=("daily_rainfall_mm", "count"),
            positive_days=("rain_day", lambda values: int(values.eq(1).sum())),
            first_confirmed_date=("date", lambda values: values[daily.loc[values.index, "daily_rainfall_mm"].notna()].min()),
            last_confirmed_date=("date", lambda values: values[daily.loc[values.index, "daily_rainfall_mm"].notna()].max()),
        )
        .reset_index()
    )
    station["confirmed_rate"] = station["confirmed_days"] / station["target_days"]
    monthly = daily.assign(year=daily["date"].dt.year, month=daily["date"].dt.month)
    monthly = (
        monthly.groupby(["station_id", "station_name", "year", "month"], dropna=False)
        .agg(days=("date", "size"), confirmed_days=("daily_rainfall_mm", "count"), positive_days=("rain_day", lambda x: int(x.eq(1).sum())))
        .reset_index()
    )
    monthly["confirmed_rate"] = monthly["confirmed_days"] / monthly["days"]
    return station, monthly


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data/external/seoul_rainfall")
    parser.add_argument("--parts", type=Path, default=PROJECT_ROOT / "data/interim/seoul_rainfall_normalized_parts")
    parser.add_argument("--rebuild-normalized", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    reports = PROJECT_ROOT / "reports"
    processed = PROJECT_ROOT / "data/processed"
    inventory_path = reports / "seoul_rainfall_file_inventory.csv"
    if args.rebuild_normalized or not inventory_path.exists() or not args.parts.exists():
        inventory, station_parts = normalize_sources(args.source, args.parts)
        _write_csv(inventory, inventory_path)
    else:
        inventory = pd.read_csv(inventory_path, encoding="utf-8-sig")
        station_parts = {
            directory.name: sorted(directory.glob("*.parquet"))
            for directory in args.parts.iterdir()
            if directory.is_dir()
        }
    daily, duplicates = build_daily_station_data(station_parts)
    station_summary, monthly_summary = summarize_daily(daily)
    processed.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(processed / "seoul_rainfall_station_daily.parquet", index=False)
    _write_csv(duplicates, reports / "seoul_rainfall_duplicate_summary.csv")
    _write_csv(station_summary, reports / "seoul_rainfall_station_summary.csv")
    _write_csv(monthly_summary, reports / "seoul_rainfall_monthly_completeness.csv")
    summary = {
        "source_file_count": int(len(inventory)),
        "source_bytes": int(inventory["size_bytes"].sum()),
        "observation_file_count": int(inventory["processing_status"].eq("ok").sum()),
        "excluded_file_count": int(inventory["processing_status"].ne("ok").sum()),
        "source_record_count": int(inventory["row_count"].fillna(0).sum()),
        "station_count": int(daily["station_id"].nunique()),
        "confirmed_station_days": int(daily["daily_rainfall_mm"].notna().sum()),
        "target_start": str(START_DATE.date()),
        "target_end": str(END_DATE.date()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (reports / "seoul_rainfall_processing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
