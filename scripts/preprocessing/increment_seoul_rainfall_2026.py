"""2026년 신규 서울시 강우 CSV만 기존 표준화·일자료에 증분 반영한다."""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts._paths import PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.seoul_rainfall import (  # noqa: E402
    aggregate_station_daily,
    detect_csv_layout,
    load_normalized_rainfall_file,
    resolve_timestamp_duplicates,
)
from src.data.seoul_rainfall_incremental import (  # noqa: E402
    find_incremental_sources,
    merge_incremental_station_daily,
)
from scripts.preprocessing.process_seoul_rainfall import summarize_daily  # noqa: E402


def main() -> None:
    started = time.perf_counter()
    source_root = PROJECT_ROOT / "data/external/seoul_rainfall"
    parts_root = PROJECT_ROOT / "data/interim/seoul_rainfall_normalized_parts"
    reports = PROJECT_ROOT / "reports"
    processed = PROJECT_ROOT / "data/processed"
    inventory_path = reports / "seoul_rainfall_file_inventory.csv"
    old_inventory = pd.read_csv(inventory_path, encoding="utf-8-sig")
    discovery = find_incremental_sources(source_root, old_inventory)
    candidates = discovery.loc[discovery["change_status"].ne("unchanged")].copy()
    if candidates["change_status"].eq("changed").any():
        changed = candidates.loc[candidates["change_status"].eq("changed"), "source_file"].tolist()
        raise RuntimeError(f"기존 파일 변경은 안전한 부분 재구축이 필요합니다: {changed}")
    if candidates.empty:
        print(json.dumps({"status": "no_changes", "new_file_count": 0}, ensure_ascii=False))
        return

    batch_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    new_inventory_rows: list[dict[str, object]] = []
    station_parts: dict[str, list[Path]] = defaultdict(list)
    for file_index, item in enumerate(candidates.itertuples(index=False)):
        source = Path(item.absolute_path)
        layout = detect_csv_layout(source)
        normalized = load_normalized_rainfall_file(source)
        for station_id, station_frame in normalized.groupby("station_id", sort=False):
            station_dir = parts_root / station_id
            station_dir.mkdir(parents=True, exist_ok=True)
            output = station_dir / f"incremental-{item.sha256[:12]}-{file_index:02d}.parquet"
            station_frame.to_parquet(output, index=False)
            station_parts[station_id].append(output)
        new_inventory_rows.append(
            {
                "source_file": item.source_file.replace("/", "\\"),
                "size_bytes": item.size_bytes,
                "file_group": "2026.01~2026.09",
                "processing_status": "ok",
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
                "sha256": item.sha256,
                "incremental_batch": batch_id,
            }
        )

    incremental_daily: list[pd.DataFrame] = []
    duplicate_rows: list[dict[str, object]] = []
    for station_id, paths in sorted(station_parts.items()):
        raw = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
        raw = raw.loc[raw["timestamp_kst"].notna()].copy()
        resolved = resolve_timestamp_duplicates(raw)
        daily = aggregate_station_daily(resolved)
        incremental_daily.append(daily)
        duplicate_rows.append(
            {
                "station_id": station_id,
                "station_name": raw["station_name"].mode().iloc[0],
                "raw_records": len(raw),
                "unique_timestamps": len(resolved),
                "identical_duplicate_timestamps": resolved["duplicate_status"].eq("identical_duplicate").sum(),
                "conflicting_duplicate_timestamps": resolved["duplicate_status"].eq("conflicting_duplicate").sum(),
            }
        )
    daily_2026 = pd.concat(incremental_daily, ignore_index=True)
    existing_daily = pd.read_parquet(processed / "seoul_rainfall_station_daily.parquet")
    existing_station_ids = set(existing_daily["station_id"].unique())
    new_station_ids = sorted(set(daily_2026["station_id"].unique()) - existing_station_ids)
    if new_station_ids:
        raise RuntimeError(f"기존 체계에 없는 2026 관측소 ID가 있습니다: {new_station_ids}")
    updated_daily = merge_incremental_station_daily(existing_daily, daily_2026)
    updated_daily.to_parquet(processed / "seoul_rainfall_station_daily.parquet", index=False)
    station_summary, monthly_summary = summarize_daily(updated_daily)
    station_summary.to_csv(
        reports / "seoul_rainfall_station_summary.csv", index=False, encoding="utf-8-sig"
    )
    monthly_summary.to_csv(
        reports / "seoul_rainfall_monthly_completeness.csv", index=False, encoding="utf-8-sig"
    )

    new_inventory = pd.DataFrame(new_inventory_rows)
    combined_inventory = pd.concat([old_inventory, new_inventory], ignore_index=True, sort=False)
    combined_inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")
    new_inventory.to_csv(reports / "seoul_rainfall_2026_file_inventory.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(duplicate_rows).to_csv(
        reports / "seoul_rainfall_2026_duplicate_summary.csv", index=False, encoding="utf-8-sig"
    )
    daily_2026.to_parquet(processed / "seoul_rainfall_station_daily_2026_increment.parquet", index=False)
    summary = {
        "incremental_batch": batch_id,
        "new_file_count": int(len(candidates)),
        "new_source_bytes": int(candidates["size_bytes"].sum()),
        "new_source_rows": int(new_inventory["row_count"].sum()),
        "new_station_count": int(daily_2026["station_id"].nunique()),
        "new_station_ids_not_in_existing": new_station_ids,
        "incremental_daily_rows_with_records": int(len(daily_2026)),
        "incremental_complete_station_days": int(daily_2026["daily_rainfall_mm"].notna().sum()),
        "incremental_incomplete_station_days": int(daily_2026["daily_quality_status"].ne("complete").sum()),
        "updated_station_daily_rows": int(len(updated_daily)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (reports / "seoul_rainfall_2026_increment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
