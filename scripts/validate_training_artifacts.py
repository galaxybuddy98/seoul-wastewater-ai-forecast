"""생성된 학습 산출물의 행 보존·시간 정렬·누수 방지 불변식을 검증한다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.dataset import build_observation_table
from src.preprocessing.schema import normalize_quality_columns

EXTERNAL = Path("data/external")
PROCESSED = Path("data/processed")
REPORT = Path("reports/training_pipeline_validation.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    quality_path = next(
        path for path in EXTERNAL.glob("*.csv")
        if len(pd.read_csv(path, encoding="cp949", nrows=1).columns) == 11
    )
    quality_raw = pd.read_csv(quality_path, encoding="cp949", low_memory=False)
    expected = build_observation_table(normalize_quality_columns(quality_raw))
    actual = pd.read_parquet(PROCESSED / "observation_series_daily.parquet")
    keys = ["date", "facility", "plant", "stream_type"]
    targets = ["bod_mg_l", "tn_mg_l", "tp_mg_l"]

    expected_compare = expected[keys + targets].sort_values(keys).reset_index(drop=True)
    actual_compare = actual[keys + targets].sort_values(keys).reset_index(drop=True)
    quality_preserved = expected_compare.equals(actual_compare)
    duplicate_keys = int(actual.duplicated(keys, keep=False).sum())

    artifact_checks = []
    for path in sorted(PROCESSED.glob("tabular_*_w*.parquet")):
        frame = pd.read_parquet(path)
        if frame.empty:
            artifact_checks.append(
                {"artifact": path.name, "rows": 0, "date_contract": True, "split_contract": True}
            )
            continue
        date_contract = bool(
            (
                pd.to_datetime(frame["input_end_date"])
                == pd.to_datetime(frame["target_date"]) - pd.Timedelta(days=1)
            ).all()
            and (
                pd.to_datetime(frame["input_start_date"])
                == pd.to_datetime(frame["target_date"])
                - pd.to_timedelta(frame["window_days"], unit="D")
            ).all()
        )
        dates = pd.to_datetime(frame["target_date"])
        expected_split = np.where(
            dates <= pd.Timestamp("2022-12-31"),
            "train",
            np.where(dates <= pd.Timestamp("2024-12-31"), "validation", "test"),
        )
        split_contract = bool((frame["split"].astype(str).to_numpy() == expected_split).all())
        artifact_checks.append(
            {
                "artifact": path.name,
                "rows": len(frame),
                "date_contract": date_contract,
                "split_contract": split_contract,
                "series_count": int(frame["series_id"].nunique()),
            }
        )

    result = {
        "source_quality_sha256": _sha256(quality_path),
        "source_rows": len(quality_raw),
        "expected_influent_rows": len(expected),
        "combined_rows": len(actual),
        "quality_values_preserved": quality_preserved,
        "duplicate_composite_key_rows": duplicate_keys,
        "weather_linked_rows": int(actual["weather_link_status"].eq("연결").sum()),
        "flow_linked_rows": int(actual["flow_link_status"].eq("정확키 연결").sum()),
        "artifact_checks": artifact_checks,
        "all_artifacts_pass": all(
            item["date_contract"] and item["split_contract"] for item in artifact_checks
        ),
        "leakage_note": "모든 lag는 target_date보다 1일 이상 과거이며 분할은 target_date 기준이다.",
    }
    if not quality_preserved or duplicate_keys or not result["all_artifacts_pass"]:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
