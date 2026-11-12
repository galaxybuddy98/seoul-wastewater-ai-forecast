"""서울시 강우 원본의 증분 판별과 기존 일자료 갱신."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.data.seoul_rainfall import add_grouped_rainfall_features


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_incremental_sources(source_root: str | Path, inventory: pd.DataFrame) -> pd.DataFrame:
    """기존 상대경로·크기를 비교하고 신규·변경 파일만 해시한다."""
    root = Path(source_root)
    old_sizes = {
        str(path).replace("\\", "/"): int(size)
        for path, size in zip(inventory["source_file"], inventory["size_bytes"])
    }
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.csv")):
        relative = str(path.relative_to(root)).replace("\\", "/")
        size = path.stat().st_size
        if relative not in old_sizes:
            status = "new"
        elif size != old_sizes[relative]:
            status = "changed"
        else:
            status = "unchanged"
        rows.append(
            {
                "source_file": relative,
                "size_bytes": size,
                "change_status": status,
                "sha256": _sha256(path) if status != "unchanged" else pd.NA,
                "absolute_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def merge_incremental_station_daily(
    existing: pd.DataFrame,
    incremental: pd.DataFrame,
) -> pd.DataFrame:
    """신규 관측일만 교체하고 전체 관측소 시계열 파생변수를 다시 계산한다."""
    keys = ["station_id", "date"]
    if incremental.duplicated(keys).any():
        raise ValueError("증분 일자료의 관측소·날짜 키가 중복됩니다.")
    feature_columns = {
        "rainfall_lag_1d_mm", "rainfall_lag_2d_mm", "rainfall_lag_3d_mm",
        "rainfall_sum_3d_mm", "rainfall_sum_7d_mm", "rainfall_missing",
    }
    base = existing.drop(columns=feature_columns, errors="ignore").copy()
    replacement_keys = incremental[keys].drop_duplicates()
    retained = base.merge(replacement_keys.assign(_replace=1), on=keys, how="left")
    retained = retained.loc[retained["_replace"].isna()].drop(columns="_replace")
    combined = pd.concat([retained, incremental], ignore_index=True, sort=False)
    combined = combined.sort_values(keys).reset_index(drop=True)
    return add_grouped_rainfall_features(combined)
