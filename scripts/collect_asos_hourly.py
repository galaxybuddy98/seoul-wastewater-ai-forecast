"""서울 ASOS 108 시간자료를 연도별로 수집하고 원본을 보존한다."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.data.api_client import build_retrying_session
from src.data.asos_hourly_collector import collect_hourly_period, iter_year_ranges
from src.data.cache import RawPageCache

START = "20160101"
# 2026-09-30 일계를 닫는 2026-10-01 00시 관측을 포함한다.
END = "20261001"
STATION_ID = "108"
RAW = Path("data/raw")
CACHE_ROOT = RAW / "api_cache"


def main() -> None:
    load_dotenv()
    api_key = os.getenv("DATA_GO_KR_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DATA_GO_KR_API_KEY가 필요합니다")
    RAW.mkdir(parents=True, exist_ok=True)
    cache = RawPageCache(CACHE_ROOT)
    session = build_retrying_session()
    frames = []
    for start_date, end_date in iter_year_ranges(START, END):
        frame = collect_hourly_period(
            session, cache, api_key, STATION_ID, start_date, end_date
        )
        frames.append(frame)
        print(f"{start_date}-{end_date}: {len(frame):,}행")
    result = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    if result.duplicated(["station_id", "timestamp"]).any():
        raise RuntimeError("시간자료에 지점·시각 중복이 있습니다")
    result.to_parquet(RAW / "asos_hourly_108.parquet", index=False)
    print(f"저장 완료: {len(result):,}행, {result['timestamp'].min()}~{result['timestamp'].max()}")


if __name__ == "__main__":
    main()
