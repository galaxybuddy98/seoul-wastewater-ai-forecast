"""ASOS 후보 지점의 원본 페이지를 캐시하며 일자료를 수집한다."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
from dotenv import load_dotenv

from src.data.api_client import build_retrying_session
from src.data.asos_api import ASOS_DAILY_URL, build_asos_params, parse_asos_payload
from src.data.cache import RawPageCache

START = "20160101"
END = "20260930"
STATIONS = ("108", "112", "119")
ROWS_PER_PAGE = 999
RAW = Path("data/raw")
REPORTS = Path("reports")
CONFIG = Path("configs/weather_stations.json")


def collect_station(station_id: str, api_key: str, cache: RawPageCache) -> pd.DataFrame:
    session = build_retrying_session()
    service = f"ASOS_{station_id}"
    frames = []
    page_no = 1
    total_pages = 1
    while page_no <= total_pages:
        payload = cache.load_page(service, page_no, page_no)
        if payload is None:
            params = build_asos_params(
                unquote(api_key), START, END, station_id=station_id, page_no=page_no
            )
            response = session.get(ASOS_DAILY_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            cache.save_page(service, page_no, page_no, payload)
        total_count, frame = parse_asos_payload(payload)
        total_pages = max(1, math.ceil(total_count / ROWS_PER_PAGE))
        frames.append(frame)
        cache.save_checkpoint(service, next_start=page_no + 1)
        page_no += 1
    result = pd.concat(frames, ignore_index=True)
    return result.drop_duplicates(["date", "station_id"]).sort_values("date").reset_index(drop=True)


def main() -> None:
    load_dotenv()
    api_key = os.getenv("DATA_GO_KR_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DATA_GO_KR_API_KEY가 필요합니다")

    RAW.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    cache = RawPageCache(RAW / "api_cache")
    candidate_frames = [collect_station(station, api_key, cache) for station in STATIONS]
    candidates = pd.concat(candidate_frames, ignore_index=True)
    candidates.to_parquet(RAW / "asos_daily_candidates.parquet", index=False)

    expected_dates = pd.date_range("2016-01-01", "2026-09-30", freq="D")
    stats = []
    for station_id, group in candidates.groupby("station_id"):
        observed_dates = pd.DatetimeIndex(group["date"].dropna().unique())
        stats.append(
            {
                "station_id": station_id,
                "station_name": group["station_name"].iloc[0],
                "rows": len(group),
                "start_date": group["date"].min(),
                "end_date": group["date"].max(),
                "missing_calendar_dates": len(expected_dates.difference(observed_dates)),
                "temperature_missing": int(group["mean_temperature_c"].isna().sum()),
                "precipitation_blank_or_missing": int(group["precipitation_mm"].isna().sum()),
            }
        )
    pd.DataFrame(stats).to_csv(
        REPORTS / "asos_station_comparison.csv", index=False, encoding="utf-8-sig"
    )

    mapping = json.loads(CONFIG.read_text(encoding="utf-8"))
    primary_id = str(mapping["primary_station"]["id"])
    primary = candidates.loc[candidates["station_id"].eq(primary_id)].copy()
    expanded = []
    for facility, station_id in mapping["facility_mapping"].items():
        if str(station_id) != primary_id:
            raise RuntimeError("현재 구현은 공통 주 관측소 매핑만 지원합니다")
        part = primary.copy()
        part["facility"] = facility
        expanded.append(part)
    weather = pd.concat(expanded, ignore_index=True)
    weather.to_parquet(RAW / "asos_daily.parquet", index=False)


if __name__ == "__main__":
    main()
