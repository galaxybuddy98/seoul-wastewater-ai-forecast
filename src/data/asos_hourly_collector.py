"""ASOS 시간자료를 원본 페이지 캐시와 함께 수집한다."""

from __future__ import annotations

import math
from urllib.parse import unquote

import pandas as pd

from src.data.asos_api import (
    ASOS_HOURLY_URL,
    build_asos_hourly_params,
    parse_asos_hourly_payload,
)


def iter_year_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """시간자료 1회 조회 최대기간을 넘지 않도록 연도별 구간을 만든다."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    ranges: list[tuple[str, str]] = []
    current = start
    while current <= end:
        period_end = min(pd.Timestamp(current.year, 12, 31), end)
        ranges.append((current.strftime("%Y%m%d"), period_end.strftime("%Y%m%d")))
        current = period_end + pd.Timedelta(days=1)
    return ranges


def collect_hourly_period(
    session,
    cache,
    api_key: str,
    station_id: str,
    start_date: str,
    end_date: str,
    rows_per_page: int = 999,
) -> pd.DataFrame:
    """지정 기간을 페이지 단위로 수집하고 재실행 시 원본 캐시를 재사용한다."""
    service = f"ASOS_HOURLY_{station_id}_{start_date}_{end_date}"
    frames: list[pd.DataFrame] = []
    page_no, total_pages = 1, 1
    while page_no <= total_pages:
        payload = cache.load_page(service, page_no, page_no)
        if payload is None:
            params = build_asos_hourly_params(
                unquote(api_key), start_date, end_date, station_id,
                page_no=page_no, num_rows=rows_per_page,
            )
            response = session.get(ASOS_HOURLY_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            cache.save_page(service, page_no, page_no, payload)
        total_count, frame = parse_asos_hourly_payload(payload)
        total_pages = max(1, math.ceil(total_count / rows_per_page))
        frames.append(frame)
        cache.save_checkpoint(service, next_start=page_no + 1)
        page_no += 1
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not result.empty:
        result = result.sort_values("timestamp").drop_duplicates(
            ["station_id", "timestamp"], keep="last"
        ).reset_index(drop=True)
    return result
