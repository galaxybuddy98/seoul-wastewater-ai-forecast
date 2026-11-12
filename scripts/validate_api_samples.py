"""인증키를 노출하지 않고 세 공공 API의 소량 응답을 검증한다."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import unquote

from dotenv import load_dotenv

from src.data.api_client import build_retrying_session
from src.data.asos_api import ASOS_DAILY_URL, build_asos_params
from src.data.seoul_api import SeoulApiClient

REPORT = Path("reports/api_sample_validation.json")


def _validate_seoul(api_key: str, service: str) -> dict[str, object]:
    client = SeoulApiClient(api_key, session=build_retrying_session())
    page = client.fetch_page(service, 1, 5)
    return {
        "success": True,
        "total_count": page.total_count,
        "sample_rows": len(page.rows),
        "fields": sorted(page.rows[0].keys()) if page.rows else [],
    }


def _validate_asos(api_key: str) -> dict[str, object]:
    session = build_retrying_session()
    params = build_asos_params(
        unquote(api_key),
        "20260901",
        "20260903",
        station_id="108",
        page_no=1,
    )
    response = session.get(ASOS_DAILY_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    header = payload.get("response", {}).get("header", {})
    body = payload.get("response", {}).get("body", {})
    items = body.get("items", {}).get("item", [])
    if isinstance(items, dict):
        items = [items]
    code = str(header.get("resultCode", ""))
    if code not in {"00", "0"}:
        raise RuntimeError(f"ASOS API 오류 {code}: {header.get('resultMsg', '응답 오류')}")
    return {
        "success": True,
        "result_code": code,
        "total_count": int(body.get("totalCount", len(items))),
        "sample_rows": len(items),
        "fields": sorted(items[0].keys()) if items else [],
    }


def main() -> None:
    load_dotenv()
    seoul_key = os.getenv("SEOUL_API_KEY", "").strip()
    asos_key = os.getenv("DATA_GO_KR_API_KEY", "").strip()
    if not seoul_key or not asos_key:
        raise RuntimeError("SEOUL_API_KEY와 DATA_GO_KR_API_KEY가 모두 필요합니다")

    result = {}
    for name, validator in (
        ("ViewQowRep", lambda: _validate_seoul(seoul_key, "ViewQowRep")),
        ("ViewTreQty", lambda: _validate_seoul(seoul_key, "ViewTreQty")),
        ("ASOS_108", lambda: _validate_asos(asos_key)),
    ):
        try:
            result[name] = validator()
        except Exception as error:
            result[name] = {
                "success": False,
                "error_type": type(error).__name__,
                "message": str(error).splitlines()[0],
            }
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
