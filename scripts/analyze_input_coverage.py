"""처리량 연결 원인과 ASOS 강수 공란 근거를 재현 가능하게 요약한다."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

PROCESSED = Path("data/processed")
REPORTS = Path("reports")


def main() -> None:
    combined = pd.read_parquet(PROCESSED / "observation_series_daily.parquet")
    flow = (
        combined.groupby(["facility", "plant", "series_id"], dropna=False)
        .agg(
            observation_rows=("date", "size"),
            flow_linked_rows=("flow_link_status", lambda values: int(values.eq("정확키 연결").sum())),
            start_date=("date", "min"),
            end_date=("date", "max"),
        )
        .reset_index()
    )
    flow["flow_unlinked_rows"] = flow["observation_rows"] - flow["flow_linked_rows"]
    flow["flow_link_rate"] = flow["flow_linked_rows"] / flow["observation_rows"]
    flow["unlinked_reason"] = "정확키 연결 완료"
    flow.loc[
        flow["plant"].eq("처리장 구분없음"), "unlinked_reason"
    ] = "동일 처리장 키의 처리량 행 없음; 다른 처리장 임의 배정 금지"
    flow.loc[
        ~flow["plant"].eq("처리장 구분없음")
        & flow["flow_link_rate"].between(0, 1, inclusive="neither"),
        "unlinked_reason",
    ] = "처리량 관측 시작일·날짜 공백으로 부분 연결"
    flow.to_csv(REPORTS / "flow_linkage_by_series.csv", index=False, encoding="utf-8-sig")

    raw_rows = []
    for filename in glob.glob("data/raw/api_cache/ASOS_*/*.json"):
        payload = json.loads(Path(filename).read_text(encoding="utf-8"))
        if "response" not in payload:
            continue
        items = (
            payload.get("response", {}).get("body", {}).get("items", {}).get("item", [])
        )
        raw_rows.extend(items if isinstance(items, list) else [items])
    raw = pd.DataFrame(raw_rows)
    blank = raw["sumRn"].fillna("").astype(str).str.strip().eq("")
    duration = pd.to_numeric(raw["sumRnDur"], errors="coerce")
    temperature = pd.to_numeric(raw["avgTa"], errors="coerce")
    diagnostics = (
        raw.assign(
            sum_rain_blank=blank,
            blank_with_positive_duration=blank & duration.gt(0),
            blank_with_temperature_missing=blank & temperature.isna(),
        )
        .groupby(["stnId", "stnNm"], dropna=False)
        .agg(
            rows=("tm", "size"),
            sum_rain_blank=("sum_rain_blank", "sum"),
            blank_with_positive_duration=("blank_with_positive_duration", "sum"),
            blank_with_temperature_missing=("blank_with_temperature_missing", "sum"),
        )
        .reset_index()
    )
    diagnostics["decision"] = "공란 일괄 0mm 변환 보류"
    diagnostics.to_csv(
        REPORTS / "rainfall_blank_diagnostics.csv", index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()
