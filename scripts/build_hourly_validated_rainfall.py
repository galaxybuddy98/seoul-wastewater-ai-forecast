"""서울 108 시간자료로 일강수량을 검증하고 기존 데이터셋에 별도 연결한다."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.preprocessing.rainfall import (
    add_rainfall_features,
    reconstruct_daily_rainfall,
    select_hourly_intervals_for_days,
)
from src.preprocessing.windows import build_tabular_windows

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
REPORTS = Path("reports")
TARGETS = ["bod_mg_l", "tn_mg_l", "tp_mg_l"]
REQUIRED = [*TARGETS, "mean_temperature_c"]
RAIN_FEATURES = [
    "precipitation_processed_mm", "precipitation_lag_1d",
    "precipitation_lag_2d", "precipitation_lag_3d",
    "precipitation_sum_3d", "precipitation_sum_7d",
]
WINDOWS = [7, 14, 30]
KEYS = ["series_id", "target_date"]


def main() -> None:
    daily = pd.read_parquet(RAW / "asos_daily.parquet")
    daily = (
        daily.loc[daily["station_id"].astype(str).eq("108"), ["date", "precipitation_mm"]]
        .drop_duplicates("date")
        .sort_values("date")
    )
    hourly = pd.read_parquet(RAW / "asos_hourly_108.parquet")
    hourly = select_hourly_intervals_for_days(hourly, "2016-01-01", "2026-09-30")
    validated = reconstruct_daily_rainfall(daily, hourly)
    validated["precipitation_processed_mm"] = validated["validated_precipitation_mm"]
    validated["precipitation_missing"] = validated["validated_precipitation_mm"].isna().astype("int8")
    validated = add_rainfall_features(validated)
    validated.to_parquet(
        PROCESSED / "asos_daily_rainfall_hourly_validated.parquet", index=False
    )
    validated.to_csv(
        REPORTS / "asos_hourly_daily_comparison.csv", index=False, encoding="utf-8-sig"
    )

    known = validated["original_daily_precipitation_mm"].notna()
    comparable = validated.loc[known & validated["hourly_precipitation_sum_mm"].notna()].copy()
    comparable["difference_mm"] = (
        comparable["hourly_precipitation_sum_mm"]
        - comparable["original_daily_precipitation_mm"]
    )
    validated["season"] = validated["date"].dt.month.map(
        lambda month: "겨울(11~3월)" if month in {11, 12, 1, 2, 3} else "비겨울(4~10월)"
    )
    seasonal = validated.groupby("season", as_index=False).agg(
        days=("date", "size"),
        days_with_24_timestamps=("hourly_row_count", lambda x: int(x.eq(24).sum())),
        days_with_numeric_rn=("hourly_numeric_count", lambda x: int(x.gt(0).sum())),
        numeric_rn_count=("hourly_numeric_count", "sum"),
        qc_missing_count=("hourly_qc_missing_count", "sum"),
        validated_missing_days=("validated_precipitation_mm", lambda x: int(x.isna().sum())),
    )
    seasonal.to_csv(
        REPORTS / "asos_hourly_seasonal_completeness.csv", index=False, encoding="utf-8-sig"
    )

    observations = pd.read_parquet(PROCESSED / "observation_series_daily.parquet")
    old_rain_columns = [
        column for column in observations.columns
        if column.startswith("precipitation_")
    ]
    linked = observations.drop(columns=old_rain_columns, errors="ignore").merge(
        validated.drop(columns="season"), on="date", how="left", validate="many_to_one"
    )
    linked.to_parquet(
        PROCESSED / "observation_series_daily_rainfall_hourly_validated.parquet", index=False
    )

    comparison_rows = []
    for window in WINDOWS:
        reference = pd.read_parquet(PROCESSED / f"tabular_quality_temperature_w{window}.parquet")
        for suffix, features in {
            "c": [*REQUIRED, *RAIN_FEATURES],
            "d": [*REQUIRED, *RAIN_FEATURES, "precipitation_missing"],
        }.items():
            new = build_tabular_windows(
                linked, window, features, TARGETS, required_feature_columns=REQUIRED
            )
            old = pd.read_parquet(PROCESSED / f"tabular_rainfall_{suffix}_w{window}.parquet")
            new_keys = new[KEYS].sort_values(KEYS).reset_index(drop=True)
            if not new_keys.equals(reference[KEYS].sort_values(KEYS).reset_index(drop=True)):
                raise RuntimeError(f"기준 평가행 불일치: {suffix} w{window}")
            new.to_parquet(
                PROCESSED / f"tabular_rainfall_hourly_validated_{suffix}_w{window}.parquet",
                index=False,
            )
            for split in ("train", "validation", "test"):
                comparison_rows.append(
                    {
                        "variant": suffix.upper(),
                        "window_days": window,
                        "split": split,
                        "samples_before": int(old["split"].eq(split).sum()),
                        "samples_after": int(new["split"].eq(split).sum()),
                    }
                )
    pd.DataFrame(comparison_rows).to_csv(
        REPORTS / "rainfall_hourly_dataset_comparison.csv", index=False, encoding="utf-8-sig"
    )

    status_counts = validated["rainfall_status"].value_counts().to_dict()
    summary = {
        "station_id": "108",
        "period": [str(validated["date"].min().date()), str(validated["date"].max().date())],
        "hourly_rows": int(len(hourly)),
        "daily_rows": int(len(validated)),
        "original_daily_known": int(known.sum()),
        "original_daily_blank": int((~known).sum()),
        "hourly_daily_comparable": int(len(comparable)),
        "agreement_within_0_05mm": int(comparable["difference_mm"].abs().le(0.05).sum()),
        "agreement_rate": float(comparable["difference_mm"].abs().le(0.05).mean()),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "confirmed_zero_from_hourly": int(status_counts.get("B_시간자료_무강수_확인", 0)),
        "restored_from_hourly": int(status_counts.get("A_시간자료_복원", 0)),
        "remaining_nan": int(validated["validated_precipitation_mm"].isna().sum()),
        "rule": "00시 rn은 전날 귀속; 공식 sumRn 우선; 공란은 직접 입증된 경우만 복원",
    }
    (REPORTS / "asos_hourly_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
