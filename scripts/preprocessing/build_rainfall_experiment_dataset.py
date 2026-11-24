"""기존 결합 자료를 재사용해 강수 결측 허용 LightGBM 표형 산출물을 만든다."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.preprocessing.rainfall import add_rainfall_features, classify_daily_rainfall
from src.preprocessing.windows import build_tabular_windows

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
REPORTS = Path("reports")
TARGETS = ["bod_mg_l", "tn_mg_l", "tp_mg_l"]
REQUIRED = [*TARGETS, "mean_temperature_c"]
RAIN_FEATURES = [
    "precipitation_processed_mm",
    "precipitation_lag_1d",
    "precipitation_lag_2d",
    "precipitation_lag_3d",
    "precipitation_sum_3d",
    "precipitation_sum_7d",
]
WINDOWS = [7, 14, 30]
KEYS = ["series_id", "target_date"]


def _prepare_weather() -> pd.DataFrame:
    raw = pd.read_parquet(RAW / "asos_daily.parquet")
    parts = []
    for _, group in raw.groupby("facility", sort=True):
        classified = classify_daily_rainfall(group)
        parts.append(add_rainfall_features(classified))
    return pd.concat(parts, ignore_index=True)


def _same_keys(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return left[KEYS].sort_values(KEYS).reset_index(drop=True).equals(
        right[KEYS].sort_values(KEYS).reset_index(drop=True)
    )


def main() -> None:
    started = time.perf_counter()
    weather = _prepare_weather()
    combined = pd.read_parquet(PROCESSED / "observation_series_daily.parquet")
    weather_columns = [
        "mean_temperature_c",
        "precipitation_mm",
        "precipitation_lag_1d",
        "precipitation_lag_2d",
        "precipitation_lag_3d",
        "precipitation_sum_3d",
        "precipitation_sum_7d",
        "weather_link_status",
    ]
    base = combined.drop(columns=[c for c in weather_columns if c in combined], errors="ignore")
    keep = [
        "date", "facility", "mean_temperature_c", "precipitation_raw_mm",
        "precipitation_raw_blank", "precipitation_processed_mm",
        "precipitation_missing", "precipitation_processing_reason",
        "precipitation_processing_evidence", *RAIN_FEATURES[1:],
    ]
    enriched = base.merge(
        weather[keep], on=["date", "facility"], how="left", validate="many_to_one"
    )
    enriched.to_parquet(PROCESSED / "observation_series_daily_rainfall_experiment.parquet", index=False)

    rows = []
    for window in WINDOWS:
        reference = pd.read_parquet(PROCESSED / f"tabular_quality_temperature_w{window}.parquet")
        for variant, features in {
            "rainfall_c": [*REQUIRED, *RAIN_FEATURES],
            "rainfall_d": [*REQUIRED, *RAIN_FEATURES, "precipitation_missing"],
        }.items():
            frame = build_tabular_windows(
                enriched,
                window_days=window,
                feature_columns=features,
                required_feature_columns=REQUIRED,
                target_columns=TARGETS,
            )
            if not _same_keys(reference, frame):
                raise RuntimeError(f"기온 기준 모델과 평가행 불일치: {variant} w{window}")
            frame.to_parquet(PROCESSED / f"tabular_{variant}_w{window}.parquet", index=False)
            old_path = PROCESSED / f"tabular_quality_weather_w{window}.parquet"
            old = pd.read_parquet(old_path) if old_path.exists() else pd.DataFrame()
            for split, count in frame["split"].value_counts().items():
                rows.append({
                    "variant": variant,
                    "window_days": window,
                    "split": split,
                    "sample_count_after": int(count),
                    "sample_count_before": int(old["split"].eq(split).sum()) if "split" in old else 0,
                    "reference_temperature_count": int(reference["split"].eq(split).sum()),
                })

    summary = pd.DataFrame(rows)
    summary.to_csv(REPORTS / "rainfall_window_sample_counts.csv", index=False, encoding="utf-8-sig")
    primary = weather.loc[weather["station_id"].astype(str).eq("108")].drop_duplicates("date")
    processing = {
        "station_id": "108",
        "rows": int(len(primary)),
        "raw_blank": int(primary["precipitation_raw_blank"].sum()),
        "converted_blank_to_zero": int(
            (primary["precipitation_raw_blank"] & primary["precipitation_processed_mm"].eq(0)).sum()
        ),
        "blank_kept_nan": int(
            (primary["precipitation_raw_blank"] & primary["precipitation_processed_mm"].isna()).sum()
        ),
        "rule": "수치 sumRn만 보존; 공식 공란 규칙 미확인 공란은 NaN",
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    (REPORTS / "rainfall_processing_summary.json").write_text(
        json.dumps(processing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    weather.drop_duplicates(["date", "station_id"]).to_parquet(
        PROCESSED / "asos_daily_rainfall_processed.parquet", index=False
    )


if __name__ == "__main__":
    main()
