"""검증된 서울시 강우 일자료를 센터·관측계통 및 기존 윈도 설계에 연결한다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.seoul_rainfall import (  # noqa: E402
    build_center_rainfall,
    join_center_rainfall,
    summarize_center_coverage,
)
from src.preprocessing.windows import build_tabular_windows  # noqa: E402


def compare_asos(station_daily: pd.DataFrame, asos: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    original_blank = asos["original_daily_precipitation_mm"].isna()
    for item in mapping.itertuples(index=False):
        municipal = station_daily.loc[station_daily["station_id"].eq(item.station_id), ["date", "daily_rainfall_mm"]]
        merged = asos[["date", "original_daily_precipitation_mm", "precipitation_processed_mm"]].merge(
            municipal, on="date", how="left", validate="one_to_one"
        )
        blank = merged["original_daily_precipitation_mm"].isna()
        both = merged["precipitation_processed_mm"].notna() & merged["daily_rainfall_mm"].notna()
        if both.any():
            asos_event = merged.loc[both, "precipitation_processed_mm"].gt(0)
            city_event = merged.loc[both, "daily_rainfall_mm"].gt(0)
            agreement = float(asos_event.eq(city_event).mean())
            mae = float((merged.loc[both, "precipitation_processed_mm"] - merged.loc[both, "daily_rainfall_mm"]).abs().mean())
        else:
            agreement = np.nan
            mae = np.nan
        rows.append(
            {
                "facility": item.facility,
                "station_id": item.station_id,
                "station_name": item.station_name,
                "asos_blank_days": int(blank.sum()),
                "asos_blank_municipal_confirmed_days": int((blank & merged["daily_rainfall_mm"].notna()).sum()),
                "asos_blank_municipal_zero_days": int((blank & merged["daily_rainfall_mm"].eq(0)).sum()),
                "asos_blank_municipal_positive_days": int((blank & merged["daily_rainfall_mm"].gt(0)).sum()),
                "both_numeric_days": int(both.sum()),
                "rain_occurrence_agreement": agreement,
                "numeric_mae_mm": mae,
            }
        )
    network = (
        station_daily.groupby("date")
        .agg(
            municipal_confirmed_station_count=("daily_rainfall_mm", "count"),
            municipal_positive_station_count=("rain_day", lambda x: int(x.eq(1).sum())),
            municipal_zero_station_count=("rain_day", lambda x: int(x.eq(0).sum())),
        )
        .reset_index()
    )
    network["municipal_network_status"] = np.select(
        [
            network["municipal_confirmed_station_count"].eq(0),
            network["municipal_positive_station_count"].eq(0),
            network["municipal_zero_station_count"].eq(0),
        ],
        ["unavailable", "zero_at_all_available_stations", "positive_at_all_available_stations"],
        default="spatially_mixed",
    )
    network = asos[["date", "original_daily_precipitation_mm", "precipitation_processed_mm"]].merge(
        network, on="date", how="left", validate="one_to_one"
    )
    network["asos_original_blank"] = network["original_daily_precipitation_mm"].isna()
    return pd.DataFrame(rows), network


def build_windows(joined: pd.DataFrame, processed: Path) -> pd.DataFrame:
    quality = ["bod_mg_l", "tn_mg_l", "tp_mg_l"]
    optional_rain = [
        "seoul_rainfall_mm", "seoul_rainfall_max_1h_mm", "seoul_rain_day",
        "seoul_rainfall_lag_1d_mm", "seoul_rainfall_lag_2d_mm",
        "seoul_rainfall_lag_3d_mm", "seoul_rainfall_sum_3d_mm",
        "seoul_rainfall_sum_7d_mm", "seoul_rainfall_missing",
    ]
    features = quality + ["mean_temperature_c"] + optional_rain
    counts: list[dict[str, object]] = []
    for days in (7, 14, 30):
        window = build_tabular_windows(
            joined,
            window_days=days,
            feature_columns=features,
            target_columns=quality,
            required_feature_columns=quality + ["mean_temperature_c"],
        )
        output = processed / f"tabular_seoul_rainfall_w{days}.parquet"
        window.to_parquet(output, index=False)
        for split, count in window["split"].value_counts(dropna=False).items():
            counts.append({"window_days": days, "split": split, "sample_count": int(count)})
    return pd.DataFrame(counts)


def main() -> None:
    processed = PROJECT_ROOT / "data/processed"
    reports = PROJECT_ROOT / "reports"
    mapping = pd.read_csv(PROJECT_ROOT / "configs/seoul_rainfall_center_mapping.csv")
    station_daily = pd.read_parquet(processed / "seoul_rainfall_station_daily.parquet")
    station_daily["date"] = pd.to_datetime(station_daily["date"])
    center_daily = build_center_rainfall(station_daily, mapping)
    center_daily.to_parquet(processed / "seoul_rainfall_center_daily.parquet", index=False)
    annual_coverage = summarize_center_coverage(center_daily)
    annual_coverage.to_csv(
        reports / "seoul_rainfall_center_annual_coverage.csv", index=False, encoding="utf-8-sig"
    )

    observations = pd.read_parquet(processed / "observation_series_daily.parquet")
    observations["date"] = pd.to_datetime(observations["date"])
    joined = join_center_rainfall(observations, center_daily)
    joined.to_parquet(processed / "observation_series_daily_seoul_rainfall.parquet", index=False)
    window_counts = build_windows(joined, processed)
    window_counts.to_csv(reports / "seoul_rainfall_window_sample_counts.csv", index=False, encoding="utf-8-sig")

    asos = pd.read_parquet(processed / "asos_daily_rainfall_hourly_validated.parquet")
    asos["date"] = pd.to_datetime(asos["date"])
    comparison, network = compare_asos(station_daily, asos, mapping)
    comparison.to_csv(reports / "seoul_rainfall_asos_center_comparison.csv", index=False, encoding="utf-8-sig")
    network.to_parquet(processed / "seoul_rainfall_asos_network_comparison.parquet", index=False)
    center_quality = (
        center_daily.groupby(["facility", "station_id", "station_name", "selection_status"], dropna=False)
        .agg(days=("date", "size"), confirmed_days=("daily_rainfall_mm", "count"),
             lag1_days=("rainfall_lag_1d_mm", "count"), sum3_days=("rainfall_sum_3d_mm", "count"),
             sum7_days=("rainfall_sum_7d_mm", "count"))
        .reset_index()
    )
    for column in ("confirmed_days", "lag1_days", "sum3_days", "sum7_days"):
        center_quality[f"{column}_rate"] = center_quality[column] / center_quality["days"]
    center_quality.to_csv(reports / "seoul_rainfall_center_quality.csv", index=False, encoding="utf-8-sig")
    summary = {
        "observation_rows_before": int(len(observations)),
        "observation_rows_after": int(len(joined)),
        "observation_duplicate_keys": int(joined.duplicated(["series_id", "date"]).sum()),
        "center_daily_rows": int(len(center_daily)),
        "center_confirmed_days": int(center_daily["daily_rainfall_mm"].notna().sum()),
        "asos_original_blank_days": int(asos["original_daily_precipitation_mm"].isna().sum()),
        "asos_blank_network_available_days": int((network["asos_original_blank"] & network["municipal_confirmed_station_count"].gt(0)).sum()),
        "asos_blank_network_all_zero_days": int((network["asos_original_blank"] & network["municipal_network_status"].eq("zero_at_all_available_stations")).sum()),
        "asos_blank_network_any_positive_days": int((network["asos_original_blank"] & network["municipal_positive_station_count"].gt(0)).sum()),
        "center_2026_daily_available_days": int(
            center_daily.loc[center_daily["date"].dt.year.eq(2026), "daily_rainfall_mm"].notna().sum()
        ),
        "center_2026_calendar_days": int(center_daily["date"].dt.year.eq(2026).sum()),
        "test_period_center_daily_available_rate": float(
            center_daily.loc[center_daily["date"].ge("2025-01-01"), "daily_rainfall_mm"].notna().mean()
        ),
        "year_boundary_lag_checks": int(
            center_daily.loc[center_daily["date"].eq("2026-01-01"), "rainfall_lag_1d_mm"].notna().sum()
        ),
    }
    (reports / "seoul_rainfall_integration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
