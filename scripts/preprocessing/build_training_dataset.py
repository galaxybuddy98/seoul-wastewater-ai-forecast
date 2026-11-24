"""CSV 참고 원본으로 처리장·유입 관측 계통별 학습 데이터셋을 생성한다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.dataset import (
    add_weather_history,
    build_observation_table,
    join_exact_flow,
    join_weather,
)
from src.preprocessing.flow import add_treatment_total_hypothesis
from src.preprocessing.schema import normalize_flow_columns, normalize_quality_columns
from src.preprocessing.windows import (
    build_tabular_windows,
    count_target_window_samples,
    tabular_to_sequence,
)

EXTERNAL = Path("data/external")
RAW = Path("data/raw")
PROCESSED = Path("data/processed")
REPORTS = Path("reports")
TARGETS = ["bod_mg_l", "tn_mg_l", "tp_mg_l"]
BASE_FEATURES = TARGETS
FEATURE_VARIANTS = {
    "quality": TARGETS,
    "quality_temperature": [*TARGETS, "mean_temperature_c"],
    "quality_weather": [
        *TARGETS,
        "mean_temperature_c",
        "precipitation_mm",
        "precipitation_lag_1d",
        "precipitation_lag_2d",
        "precipitation_lag_3d",
    ],
    "quality_flow": [
        *TARGETS,
        "primary_treatment_volume",
        "secondary_treatment_volume",
        "total_treatment_volume_hypothesis",
    ],
    "quality_temperature_flow": [
        *TARGETS,
        "mean_temperature_c",
        "primary_treatment_volume",
        "secondary_treatment_volume",
        "total_treatment_volume_hypothesis",
    ],
}
WINDOWS = [7, 14, 30]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_external_csvs() -> tuple[Path, pd.DataFrame, Path, pd.DataFrame]:
    loaded = [
        (path, pd.read_csv(path, encoding="cp949", low_memory=False))
        for path in sorted(EXTERNAL.glob("*.csv"))
    ]
    quality = next(item for item in loaded if len(item[1].columns) == 11)
    flow = next(item for item in loaded if len(item[1].columns) == 5)
    return quality[0], quality[1], flow[0], flow[1]


def _load_weather_if_available(facilities: list[str]) -> tuple[pd.DataFrame, str]:
    parquet_path = RAW / "asos_daily.parquet"
    csv_path = RAW / "asos_daily.csv"
    if parquet_path.exists():
        weather = pd.read_parquet(parquet_path)
        return weather, "ASOS API 원본 정규화 자료"
    if csv_path.exists():
        weather = pd.read_csv(csv_path)
        weather["date"] = pd.to_datetime(weather["date"])
        return weather, "ASOS API 원본 정규화 자료"

    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "facility": pd.Series(dtype="object"),
            "station_id": pd.Series(dtype="object"),
            "mean_temperature_c": pd.Series(dtype="float64"),
            "precipitation_mm": pd.Series(dtype="float64"),
        }
    ), "미수집: DATA_GO_KR_API_KEY 없음"


def _series_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            ["facility", "center_id", "source_plant", "series_id"],
            dropna=False,
        )
        .agg(
            observation_count=("date", "size"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            bod_missing=("bod_mg_l", lambda values: int(values.isna().sum())),
            tn_missing=("tn_mg_l", lambda values: int(values.isna().sum())),
            tp_missing=("tp_mg_l", lambda values: int(values.isna().sum())),
            flow_linked=("flow_link_status", lambda values: int(values.eq("정확키 연결").sum())),
            weather_linked=("weather_link_status", lambda values: int(values.eq("연결").sum())),
        )
        .reset_index()
    )


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    quality_path, quality_raw, flow_path, flow_raw = _load_external_csvs()
    quality = normalize_quality_columns(quality_raw)
    flow = add_treatment_total_hypothesis(normalize_flow_columns(flow_raw))

    observations = build_observation_table(quality)
    combined = join_exact_flow(observations, flow)
    weather, weather_source = _load_weather_if_available(sorted(combined["facility"].unique()))
    if not weather.empty:
        weather = add_weather_history(weather)
    combined = join_weather(combined, weather)
    combined = combined.sort_values(["series_id", "date"]).reset_index(drop=True)

    combined.to_parquet(PROCESSED / "observation_series_daily.parquet", index=False)
    series_summary = _series_summary(combined)
    series_summary.to_csv(
        REPORTS / "training_dataset_series_summary.csv", index=False, encoding="utf-8-sig"
    )

    count_parts = []
    artifact_rows = []
    for variant, features in FEATURE_VARIANTS.items():
        for window in WINDOWS:
            common = build_tabular_windows(combined, window, features, TARGETS)
            common_path = PROCESSED / f"tabular_{variant}_w{window}.parquet"
            common.to_parquet(common_path, index=False)

            sequence = tabular_to_sequence(common, window, features, TARGETS)
            np.savez_compressed(
                PROCESSED / f"sequence_{variant}_w{window}.npz",
                X=sequence["X"],
                y=sequence["y"],
            )
            sequence["metadata"].to_parquet(
                PROCESSED / f"sequence_{variant}_w{window}_metadata.parquet", index=False
            )
            artifact_rows.append(
                {
                    "variant": variant,
                    "window_days": window,
                    "common_samples": len(common),
                    "tabular_columns": len(common.columns),
                    "sequence_shape": list(sequence["X"].shape),
                }
            )
            counts = count_target_window_samples(
                combined, window, features, TARGETS
            )
            counts["variant"] = variant
            count_parts.append(counts)

    sample_counts = pd.concat(count_parts, ignore_index=True)
    sample_counts.to_csv(
        REPORTS / "training_sample_counts.csv", index=False, encoding="utf-8-sig"
    )

    manifest = {
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "prediction_contract": {
            "grain": "날짜×물재생센터×원본 처리장/유입 관측 계통",
            "horizon": "t+1일",
            "targets": TARGETS,
            "windows": WINDOWS,
            "split_by": "target_date",
            "train_end": "2022-12-31",
            "validation_end": "2024-12-31",
            "test_end": "2026-09-30",
        },
        "sources": {
            "quality_csv": {
                "path": quality_path.name,
                "rows": len(quality_raw),
                "sha256": _sha256(quality_path),
            },
            "flow_csv": {
                "path": flow_path.name,
                "rows": len(flow_raw),
                "sha256": _sha256(flow_path),
            },
            "seoul_api": "미수집: TLS 기본 포트 미지원, 평문 HTTP 8088은 인증키 보호를 위해 사용하지 않음",
            "weather": weather_source,
        },
        "combined": {
            "rows": len(combined),
            "series": int(combined["series_id"].nunique()),
            "flow_linked_rows": int(combined["flow_link_status"].eq("정확키 연결").sum()),
            "weather_linked_rows": int(combined["weather_link_status"].eq("연결").sum()),
        },
        "artifacts": artifact_rows,
        "assumptions": [
            "CSV는 API 미인증 상태의 파이프라인 개발용 참고 원본이다.",
            "처리량은 날짜·센터·원본 처리장 정확키만 연결한다.",
            "처리장 구분없음은 별도 관측 계통이며 센터 대표값으로 단정하지 않는다.",
            "ASOS 실측의 운영 시점 가용성은 미확인으로 연구용 가정 검토가 필요하다.",
            "모델 학습·스케일링·결측 대치는 수행하지 않았다.",
        ],
    }
    (REPORTS / "training_dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
