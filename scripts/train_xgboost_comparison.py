"""공통 B 및 서울시 강우 추천 구성의 XGBoost를 학습·평가한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import joblib
import pandas as pd
import pyarrow.parquet as pq
import xgboost

from scripts.train_lightgbm_baseline import _assert_same_rows, _metric_rows, _split
from src.models.lightgbm_baseline import add_series_code, select_lag_features
from src.models.seoul_rainfall_experiment import select_seoul_rainfall_features
from src.models.xgboost_forecaster import RANDOM_SEED, fit_xgboost

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
MODELS = Path("results/models")
PREDICTIONS = Path("results/predictions")
TARGETS = ["bod_mg_l", "tn_mg_l", "tp_mg_l"]
WINDOWS = [7, 14, 30]
RAIN_CONFIG = {"bod_mg_l": "C", "tn_mg_l": "E", "tp_mg_l": "C"}


def _source_path(config: str, window: int) -> Path:
    prefix = "tabular_quality_temperature" if config == "B" else "tabular_seoul_rainfall"
    return PROCESSED / f"{prefix}_w{window}.parquet"


def _feature_names(columns: list[str], config: str) -> list[str]:
    empty = pd.DataFrame(columns=columns)
    if config == "B":
        return select_lag_features(empty, include_flow=False)
    return select_seoul_rainfall_features(empty, config)


def _model_path(config: str, window: int, target: str) -> Path:
    return MODELS / f"xgboost_comparison_{config}_w{window}_{target}.joblib"


def _load_checkpoint(path: Path, expected_features: list[str]) -> dict[str, object]:
    payload = joblib.load(path)
    if payload.get("features") != expected_features:
        raise ValueError(f"XGBoost 체크포인트 피처 계약 불일치: {path}")
    return payload


def _train_one(config: str, window: int, target: str) -> None:
    source = _source_path(config, window)
    schema = pq.ParquetFile(source).schema_arrow.names
    features = _feature_names(schema, config)
    target_column = f"target_{target}"
    frame = pd.read_parquet(source, columns=["series_id", "split", target_column, *features])
    for column in features:
        frame[column] = frame[column].astype("float32")
    train, validation, _ = add_series_code(*_split(frame))
    features = [*features, "series_code"]
    started = perf_counter()
    model = fit_xgboost(train, validation, target_column, features)
    elapsed = perf_counter() - started
    joblib.dump(
        {
            "model": model,
            "features": features,
            "target": target_column,
            "config": config,
            "window_days": window,
            "random_seed": RANDOM_SEED,
            "training_seconds": elapsed,
            "xgboost_version": xgboost.__version__,
            "numeric_input_dtype": "float32",
        },
        _model_path(config, window, target),
    )


def _specs() -> list[tuple[str, int, str]]:
    baseline = [("B", window, target) for window in WINDOWS for target in TARGETS]
    rainfall = [(config, 30, target) for target, config in RAIN_CONFIG.items()]
    return [*baseline, *rainfall]


def _ensure_checkpoints() -> None:
    for config, window, target in _specs():
        schema = pq.ParquetFile(_source_path(config, window)).schema_arrow.names
        features = [*_feature_names(schema, config), "series_code"]
        path = _model_path(config, window, target)
        if path.exists():
            _load_checkpoint(path, features)
            continue
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.train_xgboost_comparison",
                "--train-one",
                "--config",
                config,
                "--window",
                str(window),
                "--target",
                target,
            ],
            check=True,
        )


def main() -> None:
    for path in (REPORTS, MODELS, PREDICTIONS):
        path.mkdir(parents=True, exist_ok=True)
    _ensure_checkpoints()
    metric_rows: list[dict[str, object]] = []
    series_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    training_rows: list[dict[str, object]] = []
    reference_cache: dict[int, pd.DataFrame] = {}

    for config, window, target in _specs():
        frame = pd.read_parquet(_source_path(config, window))
        if window not in reference_cache:
            reference_cache[window] = pd.read_parquet(
                PROCESSED / f"tabular_quality_temperature_w{window}.parquet",
                columns=["series_id", "target_date", "split"],
            )
        for split_name in ("train", "validation", "test"):
            _assert_same_rows(
                reference_cache[window].loc[reference_cache[window]["split"].eq(split_name)],
                frame.loc[frame["split"].eq(split_name)],
                f"XGBoost {config} w{window} {target} {split_name}",
            )
        train, validation, test = add_series_code(*_split(frame))
        schema_features = _feature_names(frame.columns.tolist(), config)
        features = [*schema_features, "series_code"]
        payload = _load_checkpoint(_model_path(config, window, target), features)
        model = payload["model"]
        target_column = f"target_{target}"
        training_rows.append(
            {
                "algorithm": "xgboost",
                "input_config": config,
                "window_days": window,
                "target": target,
                "training_seconds": payload["training_seconds"],
                "best_iteration": int(model.best_iteration),
            }
        )
        for split_name, evaluation in (("validation", validation), ("test", test)):
            metadata = {
                "experiment": "xgboost_lstm_comparison",
                "model_type": "xgboost",
                "variant": config,
                "input_config": config,
                "window_days": window,
                "target": target,
                "split": split_name,
                "best_iteration": int(model.best_iteration),
                "training_seconds": payload["training_seconds"],
            }
            overall, per_series, predictions = _metric_rows(
                evaluation,
                model.predict(evaluation[features]),
                target_column,
                metadata,
            )
            predictions["algorithm"] = "xgboost"
            metric_rows.append(overall)
            series_rows.extend(per_series)
            prediction_parts.append(predictions)

    metrics = pd.DataFrame(metric_rows)
    by_series = pd.DataFrame(series_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    training = pd.DataFrame(training_rows)
    metrics.to_csv(REPORTS / "xgboost_comparison_metrics.csv", index=False, encoding="utf-8-sig")
    by_series.to_csv(REPORTS / "xgboost_comparison_metrics_by_series.csv", index=False, encoding="utf-8-sig")
    training.to_csv(REPORTS / "xgboost_comparison_training_times.csv", index=False, encoding="utf-8-sig")
    predictions.to_parquet(PREDICTIONS / "xgboost_comparison_predictions.parquet", index=False)
    selection = {}
    baseline_validation = metrics.loc[
        metrics["variant"].eq("B") & metrics["split"].eq("validation")
    ]
    for target, group in baseline_validation.groupby("target"):
        best = group.sort_values(["mae", "rmse"]).iloc[0]
        selection[target] = {
            "window_days": int(best["window_days"]),
            "validation_mae": float(best["mae"]),
            "selection_basis": "Validation MAE; Test 미사용",
        }
    config = {
        "algorithm": "xgboost",
        "version": xgboost.__version__,
        "random_seed": RANDOM_SEED,
        "device": "cpu",
        "new_model_count": len(_specs()),
        "baseline_selection": selection,
        "rainfall_configs": RAIN_CONFIG,
        "test_interpretation": "반복 확인된 참고 구간; 독립 최종 홀드아웃 아님",
    }
    (REPORTS / "xgboost_comparison_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-one", action="store_true")
    parser.add_argument("--config", choices=["B", "C", "E"])
    parser.add_argument("--window", type=int, choices=WINDOWS)
    parser.add_argument("--target", choices=TARGETS)
    args = parser.parse_args()
    if args.train_one:
        if args.config is None or args.window is None or args.target is None:
            parser.error("--train-one에는 config, window, target이 필요합니다.")
        _train_one(args.config, args.window, args.target)
    else:
        main()
