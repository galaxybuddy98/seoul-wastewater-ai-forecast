"""기존 NPZ와 서울시 강우 시퀀스로 CPU 경량 LSTM을 학습·평가한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from scripts.training.lightgbm.train_lightgbm_baseline import _assert_same_rows, _metric_rows
from src.models.shared.comparative_models import (
    SequencePreprocessor,
    encode_series_ids,
    fit_sequence_preprocessor,
    tabular_to_sequence,
)
from src.models.lstm.lstm_forecaster import (
    LSTMConfig,
    RANDOM_SEED,
    SeriesLSTM,
    fit_lstm,
    predict_lstm,
)

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
MODELS = Path("results/models")
PREDICTIONS = Path("results/predictions")
TARGETS = ["bod_mg_l", "tn_mg_l", "tp_mg_l"]
TARGET_INDEX = {target: index for index, target in enumerate(TARGETS)}
WINDOWS = [7, 14, 30]
RAIN_CONFIG = {"bod_mg_l": "C", "tn_mg_l": "E", "tp_mg_l": "C"}
BASE_STEMS = ["bod_mg_l", "tn_mg_l", "tp_mg_l", "mean_temperature_c"]
RAIN_STEMS = {
    "C": ["seoul_rainfall_mm"],
    "E": [
        "seoul_rainfall_mm",
        "seoul_rainfall_lag_1d_mm",
        "seoul_rainfall_lag_2d_mm",
        "seoul_rainfall_lag_3d_mm",
        "seoul_rainfall_sum_3d_mm",
        "seoul_rainfall_sum_7d_mm",
        "seoul_rainfall_max_1h_mm",
    ],
}


def _specs() -> list[tuple[str, int, str]]:
    baseline = [("B", window, target) for window in WINDOWS for target in TARGETS]
    rainfall = [(config, 30, target) for target, config in RAIN_CONFIG.items()]
    return [*baseline, *rainfall]


def _model_path(config: str, window: int, target: str) -> Path:
    return MODELS / f"lstm_comparison_{config}_w{window}_{target}.pt"


def _load_data(
    config: str,
    window: int,
    target: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], list[int]]:
    if config == "B":
        source = PROCESSED / f"sequence_quality_temperature_w{window}.npz"
        arrays = np.load(source)
        values = arrays["X"]
        target_values = arrays["y"][:, TARGET_INDEX[target]]
        metadata = pd.read_parquet(
            PROCESSED / f"sequence_quality_temperature_w{window}_metadata.parquet"
        )
        return values, target_values, metadata, BASE_STEMS, []
    stems = [*BASE_STEMS, *RAIN_STEMS[config]]
    columns = [
        "series_id", "center_id", "facility", "plant", "input_start_date",
        "input_end_date", "target_date", "window_days", "split", f"target_{target}",
        *[f"{stem}_lag_{lag}" for lag in range(window) for stem in stems],
    ]
    frame = pd.read_parquet(
        PROCESSED / f"tabular_seoul_rainfall_w{window}.parquet", columns=columns
    )
    values = tabular_to_sequence(frame, stems, window)
    metadata = frame[[
        "series_id", "center_id", "facility", "plant", "input_start_date",
        "input_end_date", "target_date", "window_days", "split",
    ]].copy()
    rain_indices = list(range(len(BASE_STEMS), len(stems)))
    return values, frame[f"target_{target}"].to_numpy(), metadata, stems, rain_indices


def _checkpoint(config: str, window: int, target: str) -> dict[str, object]:
    path = _model_path(config, window, target)
    return torch.load(path, map_location="cpu", weights_only=False)


def _validate_checkpoint(
    payload: dict[str, object], config: str, window: int, target: str, stems: list[str]
) -> None:
    expected = (config, window, target, stems)
    actual = (
        payload.get("input_config"), payload.get("window_days"),
        payload.get("target"), payload.get("feature_stems"),
    )
    if actual != expected:
        raise ValueError(f"LSTM 체크포인트 계약 불일치: expected={expected}, actual={actual}")


def _train_one(config: str, window: int, target: str) -> None:
    values, target_values, metadata, stems, rain_indices = _load_data(config, window, target)
    train_mask = metadata["split"].eq("train").to_numpy()
    validation_mask = metadata["split"].eq("validation").to_numpy()
    preprocessor = fit_sequence_preprocessor(values[train_mask], rain_indices)
    train_x = preprocessor.transform(values[train_mask])
    validation_x = preprocessor.transform(values[validation_mask])
    train_series, validation_series = encode_series_ids(
        metadata.loc[train_mask, "series_id"].to_numpy(),
        metadata.loc[validation_mask, "series_id"].to_numpy(),
    )
    categories = sorted(metadata.loc[train_mask, "series_id"].unique().tolist())
    result = fit_lstm(
        train_x,
        train_series,
        target_values[train_mask],
        validation_x,
        validation_series,
        target_values[validation_mask],
        num_series=len(categories),
        config=LSTMConfig(),
    )
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "input_size": train_x.shape[2],
            "num_series": len(categories),
            "series_categories": categories,
            "feature_stems": stems,
            "rain_missing_indicator_indices": rain_indices,
            "preprocessor": {
                "medians": preprocessor.medians,
                "means": preprocessor.means,
                "stds": preprocessor.stds,
                "missing_indicator_indices": preprocessor.missing_indicator_indices,
            },
            "target_mean": result.target_mean,
            "target_std": result.target_std,
            "best_epoch": result.best_epoch,
            "history": result.history,
            "training_seconds": result.training_seconds,
            "config": result.config,
            "input_config": config,
            "window_days": window,
            "target": target,
            "random_seed": RANDOM_SEED,
            "torch_version": torch.__version__,
            "device": "cpu",
        },
        _model_path(config, window, target),
    )


def _ensure_checkpoints() -> None:
    for config, window, target in _specs():
        path = _model_path(config, window, target)
        stems = BASE_STEMS if config == "B" else [*BASE_STEMS, *RAIN_STEMS[config]]
        if path.exists():
            _validate_checkpoint(_checkpoint(config, window, target), config, window, target, stems)
            continue
        subprocess.run(
            [
                sys.executable, "-m", "scripts.training.lstm.train_lstm_comparison", "--train-one",
                "--config", config, "--window", str(window), "--target", target,
            ],
            check=True,
        )


def _restore_preprocessor(payload: dict[str, object]) -> SequencePreprocessor:
    saved = payload["preprocessor"]
    return SequencePreprocessor(
        medians=np.asarray(saved["medians"]),
        means=np.asarray(saved["means"]),
        stds=np.asarray(saved["stds"]),
        missing_indicator_indices=tuple(saved["missing_indicator_indices"]),
    )


def _restore_model(payload: dict[str, object]) -> SeriesLSTM:
    config = payload["config"]
    model = SeriesLSTM(
        input_size=int(payload["input_size"]),
        num_series=int(payload["num_series"]),
        hidden_size=int(config["hidden_size"]),
        embedding_size=int(config["embedding_size"]),
    )
    model.load_state_dict(payload["state_dict"])
    return model


def main() -> None:
    for path in (REPORTS, MODELS, PREDICTIONS):
        path.mkdir(parents=True, exist_ok=True)
    _ensure_checkpoints()
    metric_rows: list[dict[str, object]] = []
    series_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    training_rows: list[dict[str, object]] = []
    histories: dict[str, object] = {}

    for config, window, target in _specs():
        values, target_values, metadata, stems, _ = _load_data(config, window, target)
        reference = pd.read_parquet(
            PROCESSED / f"sequence_quality_temperature_w{window}_metadata.parquet"
        )
        for split_name in ("train", "validation", "test"):
            _assert_same_rows(
                reference.loc[reference["split"].eq(split_name)],
                metadata.loc[metadata["split"].eq(split_name)],
                f"LSTM {config} w{window} {target} {split_name}",
            )
        payload = _checkpoint(config, window, target)
        _validate_checkpoint(payload, config, window, target, stems)
        preprocessor = _restore_preprocessor(payload)
        train_mask = metadata["split"].eq("train").to_numpy()
        validation_mask = metadata["split"].eq("validation").to_numpy()
        test_mask = metadata["split"].eq("test").to_numpy()
        train_ids = metadata.loc[train_mask, "series_id"].to_numpy()
        _, validation_series, test_series = encode_series_ids(
            train_ids,
            metadata.loc[validation_mask, "series_id"].to_numpy(),
            metadata.loc[test_mask, "series_id"].to_numpy(),
        )
        model = _restore_model(payload)
        key = f"{config}_w{window}_{target}"
        histories[key] = payload["history"]
        training_rows.append(
            {
                "algorithm": "lstm",
                "input_config": config,
                "window_days": window,
                "target": target,
                "training_seconds": payload["training_seconds"],
                "best_epoch": payload["best_epoch"],
                "epochs_ran": len(payload["history"]),
            }
        )
        for split_name, mask, series_codes in (
            ("validation", validation_mask, validation_series),
            ("test", test_mask, test_series),
        ):
            transformed = preprocessor.transform(values[mask])
            prediction = predict_lstm(
                model,
                transformed,
                series_codes,
                float(payload["target_mean"]),
                float(payload["target_std"]),
                batch_size=int(payload["config"]["batch_size"]),
            )
            evaluation = metadata.loc[mask].copy()
            target_column = f"target_{target}"
            evaluation[target_column] = target_values[mask]
            metadata_values = {
                "experiment": "xgboost_lstm_comparison",
                "model_type": "lstm",
                "variant": config,
                "input_config": config,
                "window_days": window,
                "target": target,
                "split": split_name,
                "best_iteration": int(payload["best_epoch"]),
                "training_seconds": payload["training_seconds"],
            }
            overall, per_series, prediction_frame = _metric_rows(
                evaluation, prediction, target_column, metadata_values
            )
            prediction_frame["algorithm"] = "lstm"
            metric_rows.append(overall)
            series_rows.extend(per_series)
            prediction_parts.append(prediction_frame)

    metrics = pd.DataFrame(metric_rows)
    by_series = pd.DataFrame(series_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    training = pd.DataFrame(training_rows)
    metrics.to_csv(REPORTS / "lstm_comparison_metrics.csv", index=False, encoding="utf-8-sig")
    by_series.to_csv(REPORTS / "lstm_comparison_metrics_by_series.csv", index=False, encoding="utf-8-sig")
    training.to_csv(REPORTS / "lstm_comparison_training_times.csv", index=False, encoding="utf-8-sig")
    predictions.to_parquet(PREDICTIONS / "lstm_comparison_predictions.parquet", index=False)
    (REPORTS / "lstm_comparison_training_history.json").write_text(
        json.dumps(histories, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    baseline_validation = metrics.loc[
        metrics["variant"].eq("B") & metrics["split"].eq("validation")
    ]
    selections = {}
    for target, group in baseline_validation.groupby("target"):
        best = group.sort_values(["mae", "rmse"]).iloc[0]
        selections[target] = {
            "window_days": int(best["window_days"]),
            "validation_mae": float(best["mae"]),
            "selection_basis": "Validation MAE; Test 미사용",
        }
    config = {
        "algorithm": "lstm",
        "torch_version": torch.__version__,
        "random_seed": RANDOM_SEED,
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "new_model_count": len(_specs()),
        "baseline_selection": selections,
        "rainfall_configs": RAIN_CONFIG,
        "rainfall_missing_policy": "Train 중앙값 대치 + 변수별 결측 표시",
        "test_interpretation": "반복 확인된 참고 구간; 독립 최종 홀드아웃 아님",
    }
    (REPORTS / "lstm_comparison_config.json").write_text(
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
