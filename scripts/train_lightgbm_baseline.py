"""Persistence와 고정 설정 LightGBM 기준 모델을 시간순 평가한다."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.lightgbm_baseline import (
    RANDOM_SEED,
    add_series_code,
    fit_lightgbm,
    regression_metrics,
    select_lag_features,
)

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
RESULTS = Path("results")
MODELS = RESULTS / "models"
PREDICTIONS = RESULTS / "predictions"
FIGURES = RESULTS / "figures"
TARGETS = ["bod_mg_l", "tn_mg_l", "tp_mg_l"]
WINDOWS = [7, 14, 30]
STANDARD_VARIANTS = ["quality", "quality_temperature"]
KEYS = ["series_id", "target_date"]


def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        frame.loc[frame["split"].eq("train")].copy(),
        frame.loc[frame["split"].eq("validation")].copy(),
        frame.loc[frame["split"].eq("test")].copy(),
    )


def _metric_rows(
    evaluation: pd.DataFrame,
    predictions: np.ndarray,
    target_column: str,
    metadata: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]], pd.DataFrame]:
    actual = evaluation[target_column].to_numpy(dtype=float)
    overall = {**metadata, "eval_rows": len(evaluation), **regression_metrics(actual, predictions)}
    by_series = []
    prediction_frame = evaluation[
        ["series_id", "facility", "plant", "target_date", "split"]
    ].copy()
    prediction_frame["actual"] = actual
    prediction_frame["prediction"] = predictions
    for (series_id, facility, plant), group in prediction_frame.groupby(
        ["series_id", "facility", "plant"], dropna=False
    ):
        metrics = regression_metrics(
            group["actual"].to_numpy(), group["prediction"].to_numpy()
        )
        by_series.append(
            {
                **metadata,
                "series_id": series_id,
                "facility": facility,
                "plant": plant,
                "eval_rows": len(group),
                **metrics,
            }
        )
    for key, value in metadata.items():
        prediction_frame[key] = value
    return overall, by_series, prediction_frame


def _assert_same_rows(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    left_keys = left[KEYS].sort_values(KEYS).reset_index(drop=True)
    right_keys = right[KEYS].sort_values(KEYS).reset_index(drop=True)
    if not left_keys.equals(right_keys):
        raise RuntimeError(f"동일 평가행 조건 불충족: {label}")


def _save_selected_plot(
    predictions: pd.DataFrame,
    target: str,
    selected: dict[str, object],
) -> None:
    chosen = predictions.loc[
        predictions["target"].eq(target)
        & predictions["variant"].eq(selected["variant"])
        & predictions["window_days"].eq(selected["window_days"])
        & predictions["model_type"].eq("lightgbm")
        & predictions["split"].eq("test")
    ].copy()
    if chosen.empty:
        return
    largest_series = chosen["series_id"].value_counts().index[0]
    timeline = chosen.loc[chosen["series_id"].eq(largest_series)].sort_values("target_date")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(chosen["actual"], chosen["prediction"], s=8, alpha=0.35)
    lower = min(chosen["actual"].min(), chosen["prediction"].min())
    upper = max(chosen["actual"].max(), chosen["prediction"].max())
    axes[0].plot([lower, upper], [lower, upper], color="black", linewidth=1)
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Prediction")
    axes[0].set_title(f"{target} test: actual vs prediction")
    axes[1].plot(timeline["target_date"], timeline["actual"], label="actual", linewidth=1)
    axes[1].plot(timeline["target_date"], timeline["prediction"], label="prediction", linewidth=1)
    axes[1].set_title(f"Largest series timeline ({largest_series})")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"{target}_selected_test.png", dpi=150)
    plt.close(fig)


def main() -> None:
    for path in (MODELS, PREDICTIONS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    series_rows = []
    prediction_parts = []

    for window in WINDOWS:
        quality = pd.read_parquet(PROCESSED / f"tabular_quality_w{window}.parquet")
        temperature = pd.read_parquet(
            PROCESSED / f"tabular_quality_temperature_w{window}.parquet"
        )
        for split_name in ("train", "validation", "test"):
            _assert_same_rows(
                quality.loc[quality["split"].eq(split_name)],
                temperature.loc[temperature["split"].eq(split_name)],
                f"quality-temperature w{window} {split_name}",
            )

        _, quality_validation, quality_test = _split(quality)
        for target in TARGETS:
            target_column = f"target_{target}"
            persistence_feature = f"{target}_lag_0"
            for split_name, evaluation in (
                ("validation", quality_validation),
                ("test", quality_test),
            ):
                metadata = {
                    "experiment": "persistence",
                    "model_type": "persistence",
                    "variant": "quality",
                    "window_days": window,
                    "target": target,
                    "split": split_name,
                    "best_iteration": 0,
                }
                overall, per_series, pred = _metric_rows(
                    evaluation,
                    evaluation[persistence_feature].to_numpy(dtype=float),
                    target_column,
                    metadata,
                )
                metric_rows.append(overall)
                series_rows.extend(per_series)
                prediction_parts.append(pred)

        for variant, frame in (("quality", quality), ("quality_temperature", temperature)):
            train, validation, test = _split(frame)
            train, validation, test = add_series_code(train, validation, test)
            features = [*select_lag_features(train), "series_code"]
            for target in TARGETS:
                target_column = f"target_{target}"
                model = fit_lightgbm(train, validation, target_column, features)
                model_name = f"{variant}_w{window}_{target}"
                joblib.dump(
                    {"model": model, "features": features, "target": target_column},
                    MODELS / f"{model_name}.joblib",
                )
                for split_name, evaluation in (("validation", validation), ("test", test)):
                    metadata = {
                        "experiment": "standard",
                        "model_type": "lightgbm",
                        "variant": variant,
                        "window_days": window,
                        "target": target,
                        "split": split_name,
                        "best_iteration": int(model.best_iteration_ or model.n_estimators),
                    }
                    overall, per_series, pred = _metric_rows(
                        evaluation,
                        model.predict(evaluation[features]),
                        target_column,
                        metadata,
                    )
                    metric_rows.append(overall)
                    series_rows.extend(per_series)
                    prediction_parts.append(pred)

    metrics = pd.DataFrame(metric_rows)
    standard_validation = metrics.loc[
        metrics["experiment"].eq("standard")
        & metrics["model_type"].eq("lightgbm")
        & metrics["split"].eq("validation")
    ]
    selections = {}
    for target, group in standard_validation.groupby("target"):
        best = group.sort_values(["mae", "rmse"]).iloc[0]
        selections[target] = {
            "variant": best["variant"],
            "window_days": int(best["window_days"]),
            "validation_mae": float(best["mae"]),
            "selection_basis": "Validation MAE; Test 미사용",
        }

    for target, selected in selections.items():
        window = int(selected["window_days"])
        base_variant = str(selected["variant"])
        flow_variant = (
            "quality_temperature_flow"
            if base_variant == "quality_temperature"
            else "quality_flow"
        )
        frame = pd.read_parquet(PROCESSED / f"tabular_{flow_variant}_w{window}.parquet")
        train, validation, test = _split(frame)
        train, validation, test = add_series_code(train, validation, test)
        all_features = [*select_lag_features(train, include_flow=True), "series_code"]
        reference_features = [*select_lag_features(train, include_flow=False), "series_code"]
        target_column = f"target_{target}"
        for model_type, features in (
            ("lightgbm_reference_on_flow_rows", reference_features),
            ("lightgbm_flow", all_features),
        ):
            model = fit_lightgbm(train, validation, target_column, features)
            joblib.dump(
                {"model": model, "features": features, "target": target_column},
                MODELS / f"{model_type}_w{window}_{target}.joblib",
            )
            for split_name, evaluation in (("validation", validation), ("test", test)):
                metadata = {
                    "experiment": "flow_fair_comparison",
                    "model_type": model_type,
                    "variant": flow_variant,
                    "window_days": window,
                    "target": target,
                    "split": split_name,
                    "best_iteration": int(model.best_iteration_ or model.n_estimators),
                }
                overall, per_series, pred = _metric_rows(
                    evaluation,
                    model.predict(evaluation[features]),
                    target_column,
                    metadata,
                )
                metric_rows.append(overall)
                series_rows.extend(per_series)
                prediction_parts.append(pred)

    metrics = pd.DataFrame(metric_rows)
    series_metrics = pd.DataFrame(series_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics.to_csv(REPORTS / "model_metrics.csv", index=False, encoding="utf-8-sig")
    series_metrics.to_csv(
        REPORTS / "model_metrics_by_series.csv", index=False, encoding="utf-8-sig"
    )
    predictions.to_parquet(PREDICTIONS / "predictions.parquet", index=False)

    for target, selected in selections.items():
        selected_test = metrics.loc[
            metrics["model_type"].eq("lightgbm")
            & metrics["variant"].eq(selected["variant"])
            & metrics["window_days"].eq(selected["window_days"])
            & metrics["target"].eq(target)
            & metrics["split"].eq("test")
        ].iloc[0]
        selections[target]["test_metrics_locked"] = {
            key: float(selected_test[key]) for key in ("mae", "rmse", "r2")
        }
        _save_selected_plot(predictions, target, selected)

    config = {
        "random_seed": RANDOM_SEED,
        "train": "2016-01-01~2022-12-31",
        "validation": "2023-01-01~2024-12-31",
        "test_locked": "2025-01-01~2026-09-30",
        "selection_metric": "Validation MAE",
        "models": "Persistence, LightGBM fixed configuration",
        "rainfall_experiment": "제외: sumRn 공란을 무강수 0mm로 확정할 수 없음",
        "flow_experiment": "정확키 연결 및 연속 완전사례만; 동일 행 reference와 비교",
        "selections": selections,
    }
    (REPORTS / "model_selection.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
