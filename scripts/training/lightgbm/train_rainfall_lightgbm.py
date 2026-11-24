"""강수 결측 허용 LightGBM C·D를 기존 A·B·Persistence와 공정 비교한다."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scripts.training.lightgbm.train_lightgbm_baseline import _assert_same_rows, _metric_rows, _split
from src.models.lightgbm.lightgbm_baseline import (
    RANDOM_SEED,
    add_series_code,
    fit_lightgbm,
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
VARIANTS = ["rainfall_c", "rainfall_d"]


def main() -> None:
    for path in (MODELS, PREDICTIONS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
    metric_rows, series_rows, prediction_parts = [], [], []
    for window in WINDOWS:
        reference = pd.read_parquet(PROCESSED / f"tabular_quality_temperature_w{window}.parquet")
        for variant in VARIANTS:
            frame = pd.read_parquet(PROCESSED / f"tabular_{variant}_w{window}.parquet")
            for split_name in ("train", "validation", "test"):
                _assert_same_rows(
                    reference.loc[reference["split"].eq(split_name)],
                    frame.loc[frame["split"].eq(split_name)],
                    f"{variant} w{window} {split_name}",
                )
            train, validation, test = add_series_code(*_split(frame))
            features = [*select_lag_features(train), "series_code"]
            for target in TARGETS:
                target_column = f"target_{target}"
                model = fit_lightgbm(train, validation, target_column, features)
                joblib.dump(
                    {"model": model, "features": features, "target": target_column},
                    MODELS / f"rainfall_experiment_{variant}_w{window}_{target}.joblib",
                )
                for split_name, evaluation in (("validation", validation), ("test", test)):
                    metadata = {
                        "experiment": "rainfall_missing_tolerant",
                        "model_type": "lightgbm",
                        "variant": variant,
                        "window_days": window,
                        "target": target,
                        "split": split_name,
                        "best_iteration": int(model.best_iteration_ or model.n_estimators),
                    }
                    overall, per_series, predictions = _metric_rows(
                        evaluation, model.predict(evaluation[features]), target_column, metadata
                    )
                    metric_rows.append(overall)
                    series_rows.extend(per_series)
                    prediction_parts.append(predictions)

    rain_metrics = pd.DataFrame(metric_rows)
    rain_series = pd.DataFrame(series_rows)
    rain_predictions = pd.concat(prediction_parts, ignore_index=True)
    baseline_metrics = pd.read_csv(REPORTS / "model_metrics.csv")
    baseline_series = pd.read_csv(REPORTS / "model_metrics_by_series.csv")
    comparison_metrics = pd.concat(
        [baseline_metrics.loc[baseline_metrics["experiment"].isin(["persistence", "standard"])], rain_metrics],
        ignore_index=True,
    )
    comparison_series = pd.concat(
        [baseline_series.loc[baseline_series["experiment"].isin(["persistence", "standard"])], rain_series],
        ignore_index=True,
    )
    rain_metrics.to_csv(REPORTS / "rainfall_model_metrics_new.csv", index=False, encoding="utf-8-sig")
    comparison_metrics.to_csv(REPORTS / "rainfall_model_metrics.csv", index=False, encoding="utf-8-sig")
    comparison_series.to_csv(REPORTS / "rainfall_model_metrics_by_series.csv", index=False, encoding="utf-8-sig")
    rain_predictions.to_parquet(PREDICTIONS / "rainfall_experiment_predictions.parquet", index=False)

    candidates = comparison_metrics.loc[
        comparison_metrics["split"].eq("validation")
        & comparison_metrics["model_type"].eq("lightgbm")
        & comparison_metrics["variant"].isin(["quality", "quality_temperature", *VARIANTS])
    ]
    selections = {}
    for target, group in candidates.groupby("target"):
        best = group.sort_values(["mae", "rmse"]).iloc[0]
        test = comparison_metrics.loc[
            comparison_metrics["split"].eq("test")
            & comparison_metrics["model_type"].eq("lightgbm")
            & comparison_metrics["variant"].eq(best["variant"])
            & comparison_metrics["window_days"].eq(best["window_days"])
            & comparison_metrics["target"].eq(target)
        ].iloc[0]
        selections[target] = {
            "variant": best["variant"],
            "window_days": int(best["window_days"]),
            "validation_mae": float(best["mae"]),
            "locked_test": {key: float(test[key]) for key in ("mae", "rmse", "r2")},
        }
    config = {
        "random_seed": RANDOM_SEED,
        "selection_metric": "Validation MAE",
        "test_locked": "2025-01-01~2026-09-30; 선택에 미사용",
        "baseline_replaced": False,
        "rain_blank_policy": "공식 의미 미확인 공란은 NaN; LightGBM native missing 처리",
        "selections_experimental": selections,
    }
    (REPORTS / "rainfall_model_selection.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for target, selected in selections.items():
        variant, window = selected["variant"], selected["window_days"]
        if variant in VARIANTS:
            chosen = rain_predictions.loc[
                rain_predictions["target"].eq(target)
                & rain_predictions["variant"].eq(variant)
                & rain_predictions["window_days"].eq(window)
                & rain_predictions["split"].eq("test")
            ]
        else:
            existing = pd.read_parquet(PREDICTIONS / "predictions.parquet")
            chosen = existing.loc[
                existing["target"].eq(target)
                & existing["variant"].eq(variant)
                & existing["window_days"].eq(window)
                & existing["model_type"].eq("lightgbm")
                & existing["split"].eq("test")
            ]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(chosen["actual"], chosen["prediction"], s=8, alpha=0.3)
        lo, hi = chosen[["actual", "prediction"]].min().min(), chosen[["actual", "prediction"]].max().max()
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set(title=f"{target}: {variant}, {window}d", xlabel="Actual", ylabel="Prediction")
        fig.tight_layout()
        fig.savefig(FIGURES / f"rainfall_experiment_{target}_selected_test.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
