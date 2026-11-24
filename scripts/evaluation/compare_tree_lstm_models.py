"""Persistence·LightGBM·XGBoost·LSTM을 공통 평가행에서 비교한다."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.shared.comparative_models import (
    common_algorithm_rows,
    high_concentration_metrics,
    summarize_prediction_metrics,
)

REPORTS = Path("reports")
PREDICTIONS = Path("results/predictions")
FIGURES = Path("results/figures")
PROCESSED = Path("data/processed")
TARGETS = ["bod_mg_l", "tn_mg_l", "tp_mg_l"]
WINDOWS = [7, 14, 30]
ALGORITHMS = ["persistence", "lightgbm", "xgboost", "lstm"]
RAIN_CONFIG = {"bod_mg_l": "C", "tn_mg_l": "E", "tp_mg_l": "C"}


def _baseline_predictions() -> pd.DataFrame:
    existing = pd.read_parquet(PREDICTIONS / "predictions.parquet")
    persistence = existing.loc[
        existing["experiment"].eq("persistence")
        & existing["model_type"].eq("persistence")
    ].copy()
    persistence["algorithm"] = "persistence"
    persistence["input_config"] = "B"
    lightgbm = existing.loc[
        existing["experiment"].eq("standard")
        & existing["model_type"].eq("lightgbm")
        & existing["variant"].eq("quality_temperature")
    ].copy()
    lightgbm["algorithm"] = "lightgbm"
    lightgbm["input_config"] = "B"
    xgboost = pd.read_parquet(PREDICTIONS / "xgboost_comparison_predictions.parquet")
    lstm = pd.read_parquet(PREDICTIONS / "lstm_comparison_predictions.parquet")
    return pd.concat(
        [
            persistence,
            lightgbm,
            xgboost.loc[xgboost["variant"].eq("B")],
            lstm.loc[lstm["variant"].eq("B")],
        ],
        ignore_index=True,
    )


def _within_window_common(predictions: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in predictions.groupby(["target", "split", "window_days"], dropna=False):
        parts.append(common_algorithm_rows(group, ALGORITHMS, require_identical_keys=True))
    return pd.concat(parts, ignore_index=True)


def _validation_selections(metrics: pd.DataFrame) -> dict[str, dict[str, int]]:
    selections: dict[str, dict[str, int]] = {}
    validation = metrics.loc[
        metrics["split"].eq("validation") & metrics["algorithm"].ne("persistence")
    ]
    for target, target_group in validation.groupby("target"):
        selections[target] = {}
        for algorithm, group in target_group.groupby("algorithm"):
            best = group.sort_values(["mae", "rmse"]).iloc[0]
            selections[target][algorithm] = int(best["window_days"])
    return selections


def _selected_common(
    predictions: pd.DataFrame,
    selections: dict[str, dict[str, int]],
) -> pd.DataFrame:
    parts = []
    for target in TARGETS:
        for split in ("validation", "test"):
            chosen = [
                predictions.loc[
                    predictions["target"].eq(target)
                    & predictions["split"].eq(split)
                    & predictions["algorithm"].eq("persistence")
                    & predictions["window_days"].eq(30)
                ]
            ]
            for algorithm in ("lightgbm", "xgboost", "lstm"):
                chosen.append(
                    predictions.loc[
                        predictions["target"].eq(target)
                        & predictions["split"].eq(split)
                        & predictions["algorithm"].eq(algorithm)
                        & predictions["window_days"].eq(selections[target][algorithm])
                    ]
                )
            parts.append(common_algorithm_rows(pd.concat(chosen, ignore_index=True), ALGORITHMS))
    return pd.concat(parts, ignore_index=True)


def _rainfall_common(baseline: pd.DataFrame) -> pd.DataFrame:
    lightgbm_rain = pd.read_parquet(
        PREDICTIONS / "seoul_rainfall_lightgbm_predictions.parquet"
    )
    lightgbm_rain["algorithm"] = "lightgbm"
    xgboost = pd.read_parquet(PREDICTIONS / "xgboost_comparison_predictions.parquet")
    lstm = pd.read_parquet(PREDICTIONS / "lstm_comparison_predictions.parquet")
    parts = []
    identifiers = [
        "persistence", "lightgbm_B", "lightgbm_rain",
        "xgboost_B", "xgboost_rain", "lstm_B", "lstm_rain",
    ]
    for target, config in RAIN_CONFIG.items():
        for split in ("validation", "test"):
            persistence = baseline.loc[
                baseline["target"].eq(target)
                & baseline["split"].eq(split)
                & baseline["algorithm"].eq("persistence")
                & baseline["window_days"].eq(30)
            ].copy()
            persistence["algorithm"] = "persistence"
            candidates = [persistence]
            for algorithm, source in (
                ("lightgbm", baseline), ("xgboost", xgboost), ("lstm", lstm)
            ):
                base = source.loc[
                    source["target"].eq(target)
                    & source["split"].eq(split)
                    & source["window_days"].eq(30)
                    & source["algorithm"].eq(algorithm)
                    & source["variant"].isin(["quality_temperature", "B"])
                ].copy()
                base["algorithm"] = f"{algorithm}_B"
                candidates.append(base)
            for algorithm, source in (
                ("lightgbm", lightgbm_rain), ("xgboost", xgboost), ("lstm", lstm)
            ):
                rain = source.loc[
                    source["target"].eq(target)
                    & source["split"].eq(split)
                    & source["window_days"].eq(30)
                    & source["algorithm"].eq(algorithm)
                    & source["variant"].eq(config)
                ].copy()
                rain["algorithm"] = f"{algorithm}_rain"
                candidates.append(rain)
            parts.append(common_algorithm_rows(
                pd.concat(candidates, ignore_index=True),
                identifiers,
                require_identical_keys=True,
            ))
    return pd.concat(parts, ignore_index=True)


def _metrics_by_series(predictions: pd.DataFrame) -> pd.DataFrame:
    return summarize_prediction_metrics(
        predictions,
        ["target", "split", "series_id", "facility", "plant", "algorithm"],
    )


def _high_bod_diagnostics(rainfall_common: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    train = pd.read_parquet(
        PROCESSED / "tabular_seoul_rainfall_w30.parquet",
        columns=["split", "target_bod_mg_l"],
    )
    threshold = float(
        train.loc[train["split"].eq("train"), "target_bod_mg_l"].quantile(0.90)
    )
    chosen_ids = ["persistence", "lightgbm_rain", "xgboost_rain", "lstm_rain"]
    test = rainfall_common.loc[
        rainfall_common["target"].eq("bod_mg_l")
        & rainfall_common["split"].eq("test")
        & rainfall_common["algorithm"].isin(chosen_ids)
    ]
    rows = []
    for algorithm, group in test.groupby("algorithm"):
        rows.append(
            {
                "algorithm": algorithm.replace("_rain", ""),
                "threshold": threshold,
                **high_concentration_metrics(
                    group["actual"].to_numpy(), group["prediction"].to_numpy(), threshold
                ),
            }
        )
    return pd.DataFrame(rows), threshold


def _save_plots(
    selected: pd.DataFrame,
    rainfall_common: pd.DataFrame,
    threshold: float,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        test = selected.loc[selected["target"].eq(target) & selected["split"].eq("test")]
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        for ax, algorithm in zip(axes.flat, ALGORITHMS):
            group = test.loc[test["algorithm"].eq(algorithm)]
            ax.scatter(group["actual"], group["prediction"], s=7, alpha=0.25)
            lower = min(group["actual"].min(), group["prediction"].min())
            upper = max(group["actual"].max(), group["prediction"].max())
            ax.plot([lower, upper], [lower, upper], color="black", linewidth=1)
            ax.set(title=algorithm, xlabel="Actual", ylabel="Prediction")
        fig.suptitle(f"{target} common Test rows")
        fig.tight_layout()
        fig.savefig(FIGURES / f"algorithm_comparison_{target}_test.png", dpi=150)
        plt.close(fig)

    high = rainfall_common.loc[
        rainfall_common["target"].eq("bod_mg_l")
        & rainfall_common["split"].eq("test")
        & rainfall_common["algorithm"].isin(
            ["persistence", "lightgbm_rain", "xgboost_rain", "lstm_rain"]
        )
        & rainfall_common["actual"].ge(threshold)
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, identifier in zip(
        axes.flat, ["persistence", "lightgbm_rain", "xgboost_rain", "lstm_rain"]
    ):
        group = high.loc[high["algorithm"].eq(identifier)]
        ax.scatter(group["actual"], group["prediction"], s=10, alpha=0.4)
        lower = min(group["actual"].min(), group["prediction"].min())
        upper = max(group["actual"].max(), group["prediction"].max())
        ax.plot([lower, upper], [lower, upper], color="black", linewidth=1)
        ax.set(title=identifier.replace("_rain", ""), xlabel="Actual BOD", ylabel="Prediction")
    fig.suptitle(f"High BOD Test rows (Train p90 >= {threshold:.1f})")
    fig.tight_layout()
    fig.savefig(FIGURES / "algorithm_comparison_high_bod_test.png", dpi=150)
    plt.close(fig)


def _training_times() -> pd.DataFrame:
    xgboost = pd.read_csv(REPORTS / "xgboost_comparison_training_times.csv")
    lstm = pd.read_csv(REPORTS / "lstm_comparison_training_times.csv")
    rows = [xgboost, lstm]
    detail = pd.concat(rows, ignore_index=True)
    summary = detail.groupby(["algorithm", "input_config"], as_index=False).agg(
        model_count=("target", "size"),
        total_training_seconds=("training_seconds", "sum"),
        mean_training_seconds=("training_seconds", "mean"),
        max_training_seconds=("training_seconds", "max"),
    )
    lightgbm_note = pd.DataFrame(
        [{
            "algorithm": "lightgbm",
            "input_config": "historical_full_run",
            "model_count": np.nan,
            "total_training_seconds": 55.61,
            "mean_training_seconds": np.nan,
            "max_training_seconds": np.nan,
        }]
    )
    return pd.concat([summary, lightgbm_note], ignore_index=True)


def _write_report(
    within_metrics: pd.DataFrame,
    selected_metrics: pd.DataFrame,
    rainfall_metrics: pd.DataFrame,
    high_bod: pd.DataFrame,
    training_times: pd.DataFrame,
    selections: dict[str, dict[str, int]],
) -> None:
    validation_selected = selected_metrics.loc[selected_metrics["split"].eq("validation")]
    test_selected = selected_metrics.loc[selected_metrics["split"].eq("test")]
    rain_validation = rainfall_metrics.loc[rainfall_metrics["split"].eq("validation")]
    report = f"""# XGBoost·LSTM 및 LightGBM 3종 모델 비교

## 평가 계약

- 목표: 처리장·유입 관측 계통별 다음 날 BOD·T-N·T-P.
- 공통 입력: 과거 수질 3종+ASOS 기온(B), 7·14·30일.
- 분할: Train 2016~2022, Validation 2023~2024, Test 2025~2026-09-30 (`target_date` 기준).
- 원본 11개 관측 계통 중 유효 연속 입력창은 Train 10개, Validation/Test 7개 계통에서 생성됐다.
- 모델 선택은 Validation MAE만 사용했다. Test는 반복 확인된 참고 구간으로 완전히 독립적인 최종 홀드아웃이 아니다.
- XGBoost 3.4.1과 CPU PyTorch 2.15.0을 사용했으며 CUDA 장치는 없었다.

## Validation 선택 결과

선택 윈도: `{json.dumps(selections, ensure_ascii=False)}`

{validation_selected.to_markdown(index=False)}

## 동일 평가 표본 Test 참고 성능

{test_selected.to_markdown(index=False)}

전체 7·14·30일 윈도별 공통 성능은 `reports/model_algorithm_common_metrics.csv`에 보존했다.

## 서울시 강우량 추가 효과

{rain_validation.to_markdown(index=False)}

강우 비교는 BOD C/30일, T-N E/30일, T-P C/30일을 사용했다. XGBoost와 LightGBM은 NaN native 처리를 사용했고 LSTM은 Train 중앙값 대치와 변수별 결측 표시를 함께 사용했다. ASOS 강수량은 제외했다.

## BOD 고농도 진단

{high_bod.to_markdown(index=False)}

고농도 기준은 기존과 동일한 Train BOD 90백분위다. 이 단계에서는 원인 진단만 수행했으며 가중 손실·표본 가중은 적용하지 않았다.

## 학습시간

{training_times.to_markdown(index=False)}

LightGBM 55.61초는 기존 전체 기준모델 실행 기록으로 B 구성만의 순수 합계가 아니므로 XGBoost·LSTM 합계와 직접 등가 비교할 수 없다.

## 해석과 제한

- 강우량 추가 효과는 예측 연관성이지 인과효과가 아니다.
- 센터별 서울시 강우 관측소 매핑은 `provisional`이다.
- LSTM 단일 seed 결과이므로 신경망 seed 안정성은 추가 검증이 필요하다.
- 실제 운영에서 입력일 t 강우·기온의 공개 지연을 별도로 확인해야 한다.
"""
    (REPORTS / "xgboost_lstm_comparison_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    baseline = _baseline_predictions()
    within = _within_window_common(baseline)
    within_metrics = summarize_prediction_metrics(
        within, ["target", "split", "window_days", "algorithm"]
    )
    selections = _validation_selections(within_metrics)
    selected = _selected_common(baseline, selections)
    selected_metrics = summarize_prediction_metrics(
        selected, ["target", "split", "algorithm"]
    )
    selected_series = _metrics_by_series(selected)

    rainfall = _rainfall_common(baseline)
    rainfall_metrics = summarize_prediction_metrics(
        rainfall, ["target", "split", "algorithm"]
    )
    rainfall_metrics[["base_algorithm", "input_config"]] = rainfall_metrics["algorithm"].str.extract(
        r"^(persistence|lightgbm|xgboost|lstm)(?:_(B|rain))?$"
    )
    rainfall_metrics["input_config"] = rainfall_metrics.apply(
        lambda row: "P" if row["algorithm"] == "persistence" else (
            "B" if row["input_config"] == "B" else RAIN_CONFIG[row["target"]]
        ),
        axis=1,
    )
    high_bod, threshold = _high_bod_diagnostics(rainfall)
    training_times = _training_times()
    _save_plots(selected, rainfall, threshold)

    within_metrics.to_csv(REPORTS / "model_algorithm_common_metrics.csv", index=False, encoding="utf-8-sig")
    selected_metrics.to_csv(REPORTS / "model_algorithm_selected_metrics.csv", index=False, encoding="utf-8-sig")
    selected_series.to_csv(REPORTS / "model_algorithm_metrics_by_series.csv", index=False, encoding="utf-8-sig")
    rainfall_metrics.to_csv(REPORTS / "model_algorithm_rainfall_effect.csv", index=False, encoding="utf-8-sig")
    high_bod.to_csv(REPORTS / "model_algorithm_high_bod.csv", index=False, encoding="utf-8-sig")
    training_times.to_csv(REPORTS / "model_algorithm_training_times.csv", index=False, encoding="utf-8-sig")
    selected.to_parquet(PREDICTIONS / "model_algorithm_selected_common_predictions.parquet", index=False)
    rainfall.to_parquet(PREDICTIONS / "model_algorithm_rainfall_common_predictions.parquet", index=False)
    config = {
        "selection_metric": "Validation MAE",
        "baseline_selections": selections,
        "rainfall_configs": RAIN_CONFIG,
        "high_bod_threshold": threshold,
        "algorithms": ALGORITHMS,
        "test_interpretation": "반복 확인된 참고 구간; 독립 최종 홀드아웃 아님",
    }
    (REPORTS / "model_algorithm_comparison_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(
        within_metrics,
        selected_metrics,
        rainfall_metrics,
        high_bod,
        training_times,
        selections,
    )


if __name__ == "__main__":
    main()
