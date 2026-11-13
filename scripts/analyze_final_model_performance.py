"""저장 예측만으로 최종 모델 성능과 BOD 고농도 계통 집중도를 통합 분석한다."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.compare_tree_lstm_models import _baseline_predictions
from src.models.comparative_models import high_concentration_metrics
from src.models.lightgbm_baseline import regression_metrics

REPORTS = Path("reports")
PREDICTIONS = Path("results/predictions")
FIGURES = Path("results/figures")
PROCESSED = Path("data/processed")
RUN_ID = "2026-11-13"
TARGETS = ["bod_mg_l", "tn_mg_l", "tp_mg_l"]
ALGORITHMS = ["lightgbm", "xgboost", "lstm"]
RAIN_CONFIG = {"bod_mg_l": "C", "tn_mg_l": "E", "tp_mg_l": "C"}
KEYS = ["series_id", "target_date", "target", "split"]


def _candidate_predictions() -> pd.DataFrame:
    baseline = _baseline_predictions()
    lightgbm_rain = pd.read_parquet(PREDICTIONS / "seoul_rainfall_lightgbm_predictions.parquet")
    lightgbm_rain["algorithm"] = "lightgbm"
    xgboost = pd.read_parquet(PREDICTIONS / "xgboost_comparison_predictions.parquet")
    lstm = pd.read_parquet(PREDICTIONS / "lstm_comparison_predictions.parquet")
    rain_sources = {"lightgbm": lightgbm_rain, "xgboost": xgboost, "lstm": lstm}
    parts = []
    for target in TARGETS:
        persistence = baseline.loc[
            baseline["target"].eq(target)
            & baseline["algorithm"].eq("persistence")
            & baseline["window_days"].eq(30)
        ].copy()
        persistence["input_config"] = "P"
        persistence["candidate_id"] = "persistence_P_w30"
        parts.append(persistence)
        for algorithm in ALGORITHMS:
            for window in (7, 14, 30):
                base = baseline.loc[
                    baseline["target"].eq(target)
                    & baseline["algorithm"].eq(algorithm)
                    & baseline["window_days"].eq(window)
                ].copy()
                base["input_config"] = "B"
                base["candidate_id"] = f"{algorithm}_B_w{window}"
                parts.append(base)
            config = RAIN_CONFIG[target]
            rain = rain_sources[algorithm].loc[
                rain_sources[algorithm]["target"].eq(target)
                & rain_sources[algorithm]["window_days"].eq(30)
                & rain_sources[algorithm]["variant"].eq(config)
            ].copy()
            rain["input_config"] = config
            rain["candidate_id"] = f"{algorithm}_{config}_w30"
            parts.append(rain)
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated([*KEYS, "candidate_id"]).any():
        raise RuntimeError("최종 후보 예측 키에 중복이 있습니다.")
    return result


def _all_candidate_common(predictions: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (target, split), group in predictions.groupby(["target", "split"]):
        expected = group["candidate_id"].nunique()
        counts = group.groupby(KEYS, dropna=False)["candidate_id"].nunique()
        common_keys = counts.loc[counts.eq(expected)].index.to_frame(index=False)
        common = group.merge(common_keys, on=KEYS, how="inner", validate="many_to_one")
        if common.groupby(KEYS, dropna=False)["candidate_id"].nunique().ne(expected).any():
            raise RuntimeError("최종 후보 공통 평가 키가 완전하지 않습니다.")
        if common.groupby(KEYS, dropna=False)["actual"].nunique(dropna=False).gt(1).any():
            raise RuntimeError("최종 후보 간 실제값이 일치하지 않습니다.")
        parts.append(common)
    return pd.concat(parts, ignore_index=True)


def _metric_table(common: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = ["target", "split", "algorithm", "input_config", "window_days", "candidate_id"]
    for keys, group in common.groupby(groups, dropna=False):
        rows.append({
            **dict(zip(groups, keys)),
            "eval_rows": len(group),
            **regression_metrics(group["actual"].to_numpy(), group["prediction"].to_numpy()),
        })
    result = pd.DataFrame(rows)
    persistence = result.loc[result["algorithm"].eq("persistence"), [
        "target", "split", "mae",
    ]].rename(columns={"mae": "persistence_mae"})
    result = result.merge(persistence, on=["target", "split"], how="left", validate="many_to_one")
    result["persistence_mae_improvement_pct"] = (
        (result["persistence_mae"] - result["mae"]) / result["persistence_mae"] * 100
    )
    return result


def _selections(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = metrics.loc[
        metrics["split"].eq("validation") & metrics["algorithm"].ne("persistence")
    ]
    selected_validation = validation.sort_values(["target", "mae", "rmse"]).groupby("target").head(1)
    selected_test = metrics.merge(
        selected_validation[["target", "candidate_id"]],
        on=["target", "candidate_id"],
        how="inner",
        validate="many_to_one",
    ).loc[lambda frame: frame["split"].eq("test")]
    return selected_validation.reset_index(drop=True), selected_test.reset_index(drop=True)


def _rainfall_effect(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, config in RAIN_CONFIG.items():
        for split in ("validation", "test"):
            for algorithm in ALGORITHMS:
                base = metrics.loc[
                    metrics["target"].eq(target)
                    & metrics["split"].eq(split)
                    & metrics["algorithm"].eq(algorithm)
                    & metrics["input_config"].eq("B")
                    & metrics["window_days"].eq(30)
                ].iloc[0]
                rain = metrics.loc[
                    metrics["target"].eq(target)
                    & metrics["split"].eq(split)
                    & metrics["algorithm"].eq(algorithm)
                    & metrics["input_config"].eq(config)
                ].iloc[0]
                rows.append({
                    "target": target,
                    "split": split,
                    "algorithm": algorithm,
                    "rain_config": config,
                    "base_mae": base["mae"],
                    "rain_mae": rain["mae"],
                    "rain_mae_improvement_pct": (base["mae"] - rain["mae"]) / base["mae"] * 100,
                    "base_rmse": base["rmse"],
                    "rain_rmse": rain["rmse"],
                    "base_r2": base["r2"],
                    "rain_r2": rain["r2"],
                })
    return pd.DataFrame(rows)


def _algorithm_bod_selections(metrics: pd.DataFrame) -> pd.DataFrame:
    candidates = metrics.loc[
        metrics["target"].eq("bod_mg_l")
        & metrics["split"].eq("validation")
        & metrics["algorithm"].isin(ALGORITHMS)
    ]
    return candidates.sort_values(["algorithm", "mae", "rmse"]).groupby("algorithm").head(1)


def _high_bod(
    common: pd.DataFrame,
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    train = pd.read_parquet(
        PROCESSED / "tabular_seoul_rainfall_w30.parquet",
        columns=["split", "target_bod_mg_l"],
    )
    threshold = float(train.loc[train["split"].eq("train"), "target_bod_mg_l"].quantile(0.90))
    selections = _algorithm_bod_selections(metrics)
    chosen = []
    persistence = common.loc[
        common["target"].eq("bod_mg_l")
        & common["split"].eq("test")
        & common["algorithm"].eq("persistence")
    ]
    chosen.append(persistence)
    for _, selected in selections.iterrows():
        chosen.append(common.loc[
            common["target"].eq("bod_mg_l")
            & common["split"].eq("test")
            & common["candidate_id"].eq(selected["candidate_id"])
        ])
    selected_test = pd.concat(chosen, ignore_index=True)
    high = selected_test.loc[selected_test["actual"].ge(threshold)].copy()
    high["error"] = high["prediction"] - high["actual"]
    high["absolute_error"] = high["error"].abs()
    summary_rows = []
    for algorithm, group in high.groupby("algorithm"):
        summary_rows.append({
            "algorithm": algorithm,
            "candidate_id": group["candidate_id"].iloc[0],
            "threshold": threshold,
            **high_concentration_metrics(
                group["actual"].to_numpy(), group["prediction"].to_numpy(), threshold
            ),
        })
    summary = pd.DataFrame(summary_rows)
    series_rows = []
    for keys, group in high.groupby(["algorithm", "series_id", "facility", "plant"], dropna=False):
        algorithm, series_id, facility, plant = keys
        series_rows.append({
            "algorithm": algorithm,
            "series_id": series_id,
            "facility": facility,
            "plant": plant,
            "rows": len(group),
            "high_bod_row_share": len(group) / len(high.loc[high["algorithm"].eq(algorithm)]),
            "mae": float(group["absolute_error"].mean()),
            "mean_bias": float(group["error"].mean()),
            "underprediction_rate": float(group["error"].lt(0).mean()),
            "absolute_error_share": float(
                group["absolute_error"].sum()
                / high.loc[high["algorithm"].eq(algorithm), "absolute_error"].sum()
            ),
        })
    by_series = pd.DataFrame(series_rows)
    concentration = (
        by_series.sort_values(["algorithm", "absolute_error_share"], ascending=[True, False])
        .groupby("algorithm")
        .head(3)
        .reset_index(drop=True)
    )
    return summary, by_series, concentration, threshold


def _save_figures(
    metrics: pd.DataFrame,
    selected_validation: pd.DataFrame,
    common: pd.DataFrame,
    high_summary: pd.DataFrame,
    high_by_series: pd.DataFrame,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    chosen_ids = selected_validation.set_index("target")["candidate_id"].to_dict()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, target in zip(axes, TARGETS):
        validation = metrics.loc[
            metrics["target"].eq(target)
            & metrics["split"].eq("validation")
            & (
                metrics["algorithm"].eq("persistence")
                | metrics["candidate_id"].isin(
                    _algorithm_bod_selections(metrics)["candidate_id"]
                    if target == "bod_mg_l" else
                    metrics.loc[
                        metrics["target"].eq(target)
                        & metrics["split"].eq("validation")
                        & metrics["algorithm"].isin(ALGORITHMS)
                    ].sort_values(["algorithm", "mae"]).groupby("algorithm").head(1)["candidate_id"]
                )
            )
        ].sort_values("mae")
        ax.bar(validation["candidate_id"], validation["mae"])
        ax.tick_params(axis="x", rotation=70)
        ax.set(title=target, ylabel="Validation MAE")
    fig.tight_layout()
    fig.savefig(FIGURES / f"final_model_validation_mae_{RUN_ID}.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, target in zip(axes, TARGETS):
        group = common.loc[
            common["target"].eq(target)
            & common["split"].eq("test")
            & common["candidate_id"].eq(chosen_ids[target])
        ]
        ax.scatter(group["actual"], group["prediction"], s=7, alpha=0.25)
        lower = min(group["actual"].min(), group["prediction"].min())
        upper = max(group["actual"].max(), group["prediction"].max())
        ax.plot([lower, upper], [lower, upper], color="black", linewidth=1)
        ax.set(title=f"{target}: {chosen_ids[target]}", xlabel="Actual", ylabel="Prediction")
    fig.tight_layout()
    fig.savefig(FIGURES / f"final_model_test_predictions_{RUN_ID}.png", dpi=150)
    plt.close(fig)

    pivot = high_by_series.pivot(index="series_id", columns="algorithm", values="mae")
    pivot.plot(kind="bar", figsize=(13, 6))
    plt.ylabel("High-BOD MAE")
    plt.title("High-BOD error by observation series")
    plt.tight_layout()
    plt.savefig(FIGURES / f"final_model_high_bod_series_{RUN_ID}.png", dpi=150)
    plt.close()


def _write_report(
    metrics: pd.DataFrame,
    selected_validation: pd.DataFrame,
    selected_test: pd.DataFrame,
    rainfall_effect: pd.DataFrame,
    high_summary: pd.DataFrame,
    high_concentration: pd.DataFrame,
) -> None:
    validation_leaders = metrics.loc[
        metrics["split"].eq("validation")
        & (
            metrics["algorithm"].eq("persistence")
            | metrics["candidate_id"].isin(
                metrics.loc[metrics["algorithm"].isin(ALGORITHMS)]
                .sort_values(["target", "algorithm", "mae"])
                .groupby(["target", "algorithm"])
                .head(1)["candidate_id"]
            )
        )
    ]
    report = f"""# 유입수질 예측 모델 최종 성능 통합 및 BOD 고농도 진단

## 평가 계약

- 다음 날(t+1) 처리장·유입 관측 계통별 BOD·T-N·T-P 예측이다.
- Train 2016~2022, Validation 2023~2024, Test 2025~2026-09-30 분할을 유지했다.
- B 7·14·30일과 목표별 강우 후보(BOD C, T-N E, T-P C), Persistence를 모든 후보 공통 평가 키로 재계산했다.
- 모델 선택에는 Validation MAE만 사용했고 Test는 선택 후 참고 성능으로만 제시한다.
- Test는 반복 확인된 구간이며 완전히 독립적인 최종 홀드아웃이 아니다.

## 알고리즘별 Validation 최적 후보

{validation_leaders[["target", "algorithm", "input_config", "window_days", "eval_rows", "mae", "rmse", "r2", "persistence_mae_improvement_pct"]].sort_values(["target", "mae"]).to_markdown(index=False)}

## 목표별 최종 Validation 선택

{selected_validation[["target", "algorithm", "input_config", "window_days", "eval_rows", "mae", "rmse", "r2", "persistence_mae_improvement_pct"]].to_markdown(index=False)}

## 선택 모델의 Test 참고 성능

{selected_test[["target", "algorithm", "input_config", "window_days", "eval_rows", "mae", "rmse", "r2", "persistence_mae_improvement_pct"]].to_markdown(index=False)}

## 서울시 강우량 추가 효과

{rainfall_effect.loc[rainfall_effect["split"].eq("validation")].to_markdown(index=False)}

강우 입력은 세 알고리즘 모두 Validation MAE를 개선했지만 이는 예측 연관성이며 인과효과가 아니다. 센터별 강우 관측소 매핑은 `provisional` 상태다.

## BOD 고농도 진단

{high_summary.to_markdown(index=False)}

고농도 기준은 Train BOD 90백분위 180mg/L이며 모든 모델은 동일한 Test 고농도 표본을 사용했다.

### 절대오차 기여 상위 계통

{high_concentration.to_markdown(index=False)}

`absolute_error_share`는 해당 모델의 전체 고농도 절대오차 중 계통별 비중이다. 표본 수와 함께 해석해야 한다.

## 데이터 범위와 한계

- 원본 11개 계통 중 난지 제1처리장은 2건뿐이어서 연속 윈도를 만들 수 없었다.
- 학습 윈도는 10개 계통에서 생성됐고, 2021년 종료된 서남 제1·제2처리장과 시설현대화를 제외한 7개 계통이 Validation/Test에 포함됐다.
- LSTM은 단일 seed 결과다.
- 입력일 t의 실측 기온·강우 공개 지연은 운영 적용 전에 별도 확인해야 한다.
"""
    (REPORTS / f"final_model_integration_report_{RUN_ID}.md").write_text(report, encoding="utf-8")


def main() -> None:
    predictions = _candidate_predictions()
    common = _all_candidate_common(predictions)
    metrics = _metric_table(common)
    selected_validation, selected_test = _selections(metrics)
    rainfall_effect = _rainfall_effect(metrics)
    high_summary, high_by_series, concentration, threshold = _high_bod(common, metrics)
    _save_figures(metrics, selected_validation, common, high_summary, high_by_series)
    _write_report(
        metrics,
        selected_validation,
        selected_test,
        rainfall_effect,
        high_summary,
        concentration,
    )
    metrics.to_csv(REPORTS / f"final_model_candidate_metrics_{RUN_ID}.csv", index=False, encoding="utf-8-sig")
    selected_validation.assign(selection_split="validation").to_csv(
        REPORTS / f"final_model_selected_validation_{RUN_ID}.csv", index=False, encoding="utf-8-sig"
    )
    selected_test.to_csv(
        REPORTS / f"final_model_selected_test_{RUN_ID}.csv", index=False, encoding="utf-8-sig"
    )
    rainfall_effect.to_csv(
        REPORTS / f"final_model_rainfall_effect_{RUN_ID}.csv", index=False, encoding="utf-8-sig"
    )
    high_summary.to_csv(
        REPORTS / f"final_model_high_bod_summary_{RUN_ID}.csv", index=False, encoding="utf-8-sig"
    )
    high_by_series.to_csv(
        REPORTS / f"final_model_high_bod_by_series_{RUN_ID}.csv", index=False, encoding="utf-8-sig"
    )
    common.to_parquet(
        PREDICTIONS / f"final_model_all_candidate_common_{RUN_ID}.parquet", index=False
    )
    print({
        "candidate_metrics": len(metrics),
        "common_prediction_rows": len(common),
        "high_bod_threshold": threshold,
        "selected": selected_validation[["target", "candidate_id", "mae"]].to_dict("records"),
    })


if __name__ == "__main__":
    main()
