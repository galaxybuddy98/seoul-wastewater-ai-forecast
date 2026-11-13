"""세 수질 달력 피처와 BOD Train-only 2배 가중 XGBoost 실험."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scripts.train_lightgbm_baseline import _split
from src.models.bod_improvement import (
    add_target_calendar_features,
    assert_common_evaluation_rows,
    meets_replacement_goal,
    training_sample_weights,
)
from src.models.lightgbm_baseline import add_series_code
from src.models.seoul_rainfall_experiment import select_seoul_rainfall_features
from src.models.xgboost_forecaster import RANDOM_SEED, fit_xgboost


RUN_ID = "2026-11-13"
PROCESSED = Path("data/processed/tabular_seoul_rainfall_w30.parquet")
EXISTING_PREDICTIONS = Path("results/predictions/final_model_all_candidate_common_2026-11-13.parquet")
REPORTS = Path("reports")
MODELS = Path("results/models")
PREDICTIONS = Path("results/predictions")
FIGURES = Path("results/figures")
TARGETS = ("bod_mg_l", "tn_mg_l", "tp_mg_l")
TARGET_LABELS = {"bod_mg_l": "BOD", "tn_mg_l": "T-N", "tp_mg_l": "T-P"}
BASE_CONFIG = {"bod_mg_l": "C", "tn_mg_l": "E", "tp_mg_l": "C"}
CALENDAR_VARIANTS = {
    "season": (True, False),
    "weekday": (False, True),
    "season_weekday": (True, True),
}
SPECIAL_SERIES = {
    "series_2b498f0fc9d3": "중랑 제2처리장",
    "series_5570e6bfc807": "서남 처리장 구분없음",
    "series_38b56308fe33": "탄천 처리장 구분없음",
}
KEYS = ["split", "series_id", "target_date"]


def _regression_metrics(actual: pd.Series, prediction: pd.Series) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(actual, prediction),
        "rmse": mean_squared_error(actual, prediction) ** 0.5,
        "r2": r2_score(actual, prediction) if actual.nunique() > 1 else np.nan,
        "mean_bias": (prediction - actual).mean(),
    }


def _decorate_prediction(prediction: pd.DataFrame) -> pd.DataFrame:
    result = prediction.copy()
    date = pd.to_datetime(result["target_date"])
    result["season"] = np.select(
        [date.dt.month.isin([3, 4, 5]), date.dt.month.isin([6, 7, 8]), date.dt.month.isin([9, 10, 11])],
        ["봄", "여름", "가을"],
        default="겨울",
    )
    result["weekday"] = date.dt.dayofweek.map({0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"})
    result["error"] = result["prediction"] - result["actual"]
    return result


def _existing_comparators() -> pd.DataFrame:
    source = pd.read_parquet(EXISTING_PREDICTIONS)
    parts = []
    for target in TARGETS:
        for model_name, candidate_id in (
            ("baseline", f"xgboost_{BASE_CONFIG[target]}_w30"),
            ("persistence", "persistence_P_w30"),
        ):
            part = source[(source["target"] == target) & (source["candidate_id"] == candidate_id)].copy()
            part["model_name"] = model_name
            parts.append(part[[*KEYS, "facility", "plant", "actual", "prediction", "target", "model_name"]])
    return pd.concat(parts, ignore_index=True)


def _calendar_columns(frame: pd.DataFrame, variant: str) -> list[str]:
    include_season, include_weekday = CALENDAR_VARIANTS[variant]
    result = []
    if include_season:
        result.extend(["target_doy_sin", "target_doy_cos"])
    if include_weekday:
        result.extend([f"target_dow_{day}" for day in range(7)])
    if not set(result).issubset(frame.columns):
        raise ValueError(f"달력 피처 생성 실패: {variant}")
    return result


def _train_model(
    frame: pd.DataFrame,
    target: str,
    variant: str,
    *,
    weighted: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    base_config = BASE_CONFIG[target]
    include_season, include_weekday = CALENDAR_VARIANTS.get(variant, (False, False))
    model_frame = add_target_calendar_features(
        frame, include_season=include_season, include_weekday=include_weekday
    )
    train, validation, test = add_series_code(*_split(model_frame))
    base_features = select_seoul_rainfall_features(model_frame, base_config)
    calendar_features = _calendar_columns(model_frame, variant) if variant in CALENDAR_VARIANTS else []
    features = [*base_features, *calendar_features, "series_code"]
    for split_frame in (train, validation, test):
        split_frame[features] = split_frame[features].astype("float32")
    target_column = f"target_{target}"
    weights = None
    if weighted:
        weights = training_sample_weights(
            train, target_column, threshold=180.0, high_weight=2.0
        )
    started = perf_counter()
    model = fit_xgboost(
        train,
        validation,
        target_column,
        features,
        sample_weight=weights,
    )
    elapsed = perf_counter() - started
    model_name = f"{variant}{'_weighted2' if weighted else ''}"
    model_path = MODELS / f"calendar_weighted_xgboost_{target}_{model_name}_{RUN_ID}.joblib"
    payload = {
        "model": model,
        "features": features,
        "target": target_column,
        "base_config": base_config,
        "variant": variant,
        "weighted": weighted,
        "high_threshold": 180.0 if weighted else None,
        "high_weight": 2.0 if weighted else None,
        "random_seed": RANDOM_SEED,
        "training_seconds": elapsed,
        "best_iteration": int(model.best_iteration),
        "xgboost_version": xgboost.__version__,
    }
    joblib.dump(payload, model_path)
    parts = []
    for split_name, evaluation in (("validation", validation), ("test", test)):
        part = evaluation[["series_id", "facility", "plant", "target_date", "split"]].copy()
        part["actual"] = evaluation[target_column].to_numpy(dtype=float)
        part["prediction"] = model.predict(evaluation[features])
        part["target"] = target
        part["model_name"] = model_name
        parts.append(part)
    return pd.concat(parts, ignore_index=True), payload


def _overall_metrics(predictions: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    persistence = predictions[predictions["model_name"] == "persistence"]
    persistence_mae = {
        (row.split, row.target): row.mae
        for row in (
            persistence.groupby(["split", "target"], observed=True)
            .apply(lambda x: pd.Series(_regression_metrics(x["actual"], x["prediction"])), include_groups=False)
            .reset_index()
        ).itertuples()
    }
    for (split, target, model_name), group in predictions.groupby(
        ["split", "target", "model_name"], observed=True
    ):
        metrics = _regression_metrics(group["actual"], group["prediction"])
        high = group[group["actual"] >= thresholds[target]]
        normal = group[group["actual"] < thresholds[target]]
        rows.append(
            {
                "split": split,
                "target": target,
                "model_name": model_name,
                "rows": len(group),
                **metrics,
                "persistence_mae_improvement_pct": 100
                * (persistence_mae[(split, target)] - metrics["mae"])
                / persistence_mae[(split, target)],
                "high_threshold": thresholds[target],
                "high_rows": len(high),
                "high_mae": mean_absolute_error(high["actual"], high["prediction"]),
                "high_rmse": mean_squared_error(high["actual"], high["prediction"]) ** 0.5,
                "high_mean_bias": (high["prediction"] - high["actual"]).mean(),
                "high_underprediction_rate": (high["prediction"] < high["actual"]).mean(),
                "normal_rows": len(normal),
                "normal_mae": mean_absolute_error(normal["actual"], normal["prediction"]),
            }
        )
    return pd.DataFrame(rows)


def _slice_metrics(predictions: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(
        ["split", "target", "model_name", column], observed=True, dropna=False
    ):
        split, target, model_name, slice_value = keys
        rows.append(
            {
                "split": split,
                "target": target,
                "model_name": model_name,
                column: slice_value,
                "rows": len(group),
                **_regression_metrics(group["actual"], group["prediction"]),
            }
        )
    return pd.DataFrame(rows)


def _validation_selection(metrics: pd.DataFrame) -> pd.DataFrame:
    validation = metrics[metrics["split"] == "validation"].copy()
    rows = []
    for target in TARGETS:
        group = validation[validation["target"] == target]
        baseline = group[group["model_name"] == "baseline"].iloc[0]
        calendar = group[group["model_name"].isin(CALENDAR_VARIANTS)].sort_values(["mae", "rmse"])
        calendar_best = calendar.iloc[0]
        eligible = group.copy()
        if target == "bod_mg_l":
            weighted_names = [name for name in group["model_name"] if name.endswith("weighted2")]
            for name in weighted_names:
                row = group[group["model_name"] == name].iloc[0]
                eligible.loc[eligible["model_name"] == name, "replacement_goal_met"] = meets_replacement_goal(
                    baseline["mae"], baseline["high_mae"], row["mae"], row["high_mae"]
                )
            allowed = eligible[
                ~eligible["model_name"].str.endswith("weighted2")
                | eligible["replacement_goal_met"].eq(True)
            ]
        else:
            allowed = eligible[~eligible["model_name"].str.endswith("weighted2")]
        allowed = allowed[allowed["model_name"] != "persistence"]
        best = allowed.sort_values(["mae", "rmse"]).iloc[0]
        rows.append(
            {
                "target": target,
                "baseline_mae": baseline["mae"],
                "baseline_high_mae": baseline["high_mae"],
                "calendar_best": calendar_best["model_name"],
                "calendar_best_mae": calendar_best["mae"],
                "recommended_model": best["model_name"],
                "recommended_validation_mae": best["mae"],
                "recommended_high_mae": best["high_mae"],
                "selection_basis": "Validation MAE; BOD weighted는 사전 목표 충족 시에만 후보; Test 미사용",
            }
        )
    return pd.DataFrame(rows)


def _save_figures(metrics: pd.DataFrame, season: pd.DataFrame, series: pd.DataFrame) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    val = metrics[(metrics["split"] == "validation") & (metrics["model_name"] != "persistence")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        group = val[val["target"] == target].sort_values("mae")
        ax.barh(group["model_name"], group["mae"])
        ax.set_title(f"{TARGET_LABELS[target]} Validation MAE")
        ax.set_xlabel("MAE")
    fig.savefig(FIGURES / f"calendar_weighted_validation_mae_{RUN_ID}.png", dpi=160)
    plt.close(fig)

    bod = val[val["target"] == "bod_mg_l"]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(bod["mae"], bod["high_mae"])
    for row in bod.itertuples():
        ax.annotate(row.model_name, (row.mae, row.high_mae), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Validation overall MAE")
    ax.set_ylabel("Validation BOD >= 180 MAE")
    ax.set_title("BOD overall/high-concentration trade-off")
    fig.savefig(FIGURES / f"calendar_weighted_bod_tradeoff_{RUN_ID}.png", dpi=160)
    plt.close(fig)

    val_season = season[(season["split"] == "validation") & season["model_name"].isin(["baseline", "season", "weekday", "season_weekday"])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        group = val_season[val_season["target"] == target]
        for model_name, line in group.groupby("model_name", observed=True):
            ax.plot(line["season"], line["mae"], marker="o", label=model_name)
        ax.set_title(f"{TARGET_LABELS[target]} seasonal MAE")
    axes[-1].legend(fontsize=8)
    fig.savefig(FIGURES / f"calendar_weighted_season_mae_{RUN_ID}.png", dpi=160)
    plt.close(fig)

    special = series[(series["split"] == "validation") & series["series_id"].isin(SPECIAL_SERIES)].copy()
    special["series_label"] = special["series_id"].map(SPECIAL_SERIES)
    special = special[special["model_name"].isin(["baseline", "season", "weekday", "season_weekday"])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        group = special[special["target"] == target]
        pivot = group.pivot(index="series_label", columns="model_name", values="mae")
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"{TARGET_LABELS[target]} vulnerable-series MAE")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
    fig.savefig(FIGURES / f"calendar_weighted_special_series_{RUN_ID}.png", dpi=160)
    plt.close(fig)


def refresh_training_manifest() -> list[dict[str, object]]:
    """저장된 신규 모델에서 전체 피처·설정 manifest를 재생성한다."""
    rows: list[dict[str, object]] = []
    pattern = f"calendar_weighted_xgboost_*_{RUN_ID}.joblib"
    for model_path in sorted(MODELS.glob(pattern)):
        payload = joblib.load(model_path)
        model_name = payload["variant"] + ("_weighted2" if payload["weighted"] else "")
        rows.append(
            {key: value for key, value in payload.items() if key != "model"}
            | {
                "target": payload["target"].removeprefix("target_"),
                "model_name": model_name,
                "feature_count": len(payload["features"]),
                "model_file": str(model_path),
            }
        )
    if len(rows) != 11:
        raise ValueError(f"저장 신규 모델 수가 11개가 아님: {len(rows)}")
    (REPORTS / f"calendar_weighted_xgboost_training_{RUN_ID}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows


def _write_report(
    metrics: pd.DataFrame,
    season: pd.DataFrame,
    series: pd.DataFrame,
    selection: pd.DataFrame,
    thresholds: dict[str, float],
) -> None:
    validation = metrics[metrics["split"] == "validation"]
    lines = [
        "# BOD·T-N·T-P 계절·요일 및 BOD 고농도 가중 실험",
        "",
        "## 실험 계약",
        "",
        "- 목표별 기존 30일 XGBoost 입력(BOD/T-P C, T-N E)과 동일 표본·하이퍼파라미터를 사용했다.",
        "- 목표일 day-of-year sin/cos와 요일 one-hot만 추가했다. 월·계절 구분·주말 플래그는 정보 중복 때문에 제외했다.",
        "- BOD 가중치는 Train 실제값 180mg/L 이상에만 2.0을 적용했고 Validation/Test 목표값은 가중치 결정에 쓰지 않았다.",
        "- Validation으로만 선택했으며 반복 확인된 Test는 선택 후 참고 결과다.",
        "- ASOS 강수는 제외하고 기존 서울시 강우 입력만 유지했다.",
        "",
        "## Validation 전체 결과",
        "",
        "| 목표 | 모델 | MAE | RMSE | R² | 고농도 MAE | 일반 MAE | Persistence 개선율 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in validation.sort_values(["target", "mae"]).itertuples():
        lines.append(
            f"| {TARGET_LABELS[row.target]} | {row.model_name} | {row.mae:.4f} | {row.rmse:.4f} | "
            f"{row.r2:.4f} | {row.high_mae:.4f} | {row.normal_mae:.4f} | "
            f"{row.persistence_mae_improvement_pct:.2f}% |"
        )
    lines += ["", "## Validation 추천과 Test 참고 성능", "", "| 목표 | 추천 모델 | Validation MAE | Test MAE | Test RMSE | Test R² |", "|---|---|---:|---:|---:|---:|"]
    for row in selection.itertuples():
        test = metrics[
            (metrics["split"] == "test")
            & (metrics["target"] == row.target)
            & (metrics["model_name"] == row.recommended_model)
        ].iloc[0]
        lines.append(
            f"| {TARGET_LABELS[row.target]} | {row.recommended_model} | {row.recommended_validation_mae:.4f} | "
            f"{test.mae:.4f} | {test.rmse:.4f} | {test.r2:.4f} |"
        )

    lines += [
        "",
        "## 계절·요일 피처 기여",
        "",
        "| 목표 | 계절만 MAE 변화 | 요일만 MAE 변화 | 계절+요일 MAE 변화 | 해석 |",
        "|---|---:|---:|---:|---|",
    ]
    for target in TARGETS:
        group = validation[validation["target"] == target].set_index("model_name")
        baseline_mae = group.loc["baseline", "mae"]
        changes = {
            name: 100 * (group.loc[name, "mae"] / baseline_mae - 1)
            for name in ("season", "weekday", "season_weekday")
        }
        interpretation = (
            "계절+요일의 작은 추가 개선"
            if target == "bod_mg_l"
            else "요일이 개선을 주도하고 계절의 추가 정보는 제한적"
        )
        lines.append(
            f"| {TARGET_LABELS[target]} | {changes['season']:+.2f}% | {changes['weekday']:+.2f}% | "
            f"{changes['season_weekday']:+.2f}% | {interpretation} |"
        )

    focus = {"bod_mg_l": ["겨울", "봄"], "tn_mg_l": ["여름"], "tp_mg_l": ["여름"]}
    lines += [
        "",
        "요청 계절의 Validation MAE 변화:",
        "",
        "| 목표 | 계절 | 기존 MAE | 추천 모델 MAE | 변화 |",
        "|---|---|---:|---:|---:|",
    ]
    for selection_row in selection.itertuples():
        target = selection_row.target
        for season_name in focus[target]:
            base = season[
                (season["split"] == "validation")
                & (season["target"] == target)
                & (season["model_name"] == "baseline")
                & (season["season"] == season_name)
            ].iloc[0]
            candidate = season[
                (season["split"] == "validation")
                & (season["target"] == target)
                & (season["model_name"] == selection_row.recommended_model)
                & (season["season"] == season_name)
            ].iloc[0]
            lines.append(
                f"| {TARGET_LABELS[target]} | {season_name} | {base.mae:.4f} | {candidate.mae:.4f} | "
                f"{100 * (candidate.mae / base.mae - 1):+.2f}% |"
            )
    lines += [
        "",
        "T-N은 기존 E 구성에 강우 지연·누적·최대 1시간 강우가 이미 포함되어 있고 계절만 추가했을 때 전체 MAE가 악화됐다. "
        "여름 7일 누적강우와 T-N의 기존 Validation Spearman -0.506은 연관성을 보이지만, 이번 결과에서 계절 피처의 추가 정보는 제한적이고 요일 효과가 개선을 주도했다.",
    ]

    bod_baseline = validation[(validation["target"] == "bod_mg_l") & (validation["model_name"] == "baseline")].iloc[0]
    lines += ["", "## BOD 가중 학습 사전 목표", "", "| 모델 | 전체 MAE 변화 | 고농도 MAE 변화 | 목표 충족 |", "|---|---:|---:|---|"]
    for row in validation[(validation["target"] == "bod_mg_l") & validation["model_name"].str.endswith("weighted2")].itertuples():
        overall_change = 100 * (row.mae / bod_baseline.mae - 1)
        high_change = 100 * (row.high_mae / bod_baseline.high_mae - 1)
        met = meets_replacement_goal(bod_baseline.mae, bod_baseline.high_mae, row.mae, row.high_mae)
        lines.append(f"| {row.model_name} | {overall_change:+.2f}% | {high_change:+.2f}% | {'예' if met else '아니오'} |")

    lines += ["", "## 취약 계통 변화", "", "| 목표 | 계통 | 모델 | Validation MAE | 기존 대비 변화 |", "|---|---|---|---:|---:|"]
    val_series = series[(series["split"] == "validation") & series["series_id"].isin(SPECIAL_SERIES)]
    for (target, series_id), group in val_series.groupby(["target", "series_id"], observed=True):
        baseline = group[group["model_name"] == "baseline"]
        if baseline.empty:
            continue
        base_mae = baseline.iloc[0]["mae"]
        candidates = group[group["model_name"].isin(["baseline", "season", "weekday", "season_weekday"])]
        best = candidates.sort_values("mae").iloc[0]
        lines.append(
            f"| {TARGET_LABELS[target]} | {SPECIAL_SERIES[series_id]} | {best.model_name} | {best.mae:.4f} | "
            f"{100 * (best.mae / base_mae - 1):+.2f}% |"
        )

    lines += [
        "",
        "## T-N·T-P 고농도 진단",
        "",
        f"- Train 90백분위 기준은 T-N {thresholds['tn_mg_l']:.3f}mg/L, T-P {thresholds['tp_mg_l']:.3f}mg/L이다.",
    ]
    for target in ("tn_mg_l", "tp_mg_l"):
        recommended = selection.loc[selection["target"] == target, "recommended_model"].iloc[0]
        row = validation[(validation["target"] == target) & (validation["model_name"] == recommended)].iloc[0]
        lines.append(
            f"- {TARGET_LABELS[target]} 추천 모델의 고농도 MAE {row.high_mae:.3f}, 평균 편향 "
            f"{row.high_mean_bias:.3f}, 과소예측률 {row.high_underprediction_rate:.1%}이다."
        )
    lines += [
        "- T-N은 높은 과소예측률과 Persistence보다 큰 고농도 MAE 때문에 후속 가중 실험 가치가 있다. T-P는 고농도 MAE가 Persistence보다 낮고 달력 개선폭도 작아 우선순위가 낮다.",
        "- 두 항목은 이번 범위에서는 가중 모델을 학습하지 않았다.",
        "- 고농도 여부는 사후 평가 라벨이며 예측 입력으로 사용하지 않았다.",
        "",
        "## 해석상 한계",
        "",
        "- 서울시 강우 관측소 매핑은 잠정이며 목표일 실제 기상은 사용하지 않았다.",
        "- Test는 반복 확인된 참고 구간으로 독립 최종 홀드아웃이 아니다.",
        "- 달력 피처 개선은 계절·요일과 수질의 연관을 뜻하며 인과효과를 입증하지 않는다.",
    ]
    (REPORTS / f"calendar_weighted_xgboost_report_{RUN_ID}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    for path in (REPORTS, MODELS, PREDICTIONS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(PROCESSED)
    forbidden = [column for column in frame if any(token in column.lower() for token in ("day_of", "weekday", "weekend", "month", "season", "calendar"))]
    if forbidden:
        raise ValueError(f"기존 달력 피처 중복: {forbidden}")
    train_frame = frame[frame["split"] == "train"]
    thresholds = {
        "bod_mg_l": 180.0,
        "tn_mg_l": float(train_frame["target_tn_mg_l"].quantile(0.9)),
        "tp_mg_l": float(train_frame["target_tp_mg_l"].quantile(0.9)),
    }

    existing = _existing_comparators()
    prediction_parts = [existing]
    training_rows = []
    for target in TARGETS:
        for variant in CALENDAR_VARIANTS:
            predictions, payload = _train_model(frame, target, variant, weighted=False)
            baseline = existing[(existing["target"] == target) & (existing["model_name"] == "baseline")]
            assert_common_evaluation_rows(baseline, predictions)
            prediction_parts.append(predictions)
            training_rows.append(
                {key: value for key, value in payload.items() if key != "model"}
                | {"target": target, "model_name": variant, "feature_count": len(payload["features"])}
            )

    calendar_predictions = pd.concat(prediction_parts, ignore_index=True)
    calendar_decorated = _decorate_prediction(calendar_predictions)
    calendar_metrics = _overall_metrics(calendar_decorated, thresholds)
    bod_validation = calendar_metrics[
        (calendar_metrics["split"] == "validation")
        & (calendar_metrics["target"] == "bod_mg_l")
        & (calendar_metrics["model_name"].isin(CALENDAR_VARIANTS))
    ]
    best_bod_calendar = bod_validation.sort_values(["mae", "rmse"]).iloc[0]["model_name"]

    for variant in ("base", best_bod_calendar):
        predictions, payload = _train_model(frame, "bod_mg_l", variant, weighted=True)
        baseline = existing[(existing["target"] == "bod_mg_l") & (existing["model_name"] == "baseline")]
        assert_common_evaluation_rows(baseline, predictions)
        prediction_parts.append(predictions)
        training_rows.append(
            {key: value for key, value in payload.items() if key != "model"}
            | {
                "target": "bod_mg_l",
                "model_name": f"{variant}_weighted2",
                "feature_count": len(payload["features"]),
            }
        )

    all_predictions = _decorate_prediction(pd.concat(prediction_parts, ignore_index=True))
    if all_predictions.duplicated(KEYS + ["target", "model_name"]).any():
        raise ValueError("저장 예측 중복키")
    metrics = _overall_metrics(all_predictions, thresholds)
    season_metrics = _slice_metrics(all_predictions, "season")
    weekday_metrics = _slice_metrics(all_predictions, "weekday")
    series_metrics = _slice_metrics(all_predictions, "series_id")
    names = all_predictions[["series_id", "facility", "plant"]].drop_duplicates()
    series_metrics = series_metrics.merge(names, on="series_id", validate="many_to_one")
    selection = _validation_selection(metrics)

    all_predictions.to_parquet(PREDICTIONS / f"calendar_weighted_xgboost_predictions_{RUN_ID}.parquet", index=False)
    metrics.to_csv(REPORTS / f"calendar_weighted_xgboost_metrics_{RUN_ID}.csv", index=False, encoding="utf-8-sig")
    season_metrics.to_csv(REPORTS / f"calendar_weighted_xgboost_season_metrics_{RUN_ID}.csv", index=False, encoding="utf-8-sig")
    weekday_metrics.to_csv(REPORTS / f"calendar_weighted_xgboost_weekday_metrics_{RUN_ID}.csv", index=False, encoding="utf-8-sig")
    series_metrics.to_csv(REPORTS / f"calendar_weighted_xgboost_series_metrics_{RUN_ID}.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(REPORTS / f"calendar_weighted_xgboost_selection_{RUN_ID}.csv", index=False, encoding="utf-8-sig")
    refresh_training_manifest()
    config = {
        "random_seed": RANDOM_SEED,
        "window_days": 30,
        "base_config": BASE_CONFIG,
        "calendar_variants": {key: {"season": value[0], "weekday": value[1]} for key, value in CALENDAR_VARIANTS.items()},
        "calendar_features": {"season": ["target_doy_sin", "target_doy_cos"], "weekday": [f"target_dow_{day}" for day in range(7)]},
        "excluded_redundant_features": ["month", "season_category", "is_weekend"],
        "high_thresholds": thresholds,
        "bod_high_weight": 2.0,
        "selection_basis": "Validation MAE; BOD weighted는 사전 목표 충족 시만 후보; Test 미사용",
        "test_interpretation": "반복 확인된 참고 구간; 독립 최종 홀드아웃 아님",
    }
    (REPORTS / f"calendar_weighted_xgboost_config_{RUN_ID}.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_figures(metrics, season_metrics, series_metrics)
    _write_report(metrics, season_metrics, series_metrics, selection, thresholds)
    print({"new_models": len(training_rows), "prediction_rows": len(all_predictions), "thresholds": thresholds, "selection": selection[["target", "recommended_model"]].to_dict("records")})


if __name__ == "__main__":
    main()
