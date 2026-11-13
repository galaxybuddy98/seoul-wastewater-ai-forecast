"""서울시 강우량 C·D·E LightGBM을 학습하고 기존 A·B와 공정 비교한다."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.train_lightgbm_baseline import _assert_same_rows, _metric_rows, _split
from src.models.lightgbm_baseline import (
    RANDOM_SEED,
    add_series_code,
    fit_lightgbm,
    regression_metrics,
)
from src.models.seoul_rainfall_experiment import (
    EVALUATION_KEYS,
    attach_input_day_rainfall,
    common_prediction_rows,
    load_model_checkpoint,
    select_seoul_rainfall_features,
)

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
RESULTS = Path("results")
MODELS = RESULTS / "models"
PREDICTIONS = RESULTS / "predictions"
FIGURES = RESULTS / "figures"
TARGETS = ["bod_mg_l", "tn_mg_l", "tp_mg_l"]
WINDOWS = [30, 7, 14]
NEW_VARIANTS = ["C", "D", "E"]
COMPARE_VARIANTS = ["P", "A", "B", "C", "D", "E"]
BASE_VARIANT_MAP = {
    "quality": "A",
    "quality_temperature": "B",
}


def _model_path(variant: str, window: int, target: str) -> Path:
    return MODELS / f"seoul_rainfall_lightgbm_{variant}_w{window}_{target}.joblib"


def _train_one(variant: str, window: int, target: str) -> None:
    """한 모델만 독립 프로세스에서 float32 입력으로 학습해 메모리 피크를 제한한다."""
    source = PROCESSED / f"tabular_seoul_rainfall_w{window}.parquet"
    schema_columns = pq.ParquetFile(source).schema_arrow.names
    features = select_seoul_rainfall_features(pd.DataFrame(columns=schema_columns), variant)
    target_column = f"target_{target}"
    columns = ["series_id", "split", target_column, *features]
    frame = pd.read_parquet(source, columns=columns)
    for column in features:
        frame[column] = frame[column].astype("float32")
    train, validation, _ = add_series_code(*_split(frame))
    features = [*features, "series_code"]
    model = fit_lightgbm(train, validation, target_column, features)
    joblib.dump(
        {
            "model": model,
            "features": features,
            "target": target_column,
            "variant": variant,
            "window_days": window,
            "random_seed": RANDOM_SEED,
            "numeric_input_dtype": "float32",
        },
        _model_path(variant, window, target),
    )


def _ensure_model_checkpoints() -> None:
    """미완료 모델만 독립 프로세스로 학습한다."""
    for window in WINDOWS:
        schema_columns = pq.ParquetFile(
            PROCESSED / f"tabular_seoul_rainfall_w{window}.parquet"
        ).schema_arrow.names
        for variant in NEW_VARIANTS:
            features = [
                *select_seoul_rainfall_features(pd.DataFrame(columns=schema_columns), variant),
                "series_code",
            ]
            for target in TARGETS:
                path = _model_path(variant, window, target)
                if path.exists():
                    load_model_checkpoint(path, features)
                    continue
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.train_seoul_rainfall_lightgbm",
                        "--train-one",
                        "--variant",
                        variant,
                        "--window",
                        str(window),
                        "--target",
                        target,
                    ],
                    check=True,
                )


def _existing_predictions() -> pd.DataFrame:
    frame = pd.read_parquet(PREDICTIONS / "predictions.parquet")
    standard = frame.loc[
        frame["experiment"].eq("standard")
        & frame["model_type"].eq("lightgbm")
        & frame["variant"].isin(BASE_VARIANT_MAP)
    ].copy()
    standard["variant"] = standard["variant"].map(BASE_VARIANT_MAP)
    persistence = frame.loc[
        frame["experiment"].eq("persistence")
        & frame["model_type"].eq("persistence")
    ].copy()
    persistence["variant"] = "P"
    return pd.concat([persistence, standard], ignore_index=True)


def _metric_table(predictions: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = []
    group_columns = ["split", "target", "window_days", "variant"]
    for keys, group in predictions.groupby(group_columns, dropna=False):
        metrics = regression_metrics(group["actual"].to_numpy(), group["prediction"].to_numpy())
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "comparison_scope": scope,
                "eval_rows": len(group),
                **metrics,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    persistence = result.loc[result["variant"].eq("P"), [
        "split", "target", "window_days", "mae"
    ]].rename(columns={"mae": "persistence_mae"})
    result = result.merge(
        persistence,
        on=["split", "target", "window_days"],
        how="left",
        validate="many_to_one",
    )
    result["persistence_mae_improvement_pct"] = (
        (result["persistence_mae"] - result["mae"]) / result["persistence_mae"] * 100
    )
    return result


def _common_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    within_parts = []
    for (split, target, window), group in predictions.groupby(
        ["split", "target", "window_days"], dropna=False
    ):
        common = common_prediction_rows(group, COMPARE_VARIANTS)
        within_parts.append(common)
    within = pd.concat(within_parts, ignore_index=True)
    within_metrics = _metric_table(within, "동일 윈도 내 P·A·B·C·D·E 공통 평가행")

    key_columns = ["series_id", "target_date", "target", "split"]
    counts = within.groupby(key_columns, dropna=False)["window_days"].nunique()
    across_keys = counts.loc[counts.eq(len(WINDOWS))].index.to_frame(index=False)
    across = within.merge(across_keys, on=key_columns, how="inner", validate="many_to_one")
    across_metrics = _metric_table(across, "7·14·30일 전체 공통 평가행")
    return within_metrics, across_metrics


def _save_plot(predictions: pd.DataFrame, target: str, selected: dict[str, object]) -> None:
    chosen = predictions.loc[
        predictions["target"].eq(target)
        & predictions["variant"].eq(selected["rainfall_variant"])
        & predictions["window_days"].eq(selected["rainfall_window_days"])
        & predictions["split"].eq("test")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(chosen["actual"], chosen["prediction"], s=8, alpha=0.3)
    lower = min(chosen["actual"].min(), chosen["prediction"].min())
    upper = max(chosen["actual"].max(), chosen["prediction"].max())
    axes[0].plot([lower, upper], [lower, upper], color="black", linewidth=1)
    axes[0].set(title=f"{target} Test: actual vs prediction", xlabel="Actual", ylabel="Prediction")
    series_id = chosen["series_id"].value_counts().index[0]
    timeline = chosen.loc[chosen["series_id"].eq(series_id)].sort_values("target_date")
    axes[1].plot(timeline["target_date"], timeline["actual"], label="actual", linewidth=1)
    axes[1].plot(timeline["target_date"], timeline["prediction"], label="prediction", linewidth=1)
    axes[1].set_title(f"Largest evaluation series: {series_id}")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"seoul_rainfall_lightgbm_{target}_selected_test.png", dpi=150)
    plt.close(fig)


def _environmental_analysis(
    predictions: pd.DataFrame,
    selections: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rainfall = pd.read_parquet(PROCESSED / "seoul_rainfall_center_daily.parquet")
    rainfall["date"] = pd.to_datetime(rainfall["date"])
    train_rain = rainfall.loc[rainfall["date"].le("2022-12-31"), "daily_rainfall_mm"].dropna()
    positive = train_rain.loc[train_rain.gt(0)]
    heavy_threshold = float(positive.quantile(0.95)) if not positive.empty else float("nan")
    rows = []
    detail_parts = []
    for target, selected in selections.items():
        chosen = predictions.loc[
            predictions["target"].eq(target)
            & predictions["variant"].eq(selected["rainfall_variant"])
            & predictions["window_days"].eq(selected["rainfall_window_days"])
        ].copy()
        enriched = attach_input_day_rainfall(chosen, rainfall)
        enriched["error"] = enriched["prediction"] - enriched["actual"]
        enriched["absolute_error"] = enriched["error"].abs()
        enriched["rain_class"] = np.select(
            [
                enriched["daily_rainfall_mm"].isna(),
                enriched["daily_rainfall_mm"].eq(0),
                enriched["daily_rainfall_mm"].ge(heavy_threshold),
            ],
            ["미확인", "무강우", "집중강우"],
            default="강우",
        )
        for (split, rain_class), group in enriched.groupby(["split", "rain_class"]):
            metric = regression_metrics(group["actual"].to_numpy(), group["prediction"].to_numpy())
            rows.append({
                "target": target,
                "split": split,
                "analysis": "당일 강우 상태",
                "group": rain_class,
                "rows": len(group),
                "mean_error": float(group["error"].mean()),
                **metric,
                "train_positive_rain_p95_mm": heavy_threshold,
            })
        for lag in (1, 2, 3):
            column = f"rainfall_lag_{lag}d_mm"
            if column not in enriched:
                continue
            mask = enriched[column].gt(0)
            for split, group in enriched.loc[mask].groupby("split"):
                metric = regression_metrics(group["actual"].to_numpy(), group["prediction"].to_numpy())
                rows.append({
                    "target": target,
                    "split": split,
                    "analysis": "강우 후 지연",
                    "group": f"{lag}일 전 강우",
                    "rows": len(group),
                    "mean_error": float(group["error"].mean()),
                    **metric,
                    "train_positive_rain_p95_mm": heavy_threshold,
                })
        if target == "bod_mg_l":
            source = pd.read_parquet(
                PROCESSED / f"tabular_seoul_rainfall_w{selected['rainfall_window_days']}.parquet",
                columns=["split", "target_bod_mg_l"],
            )
            high_threshold = float(
                source.loc[source["split"].eq("train"), "target_bod_mg_l"].quantile(0.90)
            )
            for split, group in enriched.loc[enriched["actual"].ge(high_threshold)].groupby("split"):
                metric = regression_metrics(group["actual"].to_numpy(), group["prediction"].to_numpy())
                rows.append({
                    "target": target,
                    "split": split,
                    "analysis": "BOD 고농도",
                    "group": f"Train 90백분위 이상({high_threshold:.3f})",
                    "rows": len(group),
                    "mean_error": float(group["error"].mean()),
                    "underprediction_rate": float(group["error"].lt(0).mean()),
                    **metric,
                    "train_positive_rain_p95_mm": heavy_threshold,
                })
        detail_parts.append(enriched)
    return pd.DataFrame(rows), pd.concat(detail_parts, ignore_index=True)


def _center_coverage() -> pd.DataFrame:
    rainfall = pd.read_parquet(PROCESSED / "seoul_rainfall_center_daily.parquet")
    rainfall["date"] = pd.to_datetime(rainfall["date"])
    rainfall = rainfall.loc[rainfall["date"].between("2016-01-01", "2026-09-30")].copy()
    rainfall["split"] = np.select(
        [rainfall["date"].le("2022-12-31"), rainfall["date"].le("2024-12-31")],
        ["train", "validation"],
        default="test",
    )
    rows = []
    fields = [
        "daily_rainfall_mm",
        "rainfall_sum_3d_mm",
        "rainfall_sum_7d_mm",
        "max_1h_rainfall_mm",
    ]
    for (facility, split), group in rainfall.groupby(["facility", "split"]):
        row = {
            "facility": facility,
            "split": split,
            "calendar_days": len(group),
            "start_date": group["date"].min(),
            "end_date": group["date"].max(),
            "mapping_status": group["selection_status"].dropna().iloc[0],
        }
        for field in fields:
            row[f"{field}_available_rate"] = float(group[field].notna().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _write_report(
    config: dict[str, object],
    comparison: pd.DataFrame,
    importance: pd.DataFrame,
    environmental: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    scope = "동일 윈도 내 P·A·B·C·D·E 공통 평가행"
    validation = comparison.loc[
        comparison["comparison_scope"].eq(scope)
        & comparison["split"].eq("validation")
        & comparison["window_days"].eq(30)
    ][["target", "variant", "eval_rows", "mae", "rmse", "r2", "persistence_mae_improvement_pct"]]
    selected_rows = []
    for target, selected in config["selections"].items():
        selected_rows.append(
            comparison.loc[
                comparison["comparison_scope"].eq(scope)
                & comparison["split"].eq("test")
                & comparison["target"].eq(target)
                & comparison["variant"].eq(selected["selected_variant"])
                & comparison["window_days"].eq(selected["selected_window_days"])
            ].iloc[0]
        )
    selected_test = pd.DataFrame(selected_rows)[
        ["target", "window_days", "variant", "eval_rows", "mae", "rmse", "r2", "persistence_mae_improvement_pct"]
    ]
    rainfall_share_rows = []
    for target, selected in config["selections"].items():
        chosen = importance.loc[
            importance["target"].eq(target)
            & importance["variant"].eq(selected["rainfall_variant"])
            & importance["window_days"].eq(selected["rainfall_window_days"])
        ]
        rainfall_share_rows.append({
            "target": target,
            "variant": selected["rainfall_variant"],
            "window_days": selected["rainfall_window_days"],
            "rainfall_gain_share": float(
                chosen.loc[chosen["feature_group"].eq("서울시 강우"), "gain_share"].sum()
            ),
        })
    rainfall_share = pd.DataFrame(rainfall_share_rows)
    test_environment = environmental.loc[environmental["split"].eq("test")]
    report = f"""# 서울시 강우량 기반 LightGBM 실험 결과

## 실험 범위

- 기존 Persistence·A(과거 수질)·B(과거 수질+ASOS 기온) 예측은 재사용했다.
- 신규 C(일강우량)·D(강우 lag·3/7일 누적)·E(일 최대 1시간 강우 추가)는 3개 목표×3개 윈도×3개 구성, 총 27개 모델을 학습했다.
- ASOS 강수량은 입력에서 제외했고 ASOS 기온만 유지했다.
- 센터별 서울시 관측소 매핑은 `provisional`이며 공간적 대표성이 공식 확정된 것은 아니다.
- 선택은 Validation MAE만 사용했다. Test는 이미 반복 확인된 기간이므로 완전히 미사용된 독립 홀드아웃으로 해석하지 않는다.

## 강우자료 확보율

{coverage.to_markdown(index=False)}

2026년 자료는 2026-09-30까지 반영되어 있다. 강우 NaN은 수질 표본 삭제 조건으로 사용하지 않고 LightGBM에 그대로 전달했다. 당일(t) 강우량은 연구용 데이터가 예측 시점에 제공된다는 가정이며, 실제 운영 공개 지연은 별도 검증이 필요하다.

## 동일 평가 표본 성능

### Validation, 30일 윈도

{validation.sort_values(["target", "variant"]).to_markdown(index=False)}

### Validation으로 선택한 설정의 Test 참고 성능

{selected_test.to_markdown(index=False)}

선택 결과는 BOD=C/30일, T-N=E/30일, T-P=C/30일이다. 같은 30일 Validation 표본에서 B 대비 MAE는 각각 약 1.32%, 4.01%, 2.51% 개선됐다. 기온 추가(B-A)는 BOD에서 소폭 악화됐고 T-N·T-P에서는 소폭 개선됐다. D·E의 추가 파생변수는 T-N에서만 C보다 더 개선됐으며 BOD·T-P에서는 일강우량 C가 더 나았다.

서로 다른 7·14·30일 윈도의 완전 공통 평가행 결과는 `reports/seoul_rainfall_lightgbm_common_metrics.csv`의 `comparison_scope=7·14·30일 전체 공통 평가행`에 별도 보존했다.

## 변수 중요도

{rainfall_share.to_markdown(index=False)}

위 값은 선택된 강우 모델의 LightGBM Gain 합계 중 서울시 강우 피처 비중이다. 상관·예측 기여를 나타낼 뿐 강우의 인과효과를 증명하지 않는다. D·E는 입력창의 일강우 이력과 lag·누적 변수가 일부 중복되므로 중요도가 여러 피처로 분산될 수 있다.

## 환경공학적 오차 분석

{test_environment[["target", "analysis", "group", "rows", "mae", "rmse", "r2", "mean_error", "underprediction_rate"]].to_markdown(index=False)}

- 집중강우 표본은 적어 지표 변동성이 크다. 특히 T-N·T-P의 집중강우 Test 오차가 커졌지만 이를 일반적 인과효과로 단정할 수 없다.
- BOD Train 90백분위 이상 구간은 Test에서 과소예측률이 매우 높아 급등 대응이 핵심 잔여 문제다.
- 계통별 오차는 `reports/seoul_rainfall_lightgbm_metrics_by_series.csv`에 보존했다. 관측소 한 곳이 하수처리구역 전체를 대표한다고 해석하지 않는다.

## 누수·재현성 검증

- 분할은 `target_date` 기준 Train(2016~2022), Validation(2023~2024), Test(2025~2026-09-30)를 유지했다.
- 표형 피처는 각 계통 내부에서 생성된 입력일 t 이하의 lag만 선택하며, 목표일 t+1 강우량은 사용하지 않는다.
- A·B·C·D·E 및 Persistence는 `target_date × series_id × target` 공통 평가행으로 재계산했다.
- 랜덤 시드는 42이며 기존 LightGBM 고정 설정과 Validation 조기종료를 유지했다.
- 모델, 피처 목록, 예측값, 지표, 중요도와 설정은 기존 파일과 구분해 저장했다.

## 다음 모델 비교 권고

XGBoost·LSTM 비교 시 BOD·T-P는 C(수질+기온+일강우량), T-N은 E(C+lag·누적+최대 1시간 강우량)를 우선 후보로 사용하되, 동일 평가 키와 Validation 중심 선택을 유지한다. 모델별 입력 차이 자체의 효과를 확인하려면 30일 공통 표본 비교를 기본으로 한다.
"""
    (REPORTS / "seoul_rainfall_lightgbm_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    for path in (MODELS, PREDICTIONS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
    _ensure_model_checkpoints()
    metric_rows: list[dict[str, object]] = []
    series_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    importance_rows: list[dict[str, object]] = []
    feature_manifest: dict[str, list[str]] = {}
    availability_rows: list[dict[str, object]] = []

    for window in WINDOWS:
        frame = pd.read_parquet(PROCESSED / f"tabular_seoul_rainfall_w{window}.parquet")
        reference = pd.read_parquet(PROCESSED / f"tabular_quality_temperature_w{window}.parquet")
        for split_name in ("train", "validation", "test"):
            _assert_same_rows(
                reference.loc[reference["split"].eq(split_name)],
                frame.loc[frame["split"].eq(split_name)],
                f"서울시 강우 w{window} {split_name}",
            )
        train, validation, test = add_series_code(*_split(frame))
        for variant in NEW_VARIANTS:
            features = [*select_seoul_rainfall_features(train, variant), "series_code"]
            feature_manifest[f"{variant}_w{window}"] = features
            rain_features = [column for column in features if column.startswith("seoul_rainfall_")]
            for split_name, split_frame in (("train", train), ("validation", validation), ("test", test)):
                availability_rows.append({
                    "window_days": window,
                    "variant": variant,
                    "split": split_name,
                    "rows": len(split_frame),
                    "rain_feature_cells": len(split_frame) * len(rain_features),
                    "rain_feature_available_rate": float(split_frame[rain_features].notna().to_numpy().mean()),
                })
            for target in TARGETS:
                target_column = f"target_{target}"
                model = load_model_checkpoint(_model_path(variant, window, target), features)
                gains = model.booster_.feature_importance(importance_type="gain")
                for feature, gain in zip(features, gains):
                    importance_rows.append({
                        "variant": variant,
                        "window_days": window,
                        "target": target,
                        "feature": feature,
                        "feature_group": "서울시 강우" if feature.startswith("seoul_rainfall_") else "수질·기온·계통",
                        "gain": float(gain),
                    })
                for split_name, evaluation in (("validation", validation), ("test", test)):
                    metadata = {
                        "experiment": "seoul_rainfall_lightgbm",
                        "model_type": "lightgbm",
                        "variant": variant,
                        "window_days": window,
                        "target": target,
                        "split": split_name,
                        "best_iteration": int(model.best_iteration_ or model.n_estimators),
                    }
                    overall, by_series, prediction = _metric_rows(
                        evaluation,
                        model.predict(evaluation[features]),
                        target_column,
                        metadata,
                    )
                    metric_rows.append(overall)
                    series_rows.extend(by_series)
                    prediction_parts.append(prediction)
                del model
                gc.collect()
        del frame, reference, train, validation, test
        gc.collect()

    new_predictions = pd.concat(prediction_parts, ignore_index=True)
    existing_predictions = _existing_predictions()
    all_predictions = pd.concat([existing_predictions, new_predictions], ignore_index=True)
    within_metrics, across_metrics = _common_metrics(all_predictions)
    comparison_metrics = pd.concat([within_metrics, across_metrics], ignore_index=True)

    validation = comparison_metrics.loc[
        comparison_metrics["comparison_scope"].eq("동일 윈도 내 P·A·B·C·D·E 공통 평가행")
        & comparison_metrics["split"].eq("validation")
        & comparison_metrics["variant"].ne("P")
    ]
    selections: dict[str, dict[str, object]] = {}
    for target, group in validation.groupby("target"):
        best = group.sort_values(["mae", "rmse"]).iloc[0]
        rainfall_best = group.loc[group["variant"].isin(NEW_VARIANTS)].sort_values(["mae", "rmse"]).iloc[0]
        selections[target] = {
            "selected_variant": str(best["variant"]),
            "selected_window_days": int(best["window_days"]),
            "validation_mae": float(best["mae"]),
            "rainfall_variant": str(rainfall_best["variant"]),
            "rainfall_window_days": int(rainfall_best["window_days"]),
            "rainfall_validation_mae": float(rainfall_best["mae"]),
            "selection_basis": "Validation MAE; Test는 선택에 미사용",
        }

    environmental, environmental_detail = _environmental_analysis(all_predictions, selections)
    new_metrics = pd.DataFrame(metric_rows)
    new_series = pd.DataFrame(series_rows)
    importance = pd.DataFrame(importance_rows)
    importance["gain_share"] = importance.groupby(["variant", "window_days", "target"])["gain"].transform(
        lambda values: values / values.sum() if values.sum() else 0.0
    )

    new_predictions.to_parquet(PREDICTIONS / "seoul_rainfall_lightgbm_predictions.parquet", index=False)
    new_metrics.to_csv(REPORTS / "seoul_rainfall_lightgbm_metrics_new.csv", index=False, encoding="utf-8-sig")
    new_series.to_csv(REPORTS / "seoul_rainfall_lightgbm_metrics_by_series.csv", index=False, encoding="utf-8-sig")
    comparison_metrics.to_csv(REPORTS / "seoul_rainfall_lightgbm_common_metrics.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(REPORTS / "seoul_rainfall_lightgbm_feature_importance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(availability_rows).to_csv(
        REPORTS / "seoul_rainfall_lightgbm_feature_availability.csv", index=False, encoding="utf-8-sig"
    )
    environmental.to_csv(REPORTS / "seoul_rainfall_lightgbm_error_analysis.csv", index=False, encoding="utf-8-sig")
    environmental_detail.to_parquet(
        PREDICTIONS / "seoul_rainfall_lightgbm_error_analysis_rows.parquet", index=False
    )
    config = {
        "random_seed": RANDOM_SEED,
        "new_model_count": len(NEW_VARIANTS) * len(WINDOWS) * len(TARGETS),
        "new_variants": NEW_VARIANTS,
        "windows": WINDOWS,
        "targets": TARGETS,
        "asos_precipitation_used": False,
        "asos_temperature_used": True,
        "center_station_mapping_status": "provisional",
        "selection_metric": "Validation MAE",
        "test_interpretation": "반복 확인된 참고 구간이며 완전히 미사용된 독립 홀드아웃이 아님",
        "same_day_rain_assumption": "입력 시점 t의 자료 가용 가정; 실제 운영 공개 지연 별도 확인 필요",
        "features": feature_manifest,
        "selections": selections,
    }
    (REPORTS / "seoul_rainfall_lightgbm_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for target, selected in selections.items():
        _save_plot(all_predictions, target, selected)
    coverage = _center_coverage()
    coverage.to_csv(
        REPORTS / "seoul_rainfall_lightgbm_center_coverage.csv", index=False, encoding="utf-8-sig"
    )
    _write_report(config, comparison_metrics, importance, environmental, coverage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-one", action="store_true")
    parser.add_argument("--variant", choices=NEW_VARIANTS)
    parser.add_argument("--window", type=int, choices=WINDOWS)
    parser.add_argument("--target", choices=TARGETS)
    args = parser.parse_args()
    if args.train_one:
        if args.variant is None or args.window is None or args.target is None:
            parser.error("--train-one에는 --variant, --window, --target이 필요합니다.")
        _train_one(args.variant, args.window, args.target)
    else:
        main()
