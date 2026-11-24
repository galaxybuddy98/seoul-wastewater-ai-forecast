"""기존 30일 C/E 데이터에 SS를 추가한 3종 알고리즘 비교 실험."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.shared.comparative_models import (
    encode_series_ids,
    fit_sequence_preprocessor,
    tabular_to_sequence,
)
from src.models.lightgbm.lightgbm_baseline import add_series_code, fit_lightgbm
from src.models.lstm.lstm_forecaster import LSTMConfig, fit_lstm, predict_lstm
from src.models.shared.seoul_rainfall_experiment import select_seoul_rainfall_features
from src.models.shared.ss_experiment import attach_ss_lags, ss_feature_columns
from src.models.xgboost.xgboost_forecaster import fit_xgboost

from scripts._paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
PROCESSED = ROOT / "data/processed"
RESULTS = ROOT / "results"
MODELS = RESULTS / "models/ss_experiment_2026-11-13"
PREDICTIONS = RESULTS / "predictions/ss_experiment_predictions_2026-11-13.parquet"
REPORTS = ROOT / "reports"
FIGURES = RESULTS / "figures/ss_experiment_2026-11-13"
WINDOW = 30
TARGET_VARIANTS = {"bod_mg_l": "C", "tn_mg_l": "E", "tp_mg_l": "C"}
ALGORITHMS = ("lightgbm", "xgboost", "lstm")
KEYS = ["series_id", "target_date", "split", "target"]


def metrics(actual: pd.Series | np.ndarray, prediction: pd.Series | np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(mean_squared_error(actual, prediction) ** 0.5),
        "r2": float(r2_score(actual, prediction)),
    }


def build_augmented_frame() -> pd.DataFrame:
    source = PROCESSED / "tabular_seoul_rainfall_w30.parquet"
    output = PROCESSED / "tabular_seoul_rainfall_ss_w30.parquet"
    base = pd.read_parquet(source)
    observations = pd.read_parquet(
        PROCESSED / "observation_series_daily_seoul_rainfall.parquet",
        columns=["series_id", "date", "ss_mg_l"],
    )
    augmented = attach_ss_lags(base, observations, WINDOW)
    if len(augmented) != len(base) or augmented.duplicated(["series_id", "target_date"]).any():
        raise RuntimeError("SS 결합 후 행 수 또는 평가 키가 변경되었습니다")
    augmented.to_parquet(output, index=False)
    return augmented


def quality_tables(observations: pd.DataFrame) -> None:
    frame = observations.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["split"] = np.select(
        [frame.date.le("2022-12-31"), frame.date.le("2024-12-31")],
        ["train", "validation"], default="test",
    )
    coverage = frame.groupby(["series_id", "facility", "plant", "split"], dropna=False).agg(
        rows=("date", "size"), ss_available=("ss_mg_l", "count"),
        ss_zero=("ss_mg_l", lambda x: int(x.eq(0).sum())),
        ss_negative=("ss_mg_l", lambda x: int(x.lt(0).sum())),
        start_date=("date", "min"), end_date=("date", "max"),
    ).reset_index()
    coverage["ss_missing_rate"] = 1 - coverage.ss_available / coverage.rows
    coverage.to_csv(REPORTS / "ss_quality_coverage_2026-11-13.csv", index=False, encoding="utf-8-sig")

    correlations = []
    for series_id, group in frame.groupby("series_id"):
        for target in TARGET_VARIANTS:
            valid = group[["ss_mg_l", target]].dropna()
            correlations.append({
                "series_id": series_id, "target": target, "n": len(valid),
                "pearson": valid.ss_mg_l.corr(valid[target], method="pearson"),
                "spearman": valid.ss_mg_l.corr(valid[target], method="spearman"),
            })
    pd.DataFrame(correlations).to_csv(
        REPORTS / "ss_quality_correlations_2026-11-13.csv", index=False, encoding="utf-8-sig"
    )
    diagnostics = []
    for series_id, group in frame.groupby("series_id"):
        values = group.ss_mg_l.dropna()
        q1, q3 = values.quantile([.25, .75]); iqr = q3 - q1
        median = values.median(); mad = (values - median).abs().median()
        iqr_outlier = values.lt(q1 - 1.5 * iqr) | values.gt(q3 + 1.5 * iqr)
        mad_outlier = ((values - median).abs() / (1.4826 * mad)).gt(3.5) if mad > 0 else pd.Series(False, index=values.index)
        daily = group.set_index("date").ss_mg_l.reindex(pd.date_range(group.date.min(), group.date.max()))
        missing = daily.isna(); run_id = missing.ne(missing.shift()).cumsum()
        max_missing_run = int(missing.groupby(run_id).sum().max()) if len(missing) else 0
        diagnostics.append({"series_id": series_id, "rows": len(group), "minimum": values.min(),
                            "maximum": values.max(), "zero_count": int(values.eq(0).sum()),
                            "iqr_outlier_count": int(iqr_outlier.sum()), "mad_outlier_count": int(mad_outlier.sum()),
                            "missing_calendar_days": int(missing.sum()), "max_consecutive_missing_days": max_missing_run})
    pd.DataFrame(diagnostics).to_csv(
        REPORTS / "ss_quality_outliers_gaps_2026-11-13.csv", index=False, encoding="utf-8-sig"
    )


def _features(frame: pd.DataFrame, variant: str) -> list[str]:
    return [*select_seoul_rainfall_features(frame, variant), *ss_feature_columns(WINDOW)]


def _prediction_frame(frame: pd.DataFrame, target: str, algorithm: str, prediction: np.ndarray, elapsed: float, best: int) -> pd.DataFrame:
    output = frame[["series_id", "facility", "plant", "target_date", "split"]].copy()
    output["actual"] = frame[f"target_{target}"].to_numpy()
    output["prediction"] = prediction
    output["target"] = target
    output["algorithm"] = algorithm
    output["variant"] = TARGET_VARIANTS[target]
    output["window_days"] = WINDOW
    output["training_seconds"] = elapsed
    output["best_iteration"] = best
    output["ss_history_missing"] = frame[ss_feature_columns(WINDOW)].isna().any(axis=1).to_numpy()
    return output


def train_tree(frame: pd.DataFrame, target: str, algorithm: str) -> pd.DataFrame:
    features = _features(frame, TARGET_VARIANTS[target])
    target_column = f"target_{target}"
    selected = frame[["series_id", "facility", "plant", "target_date", "split", target_column, *features]].copy()
    train = selected[selected.split.eq("train")]
    validation = selected[selected.split.eq("validation")]
    test = selected[selected.split.eq("test")]
    train, validation, test = add_series_code(train, validation, test)
    features = [*features, "series_code"]
    started = perf_counter()
    model = (fit_lightgbm if algorithm == "lightgbm" else fit_xgboost)(
        train, validation, target_column, features
    )
    elapsed = perf_counter() - started
    best = int(getattr(model, "best_iteration_", getattr(model, "best_iteration", -1)) or -1)
    joblib.dump({
        "model": model, "features": features, "target": target, "variant": TARGET_VARIANTS[target],
        "window_days": WINDOW, "algorithm": algorithm, "random_seed": 42,
        "training_seconds": elapsed, "ss_source": "same_series_past_30_days",
    }, MODELS / f"{algorithm}_{target}.joblib")
    return pd.concat([
        _prediction_frame(validation, target, algorithm, model.predict(validation[features]), elapsed, best),
        _prediction_frame(test, target, algorithm, model.predict(test[features]), elapsed, best),
    ], ignore_index=True)


def train_lstm(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    features = _features(frame, TARGET_VARIANTS[target])
    stems = [column.removesuffix("_lag_0") for column in features if column.endswith("_lag_0")]
    values = tabular_to_sequence(frame, stems, WINDOW)
    train_mask = frame.split.eq("train").to_numpy()
    validation_mask = frame.split.eq("validation").to_numpy()
    test_mask = frame.split.eq("test").to_numpy()
    missing_indices = [i for i, stem in enumerate(stems) if stem.startswith("seoul_rainfall") or stem == "ss_mg_l"]
    preprocessor = fit_sequence_preprocessor(values[train_mask], missing_indices)
    train_x = preprocessor.transform(values[train_mask])
    validation_x = preprocessor.transform(values[validation_mask])
    test_x = preprocessor.transform(values[test_mask])
    train_series, validation_series, test_series = encode_series_ids(
        frame.loc[train_mask, "series_id"].to_numpy(),
        frame.loc[validation_mask, "series_id"].to_numpy(),
        frame.loc[test_mask, "series_id"].to_numpy(),
    )
    y = frame[f"target_{target}"].to_numpy()
    started = perf_counter()
    result = fit_lstm(
        train_x, train_series, y[train_mask], validation_x, validation_series,
        y[validation_mask], num_series=int(pd.Series(train_series).nunique()), config=LSTMConfig(),
    )
    elapsed = perf_counter() - started
    checkpoint = {
        "state_dict": result.model.state_dict(), "input_size": train_x.shape[2],
        "num_series": int(pd.Series(train_series).nunique()),
        "series_categories": sorted(frame.loc[train_mask, "series_id"].unique().tolist()),
        "feature_stems": stems, "missing_indicator_indices": missing_indices,
        "preprocessor": {"medians": preprocessor.medians, "means": preprocessor.means,
                         "stds": preprocessor.stds, "missing_indicator_indices": preprocessor.missing_indicator_indices},
        "target_mean": result.target_mean, "target_std": result.target_std,
        "best_epoch": result.best_epoch, "history": result.history,
        "training_seconds": elapsed, "target": target, "variant": TARGET_VARIANTS[target],
        "window_days": WINDOW, "random_seed": 42,
    }
    torch.save(checkpoint, MODELS / f"lstm_{target}.pt")
    validation_prediction = predict_lstm(result.model, validation_x, validation_series, result.target_mean, result.target_std)
    test_prediction = predict_lstm(result.model, test_x, test_series, result.target_mean, result.target_std)
    return pd.concat([
        _prediction_frame(frame.loc[validation_mask], target, "lstm", validation_prediction, elapsed, result.best_epoch),
        _prediction_frame(frame.loc[test_mask], target, "lstm", test_prediction, elapsed, result.best_epoch),
    ], ignore_index=True)


def load_baselines() -> pd.DataFrame:
    parts = []
    paths = {
        "lightgbm": RESULTS / "predictions/seoul_rainfall_lightgbm_predictions.parquet",
        "xgboost": RESULTS / "predictions/xgboost_comparison_predictions.parquet",
        "lstm": RESULTS / "predictions/lstm_comparison_predictions.parquet",
    }
    for algorithm, path in paths.items():
        frame = pd.read_parquet(path)
        algorithm_column = frame.get("algorithm", pd.Series(algorithm, index=frame.index))
        keep = frame.window_days.eq(WINDOW) & frame.apply(lambda row: row.variant == TARGET_VARIANTS.get(row.target), axis=1)
        selected = frame[keep].copy()
        selected["algorithm"] = algorithm_column[keep]
        selected["source"] = "baseline"
        parts.append(selected)
    return pd.concat(parts, ignore_index=True)


def evaluate(new: pd.DataFrame, baseline: pd.DataFrame, augmented: pd.DataFrame) -> None:
    new = new.copy(); new["source"] = "ss"
    combined = pd.concat([baseline, new], ignore_index=True, sort=False)
    rows = []
    series_rows = []
    high_rows = []
    thresholds = {target: float(augmented.loc[augmented.split.eq("train"), f"target_{target}"].quantile(.9)) for target in TARGET_VARIANTS}
    for (algorithm, target, split), group in combined.groupby(["algorithm", "target", "split"]):
        base = group[group.source.eq("baseline")]
        ss = group[group.source.eq("ss")]
        common = base.merge(ss, on=KEYS, suffixes=("_base", "_ss"), validate="one_to_one")
        if not np.allclose(common.actual_base, common.actual_ss):
            raise RuntimeError("기존 모델과 SS 모델의 실제값이 다릅니다")
        for source, suffix in (("baseline", "base"), ("ss", "ss")):
            score = metrics(common.actual_base, common[f"prediction_{suffix}"])
            rows.append({"algorithm": algorithm, "target": target, "split": split, "source": source,
                         "n": len(common), **score})
            pred = common[f"prediction_{suffix}"]
            high = common.actual_base.ge(thresholds[target])
            high_rows.append({"algorithm": algorithm, "target": target, "split": split, "source": source,
                              "threshold": thresholds[target], "n": int(high.sum()),
                              "mae": float(np.abs(common.loc[high, "actual_base"] - pred[high]).mean()),
                              "underprediction_rate": float(pred[high].lt(common.loc[high, "actual_base"]).mean()),
                              "mean_bias": float((pred[high] - common.loc[high, "actual_base"]).mean())})
            for series_id, sg in common.groupby("series_id"):
                s = metrics(sg.actual_base, sg[f"prediction_{suffix}"])
                series_rows.append({"algorithm": algorithm, "target": target, "split": split,
                                    "source": source, "series_id": series_id, "n": len(sg), **s})
    metric_frame = pd.DataFrame(rows)
    wide = metric_frame.pivot(index=["algorithm", "target", "split", "n"], columns="source", values=["mae", "rmse", "r2"]).reset_index()
    wide.columns = ["_".join(str(x) for x in c if x) for c in wide.columns]
    wide["mae_improvement_pct"] = (wide.mae_baseline - wide.mae_ss) / wide.mae_baseline * 100
    wide.to_csv(REPORTS / "ss_model_comparison_2026-11-13.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(series_rows).to_csv(REPORTS / "ss_model_series_metrics_2026-11-13.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(high_rows).to_csv(REPORTS / "ss_model_high_concentration_2026-11-13.csv", index=False, encoding="utf-8-sig")

    importance = []
    for path in MODELS.glob("*.joblib"):
        artifact = joblib.load(path); model = artifact["model"]
        values = getattr(model, "feature_importances_", np.zeros(len(artifact["features"])))
        total = float(np.sum(values)) or 1.0
        for feature, value in zip(artifact["features"], values):
            importance.append({"model": path.stem, "feature": feature, "importance": float(value),
                               "importance_share": float(value) / total, "is_ss": feature.startswith("ss_mg_l_")})
    pd.DataFrame(importance).to_csv(REPORTS / "ss_tree_feature_importance_2026-11-13.csv", index=False, encoding="utf-8-sig")

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    validation = wide[wide.split.eq("validation")]
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = validation.algorithm + "-" + validation.target.str.replace("_mg_l", "")
    x = np.arange(len(validation)); ax.bar(x-.2, validation.mae_baseline, .4, label="기존"); ax.bar(x+.2, validation.mae_ss, .4, label="SS 추가")
    ax.set_xticks(x, labels, rotation=35, ha="right"); ax.set_ylabel("MAE"); ax.legend(); fig.tight_layout()
    fig.savefig(FIGURES / "validation_mae_comparison.png", dpi=150); plt.close(fig)

    report_lines = ["# SS 데이터 검증 및 3종 알고리즘 비교", "", "## 핵심 결과", "",
                    validation[["algorithm", "target", "mae_baseline", "mae_ss", "mae_improvement_pct"]].to_markdown(index=False), "",
                    "- 모델 선택은 Validation 기준이며 Test는 반복 확인된 참고 구간이다.",
                    "- SS는 동일 계통의 목표일 이전 30일만 사용했다. 목표일 SS와 미래 수질·기상은 사용하지 않았다.",
                    "- 운영 환경에서 일별 SS가 예측 시점 전에 공개되는지는 원자료에 공개 시각이 없어 미확인이다.",
                    "- SS 개선이 일관되지 않으면 최종 피처로 자동 채택하지 않는다."]
    (REPORTS / "ss_experiment_report_2026-11-13.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True); PREDICTIONS.parent.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(exist_ok=True)
    observations = pd.read_parquet(PROCESSED / "observation_series_daily_seoul_rainfall.parquet")
    if observations.duplicated(["series_id", "date"]).any() or observations.ss_mg_l.lt(0).any():
        raise RuntimeError("SS 품질 게이트 실패: 중복 키 또는 음수")
    quality_tables(observations)
    frame = build_augmented_frame()
    predictions = []
    for target in TARGET_VARIANTS:
        predictions.append(train_tree(frame, target, "lightgbm"))
        predictions.append(train_tree(frame, target, "xgboost"))
        predictions.append(train_lstm(frame, target))
    new = pd.concat(predictions, ignore_index=True)
    if new.duplicated(KEYS + ["algorithm"]).any():
        raise RuntimeError("신규 예측 키가 중복됩니다")
    new.to_parquet(PREDICTIONS, index=False)
    evaluate(new, load_baselines(), frame)
    config = {"window_days": WINDOW, "targets": TARGET_VARIANTS, "algorithms": ALGORITHMS,
              "random_seed": 42, "new_model_count": 9, "ss_timing": "target_date-1..target_date-30"}
    (REPORTS / "ss_experiment_config_2026-11-13.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
