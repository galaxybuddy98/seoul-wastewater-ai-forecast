"""계절·요일·BOD 가중 실험 산출물의 재현성과 공정 비교를 검증한다."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.xgboost.bod_improvement import assert_common_evaluation_rows


RUN_ID = "2026-11-13"
PREDICTION_PATH = Path(f"results/predictions/calendar_weighted_xgboost_predictions_{RUN_ID}.parquet")
METRICS_PATH = Path(f"reports/calendar_weighted_xgboost_metrics_{RUN_ID}.csv")
TRAINING_PATH = Path(f"reports/calendar_weighted_xgboost_training_{RUN_ID}.json")
MODEL_DIR = Path("results/models")
KEYS = ["split", "series_id", "target_date"]


def main() -> None:
    predictions = pd.read_parquet(PREDICTION_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    training = json.loads(TRAINING_PATH.read_text(encoding="utf-8"))
    if len(predictions) != 151_079:
        raise AssertionError(f"예측 행 수 불일치: {len(predictions)}")
    if predictions.duplicated(KEYS + ["target", "model_name"]).any():
        raise AssertionError("예측 평가키 중복")
    if len(training) != 11:
        raise AssertionError(f"학습 모델 수 불일치: {len(training)}")

    for target in predictions["target"].unique():
        baseline = predictions[(predictions["target"] == target) & (predictions["model_name"] == "baseline")]
        for model_name, candidate in predictions[predictions["target"] == target].groupby("model_name"):
            assert_common_evaluation_rows(baseline, candidate)

    thresholds = metrics.groupby("target")["high_threshold"].first().to_dict()
    checked = 0
    for row in metrics.itertuples():
        group = predictions[
            (predictions["split"] == row.split)
            & (predictions["target"] == row.target)
            & (predictions["model_name"] == row.model_name)
        ]
        actual = group["actual"]
        prediction = group["prediction"]
        high = group[actual >= thresholds[row.target]]
        values = {
            "mae": mean_absolute_error(actual, prediction),
            "rmse": mean_squared_error(actual, prediction) ** 0.5,
            "r2": r2_score(actual, prediction),
            "mean_bias": (prediction - actual).mean(),
            "high_mae": mean_absolute_error(high["actual"], high["prediction"]),
            "high_rmse": mean_squared_error(high["actual"], high["prediction"]) ** 0.5,
            "high_mean_bias": (high["prediction"] - high["actual"]).mean(),
            "high_underprediction_rate": (high["prediction"] < high["actual"]).mean(),
        }
        for column, value in values.items():
            if not np.isclose(value, getattr(row, column), rtol=1e-10, atol=1e-10):
                raise AssertionError(f"저장 지표 불일치: {row.target}/{row.model_name}/{row.split}/{column}")
        checked += 1

    model_paths = sorted(MODEL_DIR.glob(f"calendar_weighted_xgboost_*_{RUN_ID}.joblib"))
    if len(model_paths) != 11:
        raise AssertionError(f"저장 모델 파일 수 불일치: {len(model_paths)}")
    for item, model_path in zip(sorted(training, key=lambda row: row["model_file"]), model_paths):
        payload = joblib.load(model_path)
        if item["features"] != payload["features"]:
            raise AssertionError(f"피처 manifest 불일치: {model_path}")
        if item["feature_count"] != len(payload["features"]):
            raise AssertionError(f"피처 수 불일치: {model_path}")
        forbidden = {"target_bod_mg_l", "target_tn_mg_l", "target_tp_mg_l"}.intersection(payload["features"])
        if forbidden:
            raise AssertionError(f"미래 목표열이 피처에 포함됨: {model_path} / {sorted(forbidden)}")

    print(
        {
            "prediction_rows": len(predictions),
            "metric_rows_checked": checked,
            "model_files": len(model_paths),
            "duplicate_keys": 0,
            "actual_values_match": True,
            "future_target_features": 0,
        }
    )


if __name__ == "__main__":
    main()
