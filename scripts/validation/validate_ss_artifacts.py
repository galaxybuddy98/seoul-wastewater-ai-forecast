"""SS 실험 산출물의 키·모델 수·저장 지표·누수 계약을 검증한다."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

from scripts._paths import PROJECT_ROOT

ROOT = PROJECT_ROOT


def main() -> None:
    prediction_path = ROOT / "results/predictions/ss_experiment_predictions_2026-11-13.parquet"
    model_dir = ROOT / "results/models/ss_experiment_2026-11-13"
    predictions = pd.read_parquet(prediction_path)
    keys = ["algorithm", "target", "split", "series_id", "target_date"]
    assert not predictions.duplicated(keys).any()
    assert len(list(model_dir.glob("*.joblib"))) == 6
    assert len(list(model_dir.glob("*.pt"))) == 3
    assert set(predictions.algorithm) == {"lightgbm", "xgboost", "lstm"}
    assert set(predictions.target) == {"bod_mg_l", "tn_mg_l", "tp_mg_l"}
    counts = predictions.groupby(["split", "algorithm", "target"]).size()
    assert counts.groupby(level="split").nunique().eq(1).all()

    for path in model_dir.glob("*.joblib"):
        artifact = joblib.load(path)
        ss_features = [name for name in artifact["features"] if name.startswith("ss_mg_l_lag_")]
        assert ss_features == [f"ss_mg_l_lag_{lag}" for lag in range(30)]
    for path in model_dir.glob("*.pt"):
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        assert artifact["feature_stems"].count("ss_mg_l") == 1
        assert artifact["feature_stems"].index("ss_mg_l") in artifact["missing_indicator_indices"]

    stored = pd.read_csv(ROOT / "reports/ss_model_comparison_2026-11-13.csv")
    ss_rows = stored[["algorithm", "target", "split", "mae_ss"]]
    recalculated = predictions.groupby(["algorithm", "target", "split"]).apply(
        lambda x: mean_absolute_error(x.actual, x.prediction), include_groups=False
    ).rename("mae_recalculated").reset_index()
    checked = ss_rows.merge(recalculated, validate="one_to_one")
    assert np.allclose(checked.mae_ss, checked.mae_recalculated, rtol=0, atol=1e-10)
    print({"prediction_rows": len(predictions), "models": 9, "metric_rows_checked": len(checked)})


if __name__ == "__main__":
    main()
