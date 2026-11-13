"""저장 예측값을 위한 독립 시각화 검증 유틸리티."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

KEYS = ["split", "target", "series_id", "target_date"]
SIMPLE_LINE_COLORS = {"actual": "#0072B2", "xgboost": "#E69F00"}


def two_line_series(frame: pd.DataFrame) -> pd.DataFrame:
    """최종 XGBoost 행을 실제값·예측값 두 선용 표로 변환한다."""
    if frame.empty or not frame["model"].eq("xgboost").all():
        raise ValueError("최종 XGBoost 행만 두 선 시각화에 사용할 수 있습니다")
    result = frame[["target_date", "actual", "prediction"]].copy()
    result["target_date"] = pd.to_datetime(result["target_date"])
    result = result.rename(columns={"prediction": "xgboost"}).sort_values("target_date")
    if result["target_date"].duplicated().any():
        raise ValueError("계통 내 목표일 중복")
    return result.reset_index(drop=True)


def assert_prediction_alignment(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """공통 키의 중복과 실제값 불일치를 거부한다."""
    frames = []
    for name, frame in (("left", left), ("right", right)):
        work = frame.copy()
        work["target_date"] = pd.to_datetime(work["target_date"])
        if work.duplicated(KEYS).any():
            raise ValueError(f"{name} 예측 키 중복")
        frames.append(work)
    common = frames[0].merge(frames[1], on=KEYS, suffixes=("_l", "_r"), validate="one_to_one")
    if common.empty:
        raise ValueError("공통 평가 키 없음")
    if not np.allclose(common["actual_l"], common["actual_r"], equal_nan=True):
        raise ValueError("공통 평가 키의 실제값 불일치")


def consecutive_segments(frame: pd.DataFrame) -> list[pd.DataFrame]:
    work = frame.sort_values("target_date").copy()
    if work.empty:
        return []
    group = pd.to_datetime(work["target_date"]).diff().dt.days.ne(1).cumsum()
    return [part for _, part in work.groupby(group, sort=False)]


def metric_row(actual: pd.Series, prediction: pd.Series) -> dict[str, float | int]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(prediction, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "r2": float(r2_score(y, p)) if len(y) > 1 else np.nan,
        "bias": float(np.mean(p - y)),
    }


def select_dynamic_validation_window(frame: pd.DataFrame, days: int = 90) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validation 실제값만으로 변화량이 가장 큰 고정 길이 구간을 선택한다."""
    work = frame.loc[frame["split"].eq("validation")].copy()
    work["target_date"] = pd.to_datetime(work["target_date"])
    daily = work.groupby("target_date", as_index=True)["actual"].mean().sort_index()
    full = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()))
    change = full.diff().abs().rolling(days, min_periods=days - 1).sum()
    end = change.idxmax()
    return end - pd.Timedelta(days=days - 1), end
