"""표형 모델과 LSTM이 공유하는 누수 없는 시계열 윈도."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.dataset import DatasetError, assign_target_split


def build_tabular_windows(
    frame: pd.DataFrame,
    window_days: int,
    feature_columns: list[str],
    target_columns: list[str],
    required_feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """계통별 연속 L일 입력과 t+1 목표를 표형 피처로 만든다."""
    if window_days < 1:
        raise ValueError("window_days는 1 이상이어야 합니다")
    parts: list[pd.DataFrame] = []
    metadata_columns = ["series_id", "center_id", "facility", "plant"]

    for series_id, group in frame.groupby("series_id", sort=True):
        ordered = group.sort_values("date").copy()
        if ordered["date"].duplicated().any():
            raise DatasetError(f"계통 {series_id}에 날짜 중복이 있습니다")
        ordered = ordered.set_index("date")
        daily = ordered.reindex(
            pd.date_range(ordered.index.min(), ordered.index.max(), freq="D")
        )
        candidate_columns = {
            f"{feature}_lag_{lag}": daily[feature].shift(lag + 1)
            for feature in feature_columns
            for lag in range(window_days)
        }
        candidate_columns.update(
            {
                f"target_{target_column}": daily[target_column]
                for target_column in target_columns
            }
        )
        candidate = pd.DataFrame(candidate_columns, index=daily.index)

        required_features = (
            feature_columns if required_feature_columns is None else required_feature_columns
        )
        unknown_required = set(required_features) - set(feature_columns)
        if unknown_required:
            raise ValueError(f"필수 피처가 feature_columns에 없습니다: {sorted(unknown_required)}")
        value_columns = [
            *[f"{feature}_lag_{lag}" for feature in required_features for lag in range(window_days)],
            *[f"target_{target_column}" for target_column in target_columns],
        ]
        candidate = candidate.dropna(subset=value_columns)
        if candidate.empty:
            continue
        for column in metadata_columns:
            if column in ordered.columns:
                candidate[column] = ordered[column].dropna().iloc[0]
        candidate["input_start_date"] = candidate.index - pd.to_timedelta(window_days, unit="D")
        candidate["input_end_date"] = candidate.index - pd.Timedelta(days=1)
        candidate["target_date"] = candidate.index
        candidate["window_days"] = window_days
        candidate["split"] = assign_target_split(pd.Series(candidate.index)).to_numpy()
        parts.append(candidate.reset_index(drop=True))

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_sequence_windows(
    frame: pd.DataFrame,
    window_days: int,
    feature_columns: list[str],
    target_columns: list[str],
) -> dict[str, object]:
    """표형 윈도와 동일한 평가행을 시간순 3차원 배열로 변환한다."""
    tabular = build_tabular_windows(frame, window_days, feature_columns, target_columns)
    return tabular_to_sequence(tabular, window_days, feature_columns, target_columns)


def count_target_window_samples(
    frame: pd.DataFrame,
    window_days: int,
    feature_columns: list[str],
    target_columns: list[str],
    required_feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """전체 일자축에서 연속 과거 입력과 목표별 t+1 가용 표본을 빠르게 센다."""
    rows: list[dict[str, object]] = []
    for series_id, group in frame.groupby("series_id", sort=True):
        ordered = group.sort_values("date").set_index("date")
        if ordered.index.duplicated().any():
            raise DatasetError(f"계통 {series_id}에 날짜 중복이 있습니다")
        daily = ordered.reindex(pd.date_range(ordered.index.min(), ordered.index.max(), freq="D"))
        required_features = (
            feature_columns if required_feature_columns is None else required_feature_columns
        )
        unknown_required = set(required_features) - set(feature_columns)
        if unknown_required:
            raise ValueError(f"필수 피처가 feature_columns에 없습니다: {sorted(unknown_required)}")
        history_row_complete = daily[required_features].notna().all(axis=1)
        history_complete = (
            history_row_complete.shift(1).rolling(window_days, min_periods=window_days).sum()
            .eq(window_days)
        )
        identity = group.iloc[0]
        for target_column in target_columns:
            valid = history_complete & daily[target_column].notna()
            target_dates = pd.Series(daily.index[valid])
            if target_dates.empty:
                continue
            splits = assign_target_split(target_dates)
            for split, count in splits.value_counts(dropna=False).items():
                if pd.isna(split):
                    continue
                rows.append(
                    {
                        "facility": identity.get("facility"),
                        "plant": identity.get("plant"),
                        "series_id": series_id,
                        "window_days": window_days,
                        "target": target_column,
                        "split": split,
                        "sample_count": int(count),
                    }
                )
    return pd.DataFrame(rows)


def tabular_to_sequence(
    tabular: pd.DataFrame,
    window_days: int,
    feature_columns: list[str],
    target_columns: list[str],
) -> dict[str, object]:
    """이미 만든 표형 윈도를 동일 평가행의 시퀀스 배열로 변환한다."""
    if tabular.empty:
        return {
            "X": np.empty((0, window_days, len(feature_columns))),
            "y": np.empty((0, len(target_columns))),
            "metadata": tabular,
        }
    lag_order = list(reversed(range(window_days)))
    feature_arrays = [
        tabular[[f"{feature}_lag_{lag}" for lag in lag_order]].to_numpy(dtype=float)
        for feature in feature_columns
    ]
    x = np.stack(feature_arrays, axis=2)
    y = tabular[[f"target_{column}" for column in target_columns]].to_numpy(dtype=float)
    metadata_columns = [
        "series_id", "center_id", "facility", "plant",
        "input_start_date", "input_end_date", "target_date", "window_days", "split",
    ]
    return {
        "X": x,
        "y": y,
        "metadata": tabular[[column for column in metadata_columns if column in tabular.columns]].copy(),
    }
