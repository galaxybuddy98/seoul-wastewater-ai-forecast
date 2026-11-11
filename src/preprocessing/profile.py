"""일별 시계열의 날짜 누락과 연속 결측을 요약한다."""

from __future__ import annotations

import pandas as pd


def _group_items(frame: pd.DataFrame, group_columns: list[str]):
    grouper = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in frame.groupby(grouper, dropna=False, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        yield dict(zip(group_columns, keys)), group


def find_calendar_gaps(
    frame: pd.DataFrame, group_columns: list[str], date_column: str
) -> pd.DataFrame:
    """각 시계열의 관측 시작~종료 사이 누락 날짜를 요약한다."""
    rows = []
    for labels, group in _group_items(frame, group_columns):
        dates = pd.DatetimeIndex(pd.to_datetime(group[date_column], errors="coerce").dropna().unique()).sort_values()
        if dates.empty:
            rows.append({**labels, "start_date": pd.NaT, "end_date": pd.NaT,
                         "missing_date_count": 0, "longest_gap_days": 0,
                         "first_missing_date": pd.NaT})
            continue
        expected = pd.date_range(dates.min(), dates.max(), freq="D")
        missing = expected.difference(dates)
        longest = 0
        if len(missing):
            series = pd.Series(missing)
            run_id = series.diff().dt.days.ne(1).cumsum()
            longest = int(series.groupby(run_id).size().max())
        rows.append({**labels, "start_date": dates.min(), "end_date": dates.max(),
                     "missing_date_count": int(len(missing)),
                     "longest_gap_days": longest,
                     "first_missing_date": missing.min() if len(missing) else pd.NaT})
    return pd.DataFrame(rows)


def find_missing_runs(
    frame: pd.DataFrame,
    group_columns: list[str],
    date_column: str,
    value_columns: list[str],
) -> pd.DataFrame:
    """실제 행에서 연속된 결측 측정 구간을 반환한다."""
    records = []
    for labels, group in _group_items(frame, group_columns):
        ordered = group.sort_values(date_column)
        dates = pd.to_datetime(ordered[date_column], errors="coerce")
        for column in value_columns:
            missing = ordered[column].isna()
            run_id = (missing.ne(missing.shift()) | dates.diff().dt.days.ne(1)).cumsum()
            for _, run in ordered.loc[missing].assign(_date=dates[missing], _run=run_id[missing]).groupby("_run"):
                records.append({**labels, "variable": column,
                                "start_date": run["_date"].min(),
                                "end_date": run["_date"].max(),
                                "run_length": int(len(run))})
    columns = [*group_columns, "variable", "start_date", "end_date", "run_length"]
    return pd.DataFrame(records, columns=columns)


def count_next_day_window_samples(
    frame: pd.DataFrame,
    group_columns: list[str],
    date_column: str,
    target_columns: list[str],
    windows: list[int],
    validation_start: str = "2023-01-01",
    test_start: str = "2025-01-01",
) -> pd.DataFrame:
    """연속 과거 관측과 +1일 목표가 모두 있는 후보 표본 수를 센다."""
    validation_boundary = pd.Timestamp(validation_start)
    test_boundary = pd.Timestamp(test_start)
    rows: list[dict[str, object]] = []

    for group_key, group in frame.groupby(group_columns, dropna=False, sort=True):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        ordered = group.sort_values(date_column).drop_duplicates(date_column, keep=False)
        dates = pd.to_datetime(ordered[date_column]).reset_index(drop=True)
        targets = ordered[target_columns].reset_index(drop=True)

        for window in windows:
            counts: dict[str, int] = {}
            for end_index in range(window - 1, len(ordered) - 1):
                start_index = end_index - window + 1
                observed_dates = dates.iloc[start_index : end_index + 2]
                expected_dates = pd.date_range(
                    observed_dates.iloc[0], periods=window + 1, freq="D"
                )
                if not observed_dates.reset_index(drop=True).equals(
                    pd.Series(expected_dates)
                ):
                    continue
                if targets.iloc[start_index : end_index + 2].isna().any().any():
                    continue

                target_date = dates.iloc[end_index + 1]
                split = (
                    "test" if target_date >= test_boundary
                    else "validation" if target_date >= validation_boundary
                    else "train"
                )
                counts[split] = counts.get(split, 0) + 1

            for split, count in counts.items():
                row = dict(zip(group_columns, key_values))
                row.update(
                    {"window_days": window, "split": split, "candidate_count": count}
                )
                rows.append(row)

    return pd.DataFrame(rows)
