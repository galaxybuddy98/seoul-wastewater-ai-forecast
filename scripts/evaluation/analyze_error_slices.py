"""저장 예측값으로 계통·계절·요일·고농도·처리량 관계를 분석한다.

모델 학습이나 원본 데이터 처리는 수행하지 않는다. 모든 모델 비교는 동일한
평가 키와 실제값을 확인한 뒤 계산한다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


RUN_ID = "2026-11-13"
REPORTS = Path("reports")
FIGURES = Path("results/figures")
PREDICTIONS = Path("results/predictions/final_model_all_candidate_common_2026-11-13.parquet")
DAILY = Path("data/processed/observation_series_daily_seoul_rainfall.parquet")
TARGETS = ("bod_mg_l", "tn_mg_l", "tp_mg_l")
TARGET_LABELS = {"bod_mg_l": "BOD", "tn_mg_l": "T-N", "tp_mg_l": "T-P"}
RAIN_CONFIG = {"bod_mg_l": "C", "tn_mg_l": "E", "tp_mg_l": "C"}
ALGORITHMS = ("persistence", "lightgbm", "xgboost", "lstm")
SEASON_ORDER = ("봄", "여름", "가을", "겨울")
WEEKDAY_NAMES = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
KEYS = ["split", "target", "series_id", "target_date"]


def _season(month: pd.Series) -> pd.Categorical:
    labels = np.select(
        [month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])],
        ["봄", "여름", "가을"],
        default="겨울",
    )
    return pd.Categorical(labels, categories=SEASON_ORDER, ordered=True)


def _candidate_id(algorithm: str, target: str) -> str:
    if algorithm == "persistence":
        return "persistence_P_w30"
    return f"{algorithm}_{RAIN_CONFIG[target]}_w30"


def _metrics(group: pd.DataFrame) -> pd.Series:
    actual = group["actual"].astype(float)
    prediction = group["prediction"].astype(float)
    return pd.Series(
        {
            "rows": len(group),
            "actual_mean": actual.mean(),
            "actual_median": actual.median(),
            "actual_p90": actual.quantile(0.9),
            "actual_std": actual.std(),
            "mae": mean_absolute_error(actual, prediction),
            "rmse": mean_squared_error(actual, prediction) ** 0.5,
            "r2": r2_score(actual, prediction) if len(group) >= 2 and actual.nunique() > 1 else np.nan,
            "mean_bias": (prediction - actual).mean(),
        }
    )


def _safe_corr(pair: pd.DataFrame, left: str, right: str, method: str) -> float:
    if len(pair) < 3 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
        return np.nan
    return float(pair[left].corr(pair[right], method=method))


def _load_common_predictions() -> pd.DataFrame:
    source = pd.read_parquet(PREDICTIONS)
    pieces = []
    for target in TARGETS:
        for algorithm in ALGORITHMS:
            candidate = _candidate_id(algorithm, target)
            part = source[(source["target"] == target) & (source["candidate_id"] == candidate)].copy()
            if part.empty:
                raise ValueError(f"후보 예측 없음: {candidate} / {target}")
            part["algorithm"] = algorithm
            pieces.append(part)
    frame = pd.concat(pieces, ignore_index=True)
    if frame.duplicated(KEYS + ["algorithm"]).any():
        raise ValueError("모델별 평가 키 중복")
    counts = frame.groupby(KEYS, observed=True)["algorithm"].nunique()
    if not counts.eq(len(ALGORITHMS)).all():
        raise ValueError("공통 평가 키에 네 알고리즘이 모두 존재하지 않음")
    if not frame.groupby(KEYS, observed=True)["actual"].nunique().eq(1).all():
        raise ValueError("모델 간 실제값 불일치")

    dates = pd.to_datetime(frame["target_date"])
    frame["season"] = _season(dates.dt.month)
    frame["weekday_number"] = dates.dt.dayofweek
    frame["weekday"] = frame["weekday_number"].map(WEEKDAY_NAMES)
    frame["day_type"] = np.where(frame["weekday_number"] >= 5, "주말", "평일")
    frame["error"] = frame["prediction"] - frame["actual"]
    frame["absolute_error"] = frame["error"].abs()
    return frame


def _series_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        predictions.groupby(
            ["split", "target", "algorithm", "series_id", "facility", "plant"],
            observed=True,
            dropna=False,
        )
        .apply(_metrics, include_groups=False)
        .reset_index()
    )
    overall = (
        predictions.groupby(["split", "target", "algorithm"], observed=True)
        .apply(_metrics, include_groups=False)
        .reset_index()[["split", "target", "algorithm", "mae"]]
        .rename(columns={"mae": "overall_mae"})
    )
    grouped = grouped.merge(overall, on=["split", "target", "algorithm"], validate="many_to_one")
    grouped["mae_vs_overall_ratio"] = grouped["mae"] / grouped["overall_mae"]
    grouped["row_share"] = grouped["rows"] / grouped.groupby(
        ["split", "target", "algorithm"], observed=True
    )["rows"].transform("sum")
    total_error = grouped["mae"] * grouped["rows"]
    grouped["absolute_error_share"] = total_error / total_error.groupby(
        [grouped["split"], grouped["target"], grouped["algorithm"]], observed=True
    ).transform("sum")
    return grouped


def _slice_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    season = (
        predictions.groupby(["split", "target", "algorithm", "season"], observed=True)
        .apply(_metrics, include_groups=False)
        .reset_index()
    )
    weekday = (
        predictions.groupby(["split", "target", "algorithm", "weekday_number", "weekday"], observed=True)
        .apply(_metrics, include_groups=False)
        .reset_index()
    )
    day_type = (
        predictions.groupby(["split", "target", "algorithm", "day_type"], observed=True)
        .apply(_metrics, include_groups=False)
        .reset_index()
    )
    season_series = (
        predictions[predictions["algorithm"] == "xgboost"]
        .groupby(
            ["split", "target", "season", "series_id", "facility", "plant"],
            observed=True,
            dropna=False,
        )
        .apply(_metrics, include_groups=False)
        .reset_index()
    )
    return season, weekday, day_type, season_series


def _daily_with_calendar() -> pd.DataFrame:
    daily = pd.read_parquet(DAILY).copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year
    daily["season"] = _season(daily["date"].dt.month)
    daily["weekday_number"] = daily["date"].dt.dayofweek
    daily["weekday"] = daily["weekday_number"].map(WEEKDAY_NAMES)
    daily["day_type"] = np.where(daily["weekday_number"] >= 5, "주말", "평일")
    daily["split"] = np.select(
        [daily["date"] <= pd.Timestamp("2022-12-31"), daily["date"] <= pd.Timestamp("2024-12-31")],
        ["train", "validation"],
        default="test",
    )
    return daily


def _environment_relationships(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in ("validation", "test"):
        for target in TARGETS:
            for season in SEASON_ORDER:
                group = daily[(daily["split"] == split) & (daily["season"] == season)]
                for feature in ("mean_temperature_c", "seoul_rainfall_mm", "seoul_rainfall_sum_7d_mm"):
                    pair = group[[target, feature]].dropna()
                    rows.append(
                        {
                            "split": split,
                            "target": target,
                            "season": season,
                            "feature": feature,
                            "rows": len(pair),
                            "feature_missing_rate": group[feature].isna().mean(),
                            "pearson": _safe_corr(pair, target, feature, "pearson"),
                            "spearman": _safe_corr(pair, target, feature, "spearman"),
                        }
                    )
    return pd.DataFrame(rows)


def _high_bod_analysis(predictions: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    bod = predictions[predictions["target"] == "bod_mg_l"].copy()
    bod["high_bod"] = bod["actual"] >= 180.0
    frequency = (
        bod[bod["algorithm"] == "xgboost"]
        .groupby(["split", "season", "weekday", "day_type", "series_id", "facility", "plant"], observed=True)
        .agg(rows=("actual", "size"), high_rows=("high_bod", "sum"), actual_mean=("actual", "mean"))
        .reset_index()
    )
    frequency["high_rate"] = frequency["high_rows"] / frequency["rows"]

    high = bod[bod["high_bod"]].copy()
    model_summary = (
        high.groupby(["split", "algorithm"], observed=True)
        .apply(_metrics, include_groups=False)
        .reset_index()
    )
    under = (
        high.groupby(["split", "algorithm"], observed=True)
        .agg(underprediction_rate=("error", lambda x: (x < 0).mean()))
        .reset_index()
    )
    model_summary = model_summary.merge(under, on=["split", "algorithm"], validate="one_to_one")

    by_series = (
        high.groupby(["split", "algorithm", "series_id", "facility", "plant"], observed=True)
        .apply(_metrics, include_groups=False)
        .reset_index()
    )
    by_series["total_absolute_error"] = by_series["mae"] * by_series["rows"]
    by_series["row_share"] = by_series["rows"] / by_series.groupby(
        ["split", "algorithm"], observed=True
    )["rows"].transform("sum")
    by_series["absolute_error_share"] = by_series["total_absolute_error"] / by_series.groupby(
        ["split", "algorithm"], observed=True
    )["total_absolute_error"].transform("sum")

    # 목표일보다 앞선 날짜만 자체 결합해 예측시점 가용 전조를 계산한다.
    events = bod[bod["algorithm"] == "xgboost"][
        ["split", "series_id", "facility", "plant", "target_date", "actual", "high_bod"]
    ].copy()
    daily_index = daily.set_index(["series_id", "date"])
    precursor_parts = []
    for lag in range(1, 8):
        lookup = events[["series_id", "target_date"]].copy()
        lookup["date"] = lookup["target_date"] - pd.Timedelta(days=lag)
        indexed = pd.MultiIndex.from_frame(lookup[["series_id", "date"]])
        values = daily_index.reindex(indexed)[
            ["bod_mg_l", "mean_temperature_c", "seoul_rainfall_mm"]
        ].reset_index(drop=True)
        part = events.copy()
        part["lag_days"] = lag
        part[["prior_bod_mg_l", "prior_temperature_c", "prior_rainfall_mm"]] = values.to_numpy()
        precursor_parts.append(part)
    precursors = pd.concat(precursor_parts, ignore_index=True)
    precursor_summary = (
        precursors.groupby(["split", "high_bod", "lag_days"], observed=True)
        .agg(
            events=("actual", "size"),
            prior_bod_available=("prior_bod_mg_l", "count"),
            prior_bod_mean=("prior_bod_mg_l", "mean"),
            prior_temperature_available=("prior_temperature_c", "count"),
            prior_temperature_mean=("prior_temperature_c", "mean"),
            prior_rainfall_available=("prior_rainfall_mm", "count"),
            prior_rainfall_mean=("prior_rainfall_mm", "mean"),
        )
        .reset_index()
    )

    pivot = bod.pivot_table(
        index=["split", "series_id", "target_date", "actual", "high_bod"],
        columns="algorithm",
        values="prediction",
        aggfunc="first",
    ).reset_index()
    high_pivot = pivot[pivot["high_bod"]].copy()
    persistence_reason = (
        high_pivot.groupby("split", observed=True)
        .agg(
            rows=("actual", "size"),
            actual_mean=("actual", "mean"),
            persistence_prediction_mean=("persistence", "mean"),
            xgboost_prediction_mean=("xgboost", "mean"),
            persistence_closer_rate=("actual", lambda _: np.nan),
        )
        .reset_index()
    )
    for idx, split in persistence_reason["split"].items():
        group = high_pivot[high_pivot["split"] == split]
        persistence_reason.loc[idx, "persistence_closer_rate"] = (
            (group["persistence"] - group["actual"]).abs()
            < (group["xgboost"] - group["actual"]).abs()
        ).mean()
        persistence_reason.loc[idx, "previous_day_high_rate"] = (group["persistence"] >= 180.0).mean()
        persistence_reason.loc[idx, "mean_actual_jump_from_previous"] = (
            group["actual"] - group["persistence"]
        ).mean()
    return frequency, model_summary, by_series, precursor_summary, persistence_reason


def _flow_analysis(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    flow_columns = ["primary_treatment_volume", "secondary_treatment_volume"]
    daily = daily.copy()
    daily["flow_linked"] = daily[flow_columns].notna().any(axis=1)
    coverage = (
        daily.groupby(["series_id", "facility", "plant", "year"], dropna=False, observed=True)
        .agg(
            observation_rows=("date", "size"),
            linked_rows=("flow_linked", "sum"),
            primary_nonnull=("primary_treatment_volume", "count"),
            secondary_nonnull=("secondary_treatment_volume", "count"),
        )
        .reset_index()
    )
    coverage["link_rate"] = coverage["linked_rows"] / coverage["observation_rows"]
    coverage["flow_missing_rate"] = 1 - coverage["link_rate"]

    rows: list[dict[str, object]] = []
    for (series_id, facility, plant), group in daily.groupby(
        ["series_id", "facility", "plant"], dropna=False, observed=True
    ):
        for target in TARGETS:
            for flow in flow_columns:
                pair = group[[target, flow]].dropna()
                rows.append(
                    {
                        "series_id": series_id,
                        "facility": facility,
                        "plant": plant,
                        "target": target,
                        "flow_variable": flow,
                        "rows": len(pair),
                        "pearson": _safe_corr(pair, target, flow, "pearson"),
                        "spearman": _safe_corr(pair, target, flow, "spearman"),
                    }
                )
    return coverage, pd.DataFrame(rows)


def _save_figures(
    series_metrics: pd.DataFrame,
    season_metrics: pd.DataFrame,
    high_series: pd.DataFrame,
    precursors: pd.DataFrame,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    xgb = series_metrics[
        (series_metrics["split"] == "validation") & (series_metrics["algorithm"] == "xgboost")
    ].copy()
    xgb["label"] = xgb["facility"].astype(str) + " / " + xgb["plant"].astype(str)
    pivot = xgb.pivot(index="label", columns="target", values="mae").rename(columns=TARGET_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(15, max(5, len(pivot) * 0.55)), constrained_layout=True)
    for ax, target in zip(axes, ["BOD", "T-N", "T-P"]):
        values = pivot[target].sort_values()
        ax.barh(values.index, values.values)
        ax.set_title(f"Validation {target} MAE")
        ax.set_xlabel("MAE")
    fig.savefig(FIGURES / f"error_slice_series_mae_{RUN_ID}.png", dpi=160)
    plt.close(fig)

    val_season = season_metrics[season_metrics["split"] == "validation"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        sub = val_season[val_season["target"] == target]
        for algorithm in ALGORITHMS:
            line = sub[sub["algorithm"] == algorithm].set_index("season").reindex(SEASON_ORDER)
            ax.plot(SEASON_ORDER, line["mae"], marker="o", label=algorithm)
        ax.set_title(f"{TARGET_LABELS[target]} seasonal MAE")
        ax.set_ylabel("MAE")
    axes[-1].legend(fontsize=8)
    fig.savefig(FIGURES / f"error_slice_season_mae_{RUN_ID}.png", dpi=160)
    plt.close(fig)

    hs = high_series[(high_series["split"] == "test") & (high_series["algorithm"] == "xgboost")].copy()
    hs["label"] = hs["facility"].astype(str) + " / " + hs["plant"].astype(str)
    hs = hs.sort_values("absolute_error_share")
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.barh(hs["label"], hs["absolute_error_share"] * 100)
    ax.set_xlabel("Share of total absolute error (%)")
    ax.set_title("Test high-BOD XGBoost error concentration")
    fig.savefig(FIGURES / f"error_slice_high_bod_concentration_{RUN_ID}.png", dpi=160)
    plt.close(fig)

    pre = precursors[precursors["split"] == "validation"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    variables = [
        ("prior_bod_mean", "Prior BOD"),
        ("prior_temperature_mean", "Prior temperature"),
        ("prior_rainfall_mean", "Prior rainfall"),
    ]
    for ax, (column, title) in zip(axes, variables):
        for high_value, label in [(False, "normal target"), (True, "high target")]:
            line = pre[pre["high_bod"] == high_value].sort_values("lag_days")
            ax.plot(line["lag_days"], line[column], marker="o", label=label)
        ax.set_title(title)
        ax.set_xlabel("Days before target")
    axes[-1].legend(fontsize=8)
    fig.savefig(FIGURES / f"error_slice_high_bod_precursors_{RUN_ID}.png", dpi=160)
    plt.close(fig)


def _write_report(
    series_metrics: pd.DataFrame,
    season_metrics: pd.DataFrame,
    weekday_metrics: pd.DataFrame,
    day_type_metrics: pd.DataFrame,
    environment: pd.DataFrame,
    high_frequency: pd.DataFrame,
    high_summary: pd.DataFrame,
    high_series: pd.DataFrame,
    precursors: pd.DataFrame,
    persistence_reason: pd.DataFrame,
    flow_coverage: pd.DataFrame,
    flow_corr: pd.DataFrame,
) -> None:
    val_xgb = series_metrics[
        (series_metrics["split"] == "validation") & (series_metrics["algorithm"] == "xgboost")
    ]
    worst = val_xgb.sort_values(["target", "mae_vs_overall_ratio"], ascending=[True, False]).groupby(
        "target", observed=True
    ).head(2)
    season_xgb = season_metrics[
        (season_metrics["split"] == "validation") & (season_metrics["algorithm"] == "xgboost")
    ]
    worst_season = season_xgb.sort_values(["target", "mae"], ascending=[True, False]).groupby(
        "target", observed=True
    ).head(1)
    val_weekday = weekday_metrics[
        (weekday_metrics["split"] == "validation") & (weekday_metrics["algorithm"] == "xgboost")
    ]
    worst_weekday = val_weekday.sort_values(["target", "mae"], ascending=[True, False]).groupby(
        "target", observed=True
    ).head(1)
    val_day_type = day_type_metrics[
        (day_type_metrics["split"] == "validation") & (day_type_metrics["algorithm"] == "xgboost")
    ]
    val_environment = environment[(environment["split"] == "validation") & (environment["rows"] >= 100)].copy()
    val_environment["abs_spearman"] = val_environment["spearman"].abs()
    environment_top = val_environment.nlargest(6, "abs_spearman")
    high_test = high_summary[high_summary["split"] == "test"].sort_values("mae")
    high_top = high_series[
        (high_series["split"] == "test") & (high_series["algorithm"] == "xgboost")
    ].nlargest(3, "absolute_error_share")
    linked = flow_coverage.groupby("series_id", observed=True).agg(
        rows=("observation_rows", "sum"), linked=("linked_rows", "sum")
    )
    linked["rate"] = linked["linked"] / linked["rows"]
    usable_series = int((linked["linked"] >= 100).sum())
    strongest_flow = flow_corr[flow_corr["rows"] >= 100].copy()
    strongest_flow["abs_spearman"] = strongest_flow["spearman"].abs()
    strongest_flow = strongest_flow.nlargest(5, "abs_spearman")
    validation_high = high_frequency[high_frequency["split"] == "validation"]
    high_by_season = validation_high.groupby("season", observed=True).agg(
        rows=("rows", "sum"), high_rows=("high_rows", "sum")
    ).reset_index()
    high_by_season["high_rate"] = high_by_season["high_rows"] / high_by_season["rows"]
    high_by_weekday = validation_high.groupby("weekday", observed=True).agg(
        rows=("rows", "sum"), high_rows=("high_rows", "sum")
    ).reset_index()
    high_by_weekday["high_rate"] = high_by_weekday["high_rows"] / high_by_weekday["rows"]
    high_by_series_frequency = validation_high.groupby(
        ["series_id", "facility", "plant"], observed=True, dropna=False
    ).agg(rows=("rows", "sum"), high_rows=("high_rows", "sum")).reset_index()
    high_by_series_frequency["high_rate"] = (
        high_by_series_frequency["high_rows"] / high_by_series_frequency["rows"]
    )
    precursor_focus = precursors[
        (precursors["split"] == "validation") & (precursors["lag_days"].isin([1, 3, 7]))
    ]

    lines = [
        "# 계통별·계절별 오차 원인 분석",
        "",
        "## 분석 계약",
        "",
        "- 저장된 LightGBM·XGBoost·LSTM·Persistence 예측만 사용했으며 모델 재학습은 수행하지 않았다.",
        "- 목표별 최종 강우 구성(BOD C/30일, T-N E/30일, T-P C/30일)을 네 알고리즘의 동일 평가키에서 비교했다.",
        "- 개선 가설은 Validation을 중심으로 판단하고, 반복 확인된 Test는 참고 진단으로만 사용했다.",
        "- 공휴일 달력 산출물이 프로젝트에 없어 임의 생성하지 않았으며 요일·평일·주말만 분석했다.",
        "- 당일 기온·강우와 수질의 관계는 사후 연관 분석이다. 예측 운영 시점의 가용성을 보장하거나 인과를 뜻하지 않는다.",
        "",
        "## 계통별 핵심 결과",
        "",
        "| 목표 | 취약 계통 | Validation MAE | 전체 대비 MAE 배율 | 표본 비중 | 총절대오차 비중 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in worst.itertuples():
        lines.append(
            f"| {TARGET_LABELS[row.target]} | {row.facility} / {row.plant} | {row.mae:.4f} | "
            f"{row.mae_vs_overall_ratio:.2f} | {row.row_share:.1%} | {row.absolute_error_share:.1%} |"
        )
    worst_ids = worst[["target", "series_id"]].drop_duplicates()
    model_compare = series_metrics[series_metrics["split"] == "validation"].merge(
        worst_ids, on=["target", "series_id"], how="inner", validate="many_to_one"
    )
    lines += [
        "",
        "취약 계통의 모델별 MAE:",
        "",
        "| 목표 | 계통 | Persistence | LightGBM | XGBoost | LSTM |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (target, facility, plant), group in model_compare.groupby(
        ["target", "facility", "plant"], observed=True, dropna=False
    ):
        values = group.set_index("algorithm")["mae"]
        lines.append(
            f"| {TARGET_LABELS[target]} | {facility} / {plant} | {values.get('persistence', np.nan):.3f} | "
            f"{values.get('lightgbm', np.nan):.3f} | {values.get('xgboost', np.nan):.3f} | "
            f"{values.get('lstm', np.nan):.3f} |"
        )
    lines += [
        "",
        "표본 비중보다 총절대오차 비중이 크고 MAE 배율이 높은 계통은 독립 모델 후보지만, 먼저 계통 식별자와 계절 피처를 추가한 통합 모델에서 개선되는지 확인해야 한다.",
        "",
        "## 계절 결과",
        "",
        "| 목표 | Validation 최악 계절 | 표본 수 | 실제 평균 | 실제 90백분위 | MAE | 평균 편향 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in worst_season.itertuples():
        lines.append(
            f"| {TARGET_LABELS[row.target]} | {row.season} | {int(row.rows)} | {row.actual_mean:.3f} | "
            f"{row.actual_p90:.3f} | {row.mae:.4f} | {row.mean_bias:.4f} |"
        )
    lines += [
        "",
        "Validation 요일별 최악 슬라이스:",
        "",
        "| 목표 | 요일 | 표본 수 | 실제 평균 | MAE | 평균 편향 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in worst_weekday.itertuples():
        lines.append(
            f"| {TARGET_LABELS[row.target]} | {row.weekday} | {int(row.rows)} | {row.actual_mean:.3f} | "
            f"{row.mae:.4f} | {row.mean_bias:.4f} |"
        )
    lines += [
        "",
        "Validation 평일·주말 비교:",
        "",
        "| 목표 | 구분 | 표본 수 | 실제 평균 | MAE | 평균 편향 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in val_day_type.itertuples():
        lines.append(
            f"| {TARGET_LABELS[row.target]} | {row.day_type} | {int(row.rows)} | {row.actual_mean:.3f} | "
            f"{row.mae:.4f} | {row.mean_bias:.4f} |"
        )
    lines += [
        "",
        "계절별 기온·강우와 수질의 Validation Spearman 절대값 상위 관계:",
        "",
        "| 목표 | 계절 | 변수 | 표본 수 | Spearman | 결측률 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in environment_top.itertuples():
        lines.append(
            f"| {TARGET_LABELS[row.target]} | {row.season} | {row.feature} | {int(row.rows)} | "
            f"{row.spearman:.3f} | {row.feature_missing_rate:.1%} |"
        )
    lines += [
        "",
        "계절·요일 슬라이스와 기상 상관은 반복 패턴을 보여 주지만 달력·기상 피처의 인과 효과를 입증하지 않는다. 공휴일 효과는 검증 가능한 달력 스냅샷이 없어 미확인이다.",
        "",
        "## BOD 180mg/L 이상 진단",
        "",
        "| 모델 | Test 표본 | MAE | 과소예측률 | 평균 편향 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in high_test.itertuples():
        lines.append(
            f"| {row.algorithm} | {int(row.rows)} | {row.mae:.3f} | {row.underprediction_rate:.1%} | {row.mean_bias:.3f} |"
        )
    lines += [
        "",
        "Validation 고농도 발생률:",
        "",
        "| 구분 | 값 | 평가 표본 | 고농도 표본 | 발생률 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in high_by_season.itertuples():
        lines.append(f"| 계절 | {row.season} | {int(row.rows)} | {int(row.high_rows)} | {row.high_rate:.1%} |")
    for row in high_by_weekday.sort_values("high_rate", ascending=False).head(3).itertuples():
        lines.append(f"| 요일 상위 | {row.weekday} | {int(row.rows)} | {int(row.high_rows)} | {row.high_rate:.1%} |")
    for row in high_by_series_frequency.sort_values("high_rate", ascending=False).head(3).itertuples():
        lines.append(
            f"| 계통 상위 | {row.facility} / {row.plant} | {int(row.rows)} | {int(row.high_rows)} | {row.high_rate:.1%} |"
        )
    lines += [
        "",
        "고농도 목표일 이전 전조 평균(Validation):",
        "",
        "| 목표 상태 | 이전 일수 | 이전 BOD | 이전 기온 | 이전 강우량 | BOD 가용 표본 | 강우 가용 표본 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in precursor_focus.itertuples():
        state = "고농도" if row.high_bod else "일반"
        lines.append(
            f"| {state} | {int(row.lag_days)} | {row.prior_bod_mean:.2f} | {row.prior_temperature_mean:.2f} | "
            f"{row.prior_rainfall_mean:.2f} | {int(row.prior_bod_available)} | {int(row.prior_rainfall_available)} |"
        )
    reason = persistence_reason[persistence_reason["split"] == "test"].iloc[0]
    lines += [
        "",
        f"Persistence는 Test 고농도 표본의 {reason.persistence_closer_rate:.1%}에서 XGBoost보다 실제값에 가까웠다. "
        f"이전 관측도 180mg/L 이상인 비율은 {reason.previous_day_high_rate:.1%}였고, 실제값은 직전 관측보다 평균 "
        f"{reason.mean_actual_jump_from_previous:.2f}mg/L 높았다. 즉 일부 고농도는 지속성이 있어 직전값이 유리하고, "
        "회귀형 ML 모델은 희소 극값을 평균 쪽으로 강하게 축소한다.",
        "",
        "XGBoost Test 고농도 총절대오차 상위 계통:",
        "",
    ]
    for row in high_top.itertuples():
        lines.append(
            f"- {row.facility} / {row.plant}: 표본 {int(row.rows)}개, MAE {row.mae:.2f}, "
            f"표본 비중 {row.row_share:.1%}, 총절대오차 비중 {row.absolute_error_share:.1%}"
        )
    lines += [
        "",
        "고농도 여부는 목표일 실제 BOD로 정의한 사후 진단 라벨이다. 예측 시점에는 알 수 없으므로 운영 입력이나 라우팅 조건으로 직접 사용할 수 없다.",
        "",
        "## 처리량 활용 가능성",
        "",
        f"- 11개 원 계통 중 관측 100건 이상 정확 연결 처리량이 있는 계통은 {usable_series}개다.",
        "- 1차·2차 처리량은 공식 실제 유입유량이 아니며 상호배타성·채수 경계 일치가 확인되지 않았다.",
        "- 상관은 1차와 2차 원성분을 각각 계산한 탐색 통계이며 합산 가설은 사용하지 않았다.",
        "- 구분없음 계통은 동일 처리장 키가 없어 처리량을 다른 처리장에 배정하지 않았다.",
        "",
        "표본 100건 이상에서 절대 Spearman 상관 상위 결과:",
        "",
        "| 계통 | 목표 | 처리량 변수 | 표본 | Spearman |",
        "|---|---|---|---:|---:|",
    ]
    for row in strongest_flow.itertuples():
        lines.append(
            f"| {row.facility} / {row.plant} | {TARGET_LABELS[row.target]} | {row.flow_variable} | "
            f"{int(row.rows)} | {row.spearman:.3f} |"
        )
    lines += [
        "",
        "## 개선 방향 비교",
        "",
        "| 방향 | 예상 이점 | 데이터 충분성 | 주요 위험 | 우선순위 |",
        "|---|---|---|---|---:|",
        "| A. 계절·요일 피처 | 반복적인 계절 수준·주중 패턴을 저비용으로 반영 | 7개 평가 계통에 충분 | 공휴일 부재, 장기 추세와 계절 효과 혼동 | 2 |",
        "| B. 고농도 가중/비대칭 손실 | 92~97%에 이르는 ML 과소예측을 직접 겨냥 | 고농도 표본은 제한적이나 계통 합산 실험 가능 | 일반 구간 MAE 악화, Validation 과적합 | 1 |",
        "| C. 일부 계통 독립 모델 | 중랑 제2 등 이질 계통의 수준 차이를 분리 | 장기 계통은 가능, 짧거나 종료된 계통은 부족 | 표본 감소와 계통별 튜닝 과적합 | 3 |",
        "",
        "**다음 실험은 B를 우선 추천한다.** 고농도 과소예측이 세 ML 알고리즘에서 공통이고 Persistence가 극값에서 더 강하므로, "
        "Train에서만 산정한 가중치 또는 과소예측 비대칭 손실을 제한된 후보로 비교하는 것이 문제를 가장 직접적으로 검증한다. "
        "선택은 Validation 전체 MAE와 고농도 MAE를 함께 보고, 일반 구간 성능 훼손 한도를 사전에 정해야 한다.",
        "",
        "그 다음 A를 낮은 복잡도의 공통 개선으로 시험하고, C는 A/B 적용 후에도 Validation에서 지속적으로 취약한 장기 계통에만 제한한다.",
        "",
        "## 한계",
        "",
        "- 원본 11개 계통 중 10개가 학습 윈도에 기여했고 평가기간을 충족한 7개만 Validation/Test에 포함된다.",
        "- 서울시 강우 관측소 매핑은 잠정이며 하수처리구역 대표성이 공식 확인되지 않았다.",
        "- Test는 여러 차례 확인된 구간으로 독립 최종 홀드아웃이 아니다.",
        "- 계절·기상·처리량 상관은 교란요인을 통제하지 않은 탐색적 연관이며 인과관계가 아니다.",
    ]
    (REPORTS / f"error_slice_analysis_report_{RUN_ID}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    predictions = _load_common_predictions()
    daily = _daily_with_calendar()
    series_metrics = _series_metrics(predictions)
    season_metrics, weekday_metrics, day_type_metrics, season_series = _slice_metrics(predictions)
    environment = _environment_relationships(daily)
    high_frequency, high_summary, high_series, precursors, persistence_reason = _high_bod_analysis(
        predictions, daily
    )
    flow_coverage, flow_corr = _flow_analysis(daily)

    outputs = {
        "error_slice_series_metrics": series_metrics,
        "error_slice_season_metrics": season_metrics,
        "error_slice_weekday_metrics": weekday_metrics,
        "error_slice_day_type_metrics": day_type_metrics,
        "error_slice_season_series_metrics": season_series,
        "error_slice_environment_relationships": environment,
        "error_slice_high_bod_frequency": high_frequency,
        "error_slice_high_bod_model_metrics": high_summary,
        "error_slice_high_bod_series_metrics": high_series,
        "error_slice_high_bod_precursors": precursors,
        "error_slice_high_bod_persistence_reason": persistence_reason,
        "error_slice_flow_coverage": flow_coverage,
        "error_slice_flow_correlations": flow_corr,
    }
    for name, frame in outputs.items():
        frame.to_csv(REPORTS / f"{name}_{RUN_ID}.csv", index=False, encoding="utf-8-sig")

    _save_figures(series_metrics, season_metrics, high_series, precursors)
    _write_report(
        series_metrics,
        season_metrics,
        weekday_metrics,
        day_type_metrics,
        environment,
        high_frequency,
        high_summary,
        high_series,
        precursors,
        persistence_reason,
        flow_coverage,
        flow_corr,
    )
    print(
        {
            "prediction_rows": len(predictions),
            "evaluation_keys": predictions[KEYS].drop_duplicates().shape[0],
            "series": predictions["series_id"].nunique(),
            "csv_outputs": len(outputs),
            "figures": 4,
        }
    )


if __name__ == "__main__":
    main()
