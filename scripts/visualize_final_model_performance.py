"""저장된 예측값으로 포트폴리오용 최종 성능 그림과 보고서를 생성한다.

모델 학습, API 호출, 원본 데이터 처리를 수행하지 않는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.portfolio_visualization import (  # noqa: E402
    KEYS, assert_prediction_alignment, consecutive_segments, metric_row,
    select_dynamic_validation_window,
)

STAMP = "2026-11-14"
OUT = ROOT / "results" / "figures" / f"final_model_portfolio_{STAMP}"
REPORTS = ROOT / "reports"
TARGET_LABEL = {"bod_mg_l": "BOD", "tn_mg_l": "T-N", "tp_mg_l": "T-P"}
UNIT = "mg/L"
FINAL_NAME = {"bod_mg_l": "season_weekday", "tn_mg_l": "weekday", "tp_mg_l": "weekday"}
COLORS = {"actual": "#202124", "xgboost": "#0072B2", "persistence": "#D55E00"}
DISCLAIMER = "Test: 반복 확인된 참고 평가 구간"


def setup_font() -> str:
    candidates = ["Malgun Gothic", "맑은 고딕", "Noto Sans CJK KR", "NanumGothic"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((x for x in candidates if x in available), "DejaVu Sans")
    plt.rcParams.update({"font.family": chosen, "axes.unicode_minus": False, "figure.dpi": 110})
    return chosen


def save(fig, stem: str, representative: bool = False) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    paths = [png]
    if representative:
        svg = OUT / f"{stem}.svg"
        fig.savefig(svg, bbox_inches="tight", facecolor="white")
        paths.append(svg)
    plt.close(fig)
    return paths


def load_final() -> pd.DataFrame:
    p = ROOT / "results/predictions/calendar_weighted_xgboost_predictions_2026-11-13.parquet"
    raw = pd.read_parquet(p)
    raw["target_date"] = pd.to_datetime(raw["target_date"])
    pieces = []
    for target, name in FINAL_NAME.items():
        x = raw[(raw.target == target) & (raw.model_name.isin([name, "persistence"]))].copy()
        x["model"] = x.model_name.map({name: "xgboost", "persistence": "persistence"})
        pieces.append(x)
    data = pd.concat(pieces, ignore_index=True)
    mapping = pd.read_csv(
        REPORTS / "calendar_weighted_xgboost_series_metrics_2026-11-13.csv",
        encoding="utf-8-sig",
    )[["series_id", "facility", "plant"]].drop_duplicates("series_id")
    data = data.drop(columns=["facility", "plant"], errors="ignore").merge(
        mapping, on="series_id", how="left", validate="many_to_one"
    )
    data["series_label"] = data["facility"].fillna(data["series_id"]) + " · " + data["plant"].fillna("")
    for (_, target), group in data.groupby(["model", "target"]):
        if group.duplicated(KEYS).any():
            raise ValueError(f"중복 평가 키: {target}")
    for target in FINAL_NAME:
        assert_prediction_alignment(
            data[(data.target == target) & (data.model == "xgboost")],
            data[(data.target == target) & (data.model == "persistence")],
        )
    return data


def metrics(data: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in data.groupby(groups, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(groups, key)) | metric_row(g.actual, g.prediction))
    return pd.DataFrame(rows)


def annotation(ax, row: dict) -> None:
    ax.text(.01, .98, f"MAE {row['mae']:.3f} | RMSE {row['rmse']:.3f} | R² {row['r2']:.3f} | n={row['n']:,}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox={"facecolor": "white", "alpha": .82, "edgecolor": "#cccccc"})


def plot_lines(ax, g: pd.DataFrame) -> None:
    for model in ["actual", "xgboost", "persistence"]:
        source = g[g.model.eq("xgboost")] if model == "actual" else g[g.model.eq(model)]
        value = "actual" if model == "actual" else "prediction"
        for segment in consecutive_segments(source):
            ax.plot(segment.target_date, segment[value], color=COLORS[model], lw=1.15 if model == "actual" else .9,
                    alpha=.92 if model != "persistence" else .7,
                    label={"actual": "실제", "xgboost": "최종 XGBoost", "persistence": "Persistence"}[model])
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=7, ncol=3, loc="upper right")


def time_series_figures(data: pd.DataFrame, manifest: list[dict]) -> dict[str, tuple[str, str]]:
    windows = {}
    for target in FINAL_NAME:
        target_data = data[data.target.eq(target)]
        start, end = select_dynamic_validation_window(target_data[target_data.model.eq("xgboost")], 90)
        windows[target] = (str(start.date()), str(end.date()))
        for split in ["validation", "test"]:
            x = target_data[target_data.split.eq(split)]
            series = sorted(x.series_id.unique())
            fig, axes = plt.subplots(len(series), 1, figsize=(15, 2.2 * len(series)), sharex=True)
            for ax, sid in zip(np.atleast_1d(axes), series):
                g = x[x.series_id.eq(sid)]
                plot_lines(ax, g)
                row = metric_row(g[g.model.eq("xgboost")].actual, g[g.model.eq("xgboost")].prediction)
                annotation(ax, row); ax.set_title(str(g.series_label.iloc[0]), fontsize=9, loc="left"); ax.set_ylabel(UNIT)
            title = f"{TARGET_LABEL[target]} 계통별 전체 {split.title()} 예측"
            if split == "test": title += f" ({DISCLAIMER})"
            fig.suptitle(title, fontsize=14); axes[-1].set_xlabel("목표일"); fig.tight_layout()
            stem = f"timeseries_all_{target}_{split}"
            for p in save(fig, stem, representative=True): manifest.append({"file": str(p.relative_to(ROOT)), "kind": "전체 시계열", "target": target, "split": split})

            if split == "validation": zstart, zend = start, end
            else:
                zstart = pd.Timestamp(year=2025, month=start.month, day=start.day)
                zend = zstart + pd.Timedelta(days=89)
            z = x[x.target_date.between(zstart, zend)]
            fig, axes = plt.subplots(len(series), 1, figsize=(15, 2.2 * len(series)), sharex=True)
            for ax, sid in zip(np.atleast_1d(axes), series):
                g = z[z.series_id.eq(sid)]; plot_lines(ax, g)
                if not g.empty: annotation(ax, metric_row(g[g.model.eq("xgboost")].actual, g[g.model.eq("xgboost")].prediction))
                ax.set_title(str(g.series_label.iloc[0]) if not g.empty else str(sid), fontsize=9, loc="left"); ax.set_ylabel(UNIT)
            fig.suptitle(f"{TARGET_LABEL[target]} 90일 확대 ({zstart.date()}~{zend.date()}, Validation 실제 변화량 기준)", fontsize=13)
            axes[-1].set_xlabel("목표일"); fig.tight_layout()
            stem = f"timeseries_zoom_{target}_{split}"
            for p in save(fig, stem, representative=(split == "validation")): manifest.append({"file": str(p.relative_to(ROOT)), "kind": "90일 확대", "target": target, "split": split})

            for sid in series:
                g = x[x.series_id.eq(sid)]
                fig, ax = plt.subplots(figsize=(14, 4)); plot_lines(ax, g)
                annotation(ax, metric_row(g[g.model.eq("xgboost")].actual, g[g.model.eq("xgboost")].prediction))
                ax.set(title=f"{TARGET_LABEL[target]} | {g.series_label.iloc[0]} | {split.title()}", xlabel="목표일", ylabel=f"농도 ({UNIT})")
                fig.tight_layout(); safe = str(sid).replace("/", "_").replace(" ", "_").replace("|", "_")
                for p in save(fig, f"series_{target}_{split}_{safe}"): manifest.append({"file": str(p.relative_to(ROOT)), "kind": "개별 계통", "target": target, "split": split, "series_id": sid})
    return windows


def scatter_residual_figures(data: pd.DataFrame, manifest: list[dict]) -> None:
    for split in ["validation", "test"]:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        for col, target in enumerate(FINAL_NAME):
            g = data[(data.target == target) & (data.model == "xgboost") & (data.split == split)].copy()
            row = metric_row(g.actual, g.prediction)
            ax = axes[0, col]; ax.scatter(g.actual, g.prediction, s=7, alpha=.22, color=COLORS["xgboost"], rasterized=True)
            lo, hi = min(g.actual.min(), g.prediction.min()), max(g.actual.max(), g.prediction.max()); ax.plot([lo, hi], [lo, hi], "--", c="#666")
            ax.set(title=f"{TARGET_LABEL[target]} 실제-예측", xlabel=f"실제 ({UNIT})", ylabel=f"예측 ({UNIT})"); annotation(ax, row)
            g["residual"] = g.prediction - g.actual
            ax = axes[1, col]; ax.scatter(g.actual, g.residual, s=7, alpha=.22, c=pd.factorize(g.series_id)[0], cmap="tab10", rasterized=True)
            ax.axhline(0, ls="--", c="#666"); ax.set(title=f"{TARGET_LABEL[target]} 잔차 (예측-실제)", xlabel=f"실제 ({UNIT})", ylabel=f"잔차 ({UNIT})")
        fig.suptitle(f"최종 XGBoost 산점도와 잔차 | {split.title()}" + (f" | {DISCLAIMER}" if split == "test" else ""), fontsize=15)
        fig.tight_layout()
        for p in save(fig, f"scatter_residual_{split}", representative=True): manifest.append({"file": str(p.relative_to(ROOT)), "kind": "산점도·잔차", "split": split})


def high_bod_figure(data: pd.DataFrame, manifest: list[dict]) -> pd.DataFrame:
    bod = data[data.target.eq("bod_mg_l")].copy()
    rows = []
    for split in ["validation", "test"]:
        for model, g in bod[(bod.split == split) & (bod.actual >= 180)].groupby("model"):
            row = metric_row(g.actual, g.prediction); row.update({"split": split, "model": model, "underprediction_pct": float((g.prediction < g.actual).mean() * 100)})
            rows.append(row)
        events = bod[(bod.split == split) & (bod.model == "xgboost") & (bod.actual >= 180)].sort_values("actual", ascending=False).head(6)
        fig, axes = plt.subplots(len(events), 1, figsize=(14, 2.5 * max(len(events), 1)))
        for ax, event in zip(np.atleast_1d(axes), events.itertuples()):
            g = bod[(bod.split == split) & (bod.series_id == event.series_id) & bod.target_date.between(event.target_date - pd.Timedelta(7, "d"), event.target_date + pd.Timedelta(7, "d"))]
            plot_lines(ax, g); ax.axvline(event.target_date, color="#CC79A7", ls=":"); ax.axhline(180, color="#999", ls="--")
            label = g.series_label.iloc[0] if not g.empty else event.series_id
            ax.set_title(f"{label} | 고농도일 {event.target_date.date()} | 실제 {event.actual:.1f}", fontsize=9, loc="left"); ax.set_ylabel(UNIT)
        fig.suptitle(f"BOD 고농도(사후 실제값 ≥180) 전후 7일 | {split.title()}", fontsize=14); axes[-1].set_xlabel("목표일"); fig.tight_layout()
        for p in save(fig, f"high_bod_context_{split}", representative=True): manifest.append({"file": str(p.relative_to(ROOT)), "kind": "BOD 고농도", "split": split})
    return pd.DataFrame(rows)


def comparison_figures(manifest: list[dict]) -> None:
    alg = pd.read_csv(REPORTS / "model_algorithm_selected_metrics.csv")
    # 저장 표의 명칭 차이를 정규화한다.
    alg.columns = [c.lower() for c in alg.columns]
    split_col = "split"; target_col = "target"; model_col = "algorithm" if "algorithm" in alg else "model"
    for split in ["validation", "test"]:
        x = alg[alg[split_col].eq(split)].copy()
        if x.empty: continue
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        for row, metric in enumerate(["mae", "rmse", "r2"]):
            for col, target in enumerate(FINAL_NAME):
                g = x[x[target_col].eq(target)].sort_values(model_col)
                axes[row, col].bar(g[model_col].astype(str), g[metric], color="#56B4E9")
                axes[row, col].set_title(f"{TARGET_LABEL[target]} {metric.upper()}"); axes[row, col].tick_params(axis="x", rotation=25)
        fig.suptitle(f"동일 표본 알고리즘 비교 | {split.title()}", fontsize=15); fig.tight_layout()
        for p in save(fig, f"algorithm_metrics_{split}", representative=(split == "validation")): manifest.append({"file": str(p.relative_to(ROOT)), "kind": "알고리즘 비교", "split": split})

    effects = []
    for filename, label in [("final_model_rainfall_effect_2026-11-13.csv", "서울시 강우량"), ("calendar_weighted_xgboost_metrics_2026-11-13.csv", "계절·요일"), ("ss_model_comparison_2026-11-13.csv", "SS")]:
        p = REPORTS / filename
        if p.exists():
            d = pd.read_csv(p); d["source"] = label; effects.append(d)
    # 서로 다른 표의 수치 자체를 억지로 합치지 않고, 각 표의 MAE 열 분포를 별도 패널로 표시한다.
    fig, axes = plt.subplots(1, len(effects), figsize=(6 * max(len(effects), 1), 5))
    for ax, d in zip(np.atleast_1d(axes), effects):
        mae_cols = [c for c in d.columns if "mae" in c.lower() and pd.api.types.is_numeric_dtype(d[c])]
        vals = [d[c].dropna().mean() for c in mae_cols]
        ax.bar(range(len(vals)), vals, color="#009E73"); ax.set_xticks(range(len(vals)), [c.replace("_", "\n") for c in mae_cols], rotation=30, ha="right", fontsize=7)
        ax.set_title(d.source.iloc[0]); ax.set_ylabel("저장 지표 평균 MAE (직접 우열 비교 금지)")
    fig.suptitle("입력 피처 실험 요약 — 각 실험 내부 공통 표본 기준", fontsize=14); fig.tight_layout()
    for p in save(fig, "feature_experiment_summary", representative=True): manifest.append({"file": str(p.relative_to(ROOT)), "kind": "피처 실험 비교"})


def main() -> None:
    font = setup_font(); data = load_final(); manifest: list[dict] = []
    overall = metrics(data, ["split", "target", "model"])
    series = metrics(data, ["split", "target", "model", "series_id", "facility", "plant"])
    pmae = overall[overall.model.eq("persistence")][["split", "target", "mae"]].rename(columns={"mae": "persistence_mae"})
    overall = overall.merge(pmae, on=["split", "target"], how="left")
    overall["persistence_improvement_pct"] = (overall.persistence_mae - overall.mae) / overall.persistence_mae * 100
    windows = time_series_figures(data, manifest)
    scatter_residual_figures(data, manifest)
    high = high_bod_figure(data, manifest)
    comparison_figures(manifest)
    overall.to_csv(REPORTS / f"final_model_visualization_metrics_{STAMP}.csv", index=False, encoding="utf-8-sig")
    series.to_csv(REPORTS / f"final_model_visualization_series_metrics_{STAMP}.csv", index=False, encoding="utf-8-sig")
    high.to_csv(REPORTS / f"final_model_visualization_high_bod_{STAMP}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(manifest).to_csv(REPORTS / f"final_model_visualization_manifest_{STAMP}.csv", index=False, encoding="utf-8-sig")
    summary = {"font": font, "image_files": len(manifest), "png": sum(x["file"].endswith(".png") for x in manifest), "svg": sum(x["file"].endswith(".svg") for x in manifest), "zoom_windows": windows,
               "evaluation_series": int(data.series_id.nunique()), "key_duplicates": int(data.duplicated(KEYS + ["model"]).sum())}
    (REPORTS / f"final_model_visualization_summary_{STAMP}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
