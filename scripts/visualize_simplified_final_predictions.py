"""실제값과 최종 XGBoost 두 선만 사용하는 간소화 시계열 생성기."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.portfolio_visualization import (  # noqa: E402
    SIMPLE_LINE_COLORS, consecutive_segments, metric_row,
    select_dynamic_validation_window, two_line_series,
)

STAMP = "2026-11-14"
OUT = ROOT / "results/figures" / f"final_model_simplified_{STAMP}"
REPORTS = ROOT / "reports"
FINAL_NAME = {"bod_mg_l": "season_weekday", "tn_mg_l": "weekday", "tp_mg_l": "weekday"}
LABEL = {"bod_mg_l": "BOD", "tn_mg_l": "T-N", "tp_mg_l": "T-P"}


def setup() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = "Malgun Gothic" if "Malgun Gothic" in available else "DejaVu Sans"
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})
    OUT.mkdir(parents=True, exist_ok=True)
    return font


def load_predictions() -> pd.DataFrame:
    raw = pd.read_parquet(ROOT / "results/predictions/calendar_weighted_xgboost_predictions_2026-11-13.parquet")
    raw["target_date"] = pd.to_datetime(raw["target_date"])
    parts = []
    for target, model_name in FINAL_NAME.items():
        x = raw[(raw.target == target) & (raw.model_name == model_name)].copy()
        x["model"] = "xgboost"
        parts.append(x)
    data = pd.concat(parts, ignore_index=True)
    mapping = pd.read_csv(REPORTS / "calendar_weighted_xgboost_series_metrics_2026-11-13.csv", encoding="utf-8-sig")
    mapping = mapping[["series_id", "facility", "plant"]].drop_duplicates("series_id")
    data = data.drop(columns=["facility", "plant"], errors="ignore").merge(mapping, on="series_id", how="left", validate="many_to_one")
    data["series_label"] = data.facility + " · " + data.plant
    if data.duplicated(["split", "target", "series_id", "target_date"]).any():
        raise ValueError("최종 예측 평가 키 중복")
    return data


def plot_two_lines(ax, frame: pd.DataFrame) -> None:
    lines = two_line_series(frame)
    for segment in consecutive_segments(lines):
        ax.plot(segment.target_date, segment.actual, color=SIMPLE_LINE_COLORS["actual"], lw=1.25, label="실제 관측값")
        ax.plot(segment.target_date, segment.xgboost, color=SIMPLE_LINE_COLORS["xgboost"], lw=1.15, label="최종 XGBoost")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8, frameon=False)
    ax.grid(True, color="#d9d9d9", alpha=.45, linewidth=.6)
    ax.set_ylabel("농도 (mg/L)")


def subtitle(frame: pd.DataFrame) -> str:
    m = metric_row(frame.actual, frame.prediction)
    return f"MAE {m['mae']:.3f} · R² {m['r2']:.3f} · n={m['n']:,}"


def save(fig, stem: str) -> str:
    path = OUT / f"{stem}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path.relative_to(ROOT))


def generate(data: pd.DataFrame) -> tuple[list[dict], dict]:
    manifest: list[dict] = []
    windows = {}
    for target in FINAL_NAME:
        target_data = data[data.target == target]
        start, end = select_dynamic_validation_window(target_data, 90)
        windows[target] = [str(start.date()), str(end.date())]
        for split in ["validation", "test"]:
            x = target_data[target_data.split == split]
            series = sorted(x.series_id.unique())

            fig, axes = plt.subplots(len(series), 1, figsize=(14, 2.15 * len(series)), sharex=True)
            for ax, sid in zip(axes, series):
                g = x[x.series_id == sid]
                plot_two_lines(ax, g)
                ax.set_title(f"{g.series_label.iloc[0]}  |  {subtitle(g)}", fontsize=9, loc="left")
            axes[-1].set_xlabel("목표일")
            note = " · 반복 확인된 참고 평가" if split == "test" else ""
            fig.suptitle(f"{LABEL[target]} 실제값과 최종 XGBoost · {split.title()}{note}", fontsize=14)
            fig.tight_layout()
            manifest.append({"file": save(fig, f"overview_{target}_{split}"), "kind": "전체 계통", "target": target, "split": split})

            for sid in series:
                g = x[x.series_id == sid]
                fig, ax = plt.subplots(figsize=(13, 4.5))
                plot_two_lines(ax, g)
                ax.set_xlabel("목표일")
                ax.set_title(f"{LABEL[target]} · {g.series_label.iloc[0]} · {split.title()}\n{subtitle(g)}", fontsize=12, loc="left")
                fig.tight_layout()
                manifest.append({"file": save(fig, f"series_{target}_{split}_{sid}"), "kind": "개별 계통", "target": target, "split": split, "series_id": sid, "series_label": g.series_label.iloc[0]})

            if split == "validation":
                zstart, zend = start, end
            else:
                zstart = pd.Timestamp(2025, start.month, start.day)
                zend = zstart + pd.Timedelta(days=89)
            zoom = x[x.target_date.between(zstart, zend)]
            fig, axes = plt.subplots(len(series), 1, figsize=(14, 2.15 * len(series)), sharex=True)
            for ax, sid in zip(axes, series):
                g = zoom[zoom.series_id == sid]
                if not g.empty:
                    plot_two_lines(ax, g)
                    ax.set_title(f"{g.series_label.iloc[0]}  |  {subtitle(g)}", fontsize=9, loc="left")
            axes[-1].set_xlabel("목표일")
            fig.suptitle(f"{LABEL[target]} 90일 확대 · {zstart.date()}~{zend.date()} · {split.title()}", fontsize=14)
            fig.tight_layout()
            manifest.append({"file": save(fig, f"zoom_{target}_{split}"), "kind": "90일 확대", "target": target, "split": split, "start": str(zstart.date()), "end": str(zend.date())})

    bod = data[data.target == "bod_mg_l"]
    for split in ["validation", "test"]:
        x = bod[bod.split == split]
        events = x[x.actual >= 180].sort_values("actual", ascending=False).head(4)
        fig, axes = plt.subplots(len(events), 1, figsize=(14, 2.8 * len(events)))
        for ax, event in zip(axes, events.itertuples()):
            g = x[(x.series_id == event.series_id) & x.target_date.between(event.target_date - pd.Timedelta(days=7), event.target_date + pd.Timedelta(days=7))]
            plot_two_lines(ax, g)
            ax.axvline(event.target_date, color="#777777", ls=":", lw=.9)
            ax.set_title(f"{g.series_label.iloc[0]} · 고농도일 {event.target_date.date()} · 실제 {event.actual:.1f} mg/L", fontsize=9, loc="left")
        axes[-1].set_xlabel("목표일")
        fig.suptitle(f"BOD 고농도(사후 실제값 ≥180 mg/L) 전후 7일 · {split.title()}", fontsize=14)
        fig.tight_layout()
        manifest.append({"file": save(fig, f"high_bod_context_{split}"), "kind": "BOD 고농도", "target": "bod_mg_l", "split": split})
    return manifest, windows


def main() -> None:
    font = setup()
    data = load_predictions()
    manifest, windows = generate(data)
    pd.DataFrame(manifest).to_csv(REPORTS / f"final_model_simplified_manifest_{STAMP}.csv", index=False, encoding="utf-8-sig")
    summary = {"font": font, "png_count": len(manifest), "series_count": int(data.series_id.nunique()), "key_duplicates": int(data.duplicated(["split", "target", "series_id", "target_date"]).sum()), "zoom_windows": windows, "lines": ["actual", "xgboost"]}
    (REPORTS / f"final_model_simplified_summary_{STAMP}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
