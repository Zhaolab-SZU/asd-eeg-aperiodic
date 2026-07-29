# Ideal release name: 20b_align_exponent_movie_labels.py
# Original path: scripts/31_align_exponent_with_movie_labels.py
# Note: Align resting exponents with movie labels
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
31_align_exponent_with_movie_labels.py
--------------------------------------
将动态 aperiodic exponent 与电影事件标签对齐，并执行 Group x Event_Type 混合方差分析。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pingouin as pg
import seaborn as sns
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import save_csv  # noqa: E402


EVENT_ORDER = ["baseline", "mental", "pain", "neutral"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对齐 movie 事件并做 mixed ANOVA")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_task_movie.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--events_csv",
        type=str,
        default="data/movie_events.csv",
        help="电影事件标签 CSV 路径",
    )
    return parser.parse_args()


def assign_event_type(center_sec: float, events_df: pd.DataFrame) -> str:
    for _, row in events_df.iterrows():
        if row["onset_sec"] <= center_sec < row["end_sec"]:
            return str(row["event_type"])
    return "neutral"


def format_p(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 1e-4:
        return "<1e-4"
    return f"{value:.4f}"


def get_p_column(df: pd.DataFrame) -> str:
    for col in ("p-unc", "p_unc", "p"):
        if col in df.columns:
            return col
    raise KeyError(f"Mixed ANOVA 结果缺少 p 值列，现有列: {list(df.columns)}")


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    log = setup_logging(cfg, name="align_movie_labels")

    deriv_root = Path(cfg["paths"]["derivatives_root"])
    out_root = Path(cfg["paths"]["outputs_root"])
    ts_csv = deriv_root / "specparam" / "specparam_exponent_timeseries_global.csv"
    events_csv = (PROJECT_ROOT / args.events_csv).resolve()

    if not ts_csv.exists():
        raise FileNotFoundError(f"未找到动态时序文件: {ts_csv}")
    if not events_csv.exists():
        raise FileNotFoundError(f"未找到事件标签文件: {events_csv}")

    ts = pd.read_csv(ts_csv)
    required_ts_cols = {
        "subject_id",
        "group",
        "window_start_sec",
        "window_end_sec",
        "exponent_mean",
    }
    missing_ts = required_ts_cols - set(ts.columns)
    if missing_ts:
        raise ValueError(f"动态时序文件缺少列: {sorted(missing_ts)}")
    ts["subject_id"] = ts["subject_id"].astype(str)
    ts["group"] = ts["group"].astype(str).str.upper()

    # 与当前 config 的 participants_file 对齐，确保可按 matched 队列重跑 movie 主分析。
    part_path = Path(cfg["paths"]["participants_file"])
    if part_path.exists():
        participants = pd.read_csv(part_path)
        if {"subject_id", "group"} <= set(participants.columns):
            participants["subject_id"] = participants["subject_id"].astype(str)
            participants["group"] = participants["group"].astype(str).str.upper()
            if "included_final" in participants.columns:
                inc = pd.to_numeric(participants["included_final"], errors="coerce")
                participants = participants.loc[inc == 1, ["subject_id", "group"]]
            else:
                participants = participants[["subject_id", "group"]]
            participants = participants.drop_duplicates()
            ts = ts.merge(participants, on=["subject_id", "group"], how="inner")
            log.info("按 participants_file 过滤后时序样本: %d 被试", ts["subject_id"].nunique())

    events = pd.read_csv(events_csv)
    required_event_cols = {"onset_sec", "duration_sec", "event_type"}
    missing_ev = required_event_cols - set(events.columns)
    if missing_ev:
        raise ValueError(f"事件文件缺少列: {sorted(missing_ev)}")
    events = events.copy()
    events["event_type"] = events["event_type"].astype(str).str.strip().str.lower()
    events["end_sec"] = events["onset_sec"] + events["duration_sec"]
    events = events.sort_values("onset_sec").reset_index(drop=True)

    aligned = ts.copy()
    aligned["center_sec"] = (aligned["window_start_sec"] + aligned["window_end_sec"]) / 2.0
    aligned["event_type"] = aligned["center_sec"].apply(lambda x: assign_event_type(x, events))
    aligned["event_type"] = pd.Categorical(aligned["event_type"], categories=EVENT_ORDER, ordered=True)

    subject_event_mean = (
        aligned.groupby(["subject_id", "group", "event_type"], observed=True, as_index=False)
        .agg(
            exponent=("exponent_mean", "mean"),
            window_n=("exponent_mean", "size"),
        )
    )

    # mixed_anova 需每个被试在每个 condition 都有值，先做 complete-case
    pivot = subject_event_mean.pivot_table(
        index=["subject_id", "group"],
        columns="event_type",
        values="exponent",
        aggfunc="mean",
        observed=True,
    )
    available_event_cols = [ev for ev in EVENT_ORDER if ev in pivot.columns]
    if not available_event_cols:
        raise RuntimeError("未找到可用事件列（baseline/mental/pain/neutral）")
    complete = pivot[available_event_cols].dropna().reset_index()
    complete_long = complete.melt(
        id_vars=["subject_id", "group"],
        value_vars=available_event_cols,
        var_name="event_type",
        value_name="exponent",
    )

    anova = pg.mixed_anova(
        data=complete_long,
        dv="exponent",
        within="event_type",
        between="group",
        subject="subject_id",
    )

    stats_dir = deriv_root / "stats"
    fig_dir = out_root / "figures"
    stats_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    save_csv(aligned, stats_dir / "movie_event_aligned_timeseries.csv")
    save_csv(subject_event_mean, stats_dir / "movie_event_subject_condition_means.csv")
    save_csv(complete_long, stats_dir / "movie_event_subject_condition_means_complete.csv")
    save_csv(anova, stats_dir / "movie_event_mixed_anova.csv")

    plt.figure(figsize=(9, 5.5))
    sns.violinplot(
        data=complete_long,
        x="event_type",
        y="exponent",
        hue="group",
        cut=0,
        inner=None,
        dodge=True,
        linewidth=1,
    )
    sns.boxplot(
        data=complete_long,
        x="event_type",
        y="exponent",
        hue="group",
        dodge=True,
        width=0.25,
        showcaps=True,
        boxprops={"facecolor": "white", "alpha": 0.7},
        showfliers=False,
        whiskerprops={"linewidth": 1},
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], title="Group", frameon=True)
    plt.xlabel("Event Type")
    plt.ylabel("Aperiodic Exponent (ROI mean)")
    plt.title("Movie Events x Group: Exponent Distribution")

    src = anova["Source"].astype(str).str.lower()
    p_col = get_p_column(anova)
    p_group = anova.loc[src == "group", p_col].iloc[0] if (src == "group").any() else float("nan")
    p_event = anova.loc[src == "event_type", p_col].iloc[0] if (src == "event_type").any() else float("nan")
    p_inter = (
        anova.loc[src.str.contains("interaction"), p_col].iloc[0]
        if src.str.contains("interaction").any()
        else float("nan")
    )
    txt = (
        f"Mixed ANOVA p-values\n"
        f"Group: {format_p(p_group)}\n"
        f"Event: {format_p(p_event)}\n"
        f"Group x Event: {format_p(p_inter)}"
    )
    plt.gca().text(
        0.98,
        0.98,
        txt,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#666"},
    )
    plt.tight_layout()
    fig_path = fig_dir / "movie_event_exponent_group_violin"
    plt.savefig(fig_path.with_suffix(".png"), dpi=180)
    plt.savefig(fig_path.with_suffix(".pdf"))
    plt.close()

    log.info("对齐完成: %s", stats_dir / "movie_event_aligned_timeseries.csv")
    log.info("Mixed ANOVA 完成: %s", stats_dir / "movie_event_mixed_anova.csv")
    log.info("图已保存: %s", fig_path.with_suffix(".png"))
    log.info("Complete-case 被试数: %d", complete["subject_id"].nunique())
    log.info(
        "p-values => Group=%s, Event=%s, Interaction=%s",
        format_p(p_group),
        format_p(p_event),
        format_p(p_inter),
    )


if __name__ == "__main__":
    main()
