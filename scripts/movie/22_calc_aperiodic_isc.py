# Ideal release name: 22_calc_aperiodic_isc.py
# Original path: scripts/68_compute_aperiodic_isc.py
# Note: TD-template and within-group Aperiodic-ISC
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
68_compute_aperiodic_isc.py
---------------------------
Time-Resolved Aperiodic-ISC：基于滑动窗 exponent(t) 序列的被试间同步性。

流程：
1) 读取 specparam_exponent_timeseries_global.csv（2 s 窗 / 0.5 s 步长）
2) 计算 TD-template ISC（复现主分析口径）
3) 计算组内 LOO Aperiodic-ISC（TD↔TD 模板，ASD↔ASD 模板）
4) 计算组内 pairwise 均值 ISC 与 time-resolved ISC(t) 曲线
5) 组间 Welch t 检验 + 可视化

isc-mode:
  segmented — mental/pain/neutral 事件片段（主研究 naturalistic movie）
  overall   — 全片时间轴，无事件标签（外部电影验证等）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.aperiodic_isc_analysis import (  # noqa: E402
    EVENT_TYPES_ALL,
    build_concat_keys,
    build_concat_keys_overall,
    build_subject_concat_series,
    compute_pairwise_mean_isc,
    compute_td_template_isc,
    compute_time_resolved_group_isc,
    compute_within_group_isc,
    load_exponent_timeseries,
    summarize_group_isc_tests,
)
from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import save_csv  # noqa: E402
from src.stats_utils import fdr_correction  # noqa: E402

EVENT_TYPES_OVERALL = ("overall",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute time-resolved Aperiodic-ISC")
    parser.add_argument("--config", type=str, default="config/config_task_movie.yaml")
    parser.add_argument("--events_csv", type=str, default="data/movie_events.csv")
    parser.add_argument("--min_overlap_points", type=int, default=10)
    parser.add_argument("--local_window_bins", type=int, default=11, help="ISC(t) 局部窗口 bin 数（默认 ±5×0.5s）")
    parser.add_argument(
        "--movie_analysis_csv",
        type=str,
        default="derivatives_task_movie/participants_analysis.csv",
    )
    parser.add_argument(
        "--movie_specparam_qc_csv",
        type=str,
        default="derivatives_task_movie/specparam/specparam_qc_summary_subject.csv",
    )
    parser.add_argument(
        "--isc-mode",
        type=str,
        choices=("segmented", "overall"),
        default=None,
        help="segmented=按 movie_events 片段；overall=全片时间轴（外部 ThePresent）",
    )
    return parser.parse_args()


def _resolve_isc_mode(args: argparse.Namespace, cfg: dict) -> str:
    if args.isc_mode is not None:
        return str(args.isc_mode).strip().lower()
    return str(cfg.get("movie", {}).get("isc_mode", "segmented")).strip().lower()


def _add_fdr(stats_df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = stats_df.copy()
    out["source"] = source
    if out.empty:
        out["fdr_p"] = []
        out["significant_fdr"] = []
        return out
    reject, p_adj = fdr_correction(pd.to_numeric(out["p_value"], errors="coerce").to_numpy(dtype=float))
    out["fdr_p"] = p_adj
    out["significant_fdr"] = reject
    return out


def _plot_subject_boxplot(isc_df: pd.DataFrame, stats_df: pd.DataFrame, title: str, out_path: Path) -> None:
    plt.figure(figsize=(9, 5.5))
    sns.boxplot(data=isc_df, x="event_type", y="isc_z", hue="group", showfliers=False, width=0.6)
    sns.stripplot(
        data=isc_df,
        x="event_type",
        y="isc_z",
        hue="group",
        dodge=True,
        alpha=0.55,
        size=3.5,
        linewidth=0,
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], title="Group", frameon=True)
    plt.xlabel("Condition")
    plt.ylabel("Aperiodic-ISC (Fisher z)")
    plt.title(title)
    txt_lines = []
    for _, r in stats_df.iterrows():
        p_txt = "NA" if pd.isna(r["p_value"]) else f"{r['p_value']:.4g}"
        txt_lines.append(f"{r['event_type']}: p={p_txt}")
    plt.gca().text(
        0.98,
        0.98,
        "\n".join(txt_lines),
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#666"},
    )
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=180)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def _plot_time_resolved(tr_df: pd.DataFrame, event_types: tuple[str, ...], out_path: Path) -> None:
    fig, axes = plt.subplots(len(event_types), 1, figsize=(11, 3.2 * len(event_types)), sharex=False)
    if len(event_types) == 1:
        axes = [axes]
    colors = {"TD": "#2166ac", "ASD": "#b2182b"}
    for ax, ev in zip(axes, event_types):
        sub = tr_df[tr_df["event_type"] == ev].copy()
        for grp in ("TD", "ASD"):
            g = sub[sub["group"] == grp].sort_values("center_sec")
            ax.plot(g["center_sec"], g["mean_loo_z"], label=grp, color=colors[grp], linewidth=1.4)
            ax.fill_between(
                g["center_sec"],
                g["mean_loo_z"] - 0.05,
                g["mean_loo_z"] + 0.05,
                color=colors[grp],
                alpha=0.12,
            )
        ax.axhline(0.0, color="#888", linewidth=0.8, linestyle="--")
        ax.set_title(f"Time-resolved within-group Aperiodic-ISC — {ev}")
        ax.set_ylabel("Mean LOO z")
        ax.legend(frameon=True, loc="upper right")
    axes[-1].set_xlabel("Movie time (s)")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _run_segmented_mode(
    ts: pd.DataFrame,
    ts_bins: np.ndarray,
    events_path: Path,
    args: argparse.Namespace,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if not events_path.exists():
        raise FileNotFoundError(f"segmented 模式需要事件文件: {events_path}")
    events = pd.read_csv(events_path)
    concat_keys = build_concat_keys(ts_bins=ts_bins, events=events)
    concat_keys_overall = build_concat_keys_overall(ts_bins=ts_bins)
    subject_concat = build_subject_concat_series(ts=ts, keys_df=concat_keys)
    subject_concat_overall = build_subject_concat_series(ts=ts, keys_df=concat_keys_overall)

    td_template_isc = compute_td_template_isc(
        subject_concat=subject_concat,
        min_overlap_points=args.min_overlap_points,
        event_types=EVENT_TYPES_ALL,
    )
    within_group_isc = compute_within_group_isc(
        subject_concat=subject_concat,
        min_overlap_points=args.min_overlap_points,
        event_types=EVENT_TYPES_ALL,
    )
    within_group_overall = compute_within_group_isc(
        subject_concat=subject_concat_overall,
        min_overlap_points=args.min_overlap_points,
        event_types=EVENT_TYPES_OVERALL,
    )
    pairwise_isc = compute_pairwise_mean_isc(
        subject_concat=subject_concat,
        min_overlap_points=args.min_overlap_points,
        event_types=EVENT_TYPES_ALL,
    )
    time_resolved = compute_time_resolved_group_isc(
        subject_concat=subject_concat,
        local_window_bins=args.local_window_bins,
        min_overlap_points=max(5, args.min_overlap_points // 2),
        event_types=EVENT_TYPES_ALL,
    )
    return (
        concat_keys,
        subject_concat,
        td_template_isc,
        within_group_isc,
        within_group_overall,
        pairwise_isc,
        time_resolved,
        summarize_group_isc_tests(td_template_isc, EVENT_TYPES_ALL),
        summarize_group_isc_tests(within_group_isc, EVENT_TYPES_ALL),
    )


def _run_overall_mode(
    ts: pd.DataFrame,
    ts_bins: np.ndarray,
    args: argparse.Namespace,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    concat_keys = build_concat_keys_overall(ts_bins=ts_bins)
    subject_concat = build_subject_concat_series(ts=ts, keys_df=concat_keys)

    td_template_isc = compute_td_template_isc(
        subject_concat=subject_concat,
        min_overlap_points=args.min_overlap_points,
        event_types=EVENT_TYPES_OVERALL,
    )
    within_group_isc = compute_within_group_isc(
        subject_concat=subject_concat,
        min_overlap_points=args.min_overlap_points,
        event_types=EVENT_TYPES_OVERALL,
    )
    pairwise_isc = compute_pairwise_mean_isc(
        subject_concat=subject_concat,
        min_overlap_points=args.min_overlap_points,
        event_types=EVENT_TYPES_OVERALL,
    )
    time_resolved = compute_time_resolved_group_isc(
        subject_concat=subject_concat,
        local_window_bins=args.local_window_bins,
        min_overlap_points=max(5, args.min_overlap_points // 2),
        event_types=EVENT_TYPES_OVERALL,
    )
    empty_overall = within_group_isc.iloc[0:0].copy()
    return (
        concat_keys,
        subject_concat,
        td_template_isc,
        within_group_isc,
        empty_overall,
        pairwise_isc,
        time_resolved,
        summarize_group_isc_tests(td_template_isc, EVENT_TYPES_OVERALL),
        summarize_group_isc_tests(within_group_isc, EVENT_TYPES_OVERALL),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    log = setup_logging(cfg, name="aperiodic_isc")
    isc_mode = _resolve_isc_mode(args, cfg)
    if isc_mode not in {"segmented", "overall"}:
        raise ValueError(f"未知 isc_mode: {isc_mode}")

    deriv = Path(cfg["paths"]["derivatives_root"])
    out_root = Path(cfg["paths"]["outputs_root"])
    stats_dir = deriv / "stats" / "aperiodic_isc"
    fig_dir = out_root / "figures" / "aperiodic_isc"
    stats_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sw = cfg.get("psd", {}).get("sliding_window", {})
    log.info(
        "ISC 模式: %s | 滑动窗 window=%.1fs step=%.1fs",
        isc_mode,
        float(sw.get("window_sec", np.nan)),
        float(sw.get("step_sec", np.nan)),
    )

    ts_path = deriv / "specparam" / "specparam_exponent_timeseries_global.csv"
    if not ts_path.exists():
        raise FileNotFoundError(f"未找到动态序列: {ts_path}")

    events_path = (PROJECT_ROOT / args.events_csv).resolve()
    movie_analysis_path = (PROJECT_ROOT / args.movie_analysis_csv).resolve()
    movie_qc_path = (PROJECT_ROOT / args.movie_specparam_qc_csv).resolve()

    ts = load_exponent_timeseries(ts_path, movie_analysis_path, movie_qc_path)
    ts_bins = np.sort(ts["center_sec"].unique())
    if ts_bins.size == 0:
        raise RuntimeError("无可用时间 bin，请检查动态 exponent 序列")

    if isc_mode == "overall":
        (
            concat_keys,
            subject_concat,
            td_template_isc,
            within_group_isc,
            within_group_overall,
            pairwise_isc,
            time_resolved,
            td_template_stats,
            within_group_stats,
        ) = _run_overall_mode(ts, ts_bins, args)
        within_overall_stats = pd.DataFrame(columns=within_group_stats.columns)
        plot_event_types = EVENT_TYPES_OVERALL
        td_title = "TD-template Aperiodic-ISC (full movie timeline)"
        within_title = "Within-group Aperiodic-ISC (full movie timeline)"
        time_resolved_stem = "fig_aperiodic_isc_time_resolved_overall"
    else:
        (
            concat_keys,
            subject_concat,
            td_template_isc,
            within_group_isc,
            within_group_overall,
            pairwise_isc,
            time_resolved,
            td_template_stats,
            within_group_stats,
        ) = _run_segmented_mode(ts, ts_bins, events_path, args)
        within_overall_stats = summarize_group_isc_tests(within_group_overall, EVENT_TYPES_OVERALL)
        plot_event_types = EVENT_TYPES_ALL
        td_title = "TD-template Aperiodic-ISC (exponent time series)"
        within_title = "Within-group Aperiodic-ISC (exponent time series)"
        time_resolved_stem = "fig_aperiodic_isc_time_resolved"

    td_template_fdr = _add_fdr(td_template_stats, "td_template")
    within_group_fdr = _add_fdr(within_group_stats, "within_group")
    within_overall_fdr = _add_fdr(within_overall_stats, "within_group_overall")

    save_csv(concat_keys, stats_dir / "aperiodic_isc_concat_keys.csv")
    save_csv(subject_concat, stats_dir / "aperiodic_isc_subject_concat_timeseries.csv")
    save_csv(td_template_isc, stats_dir / "aperiodic_isc_td_template_subject_values.csv")
    save_csv(within_group_isc, stats_dir / "aperiodic_isc_within_group_subject_values.csv")
    if not within_group_overall.empty:
        save_csv(within_group_overall, stats_dir / "aperiodic_isc_within_group_subject_values_overall.csv")
    save_csv(pairwise_isc, stats_dir / "aperiodic_isc_pairwise_group_summary.csv")
    save_csv(time_resolved, stats_dir / "aperiodic_isc_time_resolved_group_curve.csv")
    save_csv(td_template_stats, stats_dir / "aperiodic_isc_td_template_group_stats.csv")
    save_csv(within_group_stats, stats_dir / "aperiodic_isc_within_group_stats.csv")
    save_csv(within_overall_stats, stats_dir / "aperiodic_isc_within_group_stats_overall.csv")
    fdr_parts = [td_template_fdr, within_group_fdr]
    if not within_overall_stats.empty:
        fdr_parts.append(within_overall_fdr)
    save_csv(
        pd.concat(fdr_parts, ignore_index=True),
        stats_dir / "aperiodic_isc_family_fdr.csv",
    )

    _plot_subject_boxplot(
        td_template_isc,
        td_template_stats,
        td_title,
        fig_dir / "fig_aperiodic_isc_td_template_boxplot",
    )
    _plot_subject_boxplot(
        within_group_isc,
        within_group_stats,
        within_title,
        fig_dir / "fig_aperiodic_isc_within_group_boxplot",
    )
    _plot_time_resolved(time_resolved, plot_event_types, fig_dir / time_resolved_stem)

    summary_specs = [
        ("td_template", td_template_stats),
        ("within_group", within_group_stats),
    ]
    if not within_overall_stats.empty:
        summary_specs.append(("within_group_overall", within_overall_stats))
    summary_rows = []
    for method, stats_df in summary_specs:
        for _, row in stats_df.iterrows():
            summary_rows.append(
                {
                    "isc_method": method,
                    "event_type": row["event_type"],
                    "n_asd": row["n_asd"],
                    "n_td": row["n_td"],
                    "asd_mean_r": row["asd_mean_r"],
                    "td_mean_r": row["td_mean_r"],
                    "asd_mean_z": row["asd_mean_z"],
                    "td_mean_z": row["td_mean_z"],
                    "mean_diff_asd_minus_td_z": row["mean_diff_asd_minus_td_z"],
                    "t_stat": row["t_stat"],
                    "p_value": row["p_value"],
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    save_csv(summary_df, stats_dir / "aperiodic_isc_summary.csv")

    log.info("Aperiodic-ISC 汇总: %s", stats_dir / "aperiodic_isc_summary.csv")
    log.info("组内 ISC 统计: %s", stats_dir / "aperiodic_isc_within_group_stats.csv")
    log.info("时间分辨曲线: %s", stats_dir / "aperiodic_isc_time_resolved_group_curve.csv")
    log.info("图: %s", fig_dir)

    print(f"\n=== Aperiodic-ISC 主要结果 (mode={isc_mode}) ===")
    print(summary_df.to_string(index=False))
    print("\n=== 组内 pairwise 均值 ISC ===")
    print(pairwise_isc.to_string(index=False))


if __name__ == "__main__":
    main()
