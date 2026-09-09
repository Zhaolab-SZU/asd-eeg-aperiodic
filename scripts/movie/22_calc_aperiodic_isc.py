# Public path: scripts/movie/22_calc_aperiodic_isc.py
# Source: scripts/68_compute_aperiodic_isc.py (film-level, min_overlap=50)

#!/usr/bin/env python
"""
68_compute_aperiodic_isc.py
---------------------------
Partly Cloudy Aperiodic-ISC on the continuous usable film timeline.

Computes:
1) TD-template Partly Cloudy Aperiodic-ISC (primary)
2) Within-group LOO Partly Cloudy Aperiodic-ISC (complementary)
3) Pairwise / time-resolved summaries
4) Welch group tests

Aperiodic-ISC is calculated across the usable duration of Partly Cloudy using
pairwise-valid time-resolved posterior exponent estimates. No event-category
partitioning is applied.

Default min_overlap_points=50 aligns with movie QC (>=50 usable 2-s epochs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "src" / "config.py").is_file():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError("Cannot locate repository root (missing src/config.py)")
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.aperiodic_isc_analysis import (  # noqa: E402
    FILM_PARTLY_CLOUDY,
    FILM_TYPES_PARTLY_CLOUDY,
    build_film_timeline_keys,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Partly Cloudy Aperiodic-ISC")
    parser.add_argument("--config", type=str, default="config/config_task_movie.yaml")
    parser.add_argument(
        "--min_overlap_points",
        type=int,
        default=50,
        help="Minimum pairwise-valid windows for a subject-level ISC estimate "
        "(aligned with movie QC usable-epoch threshold of 50).",
    )
    parser.add_argument("--local_window_bins", type=int, default=11)
    parser.add_argument(
        "--movie_analysis_csv",
        type=str,
        default="derivatives/derivatives_task_movie/participants_analysis.csv",
    )
    parser.add_argument(
        "--movie_specparam_qc_csv",
        type=str,
        default="derivatives/derivatives_task_movie/specparam/specparam_qc_summary_subject.csv",
    )
    return parser.parse_args()


def _plot_subject_boxplot(isc_df: pd.DataFrame, stats_df: pd.DataFrame, title: str, out_path: Path) -> None:
    plt.figure(figsize=(6.5, 5.0))
    plot_df = isc_df.copy()
    plot_df["film"] = "Partly Cloudy"
    pal = {"ASD": "#B22222", "TD": "#000080"}
    sns.boxplot(
        data=plot_df, x="film", y="isc_z", hue="group", hue_order=["ASD", "TD"],
        palette=pal, showfliers=False, width=0.55,
    )
    sns.stripplot(
        data=plot_df,
        x="film",
        y="isc_z",
        hue="group",
        hue_order=["ASD", "TD"],
        palette=pal,
        dodge=True,
        alpha=0.55,
        size=3.5,
        linewidth=0,
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], title="Group", frameon=True)
    plt.xlabel("")
    plt.ylabel("Partly Cloudy Aperiodic-ISC (Fisher z)")
    plt.title(title)
    if not stats_df.empty:
        r = stats_df.iloc[0]
        p_txt = "NA" if pd.isna(r["p_value"]) else f"{r['p_value']:.4g}"
        plt.gca().text(
            0.98,
            0.02,
            f"n ASD/TD={int(r['n_asd'])}/{int(r['n_td'])}; p={p_txt}",
            transform=plt.gca().transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path.with_suffix(".png"), dpi=180)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def _plot_time_resolved(curve: pd.DataFrame, out_path: Path) -> None:
    if curve.empty:
        return
    plt.figure(figsize=(9, 4.2))
    for group, color in (("TD", "#000080"), ("ASD", "#B22222")):
        sub = curve[curve["group"] == group].sort_values("center_sec")
        plt.plot(sub["center_sec"], sub["mean_loo_z"], label=group, color=color, lw=1.6)
    plt.xlabel("Partly Cloudy time (s)")
    plt.ylabel("Within-group LOO ISC (Fisher z)")
    plt.title("Partly Cloudy time-resolved within-group Aperiodic-ISC")
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path.with_suffix(".png"), dpi=180)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = load_config(cfg_path)
    log = setup_logging(cfg, name="partly_cloudy_isc")

    deriv = Path(cfg["paths"]["derivatives_root"])
    out_root = Path(cfg["paths"]["outputs_root"])
    stats_dir = deriv / "stats" / "partly_cloudy_isc"
    fig_dir = out_root / "figures" / "partly_cloudy_isc"
    table_dir = PROJECT_ROOT / "outputs" / "tables" / "partly_cloudy_isc"
    stats_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    sw = cfg.get("psd", {}).get("sliding_window", {})
    log.info(
        "Partly Cloudy Aperiodic-ISC | window=%.1fs step=%.1fs | min_overlap_points=%d",
        float(sw.get("window_sec", np.nan)),
        float(sw.get("step_sec", np.nan)),
        int(args.min_overlap_points),
    )

    ts_path = deriv / "specparam" / "specparam_exponent_timeseries_global.csv"
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing dynamic exponent series: {ts_path}")

    movie_analysis_path = (PROJECT_ROOT / args.movie_analysis_csv).resolve()
    movie_qc_path = (PROJECT_ROOT / args.movie_specparam_qc_csv).resolve()
    ts = load_exponent_timeseries(ts_path, movie_analysis_path, movie_qc_path)
    ts_bins = np.sort(ts["center_sec"].unique())
    if ts_bins.size == 0:
        raise RuntimeError("No usable time bins after movie spectral QC filtering")

    concat_keys = build_film_timeline_keys(ts_bins, film_id=FILM_PARTLY_CLOUDY)
    subject_concat = build_subject_concat_series(ts=ts, keys_df=concat_keys)

    td_template_isc = compute_td_template_isc(
        subject_concat=subject_concat,
        min_overlap_points=args.min_overlap_points,
        event_types=FILM_TYPES_PARTLY_CLOUDY,
    )
    within_group_isc = compute_within_group_isc(
        subject_concat=subject_concat,
        min_overlap_points=args.min_overlap_points,
        event_types=FILM_TYPES_PARTLY_CLOUDY,
    )
    pairwise_isc = compute_pairwise_mean_isc(
        subject_concat=subject_concat,
        min_overlap_points=args.min_overlap_points,
        event_types=FILM_TYPES_PARTLY_CLOUDY,
    )
    time_resolved = compute_time_resolved_group_isc(
        subject_concat=subject_concat,
        local_window_bins=args.local_window_bins,
        min_overlap_points=max(5, args.min_overlap_points // 2),
        event_types=FILM_TYPES_PARTLY_CLOUDY,
    )

    # Exclude subjects failing min-overlap (isc_r NaN)
    n_before = len(td_template_isc)
    td_valid = td_template_isc[np.isfinite(td_template_isc["isc_z"])].copy()
    n_excluded_overlap = int(n_before - len(td_valid))
    within_valid = within_group_isc[np.isfinite(within_group_isc["isc_z"])].copy()

    td_stats = summarize_group_isc_tests(td_valid, FILM_TYPES_PARTLY_CLOUDY)
    within_stats = summarize_group_isc_tests(within_valid, FILM_TYPES_PARTLY_CLOUDY)

    overlap_summary = {
        "min_overlap_points_criterion": int(args.min_overlap_points),
        "n_subjects_before_overlap_filter": int(n_before),
        "n_subjects_excluded_by_overlap": int(n_excluded_overlap),
        "n_overlap_points_min": float(td_template_isc["n_overlap_points"].min()) if len(td_template_isc) else np.nan,
        "n_overlap_points_median": float(td_template_isc["n_overlap_points"].median()) if len(td_template_isc) else np.nan,
        "n_overlap_points_max": float(td_template_isc["n_overlap_points"].max()) if len(td_template_isc) else np.nan,
        "n_asd": int((td_valid["group"] == "ASD").sum()),
        "n_td": int((td_valid["group"] == "TD").sum()),
    }
    (stats_dir / "partly_cloudy_isc_overlap_qc.json").write_text(
        json.dumps(overlap_summary, indent=2), encoding="utf-8"
    )

    # Canonical outputs
    save_csv(concat_keys, stats_dir / "partly_cloudy_timeline_keys.csv")
    save_csv(subject_concat, stats_dir / "partly_cloudy_exponent_concat_timeseries.csv")
    save_csv(td_valid, stats_dir / "partly_cloudy_aperiodic_isc_subject_level.csv")
    save_csv(td_stats, stats_dir / "partly_cloudy_aperiodic_isc_group_stats.csv")
    save_csv(within_valid, stats_dir / "partly_cloudy_within_group_isc_subject_level.csv")
    save_csv(within_stats, stats_dir / "partly_cloudy_within_group_isc_group_stats.csv")
    save_csv(pairwise_isc, stats_dir / "partly_cloudy_pairwise_group_summary.csv")
    save_csv(time_resolved, stats_dir / "partly_cloudy_time_resolved_group_curve.csv")

    # Mirror into outputs/tables
    save_csv(td_valid, table_dir / "partly_cloudy_aperiodic_isc_subject_level.csv")
    save_csv(td_stats, table_dir / "partly_cloudy_aperiodic_isc_group_stats.csv")
    save_csv(within_valid, table_dir / "partly_cloudy_within_group_isc_subject_level.csv")
    save_csv(within_stats, table_dir / "partly_cloudy_within_group_isc_group_stats.csv")
    (table_dir / "partly_cloudy_isc_overlap_qc.json").write_text(
        json.dumps(overlap_summary, indent=2), encoding="utf-8"
    )

    _plot_subject_boxplot(
        td_valid,
        td_stats,
        "Partly Cloudy TD-template Aperiodic-ISC",
        fig_dir / "fig_partly_cloudy_td_template_isc",
    )
    _plot_subject_boxplot(
        within_valid,
        within_stats,
        "Partly Cloudy within-group Aperiodic-ISC (complementary)",
        fig_dir / "fig_partly_cloudy_within_group_isc",
    )
    _plot_time_resolved(time_resolved, fig_dir / "fig_partly_cloudy_time_resolved_isc")

    log.info("Overlap QC: %s", overlap_summary)
    log.info("Saved subject-level: %s", stats_dir / "partly_cloudy_aperiodic_isc_subject_level.csv")
    print("\n=== Partly Cloudy Aperiodic-ISC (TD-template) ===")
    print(td_stats.to_string(index=False))
    print("\n=== Within-group (complementary) ===")
    print(within_stats.to_string(index=False))
    print("\n=== Overlap QC ===")
    print(json.dumps(overlap_summary, indent=2))


if __name__ == "__main__":
    main()
