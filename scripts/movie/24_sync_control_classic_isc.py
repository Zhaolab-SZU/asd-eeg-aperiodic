# Ideal release name: 24_sync_control_classic_isc.py
# Original path: scripts/69_compute_classic_isc_controls.py
# Note: Envelope ISC / alpha PLV controls
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
69_compute_classic_isc_controls.py
--------------------------------
Classic within-group ISC controls matched to Aperiodic-ISC:
  1) Broadband envelope ISC (0.5–45 Hz Hilbert amplitude, Pearson LOO)
  2) Alpha phase PLV ISC (8–13 Hz Hilbert phase, LOO PLV)

Sliding window: 2 s / 0.5 s step; posterior ROI E33/E36/E37/E38.
Outputs supplementary comparison tables and figures.
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
    build_subject_concat_series,
    compute_within_group_isc,
    load_exponent_timeseries,
    summarize_group_isc_tests,
)
from src.classic_isc_analysis import (  # noqa: E402
    build_control_comparison_table,
    build_subject_timeseries_table,
    compute_alpha_phase_timeseries,
    compute_envelope_timeseries,
    compute_within_group_alpha_plv,
    summarize_group_envelope_tests,
    summarize_group_plv_tests,
)
from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import save_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classic envelope/alpha ISC controls")
    parser.add_argument("--config", type=str, default="config/config_task_movie.yaml")
    parser.add_argument("--events_csv", type=str, default="data/movie_events.csv")
    parser.add_argument("--min_overlap_points", type=int, default=10)
    parser.add_argument("--overwrite-timeseries", action="store_true")
    return parser.parse_args()


def _plot_pain_comparison(comp: pd.DataFrame, out_path: Path) -> None:
    pain = comp[comp["event_type"] == "pain"].copy()
    if pain.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = range(len(pain))
    width = 0.35
    ax.bar([i - width / 2 for i in x], pain["td_effect"], width, label="TD", color="#1B9E77")
    ax.bar([i + width / 2 for i in x], pain["asd_effect"], width, label="ASD", color="#D95F02")
    ax.set_xticks(list(x))
    ax.set_xticklabels(pain["isc_type"], rotation=15, ha="right")
    ax.set_ylabel("Within-group synchrony (r or PLV)")
    ax.set_title("Pain segment: Aperiodic-ISC vs classic controls")
    for i, row in pain.iterrows():
        idx = list(pain.index).index(i)
        p_txt = "NA" if pd.isna(row["p_value"]) else f"p={row['p_value']:.3g}"
        ax.text(idx, max(row["td_effect"], row["asd_effect"]) + 0.01, p_txt, ha="center", fontsize=8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_box_by_event(isc_df: pd.DataFrame, title: str, ylabel: str, out_path: Path, value_col: str = "isc_z") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4))
    for ax, ev in zip(axes, EVENT_TYPES_ALL):
        d = isc_df[isc_df["event_type"] == ev]
        sns.boxplot(data=d, x="group", y=value_col, order=["ASD", "TD"], ax=ax)
        ax.set_title(ev.capitalize())
        ax.set_xlabel("")
        if ax is axes[0]:
            ax.set_ylabel(ylabel)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    log = setup_logging(cfg, name="classic_isc_controls")

    deriv = Path(cfg["paths"]["derivatives_root"])
    out_root = Path(cfg["paths"]["outputs_root"])
    preproc_dir = deriv / "preprocessed"
    stats_dir = deriv / "stats" / "classic_isc"
    fig_dir = out_root / "figures" / "classic_isc"
    stats_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sw = cfg.get("psd", {}).get("sliding_window", {})
    window_sec = float(sw.get("window_sec", 2.0))
    step_sec = float(sw.get("step_sec", 0.5))

    movie_analysis_path = deriv / "participants_analysis.csv"
    movie_qc_path = deriv / "specparam" / "specparam_qc_summary_subject.csv"
    ts_ref = load_exponent_timeseries(
        deriv / "specparam" / "specparam_exponent_timeseries_global.csv",
        movie_analysis_path,
        movie_qc_path,
    )
    participants = ts_ref[["subject_id", "group"]].drop_duplicates().reset_index(drop=True)
    log.info("Classic ISC cohort: %d subjects", len(participants))

    env_ts_path = stats_dir / "envelope_timeseries_posterior.csv"
    alpha_ts_path = stats_dir / "alpha_phase_timeseries_posterior.csv"

    if args.overwrite_timeseries or not env_ts_path.exists():
        log.info("Computing broadband envelope time series...")
        env_ts = build_subject_timeseries_table(
            participants, preproc_dir, compute_envelope_timeseries, window_sec, step_sec
        )
        save_csv(env_ts, env_ts_path)
    else:
        env_ts = pd.read_csv(env_ts_path)

    if args.overwrite_timeseries or not alpha_ts_path.exists():
        log.info("Computing alpha phase time series...")
        alpha_ts = build_subject_timeseries_table(
            participants, preproc_dir, compute_alpha_phase_timeseries, window_sec, step_sec
        )
        save_csv(alpha_ts, alpha_ts_path)
    else:
        alpha_ts = pd.read_csv(alpha_ts_path)

    events = pd.read_csv(PROJECT_ROOT / args.events_csv)
    ts_bins = np.sort(env_ts["center_sec"].unique()) if len(env_ts) else np.array([])
    concat_keys = build_concat_keys(ts_bins=ts_bins, events=events)

    env_concat = build_subject_concat_series(
        env_ts.rename(columns={"signal_mean": "exponent_mean"}),
        keys_df=concat_keys,
    ).rename(columns={"exponent_mean": "signal_mean"})
    alpha_concat = build_subject_concat_series(
        alpha_ts.rename(columns={"signal_mean": "exponent_mean"}),
        keys_df=concat_keys,
    ).rename(columns={"exponent_mean": "signal_mean"})

    envelope_isc = compute_within_group_isc(
        env_concat,
        min_overlap_points=args.min_overlap_points,
        event_types=EVENT_TYPES_ALL,
        value_col="signal_mean",
    )
    envelope_isc["isc_method"] = "within_group_envelope"
    alpha_plv = compute_within_group_alpha_plv(
        alpha_concat,
        min_overlap_points=args.min_overlap_points,
        event_types=EVENT_TYPES_ALL,
        phase_col="signal_mean",
    )

    envelope_stats = summarize_group_envelope_tests(envelope_isc, EVENT_TYPES_ALL)
    alpha_stats = summarize_group_plv_tests(alpha_plv, EVENT_TYPES_ALL)

    aperiodic_path = deriv / "stats" / "aperiodic_isc" / "aperiodic_isc_within_group_subject_values.csv"
    if aperiodic_path.exists():
        aperiodic_isc = pd.read_csv(aperiodic_path)
        aperiodic_stats = summarize_group_isc_tests(aperiodic_isc, EVENT_TYPES_ALL)
    else:
        log.warning("Aperiodic-ISC file missing; comparison table will omit aperiodic rows.")
        aperiodic_stats = pd.DataFrame()

    comparison = build_control_comparison_table(aperiodic_stats, envelope_stats, alpha_stats)

    save_csv(env_concat, stats_dir / "envelope_subject_concat_timeseries.csv")
    save_csv(alpha_concat, stats_dir / "alpha_phase_subject_concat_timeseries.csv")
    save_csv(envelope_isc, stats_dir / "envelope_within_group_subject_values.csv")
    save_csv(envelope_stats, stats_dir / "envelope_within_group_stats.csv")
    save_csv(alpha_plv, stats_dir / "alpha_plv_within_group_subject_values.csv")
    save_csv(alpha_stats, stats_dir / "alpha_plv_within_group_stats.csv")
    save_csv(comparison, stats_dir / "classic_vs_aperiodic_within_group_comparison.csv")

    _plot_box_by_event(
        envelope_isc,
        "Within-group broadband envelope ISC (0.5–45 Hz)",
        "Envelope ISC (Fisher z)",
        fig_dir / "fig_envelope_within_group_isc",
    )
    _plot_box_by_event(
        alpha_plv,
        "Within-group alpha phase PLV (8–13 Hz, LOO)",
        "Alpha PLV",
        fig_dir / "fig_alpha_plv_within_group",
        value_col="isc_r",
    )
    _plot_pain_comparison(comparison, fig_dir / "fig_pain_aperiodic_vs_classic_controls")

    log.info("Comparison table: %s", stats_dir / "classic_vs_aperiodic_within_group_comparison.csv")
    print("\n=== Within-group ISC controls vs Aperiodic-ISC ===")
    print(comparison.to_string(index=False))
    pain = comparison[comparison["event_type"] == "pain"]
    if not pain.empty:
        print("\n=== Pain segment summary ===")
        print(pain.to_string(index=False))


if __name__ == "__main__":
    main()
