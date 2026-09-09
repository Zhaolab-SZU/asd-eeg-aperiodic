# Public path: scripts/movie/24_sync_control_classic_isc.py
# Source: scripts/69_compute_classic_isc_controls.py (film-level synchrony)

#!/usr/bin/env python
"""
69_compute_classic_isc_controls.py
--------------------------------
Partly Cloudy classic synchrony controls on the continuous film timeline:
  1) Broadband envelope ISC (TD-template; same posterior ROI / windows)
  2) Alpha phase PLV ISC (within-group LOO)
  3) Shared variance with Partly Cloudy Aperiodic-ISC
  4) Envelope-adjusted Partly Cloudy Aperiodic-ISC group model

No event-category partitioning.
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
import statsmodels.formula.api as smf
from scipy import stats

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
    compute_td_template_isc,
    compute_within_group_isc,
    load_exponent_timeseries,
    summarize_group_isc_tests,
)
from src.classic_isc_analysis import (  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Partly Cloudy classic ISC controls")
    parser.add_argument("--config", type=str, default="config/config_task_movie.yaml")
    parser.add_argument("--min_overlap_points", type=int, default=50)
    parser.add_argument("--overwrite-timeseries", action="store_true")
    return parser.parse_args()


def _plot_controls(comp: pd.DataFrame, out_path: Path) -> None:
    if comp.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(comp))
    width = 0.35
    ax.bar(x - width / 2, comp["td_effect"], width, label="TD", color="#1B9E77")
    ax.bar(x + width / 2, comp["asd_effect"], width, label="ASD", color="#D95F02")
    ax.set_xticks(x)
    ax.set_xticklabels(comp["isc_type"], rotation=15, ha="right")
    ax.set_ylabel("Synchrony (r or PLV)")
    ax.set_title("Partly Cloudy synchrony controls")
    for i, row in comp.iterrows():
        idx = list(comp.index).index(i)
        p_txt = "NA" if pd.isna(row["p_value"]) else f"p={row['p_value']:.3g}"
        ax.text(idx, max(row["td_effect"], row["asd_effect"]) + 0.01, p_txt, ha="center", fontsize=8)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = load_config(cfg_path)
    log = setup_logging(cfg, name="partly_cloudy_synchrony")

    deriv = Path(cfg["paths"]["derivatives_root"])
    out_root = Path(cfg["paths"]["outputs_root"])
    preproc_dir = deriv / "preprocessed"
    stats_dir = deriv / "stats" / "partly_cloudy_synchrony"
    table_dir = PROJECT_ROOT / "outputs" / "tables" / "partly_cloudy_synchrony"
    fig_dir = out_root / "figures" / "partly_cloudy_synchrony"
    for d in (stats_dir, table_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

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
    log.info("Partly Cloudy synchrony cohort: %d subjects", len(participants))

    env_ts_path = stats_dir / "partly_cloudy_envelope_timeseries_posterior.csv"
    alpha_ts_path = stats_dir / "partly_cloudy_alpha_phase_timeseries_posterior.csv"

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

    if env_ts.empty:
        raise RuntimeError(
            "[ANALYSIS BLOCKED: empty envelope timeseries — check preprocessed FIF availability]"
        )

    ts_bins = np.sort(env_ts["center_sec"].unique())
    concat_keys = build_film_timeline_keys(ts_bins, film_id=FILM_PARTLY_CLOUDY)

    env_concat = build_subject_concat_series(
        env_ts.rename(columns={"signal_mean": "exponent_mean"}),
        keys_df=concat_keys,
    ).rename(columns={"exponent_mean": "signal_mean"})
    alpha_concat = build_subject_concat_series(
        alpha_ts.rename(columns={"signal_mean": "exponent_mean"}),
        keys_df=concat_keys,
    ).rename(columns={"exponent_mean": "signal_mean"})

    # Envelope: TD-template (aligned with primary Aperiodic-ISC)
    envelope_isc = compute_td_template_isc(
        subject_concat=env_concat,
        min_overlap_points=args.min_overlap_points,
        event_types=FILM_TYPES_PARTLY_CLOUDY,
        value_col="signal_mean",
    )
    envelope_isc = envelope_isc[np.isfinite(envelope_isc["isc_z"])].copy()
    envelope_isc["isc_method"] = "td_template_envelope"

    alpha_plv = compute_within_group_alpha_plv(
        alpha_concat,
        min_overlap_points=args.min_overlap_points,
        event_types=FILM_TYPES_PARTLY_CLOUDY,
        phase_col="signal_mean",
    )
    alpha_plv = alpha_plv[np.isfinite(alpha_plv["isc_r"])].copy()

    envelope_stats = summarize_group_envelope_tests(envelope_isc, FILM_TYPES_PARTLY_CLOUDY)
    alpha_stats = summarize_group_plv_tests(alpha_plv, FILM_TYPES_PARTLY_CLOUDY)

    aperiodic_path = (
        deriv / "stats" / "partly_cloudy_isc" / "partly_cloudy_aperiodic_isc_subject_level.csv"
    )
    if not aperiodic_path.exists():
        raise FileNotFoundError(
            f"[ANALYSIS BLOCKED: missing {aperiodic_path}; run 68_compute_aperiodic_isc.py first]"
        )
    aperiodic_isc = pd.read_csv(aperiodic_path)
    aperiodic_isc["subject_id"] = aperiodic_isc["subject_id"].astype(str)
    aperiodic_stats = summarize_group_isc_tests(aperiodic_isc, FILM_TYPES_PARTLY_CLOUDY)

    # Shared variance / envelope-adjusted model
    merged = aperiodic_isc[["subject_id", "group", "isc_z"]].rename(
        columns={"isc_z": "aperiodic_isc_z"}
    )
    merged = merged.merge(
        envelope_isc[["subject_id", "group", "isc_z"]].rename(columns={"isc_z": "envelope_isc_z"}),
        on=["subject_id", "group"],
        how="inner",
    )
    if len(merged) >= 10:
        r_shared, p_shared = stats.pearsonr(merged["aperiodic_isc_z"], merged["envelope_isc_z"])
        shared_var = float(r_shared**2)
        ols = smf.ols(
            "aperiodic_isc_z ~ C(group, Treatment(reference='ASD')) + envelope_isc_z",
            data=merged,
        ).fit(cov_type="HC3")
        key = [k for k in ols.params.index if "group" in k and "TD" in k][0]
        adj = {
            "n_overlap_subjects": int(len(merged)),
            "pearson_r_aperiodic_vs_envelope": float(r_shared),
            "shared_variance_r2": shared_var,
            "pearson_p": float(p_shared),
            "envelope_adjusted_beta_td_minus_asd": float(ols.params[key]),
            "envelope_adjusted_se": float(ols.bse[key]),
            "envelope_adjusted_p": float(ols.pvalues[key]),
            "envelope_adjusted_ci95_low": float(ols.conf_int().loc[key, 0]),
            "envelope_adjusted_ci95_high": float(ols.conf_int().loc[key, 1]),
        }
    else:
        adj = {
            "n_overlap_subjects": int(len(merged)),
            "note": "[ANALYSIS BLOCKED: insufficient overlapping subjects for envelope adjustment]",
        }

    comparison_rows = []
    if len(aperiodic_stats):
        r = aperiodic_stats.iloc[0]
        comparison_rows.append(
            {
                "film_id": FILM_PARTLY_CLOUDY,
                "isc_type": "Partly Cloudy Aperiodic-ISC",
                "n_asd": r["n_asd"],
                "n_td": r["n_td"],
                "asd_effect": r["asd_mean_r"],
                "td_effect": r["td_mean_r"],
                "p_value": r["p_value"],
            }
        )
    if len(envelope_stats):
        r = envelope_stats.iloc[0]
        comparison_rows.append(
            {
                "film_id": FILM_PARTLY_CLOUDY,
                "isc_type": "Partly Cloudy broadband envelope ISC",
                "n_asd": r["n_asd"],
                "n_td": r["n_td"],
                "asd_effect": r["asd_mean_r"],
                "td_effect": r["td_mean_r"],
                "p_value": r["p_value"],
            }
        )
    if len(alpha_stats):
        r = alpha_stats.iloc[0]
        comparison_rows.append(
            {
                "film_id": FILM_PARTLY_CLOUDY,
                "isc_type": "Partly Cloudy alpha PLV ISC",
                "n_asd": r["n_asd"],
                "n_td": r["n_td"],
                "asd_effect": r["asd_mean_plv"],
                "td_effect": r["td_mean_plv"],
                "p_value": r["p_value"],
            }
        )
    comparison = pd.DataFrame(comparison_rows)

    save_csv(envelope_isc, stats_dir / "partly_cloudy_envelope_isc.csv")
    save_csv(envelope_stats, stats_dir / "partly_cloudy_envelope_isc_group_stats.csv")
    save_csv(alpha_plv, stats_dir / "partly_cloudy_alpha_plv_isc.csv")
    save_csv(alpha_stats, stats_dir / "partly_cloudy_alpha_plv_isc_group_stats.csv")
    save_csv(comparison, stats_dir / "partly_cloudy_synchrony_controls.csv")
    save_csv(merged, stats_dir / "partly_cloudy_aperiodic_vs_envelope_subject.csv")
    (stats_dir / "partly_cloudy_envelope_adjusted_summary.json").write_text(
        json.dumps(adj, indent=2), encoding="utf-8"
    )

    for name in (
        "partly_cloudy_envelope_isc.csv",
        "partly_cloudy_alpha_plv_isc.csv",
        "partly_cloudy_synchrony_controls.csv",
    ):
        save_csv(pd.read_csv(stats_dir / name), table_dir / name)
    (table_dir / "partly_cloudy_envelope_adjusted_summary.json").write_text(
        json.dumps(adj, indent=2), encoding="utf-8"
    )

    _plot_controls(comparison, fig_dir / "fig_partly_cloudy_synchrony_controls")

    print("\n=== Partly Cloudy synchrony controls ===")
    print(comparison.to_string(index=False))
    print("\n=== Envelope-adjusted summary ===")
    print(json.dumps(adj, indent=2))
    log.info("Wrote synchrony controls to %s", stats_dir)


if __name__ == "__main__":
    main()
