# Ideal release name: 14c_iclabel_artifact_sensitivity.py
# Original path: scripts/23_iclabel_artifact_sensitivity.py
# Note: ICLabel artifact sensitivity
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
23_iclabel_artifact_sensitivity.py
----------------------------------
Artifact-control sensitivity analysis using mne-icalabel (ICLabel).

Does NOT overwrite primary preprocessed / epochs / specparam / stats outputs.

Usage:
  python scripts/23_iclabel_artifact_sensitivity.py --threshold 0.80 --overwrite
  python scripts/23_iclabel_artifact_sensitivity.py --subjects S001 T002 --threshold 0.80
  python scripts/23_iclabel_artifact_sensitivity.py --threshold 0.80 --threshold 0.70 --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.iclabel_sensitivity import (  # noqa: E402
    ICLabelPaths,
    build_comparison_table,
    default_iclabel_cfg,
    fit_iclabel_group_models,
    load_main_cohort,
    load_primary_group_results,
    plot_forest_comparison,
    process_subject_iclabel,
    run_iclabel_removed_component_sensitivity,
    run_specparam_pipeline_iclabel,
    write_manuscript_snippets,
    write_summary_report,
)
from src.io_utils import load_participants, save_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ICLabel artifact-control sensitivity (parallel to primary analysis).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        action="append",
        dest="thresholds",
        help="Artifact probability threshold (repeat for multiple, e.g. 0.80 and 0.70).",
    )
    p.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject IDs for trial run (e.g. S001 T002).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute existing ICLabel outputs.",
    )
    p.add_argument(
        "--skip-ica",
        action="store_true",
        help="Skip ICA/epoching; only rerun PSD/specparam/stats from saved ICLabel epochs.",
    )
    return p.parse_args()


def check_icalabel_installed() -> None:
    try:
        import mne_icalabel  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "mne-icalabel is required: pip install mne-icalabel"
        ) from exc


def run_threshold(
    cfg: dict,
    log,
    threshold: float,
    subjects_df: pd.DataFrame,
    overwrite: bool,
    skip_ica: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    ic_cfg = default_iclabel_cfg(cfg)
    paths = ICLabelPaths.from_cfg(cfg, threshold)
    deriv = Path(cfg["paths"]["derivatives_root"])
    preproc_primary = deriv / "preprocessed"
    participants_all = load_participants(
        Path(cfg["paths"]["participants_file"]), included_only=True,
    )
    part_index = participants_all.set_index("subject_id")

    qc_records: list[dict] = []
    n_ok = 0
    n_fail = 0

    if not skip_ica:
        for idx, (_, row) in enumerate(subjects_df.iterrows()):
            sid = str(row["subject_id"])
            if sid not in part_index.index:
                log.warning("%s not in participants.csv", sid)
                n_fail += 1
                continue
            full_row = part_index.loc[sid]
            log.info(
                "[%d/%d] ICLabel ICA %s (threshold=%.2f)",
                idx + 1, len(subjects_df), sid, threshold,
            )
            rec = process_subject_iclabel(
                sid, full_row, cfg, paths, threshold, ic_cfg, overwrite, preproc_primary,
            )
            qc_records.append(rec)
            if str(rec["status"]).startswith("ok"):
                n_ok += 1
            else:
                n_fail += 1

    qc_summary = pd.DataFrame(qc_records)
    if len(qc_summary):
        save_csv(qc_summary, paths.tables_dir / f"iclabel_qc_summary_{paths.threshold_tag}.csv")

    cohort = load_main_cohort(cfg)
    cohort = cohort[cohort["subject_id"].isin(subjects_df["subject_id"])]

    global_df = run_specparam_pipeline_iclabel(cohort, cfg, paths, overwrite)
    if global_df.empty:
        log.warning("No specparam global metrics for threshold %.2f", threshold)
        return qc_summary, pd.DataFrame(), n_ok, n_fail

    max_ratio = float(cfg.get("fit_quality", {}).get("subject_invalid_channel_ratio_max", 0.20))
    global_df = global_df[global_df["invalid_channel_ratio"] <= max_ratio].copy()
    save_csv(global_df, paths.specparam_dir / "specparam_subject_global_iclabel.csv")

    model_df = cohort.merge(global_df, on=["subject_id", "group"], how="inner")
    iclabel_models = fit_iclabel_group_models(model_df, threshold, deriv)
    if len(iclabel_models):
        save_csv(
            iclabel_models,
            paths.tables_dir / f"iclabel_main_group_analysis_{paths.threshold_tag}.csv",
        )
    return qc_summary, iclabel_models, n_ok, n_fail


def main() -> None:
    args = parse_args()
    check_icalabel_installed()

    cfg = load_config()
    log = setup_logging(cfg, name="iclabel_sensitivity")
    thresholds = args.thresholds or [0.80]
    thresholds = sorted(set(thresholds))

    if args.subjects:
        subjects_df = pd.DataFrame({"subject_id": [str(s) for s in args.subjects]})
        log.info("Trial mode: %s", list(subjects_df["subject_id"]))
    else:
        subjects_df = load_main_cohort(cfg)[["subject_id", "group"]]
        log.info("Main cohort N=%d (post specparam QC)", len(subjects_df))

    all_models: list[pd.DataFrame] = []
    all_qc: list[pd.DataFrame] = []
    qc_all = pd.DataFrame()
    total_ok = 0
    total_fail = 0

    for thr in thresholds:
        log.info("=== artifact-control sensitivity | threshold %.2f ===", thr)
        qc, models, n_ok, n_fail = run_threshold(
            cfg, log, thr, subjects_df, args.overwrite, args.skip_ica,
        )
        total_ok += n_ok
        total_fail += n_fail
        if len(qc):
            all_qc.append(qc)
        if len(models):
            all_models.append(models)

    paths0 = ICLabelPaths.from_cfg(cfg, thresholds[0])
    ensure_tables = paths0.tables_dir
    ensure_tables.mkdir(parents=True, exist_ok=True)

    if all_qc:
        qc_all = pd.concat(all_qc, ignore_index=True)
        save_csv(qc_all, ensure_tables / "iclabel_qc_summary.csv")

    if not all_models:
        if args.subjects:
            log.warning(
                "No group models (trial with %d subject(s); need full cohort). "
                "ICA/epoch outputs may still be under derivatives/*/iclabel_cleaned/",
                len(args.subjects),
            )
            return
        log.error("No ICLabel group models fitted.")
        sys.exit(1)

    iclabel_all = pd.concat(all_models, ignore_index=True)
    save_csv(iclabel_all, ensure_tables / "iclabel_main_group_analysis.csv")

    primary = load_primary_group_results(cfg)
    comparison = build_comparison_table(primary, iclabel_all)
    save_csv(comparison, ensure_tables / "iclabel_vs_primary_comparison.csv")

    plot_forest_comparison(
        primary, iclabel_all,
        paths0.figures_dir / "fig_iclabel_vs_primary_forest",
    )

    qc_summary = qc_all if all_qc else pd.DataFrame()
    write_summary_report(
        paths0, qc_summary, iclabel_all, comparison, total_ok, total_fail,
    )
    write_manuscript_snippets(paths0, qc_summary, comparison, iclabel_all)

    try:
        cohort_stats = load_main_cohort(cfg)
    except FileNotFoundError:
        from src.iclabel_sensitivity import load_iclabel_stats_cohort

        cohort_stats = load_iclabel_stats_cohort(cfg)
    removed_results = run_iclabel_removed_component_sensitivity(cfg, threshold=thresholds[0], cohort_df=cohort_stats)
    if not removed_results["summary"].empty:
        save_csv(
            removed_results["summary"],
            ensure_tables / "iclabel_posterior_removed_component_models.csv",
        )
        save_csv(
            removed_results["detail"],
            ensure_tables / "iclabel_posterior_removed_component_models_detail.csv",
        )
        log.info("ICLabel posterior models ± n_components_removed written.")

    ok_rows = qc_summary[qc_summary["status"].str.startswith("ok")] if len(qc_summary) else pd.DataFrame()
    mean_removed = ok_rows["n_components_removed"].mean() if len(ok_rows) else float("nan")

    log.info("ICLabel sensitivity complete.")
    log.info("  Success: %d | Failed: %d", total_ok, total_fail)
    log.info("  Mean removed components (ok): %.2f", mean_removed)
    for _, row in iclabel_all.iterrows():
        log.info(
            "  %s @ threshold %.2f: β=%.4f, p=%.4f, n=%d",
            row["outcome"], row["threshold"], row["coef_TD_vs_ASD"], row["p"], row["n_total"],
        )
    log.info("  Tables: %s", ensure_tables)
    log.info("  Figures: %s", paths0.figures_dir)


if __name__ == "__main__":
    main()
