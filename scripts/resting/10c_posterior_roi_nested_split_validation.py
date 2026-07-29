# Ideal release name: 10c_posterior_roi_nested_split_validation.py
# Original path: scripts/28_posterior_roi_nested_split_validation.py
# Note: Nested / repeated split-sample ROI validation (Supp Table S4a)
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
28_posterior_roi_nested_split_validation.py
-------------------------------------------
Split-sample / nested validation for data-driven posterior ROI selection.

For each train/test split:
  1) Fit full-scalp channel OLS + BH-FDR on the training subjects only
  2) Form an ROI from selected channels (FDR; fallback top-|β|)
  3) Estimate the ROI-mean group effect only on held-out test subjects

Also runs stratified K-fold and a single 50/50 discovery/validation split.

Outputs under outputs/tables|figures|reports/robustness/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _project_root() -> Path:
    """Resolve analysis repo when this copy lives under github_release/scripts/resting/."""
    here = Path(__file__).resolve()
    bundle = here.parents[2]  # .../github_release or repo root
    parent = bundle.parent
    for cand in (parent, bundle):
        if (cand / "src" / "posterior_roi_nested_split.py").exists() and (
            (cand / "config").exists() or (cand / "derivatives").exists()
        ):
            return cand
    return bundle


PROJECT_ROOT = _project_root()
sys.path.insert(0, str(PROJECT_ROOT))
_BUNDLE = Path(__file__).resolve().parents[2]
if str(_BUNDLE) not in sys.path:
    sys.path.insert(0, str(_BUNDLE))

import pandas as pd

from src.config import load_config, setup_logging  # noqa: E402
from src.posterior_roi_loocv import load_resting_channel_cohort  # noqa: E402
from src.posterior_roi_nested_split import run_posterior_nested_split_pipeline  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nested/repeated split ROI selection validation")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--n-repeats", type=int, default=200, help="Repeated stratified splits")
    p.add_argument("--test-size", type=float, default=0.30, help="Test fraction for repeats")
    p.add_argument("--n-folds", type=int, default=5, help="Stratified K for nested CV")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    log = setup_logging(cfg, name="posterior_nested_split")

    deriv = Path(cfg["paths"]["derivatives_root"])
    ch_path = deriv / "specparam" / "specparam_channel_results_qc.csv"
    if not ch_path.exists():
        log.error("未找到 %s，请先运行 specparam / QC 流程", ch_path)
        sys.exit(1)

    channel_df = pd.read_csv(ch_path)
    if "fit_valid" in channel_df.columns:
        channel_df = channel_df[channel_df["fit_valid"]].copy()
    participants = load_resting_channel_cohort(cfg)
    log.info(
        "Cohort N=%d (ASD=%d, TD=%d); channels rows=%d",
        participants["subject_id"].nunique(),
        int((participants["group"] == "ASD").sum()),
        int((participants["group"] == "TD").sum()),
        len(channel_df),
    )

    paths = run_posterior_nested_split_pipeline(
        channel_df,
        participants,
        Path(cfg["paths"]["outputs_root"]),
        n_repeats=args.n_repeats,
        test_size=args.test_size,
        n_folds=args.n_folds,
        random_state=args.seed,
    )
    for name, path in paths.items():
        log.info("%s -> %s", name, path)
    log.info("Done.")


if __name__ == "__main__":
    main()
