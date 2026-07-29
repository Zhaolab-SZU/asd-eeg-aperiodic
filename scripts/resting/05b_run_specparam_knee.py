# Ideal release name: 05b_run_specparam_knee.py
# Original path: scripts/93_posterior_knee_mode_sensitivity.py
# Note: Knee-mode sensitivity (idealized 05_run_specparam_knee)
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
93_posterior_knee_mode_sensitivity.py
-------------------------------------
Primary-cohort posterior exponent knee-mode sensitivity (locked E33/E36/E37/E38).

Does NOT modify fixed-mode primary derivatives or config.

Prerequisites:
  python scripts/03_compute_psd.py
  python scripts/04_run_specparam.py
  python scripts/05_specparam_qc.py

Usage:
  python scripts/93_posterior_knee_mode_sensitivity.py
  python scripts/93_posterior_knee_mode_sensitivity.py --skip-refit
  python scripts/93_posterior_knee_mode_sensitivity.py --limit-subjects 3 --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.posterior_knee_sensitivity import run_posterior_knee_sensitivity  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Primary cohort posterior knee-mode sensitivity analysis.",
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run knee specparam refit and QC even if outputs exist.",
    )
    parser.add_argument(
        "--skip-refit",
        action="store_true",
        help="Skip knee refit; require existing sens_freq_1.0_40.0_mode_knee*.csv.",
    )
    parser.add_argument(
        "--limit-subjects",
        type=int,
        default=None,
        help="Process first N primary-cohort subjects (smoke test).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    log = setup_logging(cfg, name="posterior_knee_sensitivity")

    deriv = Path(cfg["paths"]["derivatives_root"])
    fixed_qc = deriv / "specparam" / "specparam_channel_results_qc.csv"
    psd_dir = deriv / "psd"
    analysis_parts = deriv / "participants_analysis.csv"
    preproc = deriv / "qc" / "preproc_summary.csv"
    if not psd_dir.exists() or not any(psd_dir.glob("*_psd.csv")):
        log.error("未找到 PSD 文件。请先运行 scripts/03_compute_psd.py")
        sys.exit(1)
    if not fixed_qc.exists():
        log.error("未找到 fixed QC 文件。请先运行 scripts/04_run_specparam.py 与 05_specparam_qc.py")
        sys.exit(1)
    if not analysis_parts.exists() and not preproc.exists():
        log.error("未找到分析队列文件。请先运行 scripts/02_preprocess_eeg.py")
        sys.exit(1)

    outputs = run_posterior_knee_sensitivity(
        cfg,
        overwrite=args.overwrite,
        skip_refit=args.skip_refit,
        limit_subjects=args.limit_subjects,
    )
    for name, path in outputs.items():
        log.info("  %s: %s", name, path)


if __name__ == "__main__":
    main()
