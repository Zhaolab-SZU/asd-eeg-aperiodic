# Ideal release name: 10b_loocv_fdr_survival.py
# Original path: scripts/27_posterior_roi_loocv_fdr.py
# Note: Leave-one-subject-out FDR survival (Supp Fig S1)
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
27_posterior_roi_loocv_fdr.py
-----------------------------
Leave-one-subject-out cross-validation of channel-level FDR localization.

For each of the N = 138 participants, one subject is excluded and full-brain
channel-wise group models are re-fit with Benjamini–Hochberg FDR correction.
We quantify how often the data-driven posterior cluster (E33, E36, E37, E38)
remains FDR-significant — a non-circular internal validation of spatial stability.

Outputs:
  outputs/tables/robustness/posterior_roi_loocv_fdr_folds.csv
  outputs/tables/robustness/posterior_roi_loocv_fdr_summary.csv
  outputs/reports/posterior_roi_loocv_fdr_report.txt
  outputs/figures/robustness/fig_posterior_roi_loocv_fdr_stability.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.posterior_roi_loocv import (  # noqa: E402
    load_resting_channel_cohort,
    run_posterior_loocv_pipeline,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOOCV FDR stability for posterior ROI")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    log = setup_logging(cfg, name="posterior_loocv_fdr")

    deriv = Path(cfg["paths"]["derivatives_root"])
    ch_path = deriv / "specparam" / "specparam_channel_results_qc.csv"
    if not ch_path.exists():
        log.error("未找到 %s，请先运行 specparam 流程", ch_path)
        sys.exit(1)

    channel_df = __import__("pandas").read_csv(ch_path)
    if "fit_valid" in channel_df.columns:
        channel_df = channel_df[channel_df["fit_valid"]]

    participants = load_resting_channel_cohort(cfg)

    log.info("分析队列: N = %d（与 channel-level / normative 主分析一致）", len(participants))
    paths = run_posterior_loocv_pipeline(
        channel_df,
        participants,
        Path(cfg["paths"]["outputs_root"]),
    )
    for name, path in paths.items():
        log.info("%s -> %s", name, path)


if __name__ == "__main__":
    main()
