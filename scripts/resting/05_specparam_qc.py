# Ideal release name: 05_specparam_qc.py
# Original path: scripts/05_specparam_qc.py
# Note: specparam fit QC
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
05_specparam_qc.py
------------------
specparam 拟合质量控制。

输入: derivatives/specparam/specparam_channel_results.csv
输出: derivatives/specparam/specparam_channel_results_qc.csv
      derivatives/specparam/specparam_qc_summary_subject.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import load_csv  # noqa: E402
from src.qc_utils import run_specparam_qc  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="specparam 拟合质量控制")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（例如 config/config_task_movie.yaml）",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    log = setup_logging(cfg, name="specparam_qc")

    deriv = Path(cfg["paths"]["derivatives_root"]) / "specparam"
    dynamic_mode = bool(cfg.get("psd", {}).get("sliding_window", {}).get("enabled", False))
    dynamic_csv = deriv / "specparam_channel_time_resolved_results.csv"
    static_csv = deriv / "specparam_channel_results.csv"
    in_csv = dynamic_csv if dynamic_mode and dynamic_csv.exists() else static_csv
    out_ch = deriv / "specparam_channel_results_qc.csv"
    out_sub = deriv / "specparam_qc_summary_subject.csv"

    channel_df = load_csv(in_csv)
    flagged, summary = run_specparam_qc(channel_df, cfg, out_ch, out_sub)

    n_low = summary["low_quality_subject"].sum()
    log.info(
        "QC 完成: %d 通道记录, %d 名低质量被试",
        len(flagged),
        n_low,
    )


if __name__ == "__main__":
    main()
