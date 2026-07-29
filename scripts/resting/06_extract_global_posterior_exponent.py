# Ideal release name: 06_extract_global_posterior_exponent.py
# Original path: scripts/06_compute_roi_metrics.py
# Note: Global / ROI / posterior exponent extraction
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
06_compute_roi_metrics.py
-------------------------
将通道级 specparam 结果聚合为 ROI / global 指标。

输入: derivatives/specparam/specparam_channel_results_qc.csv
      config/roi_channels.yaml
输出: derivatives/roi/specparam_subject_global.csv
      derivatives/roi/specparam_subject_roi_long.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.roi_utils import run_roi_pipeline  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="聚合 specparam 到 ROI/global")
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
    log = setup_logging(cfg, name="compute_roi")

    deriv = Path(cfg["paths"]["derivatives_root"])
    in_csv = deriv / "specparam" / "specparam_channel_results_qc.csv"
    out_global = deriv / "roi" / "specparam_subject_global.csv"
    out_long = deriv / "roi" / "specparam_subject_roi_long.csv"

    if not in_csv.exists():
        log.error("未找到 %s，请先运行 05_specparam_qc.py", in_csv)
        sys.exit(1)

    layout_override = cfg.get("eeg", {}).get("roi_layout")
    if layout_override:
        log.info("使用配置 roi_layout=%s", layout_override)
    wide, long = run_roi_pipeline(in_csv, out_global, out_long, layout_override=layout_override)
    log.info("ROI 聚合完成: %d 被试, global 列: %s", len(wide), list(wide.columns))


if __name__ == "__main__":
    main()
