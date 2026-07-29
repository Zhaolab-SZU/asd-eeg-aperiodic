# Ideal release name: hbn_specparam_roi.py
# Original path: scripts/102_hbn_specparam_roi.py
# Note: HBN ROI specparam
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
102_hbn_specparam.py
--------------------
HBN：PSD → specparam（通道级）。

默认不做手工 ROI；空间推断见 105_hbn_spatial_cluster_stats.py。

输出: derivatives/hbn_external/specparam/
      derivatives/hbn_external/roi/  （仅 --with-roi 时）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.hbn_external import run_hbn_psd_specparam, run_hbn_roi_aggregation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="HBN specparam（通道级）")
    parser.add_argument("--config", default="config/config_hbn_external.yaml")
    parser.add_argument("--limit-subjects", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--with-roi",
        action="store_true",
        help="可选：额外跑手工 occipital_posterior ROI 聚合（默认关闭）",
    )
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    log = setup_logging(cfg, name="hbn_specparam")
    run_hbn_psd_specparam(cfg, limit=args.limit_subjects, overwrite=args.overwrite)
    if args.with_roi:
        run_hbn_roi_aggregation(cfg)
        log.info("specparam + ROI 完成")
    else:
        log.info("specparam 完成（未做手工 ROI；请运行 105 做空间统计）")


if __name__ == "__main__":
    main()
