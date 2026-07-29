# Ideal release name: hbn_isc_matched_global.py
# Original path: scripts/115_hbn_thepresent_isc_matched_global.py
# Note: Matched HBN ISC global variant
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
115_hbn_thepresent_isc_matched_global.py
------------------------------------------
HBN ThePresent 匹配队列：全通道均值 exponent 的 overall Aperiodic ISC（无 ROI）。

输出:
  derivatives/hbn_external_movie/replication/isc_matched_global/
  outputs/hbn_external_movie/isc_report_matched_global_zh.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.hbn_aperiodic_isc import run_hbn_thepresent_matched_global_aperiodic_isc  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="HBN ThePresent matched global Aperiodic ISC")
    parser.add_argument("--config", default="config/config_hbn_thepresent.yaml")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    log = setup_logging(cfg, name="hbn_thepresent_isc_matched_global")
    res = run_hbn_thepresent_matched_global_aperiodic_isc(cfg)
    log.info("匹配 %d 对; 报告: %s", res["n_pairs"], res["report"])


if __name__ == "__main__":
    main()
