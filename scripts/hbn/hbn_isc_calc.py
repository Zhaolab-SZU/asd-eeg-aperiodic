# Ideal release name: hbn_isc_calc.py
# Original path: scripts/113_hbn_thepresent_aperiodic_isc.py
# Note: HBN The Present Aperiodic-ISC
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
113_hbn_thepresent_aperiodic_isc.py
------------------------------------
HBN ThePresent：非周期 exponent overall ISC（TD-template，对齐主研究 movie 方法）。

输出:
  derivatives/hbn_external_movie/replication/isc/
  outputs/hbn_external_movie/tables/isc_group_stats.csv
  outputs/hbn_external_movie/figures/fig_isc_*.png
  outputs/hbn_external_movie/isc_report_zh.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.hbn_aperiodic_isc import run_hbn_thepresent_aperiodic_isc  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="HBN ThePresent overall aperiodic ISC")
    parser.add_argument("--config", default="config/config_hbn_thepresent.yaml")
    parser.add_argument("--limit-subjects", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    log = setup_logging(cfg, name="hbn_thepresent_isc")
    paths = run_hbn_thepresent_aperiodic_isc(cfg, limit=args.limit_subjects)
    for k, p in paths.items():
        log.info("%s: %s", k, p)


if __name__ == "__main__":
    main()
