# Ideal release name: hbn_isc_matched.py
# Original path: scripts/114_hbn_thepresent_isc_matched.py
# Note: Age/IQ/sex-matched HBN ISC
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
114_hbn_thepresent_isc_matched.py
---------------------------------
HBN ThePresent：年龄/性别/IQ 1:1 匹配队列上的 overall Aperiodic ISC。

输出:
  derivatives/hbn_external_movie/replication/matched/
  derivatives/hbn_external_movie/replication/isc_matched/
  outputs/hbn_external_movie/isc_report_matched_zh.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.hbn_aperiodic_isc import run_hbn_thepresent_matched_aperiodic_isc  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="HBN ThePresent matched Aperiodic ISC")
    parser.add_argument("--config", default="config/config_hbn_thepresent.yaml")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    log = setup_logging(cfg, name="hbn_thepresent_isc_matched")
    res = run_hbn_thepresent_matched_aperiodic_isc(cfg)
    log.info("匹配 %d 对; 报告: %s", res["n_pairs"], res["report"])


if __name__ == "__main__":
    main()
