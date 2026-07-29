# Ideal release name: hbn_preprocess_present.py
# Original path: scripts/101_hbn_preprocess_thepresent.py
# Note: HBN The Present movie preprocess
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
101_hbn_preprocess_thepresent.py
--------------------------------
HBN ThePresent：读 .set → 裁 video_start/stop → 滤波 → epoch。

输出: derivatives/hbn_external_movie/epochs/
      derivatives/hbn_external_movie/qc/preproc_summary.csv
      derivatives/hbn_external_movie/participants_analysis.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.hbn_external import (  # noqa: E402
    batch_preprocess_hbn_thepresent,
    build_hbn_thepresent_participants,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="HBN ThePresent movie 预处理")
    parser.add_argument("--config", default="config/config_hbn_thepresent.yaml")
    parser.add_argument("--limit-subjects", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    log = setup_logging(cfg, name="hbn_preprocess_thepresent")
    build_hbn_thepresent_participants(cfg)
    summary = batch_preprocess_hbn_thepresent(
        cfg,
        limit=args.limit_subjects,
        overwrite=args.overwrite,
    )
    log.info("ThePresent 预处理完成: %d 被试", len(summary))


if __name__ == "__main__":
    main()
