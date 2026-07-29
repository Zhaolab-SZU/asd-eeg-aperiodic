# Ideal release name: hbn_preprocess_resting.py
# Original path: scripts/101_hbn_preprocess_resting.py
# Note: HBN resting preprocess
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
101_hbn_preprocess_resting.py
-----------------------------
HBN RestingState：读 .set → 提取 eyes-open → 滤波 → epoch。

输出: derivatives/hbn_external/epochs/
      derivatives/hbn_external/qc/preproc_summary.csv
      derivatives/hbn_external/participants_analysis.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.hbn_external import batch_preprocess_hbn, build_hbn_participants  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="HBN resting 预处理")
    parser.add_argument("--config", default="config/config_hbn_external.yaml")
    parser.add_argument("--limit-subjects", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    log = setup_logging(cfg, name="hbn_preprocess")
    build_hbn_participants(cfg)
    summary = batch_preprocess_hbn(cfg, limit=args.limit_subjects, overwrite=args.overwrite)
    log.info("预处理完成: %d 被试", len(summary))


if __name__ == "__main__":
    main()
