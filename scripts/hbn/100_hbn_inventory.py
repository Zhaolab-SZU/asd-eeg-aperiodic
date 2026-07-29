# Ideal release name: 100_hbn_inventory.py
# Original path: scripts/100_hbn_inventory.py
# Note: HBN file inventory
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
100_hbn_inventory.py
--------------------
HBN Manifest A 下载完整性 + eyes-open 段 inventory。

输出: outputs/hbn_external/hbn_inventory.csv
      data/hbn_external/participants.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.hbn_external import run_hbn_inventory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="HBN 下载 inventory")
    parser.add_argument("--config", default="config/config_hbn_external.yaml")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    log = setup_logging(cfg, name="hbn_inventory")
    inv = run_hbn_inventory(cfg)
    log.info("ready=%d / %d", inv["ready"].sum(), len(inv))


if __name__ == "__main__":
    main()
