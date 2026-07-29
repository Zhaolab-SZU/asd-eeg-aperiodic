# Ideal release name: hbn_external_validation.py
# Original path: scripts/103_hbn_external_validation.py
# Note: HBN external validation entry
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
103_hbn_external_validation.py
--------------------------------
HBN 外部复现主模型 + 中文报告。

模型:
  posterior_exponent ~ group + age_months + sex + IQ_total + usable_epochs

输出: outputs/hbn_external/tables/
      outputs/hbn_external/validation_report_zh.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.hbn_external import run_hbn_external_validation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="HBN 外部复现统计")
    parser.add_argument("--config", default="config/config_hbn_external.yaml")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    log = setup_logging(cfg, name="hbn_external_validation")
    report = run_hbn_external_validation(cfg)
    log.info("报告: %s", report)


if __name__ == "__main__":
    main()
