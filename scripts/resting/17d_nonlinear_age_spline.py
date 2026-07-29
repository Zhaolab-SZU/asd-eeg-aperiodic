# Ideal release name: 17d_nonlinear_age_spline.py
# Original path: scripts/92_nonlinear_age_sensitivity.py
# Note: Nonlinear age / spline sensitivity
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
92_nonlinear_age_sensitivity.py
---------------------------------
补充分析：global / posterior exponent 的 spline vs linear group×age 敏感性。

输出:
  outputs/tables/nonlinear_age_sensitivity/*.csv
  outputs/figures/nonlinear_age_sensitivity/*.png
  outputs/reports/nonlinear_age_sensitivity_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.nonlinear_age_sensitivity import run_nonlinear_age_sensitivity  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spline vs linear age sensitivity analysis")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    log = setup_logging(cfg, name="nonlinear_age_sensitivity")

    paths = run_nonlinear_age_sensitivity(cfg)
    log.info("非线性年龄敏感性分析完成")
    log.info("  表格: %s", paths["tables"])
    log.info("  图形: %s", paths["figures"])
    log.info("  汇报: %s", paths["report"])


if __name__ == "__main__":
    main()
