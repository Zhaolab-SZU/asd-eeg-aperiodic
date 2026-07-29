# Ideal release name: 17b_normative_td_reference.py
# Original path: scripts/90_normative_exponent_analysis.py
# Note: TD-reference normative deviation
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
90_normative_exponent_analysis.py
---------------------------------
补充分析：在 TD 组建立 global exponent 的年龄规范模型（样条 age），
计算 ASD 相对典型发育轨迹的 deviation z-score。

输出:
  derivatives/stats/normative_exponent_scores.csv
  outputs/tables/normative_exponent/*.csv
  outputs/figures/normative_exponent/*.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.normative_analysis import run_normative_analysis  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normative modeling: global exponent")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    log = setup_logging(cfg, name="normative_exponent_analysis")

    paths = run_normative_analysis(cfg)
    log.info("Normative 分析完成")
    log.info("  个体 z-score: %s", paths["scores"])
    log.info("  表格: %s", paths["tables"])
    log.info("  图形: %s", paths["figures"])


if __name__ == "__main__":
    main()
