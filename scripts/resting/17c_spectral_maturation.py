# Ideal release name: 17c_spectral_maturation.py
# Original path: scripts/91_spectral_maturation_joint_model.py
# Note: Spectral maturation joint model
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
91_spectral_maturation_joint_model.py
------------------------------------
IAF (alpha_cf) + aperiodic exponent 联合发育模型。

模型:
  alpha_cf / posterior_alpha_cf / global_exponent / posterior_exponent
    ~ C(group) * age_months + covariates

另含 TD normative deviation、IAF–exponent 独立性检验与可视化。

输出:
  outputs/tables/spectral_maturation/*.csv
  outputs/figures/spectral_maturation/*.png
  outputs/reports/spectral_maturation_joint_model_report_zh.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.spectral_maturation_analysis import run_spectral_maturation_analysis  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IAF + exponent joint maturation models")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    log = setup_logging(cfg, name="spectral_maturation_joint_model")

    paths = run_spectral_maturation_analysis(cfg)
    log.info("联合发育模型分析完成")
    log.info("  表格: %s", paths["tables"])
    log.info("  图形: %s", paths["figures"])
    log.info("  汇报: %s", paths["report"])


if __name__ == "__main__":
    main()
