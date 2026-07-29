# Ideal release name: 09_roi_mixed_model.py
# Original path: scripts/09_roi_mixed_model.py
# Note: ROI mixed models
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
09_roi_mixed_model.py
---------------------
ROI 水平混合模型：group × roi 交互效应。

模型:
  exponent ~ group * roi + age_months + sex + IQ_total + usable_epochs
  (subject 随机截距)

输入: derivatives/roi/specparam_subject_roi_long.csv, participants
输出: derivatives/stats/roi_mixed_model.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import (  # noqa: E402
    attach_usable_epochs,
    exclude_specparam_low_quality,
    load_analysis_participants,
    save_csv,
)
from src.stats_utils import model_results_to_row, run_mixedlm  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROI 混合效应模型")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（例如 config/config_task_movie.yaml）",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    log = setup_logging(cfg, name="roi_mixed_model")

    deriv = Path(cfg["paths"]["derivatives_root"])
    roi_long_path = deriv / "roi" / "specparam_subject_roi_long.csv"
    if not roi_long_path.exists():
        log.error("未找到 %s", roi_long_path)
        sys.exit(1)

    participants = load_analysis_participants(cfg)
    roi_long = pd.read_csv(roi_long_path)
    df = participants.merge(roi_long, on=["subject_id", "group"], how="inner")
    df = attach_usable_epochs(df, deriv)
    df = exclude_specparam_low_quality(df, deriv)

    formula = (
        "exponent ~ C(group) * C(roi) + age_months + C(sex) + IQ_total + usable_epochs"
    )
    sub = df.dropna(subset=["exponent", "group", "roi", "age_months", "sex", "IQ_total", "usable_epochs"])
    log.info("ROI 混合模型 n=%d", len(sub))
    if len(sub) < 10:
        log.warning("ROI 指标在当前配置下不可用（常见于动态 GLOBAL 模式），输出空结果并跳过。")
        out_path = deriv / "stats" / "roi_mixed_model.csv"
        save_csv(pd.DataFrame([{
            "analysis": "roi_mixed",
            "status": "skipped",
            "reason": "insufficient_valid_roi_rows_under_current_mode",
            "n_rows": len(sub),
        }]), out_path)
        return

    result = run_mixedlm(formula, sub, groups="subject_id")
    if not getattr(result, "_used_mixedlm", True):
        log.warning("已降级为 OLS（无随机效应）")

    rows = model_results_to_row(result, "roi_mixed", "exponent")
    out_path = deriv / "stats" / "roi_mixed_model.csv"
    save_csv(pd.DataFrame(rows), out_path)
    log.info("ROI 混合模型结果: %s", out_path)


if __name__ == "__main__":
    main()
