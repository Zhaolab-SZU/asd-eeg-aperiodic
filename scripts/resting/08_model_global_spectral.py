# Ideal release name: 08_model_global_spectral.py
# Original path: scripts/08_main_group_analysis.py
# Note: Global exponent / offset group models
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
08_main_group_analysis.py
-------------------------
主分析：global exponent / offset 的组间差异（协变量校正）。

模型:
  global_exponent ~ group + age_months + sex + IQ_total + usable_epochs
  global_offset   ~ group + age_months + sex + IQ_total + usable_epochs

输入: derivatives/roi/specparam_subject_global.csv, participants
输出: derivatives/stats/main_group_analysis.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import attach_usable_epochs, load_analysis_participants, save_csv  # noqa: E402
from src.stats_utils import model_results_to_row, run_ols  # noqa: E402

COVARIATE_FORMULA = " + age_months + C(sex) + IQ_total + usable_epochs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="主分析：global 指标组间差异")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（例如 config/config_task_movie.yaml）",
    )
    return parser.parse_args()


def merge_analysis_df(participants: pd.DataFrame, roi_global: pd.DataFrame, deriv: Path) -> pd.DataFrame:
    df = participants.merge(roi_global, on=["subject_id", "group"], how="inner")
    df = attach_usable_epochs(df, deriv)
    preproc = deriv / "qc" / "preproc_summary.csv"
    if preproc.exists() and "usable_seconds" not in df.columns:
        df = df.merge(
            pd.read_csv(preproc)[["subject_id", "usable_seconds"]],
            on="subject_id", how="left",
        )
    return df


def main() -> None:
    args = _parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    log = setup_logging(cfg, name="main_group_analysis")

    deriv = Path(cfg["paths"]["derivatives_root"])
    roi_path = deriv / "roi" / "specparam_subject_global.csv"
    if not roi_path.exists():
        log.error("未找到 %s，请先运行 06_compute_roi_metrics.py", roi_path)
        sys.exit(1)

    participants = load_analysis_participants(cfg)
    roi_global = pd.read_csv(roi_path)
    df = merge_analysis_df(participants, roi_global, deriv)

    # 排除低质量被试
    sp_qc = deriv / "specparam" / "specparam_qc_summary_subject.csv"
    if sp_qc.exists():
        qc = pd.read_csv(sp_qc)
        bad = qc.loc[qc["low_quality_subject"] == 1, "subject_id"]
        df = df[~df["subject_id"].isin(bad)]

    models = [
        ("global_exponent", f"global_exponent ~ C(group){COVARIATE_FORMULA}"),
        ("global_offset", f"global_offset ~ C(group){COVARIATE_FORMULA}"),
    ]

    all_rows = []
    for outcome, formula in models:
        sub = df.dropna(subset=[outcome, "group", "age_months", "sex", "IQ_total", "usable_epochs"])
        if len(sub) < 10:
            log.warning("%s 有效样本不足 (n=%d)", outcome, len(sub))
            continue
        log.info("拟合 %s (n=%d): %s", outcome, len(sub), formula)
        result = run_ols(formula, sub)
        all_rows.extend(model_results_to_row(result, "main", outcome))

    if not all_rows:
        log.error("无模型成功拟合。请检查 ROI 与协变量数据。")
        sys.exit(1)

    out_path = deriv / "stats" / "main_group_analysis.csv"
    save_csv(pd.DataFrame(all_rows), out_path)
    log.info("主分析结果: %s", out_path)


if __name__ == "__main__":
    main()
