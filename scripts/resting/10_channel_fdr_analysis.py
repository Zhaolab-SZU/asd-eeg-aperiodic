# Ideal release name: 10_channel_fdr_analysis.py
# Original path: scripts/10_channel_level_analysis.py
# Note: Channel-level effects + FDR
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
10_channel_level_analysis.py
----------------------------
通道水平探索性分析（每通道一模型 + FDR 校正）。

模型:
  aperiodic_exponent ~ group + age_months + sex + IQ_total + usable_epochs

输入: derivatives/specparam/specparam_channel_results_qc.csv, participants
输出: derivatives/stats/channel_level_analysis.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
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
from src.stats_utils import fdr_correction, run_ols  # noqa: E402

FORMULA = "aperiodic_exponent ~ C(group) + age_months + C(sex) + IQ_total + usable_epochs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="通道水平探索性分析")
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
    log = setup_logging(cfg, name="channel_level")

    deriv = Path(cfg["paths"]["derivatives_root"])
    ch_path = deriv / "specparam" / "specparam_channel_results_qc.csv"
    if not ch_path.exists():
        log.error("未找到 %s", ch_path)
        sys.exit(1)

    channel_df = pd.read_csv(ch_path)
    if "fit_valid" in channel_df.columns:
        channel_df = channel_df[channel_df["fit_valid"]]

    participants = load_analysis_participants(cfg)
    participants = attach_usable_epochs(participants, deriv)
    participants = exclude_specparam_low_quality(participants, deriv)

    rows = []
    for ch, sub_ch in channel_df.groupby("channel"):
        df = participants.merge(sub_ch, on=["subject_id", "group"], how="inner")
        df = df.dropna(subset=["aperiodic_exponent", "group", "age_months", "sex", "IQ_total", "usable_epochs"])
        if len(df) < 10:
            continue
        try:
            res = run_ols(FORMULA, df)
            # 提取 group 效应（取 ASD vs TD 的第一个系数）
            group_terms = [t for t in res.params.index if t.startswith("C(group)")]
            if group_terms:
                term = group_terms[0]
                rows.append({
                    "channel": ch,
                    "term": term,
                    "coef": res.params[term],
                    "pvalue": res.pvalues[term],
                    "n_obs": int(res.nobs),
                })
        except Exception as exc:
            log.warning("通道 %s 模型失败: %s", ch, exc)

    if not rows:
        log.error("无通道模型成功拟合")
        sys.exit(1)

    result_df = pd.DataFrame(rows)
    reject, p_adj = fdr_correction(result_df["pvalue"].values)
    result_df["pvalue_fdr"] = p_adj
    result_df["significant_fdr"] = reject

    out_path = deriv / "stats" / "channel_level_analysis.csv"
    save_csv(result_df, out_path)
    log.info("通道分析完成: %d 通道, FDR 显著: %d", len(result_df), reject.sum())


if __name__ == "__main__":
    main()
