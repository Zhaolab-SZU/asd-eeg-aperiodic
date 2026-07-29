# Ideal release name: 12_iaf_control_model.py
# Original path: scripts/12_periodic_peak_analysis.py
# Note: Individual alpha frequency / periodic controls
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
12_periodic_peak_analysis.py
----------------------------
周期峰参数（alpha/theta/beta）组间比较。

模型:
  {peak_metric} ~ group + age_months + sex + IQ_total + usable_epochs

输入: derivatives/specparam/specparam_channel_results_qc.csv
输出: derivatives/stats/periodic_peak_analysis.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import attach_usable_epochs, load_analysis_participants, save_csv  # noqa: E402
from src.stats_utils import model_results_to_row, run_ols  # noqa: E402

PEAK_METRICS = ["alpha_cf", "alpha_pw", "alpha_bw", "theta_pw", "beta_pw"]
FORMULA_TEMPLATE = "{metric} ~ C(group) + age_months + C(sex) + IQ_total + usable_epochs"


def main() -> None:
    cfg = load_config()
    log = setup_logging(cfg, name="periodic_peak")

    deriv = Path(cfg["paths"]["derivatives_root"])
    ch_path = deriv / "specparam" / "specparam_channel_results_qc.csv"
    channel_df = pd.read_csv(ch_path)
    if "fit_valid" in channel_df.columns:
        channel_df = channel_df[channel_df["fit_valid"]]

    participants = load_analysis_participants(cfg)
    participants = attach_usable_epochs(participants, deriv)

    # 被试水平：各峰指标全通道均值
    subject_peak = channel_df.groupby(["subject_id", "group"])[PEAK_METRICS].mean().reset_index()
    df = participants.merge(subject_peak, on=["subject_id", "group"], how="inner")

    all_rows = []
    for metric in PEAK_METRICS:
        if metric not in df.columns:
            continue
        sub = df.dropna(subset=[metric, "group", "age_months", "sex", "IQ_total", "usable_epochs"])
        if len(sub) < 10:
            continue
        formula = FORMULA_TEMPLATE.format(metric=metric)
        try:
            res = run_ols(formula, sub)
            all_rows.extend(model_results_to_row(res, "periodic_peak", metric))
        except Exception as exc:
            log.warning("%s 失败: %s", metric, exc)

    if not all_rows:
        log.error("周期峰分析无结果（可能峰参数全为 NaN）")
        sys.exit(1)

    out_path = deriv / "stats" / "periodic_peak_analysis.csv"
    save_csv(pd.DataFrame(all_rows), out_path)
    log.info("周期峰分析完成: %s", out_path)


if __name__ == "__main__":
    main()
