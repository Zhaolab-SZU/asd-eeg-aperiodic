# Ideal release name: 11_clinical_correlation.py
# Original path: scripts/11_clinical_correlation.py
# Note: ADOS / clinical associations
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
11_clinical_correlation.py
--------------------------
ASD 组内：aperiodic 参数与临床量表关系（OLS + Spearman）。

模型示例:
  ADOS_total ~ global_exponent + age_months + sex + IQ_total + usable_epochs
  language_score ~ temporal_exponent + ...

输入: participants, ROI global
输出: derivatives/stats/clinical_correlation_ols.csv
      derivatives/stats/clinical_correlation_spearman.csv
"""

from __future__ import annotations

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
from src.stats_utils import model_results_to_row, run_ols, spearman_correlation  # noqa: E402

COV = " + age_months + C(sex) + IQ_total + usable_epochs"

MODELS = [
    ("ADOS_total", "global_exponent", f"ADOS_total ~ global_exponent{COV}"),
    ("ADOS_SA", "global_exponent", f"ADOS_SA ~ global_exponent{COV}"),
    ("ADOS_RRB", "global_exponent", f"ADOS_RRB ~ global_exponent{COV}"),
    ("SRS_total", "global_exponent", f"SRS_total ~ global_exponent{COV}"),
    ("language_score", "temporal_exponent", f"language_score ~ temporal_exponent{COV}"),
]

SPEARMAN_PAIRS = [
    ("ADOS_total", "global_exponent"),
    ("ADOS_SA", "global_exponent"),
    ("ADOS_RRB", "global_exponent"),
    ("SRS_total", "global_exponent"),
    ("language_score", "temporal_exponent"),
]


def prepare_asd_df(participants: pd.DataFrame, roi_global: pd.DataFrame, deriv: Path) -> pd.DataFrame:
    """与主分析一致：ASD + ROI + specparam 被试 QC 通过。"""
    asd_label = "ASD"
    df = participants[participants["group"] == asd_label].merge(
        roi_global, on=["subject_id", "group"], how="inner",
    )
    df = attach_usable_epochs(df, deriv)
    return exclude_specparam_low_quality(df, deriv)


def main() -> None:
    cfg = load_config()
    log = setup_logging(cfg, name="clinical_correlation")

    deriv = Path(cfg["paths"]["derivatives_root"])
    roi_path = deriv / "roi" / "specparam_subject_global.csv"
    if not roi_path.exists():
        log.error("未找到 ROI 文件")
        sys.exit(1)

    participants = load_analysis_participants(cfg)
    roi_global = pd.read_csv(roi_path)
    df = prepare_asd_df(participants, roi_global, deriv)

    ols_rows = []
    for outcome, predictor, formula in MODELS:
        if outcome not in df.columns:
            log.warning("跳过 %s: 列不存在", outcome)
            continue
        n_outcome = pd.to_numeric(df[outcome], errors="coerce").notna().sum()
        if n_outcome == 0:
            log.warning("%s 无有效数据（participants 中该列全空），已跳过", outcome)
            continue
        sub = df.dropna(subset=[outcome, predictor, "age_months", "sex", "IQ_total", "usable_epochs"])
        if len(sub) < 8:
            log.warning("%s 协变量完整样本不足 n=%d（非缺失结局 n=%d）", outcome, len(sub), n_outcome)
            continue
        try:
            res = run_ols(formula, sub)
            for row in model_results_to_row(res, "clinical_ols", outcome, [predictor]):
                row["n_asd_complete"] = len(sub)
                ols_rows.append(row)
        except Exception as exc:
            log.warning("OLS 失败 %s: %s", outcome, exc)

    spear_rows = []
    for clinical, eeg_var in SPEARMAN_PAIRS:
        if clinical not in df.columns or eeg_var not in df.columns:
            continue
        x = pd.to_numeric(df[clinical], errors="coerce")
        y = pd.to_numeric(df[eeg_var], errors="coerce")
        if x.notna().sum() < 3:
            continue
        res = spearman_correlation(x, y)
        spear_rows.append({
            "clinical": clinical,
            "eeg_variable": eeg_var,
            **res,
        })

    stats_dir = deriv / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    if ols_rows:
        save_csv(pd.DataFrame(ols_rows), stats_dir / "clinical_correlation_ols.csv")
    if spear_rows:
        save_csv(pd.DataFrame(spear_rows), stats_dir / "clinical_correlation_spearman.csv")
    log.info("临床相关分析完成 (ASD n=%d)", len(df))


if __name__ == "__main__":
    main()
