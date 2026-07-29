# Ideal release name: 14_sensitivity_analysis.py
# Original path: scripts/14_sensitivity_analysis.py
# Note: Primary sensitivity suite (IQ-balance, covariates, etc.)
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
14_sensitivity_analysis.py
--------------------------
敏感性分析：频率范围、epoch 阈值、specparam 模式、额外 QC 协变量。

输出: derivatives/stats/sensitivity_analysis_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import attach_usable_epochs, load_analysis_participants, save_csv  # noqa: E402
from src.specparam_utils import run_specparam_batch  # noqa: E402
from src.stats_utils import run_ols  # noqa: E402

COV = " + age_months + C(sex) + IQ_total + usable_epochs"


def run_group_model(roi_global: pd.DataFrame, participants: pd.DataFrame, min_epochs: int) -> dict:
    """单次敏感性条件下的主效应估计。"""
    df = participants.merge(roi_global, on=["subject_id", "group"], how="inner")
    if "usable_epochs" in df.columns:
        df = df[df["usable_epochs"] >= min_epochs]
    sub = df.dropna(subset=["global_exponent", "group", "age_months", "sex", "IQ_total", "usable_epochs"])
    if len(sub) < 10:
        return {"n": len(sub), "group_coef": np.nan, "group_p": np.nan}
    try:
        res = run_ols(f"global_exponent ~ C(group){COV}", sub)
        terms = [t for t in res.params.index if t.startswith("C(group)")]
        term = terms[0] if terms else None
        if term is None:
            return {"n": len(sub), "group_coef": np.nan, "group_p": np.nan}
        return {
            "n": int(res.nobs),
            "group_coef": float(res.params[term]),
            "group_p": float(res.pvalues[term]),
        }
    except Exception:
        return {"n": len(sub), "group_coef": np.nan, "group_p": np.nan}


def main() -> None:
    cfg = load_config()
    log = setup_logging(cfg, name="sensitivity_analysis")

    participants = load_analysis_participants(cfg)
    deriv = Path(cfg["paths"]["derivatives_root"])
    psd_dir = deriv / "psd"
    sens_cfg = cfg.get("sensitivity", {})
    roi_path = deriv / "roi" / "specparam_subject_global.csv"

    rows = []
    # --- 1. 不同频率范围 & specparam 模式（重新拟合）---
    for freq_range in sens_cfg.get("freq_ranges", [[1, 40]]):
        for mode in sens_cfg.get("aperiodic_modes", ["fixed"]):
            label = f"freq_{freq_range[0]}_{freq_range[1]}_mode_{mode}"
            tmp_csv = deriv / "specparam" / f"sens_{label}.csv"
            override = {"freq_range": freq_range, "aperiodic_mode": mode}
            try:
                run_specparam_batch(
                    participants, psd_dir, tmp_csv, cfg, sp_cfg_override=override,
                )
                # 简化：仅记录拟合完成；完整流程应重跑 ROI
                rows.append({
                    "analysis": label,
                    "type": "specparam_refit",
                    "status": "completed",
                    "note": "需结合 ROI 重算后比较 group 效应",
                })
            except Exception as exc:
                rows.append({
                    "analysis": label,
                    "type": "specparam_refit",
                    "status": "failed",
                    "note": str(exc),
                })

    # --- 2. 不同 epoch 阈值（基于已有 ROI）---
    if roi_path.exists():
        roi_global = pd.read_csv(roi_path)
        participants = attach_usable_epochs(participants, deriv)
        preproc = deriv / "qc" / "preproc_summary.csv"
        if preproc.exists() and "bad_channel_count" not in participants.columns:
            participants = participants.merge(
                pd.read_csv(preproc)[["subject_id", "bad_channel_count"]],
                on="subject_id", how="left",
            )
        for min_ep in sens_cfg.get("min_epochs_thresholds", [30, 60]):
            res = run_group_model(roi_global, participants, min_ep)
            rows.append({
                "analysis": f"min_epochs_{min_ep}",
                "type": "epoch_threshold",
                **res,
            })

        # --- 3. 额外 QC 协变量（记录可用性）---
        for cov in ["bad_channel_count", "mean_r_squared"]:
            rows.append({
                "analysis": f"extra_covariate_{cov}",
                "type": "qc_covariate",
                "available": cov in participants.columns,
            })

    out_path = deriv / "stats" / "sensitivity_analysis_summary.csv"
    save_csv(pd.DataFrame(rows), out_path)
    log.info("敏感性分析摘要: %s (%d 行)", out_path, len(rows))


if __name__ == "__main__":
    main()
