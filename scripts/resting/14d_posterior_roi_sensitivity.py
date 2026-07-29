# Ideal release name: 14d_posterior_roi_sensitivity.py
# Original path: scripts/95_posterior_roi_sensitivity.py
# Note: Posterior ROI definition sensitivity
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
95_posterior_roi_sensitivity.py
--------------------------------
审稿防御：对比 FDR 显著 4 导（posterior core）与预设 occipital ROI（13 导）的组效应。

输入: derivatives/specparam/specparam_channel_results_qc.csv
      config/roi_channels.yaml
输出: outputs/tables/posterior_roi_sensitivity/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, load_roi_config, setup_logging  # noqa: E402
from src.io_utils import (  # noqa: E402
    attach_usable_epochs,
    exclude_specparam_low_quality,
    load_analysis_participants,
    save_csv,
)
from src.spectral_maturation_analysis import POSTERIOR_CORE  # noqa: E402
from src.stats_utils import compare_groups_on_variable, model_results_to_row, run_ols  # noqa: E402

COVARIATE_FORMULA = " + age_months + C(sex) + IQ_total + usable_epochs"
FORMULA_TEMPLATE = "{outcome} ~ C(group, Treatment(reference='ASD')){cov}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="4 导 core vs 13 导 occipital 敏感性")
    p.add_argument("--config", type=str, default=None)
    return p.parse_args()


def _channel_mean(
    ch_df: pd.DataFrame,
    channels: list[str],
    min_ratio: float = 0.5,
) -> pd.DataFrame:
    if "fit_valid" in ch_df.columns:
        ch_df = ch_df[ch_df["fit_valid"].astype(bool)].copy()
    rows = []
    for (sid, grp), sub in ch_df.groupby(["subject_id", "group"]):
        roi = sub[sub["channel"].isin(channels)]
        n_req = len(channels)
        n_valid = roi["aperiodic_exponent"].notna().sum()
        if n_valid < min_ratio * n_req:
            val = np.nan
        else:
            val = float(roi["aperiodic_exponent"].mean())
        rows.append(
            {
                "subject_id": str(sid),
                "group": grp,
                "aperiodic_exponent": val,
                "n_valid_channels": int(n_valid),
                "n_required_channels": n_req,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    log = setup_logging(cfg, name="posterior_roi_sensitivity")

    deriv = Path(cfg["paths"]["derivatives_root"])
    out_dir = Path(cfg["paths"]["outputs_root"]) / "tables" / "posterior_roi_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    roi_cfg = load_roi_config()
    layout = cfg.get("eeg", {}).get("roi_layout") or roi_cfg.get("default_layout", "channels_egi64")
    occipital_ch = list(roi_cfg[layout]["occipital"])

    ch_path = deriv / "specparam" / "specparam_channel_results_qc.csv"
    if not ch_path.exists():
        log.error("未找到 %s", ch_path)
        sys.exit(1)

    ch_df = pd.read_csv(ch_path)
    core_df = _channel_mean(ch_df, POSTERIOR_CORE).rename(
        columns={"aperiodic_exponent": "posterior_core_4_exponent"}
    )
    occ_df = _channel_mean(ch_df, occipital_ch).rename(
        columns={"aperiodic_exponent": "occipital_roi_13_exponent"}
    )

    subj = core_df.merge(
        occ_df[
            [
                "subject_id",
                "occipital_roi_13_exponent",
                "n_valid_channels",
                "n_required_channels",
            ]
        ],
        on="subject_id",
        how="outer",
        suffixes=("_core", "_occ"),
    )
    subj = subj.rename(
        columns={
            "n_valid_channels_core": "n_valid_core4",
            "n_required_channels_core": "n_required_core4",
            "n_valid_channels": "n_valid_occ13",
            "n_required_channels": "n_required_occ13",
        }
    )

    participants = load_analysis_participants(cfg)
    participants = exclude_specparam_low_quality(participants, deriv)
    participants = attach_usable_epochs(participants, deriv)
    min_ep = int(cfg.get("epochs", {}).get("min_usable_epochs", 60))
    if "usable_epochs" in participants.columns:
        participants = participants[participants["usable_epochs"] >= min_ep].copy()

    df = subj.merge(
        participants[
            [
                c
                for c in (
                    "subject_id",
                    "group",
                    "age_months",
                    "sex",
                    "IQ_total",
                    "usable_epochs",
                )
                if c in participants.columns or c == "subject_id"
            ]
        ],
        on="subject_id",
        how="inner",
        suffixes=("", "_p"),
    )
    if "group_p" in df.columns:
        df["group"] = df["group"].fillna(df["group_p"])
        df = df.drop(columns=["group_p"])

    save_csv(df, out_dir / "subject_level_posterior_metrics.csv")

    mask = df["posterior_core_4_exponent"].notna() & df["occipital_roi_13_exponent"].notna()
    r, r_p = stats.pearsonr(
        df.loc[mask, "posterior_core_4_exponent"],
        df.loc[mask, "occipital_roi_13_exponent"],
    )

    desc_rows = []
    model_rows = []
    for outcome, label in (
        ("posterior_core_4_exponent", "posterior_core_4"),
        ("occipital_roi_13_exponent", "occipital_roi_13"),
    ):
        desc_rows.append(compare_groups_on_variable(df, "group", outcome, "ASD", "TD"))
        sub = df.dropna(
            subset=[outcome, "group", "age_months", "sex", "IQ_total", "usable_epochs"]
        )
        formula = FORMULA_TEMPLATE.format(outcome=outcome, cov=COVARIATE_FORMULA)
        res = run_ols(formula, sub)
        model_rows.extend(
            model_results_to_row(
                res,
                model_name=label,
                outcome=outcome,
                predictors=[t for t in res.params.index if "group" in t],
            )
        )

    desc_df = pd.DataFrame(desc_rows)
    model_df = pd.DataFrame(model_rows)
    save_csv(desc_df, out_dir / "descriptive_comparison.csv")
    save_csv(model_df, out_dir / "group_ols_models.csv")

    core_row = model_df[model_df["model"] == "posterior_core_4"].iloc[0]
    occ_row = model_df[model_df["model"] == "occipital_roi_13"].iloc[0]

    fdr_channels = ", ".join(POSTERIOR_CORE)
    occ_list = ", ".join(occipital_ch)

    note = f"""# 后部电极聚合敏感性（审稿补充）

主分析终点为 **global aperiodic exponent**；本表用于回应「E33/E36/E37/E38 是否事后挑选」。

## 定义

| 指标 | 电极 | 数量 | 依据 |
|------|------|------|------|
| posterior_core_4 | {fdr_channels} | 4 | 64 导 FDR 探索后**全部**显著通道；均属于预设 occipital ROI |
| occipital_roi_13 | {occ_list} | 13 | `config/roi_channels.yaml` 中 **事先划定** 的 occipital 组（与通道 FDR 无关） |

分析队列: **N = {len(df)}**（与主分析一致）

## 被试级相关

posterior_core_4 与 occipital_roi_13: **r = {r:.3f}, p = {r_p:.2e}**

## 未校正描述（ASD vs TD）

（见 `descriptive_comparison.csv`）

## 主模型协变量（与 script 08 一致）

`exponent ~ C(group, ref=ASD) + age_months + sex + IQ_total + usable_epochs`

| 指标 | TD − ASD β | 95% CI | p |
|------|------------|--------|---|
| posterior_core_4 | {core_row['coef']:.4f} | [{core_row['ci_low']:.4f}, {core_row['ci_high']:.4f}] | {core_row['pvalue']:.4f} |
| occipital_roi_13 | {occ_row['coef']:.4f} | [{occ_row['ci_low']:.4f}, {occ_row['ci_high']:.4f}] | {occ_row['pvalue']:.4f} |

## 解读（建议写入 Discussion / 回复审稿人）

1. 若两指标 **方向一致、均显著**：组效应反映 **顶枕区宽带偏移**，而非仅 4 个事后显著电极。
2. 四导集合是 occipital ROI 的子集；ROI 混合模型已显示显著的 **group×occipital** 交互。
3. 四导用于补充图/二次分析时应标明 **exploratory**；primary 仍为 global exponent。

## 输出文件

- `subject_level_posterior_metrics.csv`
- `descriptive_comparison.csv`
- `group_ols_models.csv`
"""
    (out_dir / "reviewer_note_zh.md").write_text(note, encoding="utf-8")
    log.info("完成: %s (N=%d, r=%.3f)", out_dir, len(df), r)


if __name__ == "__main__":
    main()
