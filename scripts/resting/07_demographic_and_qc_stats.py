# Ideal release name: 07_demographic_and_qc_stats.py
# Original path: scripts/07_demographic_and_qc_stats.py
# Note: Demographics + QC stats
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
07_demographic_and_qc_stats.py
------------------------------
生成表 1（人口学/临床）与表 2（EEG/specparam 质量）。

输入: participants.csv, preproc_summary, specparam QC
输出: outputs/tables/table1_demographics.csv
      outputs/tables/table2_eeg_qc.csv
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
from src.io_utils import load_analysis_participants, load_participants, save_csv  # noqa: E402
from src.stats_utils import compare_groups_on_variable, descriptive_table  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成人口学与EEG QC统计表")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（例如 config/config_task_movie.yaml）",
    )
    return parser.parse_args()


def build_table1(participants: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """人口学与临床变量组间比较。"""
    groups = cfg["groups"]
    cont_vars = ["age_months", "IQ_total", "IQ_verbal", "IQ_nonverbal",
                 "ADOS_total", "ADOS_SA", "ADOS_RRB", "SRS_total", "language_score"]
    cont_vars = [v for v in cont_vars if v in participants.columns]

    desc = descriptive_table(participants, "group", cont_vars, continuous=cont_vars)

    comparisons = []
    for var in cont_vars:
        res = compare_groups_on_variable(
            participants, "group", var,
            groups["asd_label"], groups["td_label"],
        )
        comparisons.append(res)
    comp_df = pd.DataFrame(comparisons)

    # 性别卡方
    if "sex" in participants.columns:
        ct = pd.crosstab(participants["group"], participants["sex"])
        from src.stats_utils import chi_square_or_fisher
        chi_res = chi_square_or_fisher(ct.values)
        desc = pd.concat([desc, pd.DataFrame([{
            "variable": "sex",
            "test": chi_res["test"],
            "pvalue": chi_res["pvalue"],
        }])], ignore_index=True)

    return desc, comp_df


def build_table2(participants: pd.DataFrame, deriv: Path, cfg: dict) -> pd.DataFrame:
    """EEG 与 specparam 质量组间比较。"""
    groups = cfg["groups"]
    qc_vars = ["EEG_usable_epochs", "EEG_usable_seconds", "usable_epochs", "usable_seconds",
               "bad_channel_count", "mean_r_squared", "invalid_channel_ratio"]

    # 合并预处理 QC
    preproc_path = deriv / "qc" / "preproc_summary.csv"
    sp_qc_path = deriv / "specparam" / "specparam_qc_summary_subject.csv"
    df = participants.copy()

    if preproc_path.exists():
        preproc = pd.read_csv(preproc_path)
        preproc_cols = ["subject_id"] + [
            c for c in ("usable_epochs", "usable_seconds", "bad_channel_count") if c in preproc.columns
        ]
        df = df.merge(
            preproc[preproc_cols],
            on="subject_id", how="left", suffixes=("", "_preproc"),
        )
    if sp_qc_path.exists():
        sp_qc = pd.read_csv(sp_qc_path)
        df = df.merge(
            sp_qc[["subject_id", "mean_r_squared", "invalid_channel_ratio", "low_quality_subject"]],
            on="subject_id", how="left",
        )

    rows = []
    for var in qc_vars:
        if var not in df.columns:
            continue
        res = compare_groups_on_variable(
            df, "group", var, groups["asd_label"], groups["td_label"],
        )
        rows.append(res)
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    log = setup_logging(cfg, name="demographic_qc_stats")

    participants = load_analysis_participants(cfg)
    deriv = Path(cfg["paths"]["derivatives_root"])
    out_dir = Path(cfg["paths"]["outputs_root"]) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    desc, comp = build_table1(participants, cfg)
    save_csv(desc, out_dir / "table1_demographics_descriptive.csv")
    save_csv(comp, out_dir / "table1_demographics_comparison.csv")

    table2 = build_table2(participants, deriv, cfg)
    save_csv(table2, out_dir / "table2_eeg_qc.csv")
    log.info("表 1、表 2 已保存至 %s", out_dir)


if __name__ == "__main__":
    main()
