# Ideal release name: 07b_table1_main_cohort.py
# Original path: scripts/07b_table1_main_cohort.py
# Note: Table 1 cohort characteristics
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
07b_table1_main_cohort.py
-------------------------
主分析队列（N=138）的表 1 / 表 2，与 08–11 纳入标准一致。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.paper_figures import load_main_cohort  # noqa: E402
from src.stats_utils import compare_groups_on_variable, descriptive_table  # noqa: E402


def main() -> None:
    cfg = load_config()
    log = setup_logging(cfg, name="table1_main_cohort")
    deriv = Path(cfg["paths"]["derivatives_root"])
    out = Path(cfg["paths"]["outputs_root"]) / "tables"
    out.mkdir(parents=True, exist_ok=True)

    cohort = load_main_cohort(cfg, deriv)
    cohort.to_csv(out / "main_cohort_subject_list.csv", index=False)

    cont_vars = [
        "age_months",
        "IQ_total",
        "ADOS_total",
        "ADOS_SA",
        "ADOS_RRB",
        "language_score",
    ]
    cont_vars = [v for v in cont_vars if v in cohort.columns]
    desc = descriptive_table(cohort, "group", cont_vars, continuous=cont_vars)
    comps = [
        compare_groups_on_variable(
            cohort, "group", v, cfg["groups"]["asd_label"], cfg["groups"]["td_label"]
        )
        for v in cont_vars
    ]
    desc.to_csv(out / "table1_main_cohort_descriptive.csv", index=False)
    pd.DataFrame(comps).to_csv(out / "table1_main_cohort_comparison.csv", index=False)

    preproc = pd.read_csv(deriv / "qc" / "preproc_summary.csv")
    sp = pd.read_csv(deriv / "specparam" / "specparam_qc_summary_subject.csv")
    df = cohort.copy()
    for src, cols in [
        (preproc, ["usable_seconds", "bad_channel_count"]),
        (sp, ["mean_r_squared", "invalid_channel_ratio"]),
    ]:
        add = [c for c in cols if c not in df.columns and c in src.columns]
        if add:
            df = df.merge(src[["subject_id"] + add], on="subject_id", how="left")

    qc_vars = [
        "usable_epochs",
        "usable_seconds",
        "bad_channel_count",
        "mean_r_squared",
        "invalid_channel_ratio",
    ]
    qc_rows = [
        compare_groups_on_variable(
            df, "group", v, cfg["groups"]["asd_label"], cfg["groups"]["td_label"]
        )
        for v in qc_vars
        if v in df.columns
    ]
    pd.DataFrame(qc_rows).to_csv(out / "table2_main_cohort_eeg_qc.csv", index=False)

    if "sex" in cohort.columns:
        ct = pd.crosstab(cohort["group"], cohort["sex"])
        _, p, _, _ = chi2_contingency(ct.values)
        log.info("主分析 N=%d; 性别卡方 p=%.4f", len(cohort), p)
    log.info("已保存 table1/table2 main cohort 至 %s", out)


if __name__ == "__main__":
    main()
