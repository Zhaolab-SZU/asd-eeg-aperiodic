# Ideal release name: 15b_rematch_after_qc.py
# Original path: scripts/66_rematch_after_qc.py
# Note: Rematch after QC exclusions
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
66_rematch_after_qc.py
----------------------
基于全样本 QC 后可分析池重做 ASD/TD matching（推荐流程 A）。

输出：
- data/participants/participants_resting_matched_postqc.csv
- outputs_matched_resting/tables/postqc_rematch_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import (  # noqa: E402
    attach_usable_epochs,
    exclude_specparam_low_quality,
    load_analysis_participants,
    save_csv,
)


def _group_pvalues(asd: pd.DataFrame, td: pd.DataFrame) -> tuple[float, float, float]:
    p_age = float(stats.ttest_ind(asd["age_months"], td["age_months"], equal_var=False).pvalue)
    p_iq = float(stats.ttest_ind(asd["IQ_total"], td["IQ_total"], equal_var=False).pvalue)
    table = np.array(
        [
            [(asd["sex"] == "M").sum(), (asd["sex"] == "F").sum()],
            [(td["sex"] == "M").sum(), (td["sex"] == "F").sum()],
        ],
    )
    p_sex = float(stats.fisher_exact(table)[1])
    return p_age, p_iq, p_sex


def _build_qc_pool() -> pd.DataFrame:
    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    deriv = Path(cfg["paths"]["derivatives_root"])
    participants = load_analysis_participants(cfg)
    roi = pd.read_csv(deriv / "roi" / "specparam_subject_global.csv")

    df = participants.merge(roi, on=["subject_id", "group"], how="inner")
    df = attach_usable_epochs(df, deriv)
    df = exclude_specparam_low_quality(df, deriv)
    df = df[df["group"].astype(str).str.upper().isin(["ASD", "TD"])].copy()
    df["group"] = df["group"].astype(str).str.upper()
    df["sex"] = df["sex"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["age_months", "IQ_total", "sex"])
    df = df[df["sex"].isin(["M", "F"])].copy()
    return df.reset_index(drop=True)


def _run_matching(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    asd = df[df["group"] == "ASD"].reset_index(drop=True)
    td = df[df["group"] == "TD"].reset_index(drop=True)

    a = asd[["age_months", "IQ_total"]].to_numpy(float)
    t = td[["age_months", "IQ_total"]].to_numpy(float)
    a_sex = (asd["sex"].to_numpy() == "M").astype(int)
    t_sex = (td["sex"].to_numpy() == "M").astype(int)

    # pooled SD 标准化，兼顾年龄与IQ。
    sd_age = np.std(np.concatenate([a[:, 0], t[:, 0]])) or 1.0
    sd_iq = np.std(np.concatenate([a[:, 1], t[:, 1]])) or 1.0
    d_age = np.abs((a[:, None, 0] - t[None, :, 0]) / sd_age)
    d_iq = np.abs((a[:, None, 1] - t[None, :, 1]) / sd_iq)
    d_sex = (a_sex[:, None] != t_sex[None, :]).astype(float)

    best = None
    for w_age in [0.6, 1.0, 1.4, 2.0]:
        for w_iq in [0.6, 1.0, 1.4, 2.0, 2.6]:
            for w_sex in [0.0, 0.5, 1.0, 2.0, 3.5]:
                cost = w_age * d_age + w_iq * d_iq + w_sex * d_sex
                r, c = linear_sum_assignment(cost)
                pair_cost = cost[r, c]
                order = np.argsort(pair_cost)
                r_ord = r[order]
                c_ord = c[order]

                for k in range(len(r_ord), 9, -1):
                    ai = r_ord[:k]
                    ti = c_ord[:k]
                    a_sub = asd.iloc[ai].copy()
                    t_sub = td.iloc[ti].copy()
                    p_age, p_iq, p_sex = _group_pvalues(a_sub, t_sub)
                    if p_age > 0.05 and p_iq > 0.05 and p_sex > 0.05:
                        score = min(p_age, p_iq, p_sex)
                        cand = (k, score, p_age, p_iq, p_sex, ai, ti, w_age, w_iq, w_sex)
                        if (best is None) or (cand[0] > best[0]) or (cand[0] == best[0] and cand[1] > best[1]):
                            best = cand
                        break

    if best is None:
        raise RuntimeError("未找到同时满足 age/IQ/sex 平衡阈值的匹配结果。")

    k, score, p_age, p_iq, p_sex, ai, ti, w_age, w_iq, w_sex = best
    matched = pd.concat([asd.iloc[ai], td.iloc[ti]], axis=0).sort_values(["group", "subject_id"]).copy()
    meta = {
        "n_per_group": int(k),
        "p_age": float(p_age),
        "p_iq": float(p_iq),
        "p_sex": float(p_sex),
        "w_age": float(w_age),
        "w_iq": float(w_iq),
        "w_sex": float(w_sex),
        "score_min_p": float(score),
    }
    return matched, meta


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    log = setup_logging(cfg, name="postqc_rematch")

    pool = _build_qc_pool()
    matched, meta = _run_matching(pool)

    out_participants = PROJECT_ROOT / "data" / "participants" / "participants_resting_matched_postqc.csv"
    matched.to_csv(out_participants, index=False, encoding="utf-8-sig")

    asd = matched[matched["group"] == "ASD"]
    td = matched[matched["group"] == "TD"]
    summary = pd.DataFrame(
        [
            {
                **meta,
                "n_total": int(len(matched)),
                "asd_male": int((asd["sex"] == "M").sum()),
                "asd_female": int((asd["sex"] == "F").sum()),
                "td_male": int((td["sex"] == "M").sum()),
                "td_female": int((td["sex"] == "F").sum()),
                "asd_age_mean": float(asd["age_months"].mean()),
                "td_age_mean": float(td["age_months"].mean()),
                "asd_iq_mean": float(asd["IQ_total"].mean()),
                "td_iq_mean": float(td["IQ_total"].mean()),
            },
        ],
    )

    out_table = PROJECT_ROOT / "outputs_matched_resting" / "tables" / "postqc_rematch_summary.csv"
    save_csv(summary, out_table)
    log.info("保存重匹配 participants: %s", out_participants)
    log.info("保存重匹配摘要: %s", out_table)


if __name__ == "__main__":
    main()
