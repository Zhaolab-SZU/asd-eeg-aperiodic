# Ideal release name: 15_match_iq_and_postqc_cohorts.py
# Original path: scripts/69_build_postqc_matched_cohorts.py
# Note: IQ / post-QC matched cohort construction
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
69_build_postqc_matched_cohorts.py
----------------------------------
构建三套“QC后再匹配”队列：
1) resting_postqc_matched
2) movie_postqc_matched
3) resting_movie_both_postqc_matched（同时通过 resting 与 movie QC）

输出：
- data/participants/participants_resting_matched_postqc.csv
- data/participants/participants_task_movie_matched_postqc.csv
- data/participants/participants_resting_movie_matched_postqc.csv
- outputs/tables/qc_postmatch_cohort_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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


def _group_pvals(a_sub: pd.DataFrame, t_sub: pd.DataFrame) -> tuple[float, float, float]:
    p_age = float(stats.ttest_ind(a_sub["age_months"], t_sub["age_months"], equal_var=False).pvalue)
    p_iq = float(stats.ttest_ind(a_sub["IQ_total"], t_sub["IQ_total"], equal_var=False).pvalue)
    table = np.array(
        [
            [(a_sub["sex"] == "M").sum(), (a_sub["sex"] == "F").sum()],
            [(t_sub["sex"] == "M").sum(), (t_sub["sex"] == "F").sum()],
        ],
    )
    p_sex = float(stats.fisher_exact(table)[1])
    return p_age, p_iq, p_sex


def _match_on_demo(cand: pd.DataFrame, min_k: int = 10) -> tuple[pd.DataFrame, dict[str, Any]]:
    cand = cand.copy()
    cand["group"] = cand["group"].astype(str).str.upper()
    cand["sex"] = cand["sex"].astype(str).str.upper().str.strip()
    cand = cand.dropna(subset=["age_months", "IQ_total", "sex"])
    cand = cand[cand["sex"].isin(["M", "F"])].copy()

    asd = cand[cand["group"] == "ASD"].reset_index(drop=True)
    td = cand[cand["group"] == "TD"].reset_index(drop=True)
    if len(asd) < min_k or len(td) < min_k:
        raise ValueError(f"样本不足: ASD={len(asd)}, TD={len(td)}")

    a = asd[["age_months", "IQ_total"]].to_numpy(float)
    t = td[["age_months", "IQ_total"]].to_numpy(float)
    a_sex = (asd["sex"].to_numpy() == "M").astype(int)
    t_sex = (td["sex"].to_numpy() == "M").astype(int)

    sd_age = np.std(np.concatenate([a[:, 0], t[:, 0]])) or 1.0
    sd_iq = np.std(np.concatenate([a[:, 1], t[:, 1]])) or 1.0
    d_age = np.abs((a[:, None, 0] - t[None, :, 0]) / sd_age)
    d_iq = np.abs((a[:, None, 1] - t[None, :, 1]) / sd_iq)
    d_sex = (a_sex[:, None] != t_sex[None, :]).astype(float)

    best_meet = None
    best_any = None
    for w_age in [0.6, 1.0, 1.4, 2.0]:
        for w_iq in [0.6, 1.0, 1.4, 2.0, 2.6]:
            for w_sex in [0.0, 0.5, 1.0, 2.0, 3.5]:
                cost = w_age * d_age + w_iq * d_iq + w_sex * d_sex
                r, c = linear_sum_assignment(cost)
                pair_cost = cost[r, c]
                order = np.argsort(pair_cost)
                r_ord = r[order]
                c_ord = c[order]

                for k in range(len(r_ord), min_k - 1, -1):
                    ai = r_ord[:k]
                    ti = c_ord[:k]
                    a_sub = asd.iloc[ai].copy()
                    t_sub = td.iloc[ti].copy()
                    p_age, p_iq, p_sex = _group_pvals(a_sub, t_sub)
                    score = min(p_age, p_iq, p_sex)
                    cand_meta = (k, score, p_age, p_iq, p_sex, ai, ti, w_age, w_iq, w_sex)
                    if (best_any is None) or (cand_meta[0] > best_any[0]) or (
                        cand_meta[0] == best_any[0] and cand_meta[1] > best_any[1]
                    ):
                        best_any = cand_meta
                    if p_age > 0.05 and p_iq > 0.05 and p_sex > 0.05:
                        if (best_meet is None) or (cand_meta[0] > best_meet[0]) or (
                            cand_meta[0] == best_meet[0] and cand_meta[1] > best_meet[1]
                        ):
                            best_meet = cand_meta
                        break

    if best_meet is not None:
        best = best_meet
        threshold_met = True
    elif best_any is not None:
        best = best_any
        threshold_met = False
    else:
        raise RuntimeError("匹配失败：未找到可行匹配解。")

    k, score, p_age, p_iq, p_sex, ai, ti, w_age, w_iq, w_sex = best
    a_sub = asd.iloc[ai].copy()
    t_sub = td.iloc[ti].copy()
    a_sub["matched_pair_id"] = np.arange(len(a_sub))
    t_sub["matched_pair_id"] = np.arange(len(t_sub))
    out = pd.concat([a_sub, t_sub], axis=0).sort_values(["matched_pair_id", "group"]).reset_index(drop=True)
    meta = {
        "n_per_group": int(k),
        "p_age": float(p_age),
        "p_iq": float(p_iq),
        "p_sex": float(p_sex),
        "score_min_p": float(score),
        "w_age": float(w_age),
        "w_iq": float(w_iq),
        "w_sex": float(w_sex),
        "threshold_met": bool(threshold_met),
    }
    return out, meta


def _build_qc_pool(cfg_path: Path) -> pd.DataFrame:
    cfg = load_config(cfg_path)
    deriv = Path(cfg["paths"]["derivatives_root"])
    participants = load_analysis_participants(cfg)
    roi = pd.read_csv(deriv / "roi" / "specparam_subject_global.csv")
    df = participants.merge(roi, on=["subject_id", "group"], how="inner")
    df = attach_usable_epochs(df, deriv)
    df = exclude_specparam_low_quality(df, deriv)
    return df.reset_index(drop=True)


def _summary_row(name: str, matched: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    asd = matched[matched["group"].astype(str).str.upper() == "ASD"]
    td = matched[matched["group"].astype(str).str.upper() == "TD"]
    return {
        "cohort": name,
        "n_total": int(len(matched)),
        "n_asd": int(len(asd)),
        "n_td": int(len(td)),
        "asd_age_mean": float(pd.to_numeric(asd["age_months"], errors="coerce").mean()),
        "td_age_mean": float(pd.to_numeric(td["age_months"], errors="coerce").mean()),
        "asd_iq_mean": float(pd.to_numeric(asd["IQ_total"], errors="coerce").mean()),
        "td_iq_mean": float(pd.to_numeric(td["IQ_total"], errors="coerce").mean()),
        "asd_male": int((asd["sex"].astype(str).str.upper() == "M").sum()),
        "asd_female": int((asd["sex"].astype(str).str.upper() == "F").sum()),
        "td_male": int((td["sex"].astype(str).str.upper() == "M").sum()),
        "td_female": int((td["sex"].astype(str).str.upper() == "F").sum()),
        **meta,
    }


def main() -> None:
    log = setup_logging(load_config(), name="build_postqc_matched_cohorts")
    p_rest = PROJECT_ROOT / "data" / "participants" / "participants.csv"
    p_movie = PROJECT_ROOT / "data" / "participants" / "participants_task_movie.csv"
    rest_src = pd.read_csv(p_rest)
    movie_src = pd.read_csv(p_movie)
    for d in (rest_src, movie_src):
        d["subject_id"] = d["subject_id"].astype(str)
        d["group"] = d["group"].astype(str).str.upper()
        d["sex"] = d["sex"].astype(str).str.upper().str.strip()

    rest_pool = _build_qc_pool(PROJECT_ROOT / "config" / "config.yaml")
    movie_pool = _build_qc_pool(PROJECT_ROOT / "config" / "config_task_movie.yaml")
    for d in (rest_pool, movie_pool):
        d["subject_id"] = d["subject_id"].astype(str)
        d["group"] = d["group"].astype(str).str.upper()

    # 1) resting QC matched (refresh)
    rest_cand = rest_src.merge(
        rest_pool[["subject_id", "group"]].drop_duplicates(),
        on=["subject_id", "group"],
        how="inner",
    )
    rest_matched, rest_meta = _match_on_demo(rest_cand)
    out_rest = PROJECT_ROOT / "data" / "participants" / "participants_resting_matched_postqc.csv"
    rest_matched.to_csv(out_rest, index=False, encoding="utf-8-sig")

    # 2) movie QC matched
    movie_cand = movie_src.merge(
        movie_pool[["subject_id", "group"]].drop_duplicates(),
        on=["subject_id", "group"],
        how="inner",
    )
    movie_matched, movie_meta = _match_on_demo(movie_cand)
    out_movie = PROJECT_ROOT / "data" / "participants" / "participants_task_movie_matched_postqc.csv"
    movie_matched.to_csv(out_movie, index=False, encoding="utf-8-sig")

    # 3) both resting+movie QC matched
    both_ids = (
        rest_pool[["subject_id", "group"]].drop_duplicates()
        .merge(movie_pool[["subject_id", "group"]].drop_duplicates(), on=["subject_id", "group"], how="inner")
    )
    both_cand = rest_src.merge(both_ids, on=["subject_id", "group"], how="inner")
    both_matched, both_meta = _match_on_demo(both_cand)

    pooled_age = pd.to_numeric(both_cand["age_months"], errors="coerce")
    pooled_iq = pd.to_numeric(both_cand["IQ_total"], errors="coerce")
    age_z = (pd.to_numeric(both_matched["age_months"], errors="coerce") - pooled_age.mean()) / (pooled_age.std() or 1.0)
    iq_z = (pd.to_numeric(both_matched["IQ_total"], errors="coerce") - pooled_iq.mean()) / (pooled_iq.std() or 1.0)
    both_out = both_matched[["subject_id", "group", "sex", "age_months", "IQ_total", "matched_pair_id"]].copy()
    both_out["age_z"] = age_z
    both_out["iq_z"] = iq_z
    both_out["has_resting"] = 1
    both_out["has_movie"] = 1
    both_out = both_out[
        [
            "subject_id",
            "group",
            "sex",
            "age_months",
            "IQ_total",
            "age_z",
            "iq_z",
            "matched_pair_id",
            "has_resting",
            "has_movie",
        ]
    ].sort_values(["matched_pair_id", "group"])
    out_both = PROJECT_ROOT / "data" / "participants" / "participants_resting_movie_matched_postqc.csv"
    both_out.to_csv(out_both, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            _summary_row("resting_postqc_matched", rest_matched, rest_meta),
            _summary_row("movie_postqc_matched", movie_matched, movie_meta),
            _summary_row("resting_movie_both_postqc_matched", both_matched, both_meta),
        ],
    )
    out_summary = PROJECT_ROOT / "outputs" / "tables" / "qc_postmatch_cohort_summary.csv"
    save_csv(summary, out_summary)

    log.info("Saved: %s", out_rest)
    log.info("Saved: %s", out_movie)
    log.info("Saved: %s", out_both)
    log.info("Saved: %s", out_summary)


if __name__ == "__main__":
    main()
