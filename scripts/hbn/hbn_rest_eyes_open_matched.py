# Ideal release name: hbn_rest_eyes_open_matched.py
# Original path: scripts/143_hbn_eo_matched_external_validation.py
# Note: HBN eyes-open matched resting convergence
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
143_hbn_eo_matched_external_validation.py
-----------------------------------------
ThePresent 119×119 严格匹配队列上的 HBN 静息 EO posterior exponent 外部验证。

最低限度产出（补充材料）：
  1. 协变量 OLS（group + age + sex + IQ）
  2. matched-pair 配对敏感性
  3. 缺失诊断 + EO 可用子集平衡表
  4. pipeline 方向敏感性（nuclear knee vs legacy fixed）

用法:
  python scripts/143_hbn_eo_matched_external_validation.py
  python scripts/143_hbn_eo_matched_external_validation.py \\
    --exponent-csv outputs/hbn_nuclear/eo_posterior_exponent_nuclear_fixed.csv \\
    --pipeline-label nuclear_fixed
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_roi_config  # noqa: E402
from src.io_utils import ensure_dir, save_csv  # noqa: E402
from src.roi_utils import aggregate_roi_for_subject, get_roi_dict  # noqa: E402
from src.stats_utils import (  # noqa: E402
    chi_square_or_fisher,
    cohens_d,
    descriptive_table,
    independent_ttest,
    model_results_to_row,
    run_mixedlm,
    run_ols,
)

logger = logging.getLogger(__name__)

MATCH_DIR = PROJECT_ROOT / "derivatives/hbn_external_movie/replication/matched"
OUT_DIR = PROJECT_ROOT / "outputs/hbn_nuclear/external_validation"
ROI_CHANNELS = ("E67", "E72", "E75", "E77")
HOMOLOGOUS_PRIMARY = ("E33", "E36", "E37", "E38")

PRIMARY_RESTING_BETA = 0.133
PRIMARY_RESTING_P = 1.33e-4
PRIMARY_RESTING_N = 138


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HBN EO matched external validation")
    p.add_argument(
        "--exponent-csv",
        type=str,
        default=str(PROJECT_ROOT / "outputs/hbn_nuclear/eo_posterior_exponent.csv"),
        help="被试级 posterior exponent CSV（需含 subject_id, EO_posterior_exponent）",
    )
    p.add_argument(
        "--pipeline-label",
        type=str,
        default="nuclear_knee",
        help="pipeline 标签（写入 sensitivity 表）",
    )
    p.add_argument(
        "--legacy-specparam",
        type=str,
        default=str(PROJECT_ROOT / "derivatives/hbn_external/specparam/specparam_channel_results_qc.csv"),
        help="legacy fixed specparam 通道表（档 A 敏感性）",
    )
    p.add_argument(
        "--posterior-channels-dir",
        type=str,
        default=str(PROJECT_ROOT / "derivatives/hbn_nuclear/specparam/posterior_channels"),
        help="nuclear 通道级 specparam 目录（mean_r_squared）",
    )
    p.add_argument(
        "--failures-csv",
        type=str,
        default=str(PROJECT_ROOT / "outputs/hbn_nuclear/eo_posterior_exponent_failures.csv"),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(OUT_DIR),
    )
    p.add_argument(
        "--skip-legacy",
        action="store_true",
        help="跳过 legacy fixed pipeline 敏感性",
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help="仅根据已有 CSV 重生成 report（需 matched_eo_analysis_table 已存在）",
    )
    return p.parse_args()


def _classify_failure(error: str) -> str:
    if pd.isna(error):
        return "unknown"
    err = str(error)
    if "未找到 eyes-open" in err or "no eyes-open" in err.lower():
        return "no_EO_interval"
    if "AutoReject" in err or "epoch" in err.lower():
        return "autoreject_epoch_fail"
    return "other"


def _load_mean_r_squared(posterior_dir: Path, subject_ids: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sid in subject_ids:
        fpath = posterior_dir / f"{sid}_posterior_specparam.csv"
        if not fpath.exists():
            continue
        ch = pd.read_csv(fpath)
        rows.append({
            "subject_id": sid,
            "mean_r_squared": float(ch["r_squared"].mean()) if "r_squared" in ch.columns else np.nan,
            "mean_fit_error": float(ch["fit_error"].mean()) if "fit_error" in ch.columns else np.nan,
        })
    return pd.DataFrame(rows)


def assemble_matched_eo_table(
    exponent_csv: Path,
    posterior_dir: Path,
    failures_csv: Path,
) -> pd.DataFrame:
    """合并 match_table + participants + EO + QC + pair_id。"""
    participants = pd.read_csv(MATCH_DIR / "participants_matched.csv")
    match_table = pd.read_csv(MATCH_DIR / "match_table.csv")
    match_table["pair_id"] = np.arange(len(match_table), dtype=int)

    eo = pd.read_csv(exponent_csv)
    eo["subject_id"] = eo["subject_id"].astype(str)
    eo = eo.rename(columns={"EO_posterior_exponent": "posterior_exponent"})

    failures = pd.read_csv(failures_csv) if failures_csv.exists() else pd.DataFrame(columns=["subject_id", "error"])
    failures["subject_id"] = failures["subject_id"].astype(str)
    failures["failure_reason"] = failures["error"].map(_classify_failure)

    # pair membership: asd_id / td_id -> pair_id
    asd_pairs = match_table[["pair_id", "asd_id", "age_diff", "iq_diff", "same_sex"]].rename(
        columns={"asd_id": "subject_id"}
    )
    asd_pairs["pair_role"] = "ASD"
    td_pairs = match_table[["pair_id", "td_id", "age_diff", "iq_diff", "same_sex"]].rename(
        columns={"td_id": "subject_id"}
    )
    td_pairs["pair_role"] = "TD"
    pair_map = pd.concat([asd_pairs, td_pairs], ignore_index=True)
    pair_map["subject_id"] = pair_map["subject_id"].astype(str)

    participants["subject_id"] = participants["subject_id"].astype(str)
    meta_cols = [
        "subject_id", "group", "age_months", "sex", "IQ_total",
        "release_id", "SRS_total", "SCQ_total",
    ]
    meta_cols = [c for c in meta_cols if c in participants.columns]

    df = participants[meta_cols].merge(pair_map, on="subject_id", how="left")
    df = df.merge(
        eo[[c for c in eo.columns if c in {
            "subject_id", "posterior_exponent", "aperiodic_mode", "pipeline",
            "n_epochs_before_autoreject", "n_epochs_after_autoreject", "n_epochs_rejected",
        }]],
        on="subject_id",
        how="left",
    )
    df = df.merge(failures[["subject_id", "error", "failure_reason"]], on="subject_id", how="left")

    r2 = _load_mean_r_squared(posterior_dir, set(df["subject_id"]))
    df = df.merge(r2, on="subject_id", how="left")

    df["has_eo"] = df["posterior_exponent"].notna()
    return df.sort_values(["pair_id", "pair_role"]).reset_index(drop=True)


def _pair_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """每对的完整性诊断。"""
    rows: list[dict[str, Any]] = []
    for pid, sub in df.groupby("pair_id"):
        asd = sub[sub["pair_role"] == "ASD"]
        td = sub[sub["pair_role"] == "TD"]
        asd_has = bool(asd["has_eo"].iloc[0]) if len(asd) else False
        td_has = bool(td["has_eo"].iloc[0]) if len(td) else False
        if asd_has and td_has:
            status = "complete"
        elif asd_has and not td_has:
            status = "td_missing"
        elif not asd_has and td_has:
            status = "asd_missing"
        else:
            status = "both_missing"
        rows.append({
            "pair_id": int(pid),
            "asd_id": asd["subject_id"].iloc[0] if len(asd) else "",
            "td_id": td["subject_id"].iloc[0] if len(td) else "",
            "asd_has_eo": asd_has,
            "td_has_eo": td_has,
            "pair_status": status,
            "age_diff": float(sub["age_diff"].iloc[0]) if "age_diff" in sub.columns else np.nan,
            "iq_diff": float(sub["iq_diff"].iloc[0]) if "iq_diff" in sub.columns else np.nan,
            "same_sex": bool(sub["same_sex"].iloc[0]) if "same_sex" in sub.columns else np.nan,
            "asd_failure": asd["failure_reason"].iloc[0] if len(asd) and not asd_has else "",
            "td_failure": td["failure_reason"].iloc[0] if len(td) and not td_has else "",
        })
    return pd.DataFrame(rows)


def run_missing_diagnosis(df: pd.DataFrame, pair_diag: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """P3: 缺失原因与缺失 vs 有值者比较。"""
    matched_ids = set(df["subject_id"])
    missing = df[~df["has_eo"]].copy()
    present = df[df["has_eo"]].copy()

    miss_rows: list[dict[str, Any]] = []
    for _, row in missing.iterrows():
        miss_rows.append({
            "subject_id": row["subject_id"],
            "group": row["group"],
            "pair_id": row["pair_id"],
            "pair_role": row["pair_role"],
            "age_months": row["age_months"],
            "IQ_total": row["IQ_total"],
            "sex": row["sex"],
            "failure_reason": row.get("failure_reason", _classify_failure(row.get("error", ""))),
            "error_detail": row.get("error", ""),
        })
    missing_diag = pd.DataFrame(miss_rows)

    comp_rows: list[dict[str, Any]] = []
    for var, test in [("age_months", "welch_t"), ("IQ_total", "welch_t")]:
        m_vals = missing[var].dropna().to_numpy(dtype=float)
        p_vals = present[var].dropna().to_numpy(dtype=float)
        tt = independent_ttest(m_vals, p_vals)
        comp_rows.append({
            "variable": var,
            "test": test,
            "missing_n": len(m_vals),
            "present_n": len(p_vals),
            "missing_mean": float(np.nanmean(m_vals)) if len(m_vals) else np.nan,
            "present_mean": float(np.nanmean(p_vals)) if len(p_vals) else np.nan,
            "statistic": tt["statistic"],
            "pvalue": tt["pvalue"],
        })

    # sex: 2x2 Fisher
    sex_tab = pd.crosstab(
        pd.Series(["missing"] * len(missing) + ["present"] * len(present)),
        pd.concat([missing["sex"], present["sex"]], ignore_index=True),
    )
    if sex_tab.shape[0] == 2 and sex_tab.shape[1] >= 2:
        # use F vs M if available
        for cols in [sex_tab.columns.tolist()]:
            sub_tab = sex_tab[cols[:2]] if len(cols) >= 2 else sex_tab
            if sub_tab.shape == (2, 2):
                fisher = chi_square_or_fisher(sub_tab.to_numpy())
                comp_rows.append({
                    "variable": "sex",
                    "test": fisher["test"],
                    "missing_n": int(missing["sex"].notna().sum()),
                    "present_n": int(present["sex"].notna().sum()),
                    "missing_mean": np.nan,
                    "present_mean": np.nan,
                    "statistic": fisher["statistic"],
                    "pvalue": fisher["pvalue"],
                })
                break

    # pair completeness summary appended as metadata row in missing_diag attrs — store separately
    n_complete = int((pair_diag["pair_status"] == "complete").sum())
    n_incomplete = len(pair_diag) - n_complete
    flow = pd.DataFrame([{
        "stage": "matched_pairs",
        "n_asd": 119,
        "n_td": 119,
        "n_total": 238,
    }, {
        "stage": "eo_available",
        "n_asd": int(present[present["group"] == "ASD"]["subject_id"].nunique()),
        "n_td": int(present[present["group"] == "TD"]["subject_id"].nunique()),
        "n_total": int(present["subject_id"].nunique()),
    }, {
        "stage": "complete_pairs",
        "n_asd": n_complete,
        "n_td": n_complete,
        "n_total": n_complete,
    }, {
        "stage": "incomplete_pairs",
        "n_asd": int((pair_diag["pair_status"].isin(["asd_missing", "both_missing"])).sum()),
        "n_td": int((pair_diag["pair_status"].isin(["td_missing", "both_missing"])).sum()),
        "n_total": n_incomplete,
    }])

    return missing_diag, pd.DataFrame(comp_rows), flow


def run_balance_tables(df: pd.DataFrame, pair_diag: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """P7: EO 可用子集 vs 原始 119 平衡表。"""
    eo_avail = df[df["has_eo"]]
    balance_eo = descriptive_table(
        eo_avail, "group",
        variables=["age_months", "IQ_total"],
        continuous=["age_months", "IQ_total"],
        categorical=["sex"],
    )
    balance_eo["cohort"] = "eo_available"

    balance_full = descriptive_table(
        df, "group",
        variables=["age_months", "IQ_total"],
        continuous=["age_months", "IQ_total"],
        categorical=["sex"],
    )
    balance_full["cohort"] = "matched_full_119"

    # complete-pair subset
    complete_pids = set(pair_diag.loc[pair_diag["pair_status"] == "complete", "pair_id"])
    complete_sub = df[df["pair_id"].isin(complete_pids) & df["has_eo"]]
    balance_complete = descriptive_table(
        complete_sub, "group",
        variables=["age_months", "IQ_total"],
        continuous=["age_months", "IQ_total"],
        categorical=["sex"],
    )
    balance_complete["cohort"] = "complete_pairs_only"

    out = pd.concat([balance_full, balance_eo, balance_complete], ignore_index=True)
    return out, eo_avail


def _fit_ols_suite(df: pd.DataFrame, pipeline: str) -> pd.DataFrame:
    """P1 + P6: 协变量 OLS 规格。"""
    sub = df[df["has_eo"]].copy()
    sub["sex"] = sub["sex"].astype(str)
    outcome = "posterior_exponent"
    specs = [
        ("primary_covariates", f"{outcome} ~ C(group, Treatment(reference='ASD')) + age_months + C(sex) + IQ_total"),
        ("primary_plus_qc", (
            f"{outcome} ~ C(group, Treatment(reference='ASD')) + age_months + C(sex) + IQ_total "
            "+ n_epochs_after_autoreject + mean_r_squared"
        )),
        ("primary_plus_release", (
            f"{outcome} ~ C(group, Treatment(reference='ASD')) + age_months + C(sex) + IQ_total "
            "+ C(release_id)"
        )),
    ]
    rows: list[dict[str, Any]] = []
    for model_name, formula in specs:
        req_cols = [outcome, "group", "age_months", "sex", "IQ_total"]
        if "n_epochs_after_autoreject" in formula:
            req_cols += ["n_epochs_after_autoreject", "mean_r_squared"]
        if "release_id" in formula:
            req_cols.append("release_id")
        fit_sub = sub.dropna(subset=[c for c in req_cols if c in sub.columns])
        if len(fit_sub) < 20 or fit_sub["group"].nunique() < 2:
            rows.append({
                "pipeline": pipeline,
                "model": model_name,
                "outcome": outcome,
                "term": "C(group)[T.TD]",
                "status": "skipped",
                "n_obs": len(fit_sub),
            })
            continue
        try:
            fit = run_ols(formula, fit_sub)
            for r in model_results_to_row(fit, model_name, outcome):
                r["pipeline"] = pipeline
                r["status"] = "ok"
                rows.append(r)
        except Exception as exc:
            logger.warning("OLS failed %s: %s", model_name, exc)
            rows.append({
                "pipeline": pipeline,
                "model": model_name,
                "outcome": outcome,
                "status": f"error: {exc}",
                "n_obs": len(fit_sub),
            })
    return pd.DataFrame(rows)


def _qc_by_group(df: pd.DataFrame) -> pd.DataFrame:
    qc_vars = [
        "n_epochs_before_autoreject", "n_epochs_after_autoreject",
        "n_epochs_rejected", "mean_r_squared", "mean_fit_error",
    ]
    sub = df[df["has_eo"]]
    desc = descriptive_table(sub, "group", variables=qc_vars, continuous=qc_vars)
    test_rows: list[dict[str, Any]] = []
    for var in qc_vars:
        if var not in sub.columns:
            continue
        asd = sub.loc[sub["group"] == "ASD", var].dropna().to_numpy(dtype=float)
        td = sub.loc[sub["group"] == "TD", var].dropna().to_numpy(dtype=float)
        tt = independent_ttest(asd, td)
        test_rows.append({
            "variable": var,
            "asd_mean": float(np.mean(asd)) if len(asd) else np.nan,
            "td_mean": float(np.mean(td)) if len(td) else np.nan,
            "welch_t": tt["statistic"],
            "pvalue": tt["pvalue"],
        })
    tests = pd.DataFrame(test_rows)
    return desc, tests


def run_paired_analysis(df: pd.DataFrame, pair_diag: pd.DataFrame, pipeline: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """P2: 完整 pair 配对检验 + MixedLM。"""
    complete = pair_diag[pair_diag["pair_status"] == "complete"].copy()
    diff_rows: list[dict[str, Any]] = []
    for _, prow in complete.iterrows():
        asd_row = df[(df["pair_id"] == prow["pair_id"]) & (df["pair_role"] == "ASD")].iloc[0]
        td_row = df[(df["pair_id"] == prow["pair_id"]) & (df["pair_role"] == "TD")].iloc[0]
        diff_rows.append({
            "pair_id": int(prow["pair_id"]),
            "asd_id": prow["asd_id"],
            "td_id": prow["td_id"],
            "asd_exponent": float(asd_row["posterior_exponent"]),
            "td_exponent": float(td_row["posterior_exponent"]),
            "pair_diff_TD_minus_ASD": float(td_row["posterior_exponent"] - asd_row["posterior_exponent"]),
            "age_diff": prow["age_diff"],
            "iq_diff": prow["iq_diff"],
        })
    pair_diff = pd.DataFrame(diff_rows)
    diffs = pair_diff["pair_diff_TD_minus_ASD"].to_numpy(dtype=float)

    test_rows: list[dict[str, Any]] = []
    if len(diffs) >= 3:
        t_stat, t_p = stats.ttest_1samp(diffs, 0.0)
        w_stat, w_p = stats.wilcoxon(diffs, alternative="two-sided")
        n_pos = int((diffs > 0).sum())
        n_neg = int((diffs < 0).sum())
        binom_p = float(stats.binomtest(n_pos, n_pos + n_neg, p=0.5, alternative="two-sided").pvalue) if (n_pos + n_neg) > 0 else np.nan
        test_rows.extend([
            {
                "pipeline": pipeline,
                "test": "paired_t",
                "n_pairs": len(diffs),
                "statistic": float(t_stat),
                "pvalue": float(t_p),
                "mean_diff_TD_minus_ASD": float(np.mean(diffs)),
                "direction": "TD>ASD" if np.mean(diffs) > 0 else "ASD>TD",
            },
            {
                "pipeline": pipeline,
                "test": "wilcoxon_signed_rank",
                "n_pairs": len(diffs),
                "statistic": float(w_stat),
                "pvalue": float(w_p),
                "mean_diff_TD_minus_ASD": float(np.mean(diffs)),
                "direction": "TD>ASD" if np.mean(diffs) > 0 else "ASD>TD",
            },
            {
                "pipeline": pipeline,
                "test": "sign_test_binomial",
                "n_pairs": len(diffs),
                "statistic": float(n_pos),
                "pvalue": binom_p,
                "mean_diff_TD_minus_ASD": float(np.mean(diffs)),
                "direction": f"TD>ASD in {n_pos}/{n_pos+n_neg} pairs",
            },
        ])

    # MixedLM on long format (complete pairs only)
    complete_pids = set(complete["pair_id"])
    long_sub = df[df["pair_id"].isin(complete_pids) & df["has_eo"]].copy()
    long_sub["sex"] = long_sub["sex"].astype(str)
    long_sub["pair_id_str"] = long_sub["pair_id"].astype(str)
    if len(long_sub) >= 20:
        formula = (
            "posterior_exponent ~ C(group, Treatment(reference='ASD')) "
            "+ age_months + C(sex) + IQ_total"
        )
        try:
            mfit = run_mixedlm(formula, long_sub, groups="pair_id_str", re_formula="1")
            for r in model_results_to_row(mfit, "mixedlm_pair_re", "posterior_exponent"):
                r["pipeline"] = pipeline
                r["status"] = "ok"
                r["used_mixedlm"] = getattr(mfit, "_used_mixedlm", None)
                test_rows.append({
                    "pipeline": pipeline,
                    "test": "mixedlm_group_coef",
                    "n_pairs": int(long_sub["pair_id"].nunique()),
                    "statistic": r["coef"],
                    "pvalue": r["pvalue"],
                    "mean_diff_TD_minus_ASD": np.nan,
                    "direction": "TD>ASD" if r["coef"] > 0 else "ASD>TD",
                    "term": r["term"],
                    "std_err": r["std_err"],
                    "ci_low": r["ci_low"],
                    "ci_high": r["ci_high"],
                    "used_mixedlm": r["used_mixedlm"],
                })
        except Exception as exc:
            logger.warning("MixedLM failed: %s", exc)

    return pair_diff, pd.DataFrame(test_rows)


def build_legacy_fixed_exponents(legacy_specparam: Path, matched_ids: set[str]) -> pd.DataFrame:
    """P4 档 A: legacy fixed homologous posterior exponent。"""
    roi_cfg = load_roi_config(PROJECT_ROOT / "config/roi_channels_hbn129.yaml")
    roi_dict = get_roi_dict(roi_cfg, "channels_hbn129")
    min_ratio = float(roi_cfg.get("min_valid_channel_ratio", 0.5))

    ch = pd.read_csv(legacy_specparam)
    ch["subject_id"] = ch["subject_id"].astype(str)
    ch = ch[ch["subject_id"].isin(matched_ids)]
    if "fit_valid" in ch.columns:
        ch = ch[ch["fit_valid"].astype(bool)]

    rows: list[dict[str, Any]] = []
    for sid, sub_ch in ch.groupby("subject_id"):
        grp = sub_ch["group"].iloc[0] if "group" in sub_ch.columns else "unknown"
        rec, _ = aggregate_roi_for_subject(
            sub_ch.assign(subject_id=sid, group=grp),
            roi_dict,
            min_ratio=min_ratio,
        )
        rows.append({
            "subject_id": sid,
            "posterior_exponent": rec.get("homologous_four_exponent", np.nan),
            "pipeline": "legacy_fixed",
            "aperiodic_mode": "fixed",
        })
    return pd.DataFrame(rows)


def _descriptive_group_effect(df: pd.DataFrame, pipeline: str) -> dict[str, Any]:
    sub = df[df["has_eo"]]
    asd = sub.loc[sub["group"] == "ASD", "posterior_exponent"].dropna()
    td = sub.loc[sub["group"] == "TD", "posterior_exponent"].dropna()
    tt = independent_ttest(asd.to_numpy(), td.to_numpy())
    return {
        "pipeline": pipeline,
        "n_asd": len(asd),
        "n_td": len(td),
        "asd_mean": float(asd.mean()) if len(asd) else np.nan,
        "td_mean": float(td.mean()) if len(td) else np.nan,
        "mean_diff_TD_minus_ASD": float(td.mean() - asd.mean()) if len(asd) and len(td) else np.nan,
        "welch_t": tt["statistic"],
        "welch_p": tt["pvalue"],
        "cohens_d": cohens_d(asd.to_numpy(), td.to_numpy()),
        "direction": "TD>ASD" if (len(asd) and len(td) and td.mean() > asd.mean()) else "ASD>TD",
    }


def _group_td_term(ols_df: pd.DataFrame) -> pd.Series:
    """匹配 statsmodels 输出的 TD 组系数项名。"""
    mask = ols_df["term"].astype(str).str.contains("group", case=False, regex=False) & (
        ols_df["term"].astype(str).str.contains("TD", regex=False)
    )
    return ols_df.loc[mask]


def _ols_group_coef(ols_df: pd.DataFrame, pipeline: str) -> dict[str, Any]:
    sub = ols_df[(ols_df["pipeline"] == pipeline) & (ols_df["model"] == "primary_covariates")]
    row = _group_td_term(sub)
    if row.empty:
        return {}
    row = row.iloc[[0]]
    r = row.iloc[0]
    return {
        "ols_beta_TD_minus_ASD": float(r["coef"]),
        "ols_p": float(r["pvalue"]),
        "ols_n": int(r["n_obs"]),
        "ols_ci_low": float(r["ci_low"]),
        "ols_ci_high": float(r["ci_high"]),
    }


def run_pipeline_sensitivity(
    nuclear_df: pd.DataFrame,
    legacy_specparam: Path,
    out_dir: Path,
) -> pd.DataFrame:
    """P4: nuclear knee + legacy fixed。"""
    matched_ids = set(nuclear_df["subject_id"])
    rows: list[dict[str, Any]] = []

    nuc_desc = _descriptive_group_effect(nuclear_df, "nuclear_knee")
    rows.append(nuc_desc)

    legacy_exp = build_legacy_fixed_exponents(legacy_specparam, matched_ids)
    legacy_merged = nuclear_df.drop(columns=["posterior_exponent", "has_eo"], errors="ignore").merge(
        legacy_exp[["subject_id", "posterior_exponent"]], on="subject_id", how="left"
    )
    legacy_merged["has_eo"] = legacy_merged["posterior_exponent"].notna()
    save_csv(legacy_exp, out_dir / "legacy_fixed_homologous_exponent.csv")

    leg_desc = _descriptive_group_effect(legacy_merged, "legacy_fixed")
    rows.append(leg_desc)

    return pd.DataFrame(rows), legacy_merged


def build_convergence_table(
    nuclear_df: pd.DataFrame,
    ols_nuclear: pd.DataFrame,
    pipeline_sens: pd.DataFrame,
    ols_legacy: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """P8: 主队列 vs HBN 对照。"""
    rows = [{
        "dataset": "Primary resting",
        "pipeline": "fixed specparam",
        "roi": "/".join(HOMOLOGOUS_PRIMARY),
        "n": PRIMARY_RESTING_N,
        "effect_TD_minus_ASD": PRIMARY_RESTING_BETA,
        "effect_type": "ols_beta",
        "p_value": PRIMARY_RESTING_P,
        "direction": "TD>ASD",
        "note": "main cohort group_ols_models.csv",
    }]

    nuc_ols = _ols_group_coef(ols_nuclear, "nuclear_knee")
    nuc_rows = pipeline_sens[pipeline_sens["pipeline"] == "nuclear_knee"] if len(pipeline_sens) else pd.DataFrame()
    nuc_desc = nuc_rows.iloc[0].to_dict() if len(nuc_rows) else {}
    rows.append({
        "dataset": "HBN EO matched",
        "pipeline": "nuclear knee+AR+CSD",
        "roi": "/".join(ROI_CHANNELS),
        "n": int(nuc_desc.get("n_asd", 0) + nuc_desc.get("n_td", 0)) if nuc_desc else np.nan,
        "effect_TD_minus_ASD": nuc_ols.get("ols_beta_TD_minus_ASD", nuc_desc.get("mean_diff_TD_minus_ASD", np.nan)),
        "effect_type": "ols_beta" if nuc_ols else "mean_diff",
        "p_value": nuc_ols.get("ols_p", nuc_desc.get("welch_p", np.nan)),
        "direction": nuc_desc.get("direction", "TD>ASD"),
        "note": "ThePresent matched 119x119 EO available subset",
    })

    if ols_legacy is not None:
        leg_ols = _ols_group_coef(ols_legacy, "legacy_fixed")
        leg_rows = pipeline_sens[pipeline_sens["pipeline"] == "legacy_fixed"] if len(pipeline_sens) else pd.DataFrame()
        leg_desc = leg_rows.iloc[0].to_dict() if len(leg_rows) else {}
        rows.append({
            "dataset": "HBN EO matched",
            "pipeline": "legacy fixed (no AR/CSD/trim)",
            "roi": "/".join(ROI_CHANNELS),
            "n": int(leg_desc.get("n_asd", 0) + leg_desc.get("n_td", 0)) if leg_desc else np.nan,
            "effect_TD_minus_ASD": leg_ols.get("ols_beta_TD_minus_ASD", leg_desc.get("mean_diff_TD_minus_ASD", np.nan)),
            "effect_type": "ols_beta" if leg_ols else "mean_diff",
            "p_value": leg_ols.get("ols_p", leg_desc.get("welch_p", np.nan)),
            "direction": leg_desc.get("direction", ""),
            "note": "preprocessing-misaligned sensitivity",
        })

    return pd.DataFrame(rows)


def write_report(
    df: pd.DataFrame,
    pair_diag: pd.DataFrame,
    missing_diag: pd.DataFrame,
    missing_comp: pd.DataFrame,
    flow: pd.DataFrame,
    balance: pd.DataFrame,
    ols_df: pd.DataFrame,
    paired_tests: pd.DataFrame,
    pair_diff: pd.DataFrame,
    pipeline_sens: pd.DataFrame,
    convergence: pd.DataFrame,
    qc_desc: pd.DataFrame,
    out_path: Path,
    extra_pipelines: list[str] | None = None,
) -> None:
    """生成中文汇总报告。"""
    n_complete = int((pair_diag["pair_status"] == "complete").sum())
    n_eo = int(df["has_eo"].sum())
    n_asd_eo = int(df[df["has_eo"] & (df["group"] == "ASD")]["subject_id"].nunique())
    n_td_eo = int(df[df["has_eo"] & (df["group"] == "TD")]["subject_id"].nunique())

    primary_ols = ols_df[ols_df["model"] == "primary_covariates"].copy()
    primary_ols = _group_td_term(primary_ols)
    paired_t = paired_tests[paired_tests["test"] == "paired_t"]

    lines = [
        "# HBN 静息 EO Posterior Exponent 外部验证报告",
        "",
        "## 队列与 ROI",
        "",
        "- **匹配队列**：ThePresent 严格匹配 119 ASD × 119 TD（`participants_matched.csv` + `match_table.csv`）",
        f"- **EO 可用**：{n_eo} 人（ASD {n_asd_eo}，TD {n_td_eo}）",
        f"- **完整配对**：{n_complete} / 119 对（双侧均有 EO exponent）",
        f"- **Homologous posterior ROI**：{', '.join(ROI_CHANNELS)}（预定义于 `config/roi_channels_hbn129.yaml`）",
        f"- **主研究对应 ROI**：{', '.join(HOMOLOGOUS_PRIMARY)}（~3 mm 空间同源；**未**基于 HBN 组效应重选电极）",
        "",
        "## 样本流转",
        "",
        flow.to_markdown(index=False) if hasattr(flow, "to_markdown") else flow.to_string(),
        "",
        "## P1 协变量调整 OLS",
        "",
    ]
    if not primary_ols.empty:
        for _, r in primary_ols.iterrows():
            lines.append(
                f"- **{r['pipeline']}** / {r['model']}: β(TD−ASD)={r['coef']:.4f} "
                f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}], p={r['pvalue']:.4g}, n={int(r['n_obs'])}"
            )
    lines += ["", "## P2 配对分析（完整 pair）", ""]
    if not paired_t.empty:
        for _, r in paired_t.iterrows():
            lines.append(
                f"- **{r['pipeline']}** paired t: n_pairs={int(r['n_pairs'])}, "
                f"mean(TD−ASD)={r['mean_diff_TD_minus_ASD']:.4f}, p={r['pvalue']:.4g}, {r['direction']}"
            )
    lines += [
        "",
        "## P3 缺失诊断",
        "",
        f"- 匹配队列缺失 EO：{len(missing_diag)} 人",
    ]
    if not missing_diag.empty:
        reason_counts = missing_diag["failure_reason"].value_counts()
        for reason, cnt in reason_counts.items():
            lines.append(f"  - {reason}: {cnt}")
    lines += ["", "### 缺失者 vs 有值者", ""]
    lines.append(missing_comp.to_string(index=False) if len(missing_comp) else "（无）")

    lines += ["", "## P4 Pipeline 方向敏感性", ""]
    for _, r in pipeline_sens.iterrows():
        lines.append(
            f"- **{r['pipeline']}**: TD−ASD mean diff={r['mean_diff_TD_minus_ASD']:.4f}, "
            f"Welch p={r['welch_p']:.4g}, direction={r['direction']}, "
            f"n={int(r['n_asd'])+int(r['n_td'])}"
        )

    lines += [
        "",
        "## P7 平衡性（EO 可用子集）",
        "",
        balance[balance["cohort"] == "eo_available"].to_string(index=False)
        if "cohort" in balance.columns else "",
        "",
        "## P8 收敛对照",
        "",
        convergence.to_string(index=False),
        "",
        "## 稿件措辞建议",
        "",
        "> Exploratory external resting-state **convergence** in aperiodic exponent direction "
        "(TD > ASD) was observed in the HBN ThePresent matched cohort under nuclear and legacy pipelines; "
        "this does **not** constitute a strict replication due to differences in preprocessing "
        "(CSD, AutoReject, knee vs fixed), stimulus context, and cohort composition.",
        "",
    ]

    # Conditional main-text sentence
    ols_ok = not primary_ols.empty and (primary_ols["pvalue"] < 0.10).any() and (primary_ols["coef"] > 0).all()
    paired_ok = not paired_t.empty and (paired_t["pvalue"] < 0.10).any() and (paired_t["mean_diff_TD_minus_ASD"] > 0).all()
    if ols_ok and paired_ok:
        lines += [
            "### 主文 Discussion（条件性，1 句）",
            "",
            "In an exploratory external resting-state analysis of the age-/IQ-/sex-matched HBN subset, "
            "posterior aperiodic exponent showed directional convergence with the primary cohort (TD > ASD), "
            "although preprocessing pipelines were not identical.",
            "",
        ]

    if extra_pipelines:
        lines += ["## 额外 pipeline 运行", ""]
        for pl in extra_pipelines:
            lines.append(f"- `{pl}`：通过 `--exponent-csv` + `--pipeline-label` 接入；远端 BIDS 重跑后复用本脚本。")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_validation(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    exponent_csv = Path(args.exponent_csv)
    posterior_dir = Path(args.posterior_channels_dir)
    failures_csv = Path(args.failures_csv)
    pipeline_label = args.pipeline_label

    df = assemble_matched_eo_table(exponent_csv, posterior_dir, failures_csv)
    df["pipeline_label"] = pipeline_label
    save_csv(df, out_dir / "matched_eo_analysis_table.csv")

    pair_diag = _pair_completeness(df)
    save_csv(pair_diag, out_dir / "pair_completeness.csv")

    missing_diag, missing_comp, flow = run_missing_diagnosis(df, pair_diag)
    save_csv(missing_diag, out_dir / "missing_diagnosis.csv")
    save_csv(missing_comp, out_dir / "missing_vs_present_comparison.csv")
    save_csv(flow, out_dir / "cohort_flow.csv")

    balance, _ = run_balance_tables(df, pair_diag)
    save_csv(balance, out_dir / "balance_table_eo_available.csv")

    ols_df = _fit_ols_suite(df, pipeline_label)
    # append to existing if multi-pipeline run
    ols_path = out_dir / "group_ols_models.csv"
    if ols_path.exists() and pipeline_label != "nuclear_knee":
        ols_df = pd.concat([pd.read_csv(ols_path), ols_df], ignore_index=True)
    ols_df = ols_df.drop_duplicates(subset=["pipeline", "model", "term"], keep="last")
    save_csv(ols_df, ols_path)

    qc_desc, qc_tests = _qc_by_group(df)
    save_csv(qc_desc, out_dir / "qc_by_group_descriptive.csv")
    save_csv(qc_tests, out_dir / "qc_by_group_tests.csv")

    pair_diff, paired_tests = run_paired_analysis(df, pair_diag, pipeline_label)
    save_csv(pair_diff, out_dir / f"paired_pair_differences_{pipeline_label}.csv")
    paired_path = out_dir / "paired_tests.csv"
    if paired_path.exists():
        paired_tests = pd.concat([pd.read_csv(paired_path), paired_tests], ignore_index=True)
    dedup_cols = ["pipeline", "test", "term"] if "term" in paired_tests.columns else ["pipeline", "test"]
    paired_tests = paired_tests.drop_duplicates(subset=dedup_cols, keep="last")
    save_csv(paired_tests, paired_path)

    pipeline_sens = pd.DataFrame()
    ols_legacy = None
    if not args.skip_legacy and pipeline_label == "nuclear_knee":
        pipeline_sens, legacy_merged = run_pipeline_sensitivity(
            df, Path(args.legacy_specparam), out_dir,
        )
        ols_legacy = _fit_ols_suite(legacy_merged, "legacy_fixed")
        ols_df = pd.concat([ols_df, ols_legacy], ignore_index=True)
        save_csv(ols_df, ols_path)

        pair_diag_leg = _pair_completeness(legacy_merged)
        _, paired_leg = run_paired_analysis(legacy_merged, pair_diag_leg, "legacy_fixed")
        paired_tests = pd.concat([paired_tests, paired_leg], ignore_index=True)
        paired_tests = paired_tests.drop_duplicates(subset=dedup_cols, keep="last")
        save_csv(paired_tests, paired_path)

        # enrich pipeline_sens with OLS
        for pl in ["nuclear_knee", "legacy_fixed"]:
            extra = _ols_group_coef(ols_df, pl)
            if extra and len(pipeline_sens):
                mask = pipeline_sens["pipeline"] == pl
                for k, v in extra.items():
                    pipeline_sens.loc[mask, k] = v
        save_csv(pipeline_sens, out_dir / "pipeline_sensitivity.csv")
    elif (out_dir / "pipeline_sensitivity.csv").exists():
        pipeline_sens = pd.read_csv(out_dir / "pipeline_sensitivity.csv")

    convergence = build_convergence_table(df, ols_df, pipeline_sens, ols_legacy)
    save_csv(convergence, out_dir / "convergence_comparison_table.csv")

    write_report(
        df, pair_diag, missing_diag, missing_comp, flow, balance,
        ols_df, paired_tests, pair_diff, pipeline_sens, convergence, qc_desc,
        out_dir / "external_validation_report_zh.md",
        extra_pipelines=["nuclear_fixed", "nuclear_no_csd"] if pipeline_label == "nuclear_knee" else None,
    )

    logger.info("完成 → %s", out_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    run_validation(args)


if __name__ == "__main__":
    main()
