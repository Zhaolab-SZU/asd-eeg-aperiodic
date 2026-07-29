# Ideal release name: 25c_rest_movie_coupling_overall_isc.py
# Original path: scripts/71_resting_to_movie_coupling_overall_isc.py
# Note: Coupling with overall ISC
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
71_resting_to_movie_coupling_overall_isc.py
-------------------------------------------
跨任务耦合（不分事件条件）：
将 movie ISC 在 subject 内跨事件取均值，作为 overall_movie_isc_z，
拟合 resting posterior_exponent 与 overall ISC 的组别交互模型。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import save_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resting-to-movie coupling using overall ISC.")
    p.add_argument("--config", type=str, required=True, help="Config path.")
    p.add_argument(
        "--resting_csv",
        type=str,
        default="outputs/tables/resting_features_locked.csv",
        help="Resting feature CSV containing posterior_exponent.",
    )
    p.add_argument(
        "--movie_isc_csv",
        type=str,
        default="",
        help="Movie ISC csv; default uses <derivatives_root>/stats/movie_isc_subject_values_with_neutral.csv",
    )
    p.add_argument(
        "--participants_csv",
        type=str,
        default="",
        help="Participants csv; default uses config paths.participants_file",
    )
    p.add_argument(
        "--exclude_subject_ids",
        type=str,
        default="",
        help="Subject IDs to exclude, comma-separated (e.g., S039,S078).",
    )
    return p.parse_args()


def winsorize_series(s: pd.Series, lo: float = 0.05, hi: float = 0.95) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce")
    q_lo = float(s2.quantile(lo))
    q_hi = float(s2.quantile(hi))
    return s2.clip(lower=q_lo, upper=q_hi)


def run(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    log = setup_logging(cfg, name="resting_movie_overall_isc")
    excluded_ids = {
        s.strip() for s in str(args.exclude_subject_ids).split(",")
        if s and s.strip()
    }

    deriv = Path(cfg["paths"]["derivatives_root"])
    out_root = Path(cfg["paths"]["outputs_root"])
    stats_dir = deriv / "stats"
    fig_dir = out_root / "figures"
    stats_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    resting_path = (PROJECT_ROOT / args.resting_csv).resolve()
    if args.movie_isc_csv.strip():
        isc_path = (PROJECT_ROOT / args.movie_isc_csv).resolve()
    else:
        isc_path = deriv / "stats" / "movie_isc_subject_values_with_neutral.csv"
    if args.participants_csv.strip():
        part_path = (PROJECT_ROOT / args.participants_csv).resolve()
    else:
        part_path = Path(cfg["paths"]["participants_file"])

    movie_analysis_path = deriv / "participants_analysis.csv"
    movie_qc_path = deriv / "specparam" / "specparam_qc_summary_subject.csv"
    preproc_path = deriv / "qc" / "preproc_summary.csv"

    rest = pd.read_csv(resting_path)[["subject_id", "group", "posterior_exponent"]].copy()
    rest["subject_id"] = rest["subject_id"].astype(str)
    rest["group"] = rest["group"].astype(str).str.upper()
    rest["posterior_exponent"] = pd.to_numeric(rest["posterior_exponent"], errors="coerce")

    isc = pd.read_csv(isc_path)[["subject_id", "group", "event_type", "isc_z"]].copy()
    isc["subject_id"] = isc["subject_id"].astype(str)
    isc["group"] = isc["group"].astype(str).str.upper()
    isc["isc_z"] = pd.to_numeric(isc["isc_z"], errors="coerce")
    isc["event_type"] = isc["event_type"].astype(str).str.lower()
    isc = isc[isc["event_type"].isin(["mental", "pain", "neutral"])].copy()
    overall = (
        isc.groupby(["subject_id", "group"], as_index=False)
        .agg(
            overall_movie_isc_z=("isc_z", "mean"),
            n_events_aggregated=("isc_z", lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())),
        )
    )

    part = pd.read_csv(part_path)
    keep_cols = ["subject_id", "group", "age_months"]
    for c in ("IQ_total", "sex", "included_final"):
        if c in part.columns:
            keep_cols.append(c)
    part = part[keep_cols].copy()
    part["subject_id"] = part["subject_id"].astype(str)
    part["group"] = part["group"].astype(str).str.upper()
    if "included_final" in part.columns:
        part["included_final"] = pd.to_numeric(part["included_final"], errors="coerce")
        part = part[part["included_final"] == 1].copy()

    pre = pd.read_csv(preproc_path)[["subject_id", "usable_epochs"]].copy()
    pre["subject_id"] = pre["subject_id"].astype(str)

    merged = overall.merge(rest, on=["subject_id", "group"], how="inner")
    merged = merged.merge(part.drop(columns=["included_final"], errors="ignore"), on=["subject_id", "group"], how="inner")
    merged = merged.merge(pre, on="subject_id", how="left")

    if movie_analysis_path.exists():
        ma = pd.read_csv(movie_analysis_path)[["subject_id", "group"]].copy()
        ma["subject_id"] = ma["subject_id"].astype(str)
        ma["group"] = ma["group"].astype(str).str.upper()
        merged = merged.merge(ma.drop_duplicates(), on=["subject_id", "group"], how="inner")

    if movie_qc_path.exists():
        mq = pd.read_csv(movie_qc_path)[["subject_id", "low_quality_subject"]].copy()
        mq["subject_id"] = mq["subject_id"].astype(str)
        bad = set(mq.loc[pd.to_numeric(mq["low_quality_subject"], errors="coerce") == 1, "subject_id"].tolist())
        merged = merged[~merged["subject_id"].isin(bad)].copy()
    if excluded_ids:
        merged = merged[~merged["subject_id"].isin(excluded_ids)].copy()

    merged["age_months"] = pd.to_numeric(merged["age_months"], errors="coerce")
    merged["IQ_total"] = pd.to_numeric(merged.get("IQ_total", np.nan), errors="coerce")
    merged["usable_epochs"] = pd.to_numeric(merged.get("usable_epochs", np.nan), errors="coerce")
    if "sex" in merged.columns:
        merged["sex"] = merged["sex"].astype(str).str.upper().replace({"NAN": np.nan, "": np.nan})

    need = ["overall_movie_isc_z", "posterior_exponent", "group", "age_months", "usable_epochs"]
    if "IQ_total" in merged.columns:
        need.append("IQ_total")
    if "sex" in merged.columns:
        need.append("sex")
    model_df = merged.dropna(subset=need).copy()

    formula = "overall_movie_isc_z ~ posterior_exponent * C(group) + age_months + usable_epochs"
    if "IQ_total" in model_df.columns:
        formula += " + IQ_total"
    if "sex" in model_df.columns:
        formula += " + C(sex)"
    ols = smf.ols(formula, data=model_df).fit()
    ols_df = ols.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    ols_df["n_obs"] = int(ols.nobs)
    ols_df["r_squared"] = float(ols.rsquared)
    ols_df["adj_r_squared"] = float(ols.rsquared_adj)
    ols_df["formula"] = formula

    robust_df = model_df.copy()
    robust_df["overall_movie_isc_z_w"] = winsorize_series(robust_df["overall_movie_isc_z"])
    robust_df["posterior_exponent_w"] = winsorize_series(robust_df["posterior_exponent"])
    r_formula = formula.replace("overall_movie_isc_z", "overall_movie_isc_z_w").replace(
        "posterior_exponent",
        "posterior_exponent_w",
    )
    rlm = smf.rlm(r_formula, robust_df).fit()
    rlm_df = pd.DataFrame(
        {
            "term": rlm.params.index,
            "Coef.": rlm.params.values,
            "Std.Err.": rlm.bse.values,
            "z": rlm.tvalues.values,
            "P>|z|": rlm.pvalues.values,
        },
    )
    rlm_df["n_obs"] = int(len(robust_df))
    rlm_df["formula"] = r_formula

    save_csv(merged, stats_dir / "resting_movie_coupling_overall_isc_merged.csv")
    save_csv(ols_df, stats_dir / "resting_movie_coupling_overall_isc_interaction_model.csv")
    save_csv(rlm_df, stats_dir / "resting_movie_coupling_overall_isc_interaction_model_rlm_winsor.csv")

    plt.figure(figsize=(7.6, 5.8))
    palette = {"ASD": "#1f77b4", "TD": "#d62728"}
    sns.scatterplot(
        data=model_df,
        x="posterior_exponent",
        y="overall_movie_isc_z",
        hue="group",
        palette=palette,
        alpha=0.72,
        s=34,
        edgecolor=None,
    )
    for grp in ["ASD", "TD"]:
        sub = model_df[model_df["group"] == grp].copy()
        if len(sub) >= 3:
            sns.regplot(
                data=sub,
                x="posterior_exponent",
                y="overall_movie_isc_z",
                scatter=False,
                ci=95,
                color=palette[grp],
                line_kws={"linewidth": 2},
            )
    p_ols = float(ols_df.loc[ols_df["term"] == "posterior_exponent:C(group)[T.TD]", "P>|t|"].iloc[0]) \
        if (ols_df["term"] == "posterior_exponent:C(group)[T.TD]").any() else np.nan
    p_rlm = float(rlm_df.loc[rlm_df["term"] == "posterior_exponent_w:C(group)[T.TD]", "P>|z|"].iloc[0]) \
        if (rlm_df["term"] == "posterior_exponent_w:C(group)[T.TD]").any() else np.nan
    plt.gca().text(
        0.98,
        0.98,
        f"Interaction p (OLS) = {p_ols:.4g}\nInteraction p (RLM) = {p_rlm:.4g}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#666"},
    )
    plt.xlabel("Resting Posterior Exponent")
    plt.ylabel("Overall Movie ISC (Fisher z, mean over events)")
    plt.title("Resting-to-Movie Coupling (Overall ISC)")
    plt.tight_layout()
    fig_path = fig_dir / "resting_to_movie_coupling_scatter_overall_isc"
    plt.savefig(fig_path.with_suffix(".png"), dpi=190)
    plt.savefig(fig_path.with_suffix(".pdf"))
    plt.close()

    log.info("Merged: %s", stats_dir / "resting_movie_coupling_overall_isc_merged.csv")
    log.info("OLS: %s", stats_dir / "resting_movie_coupling_overall_isc_interaction_model.csv")
    log.info("RLM: %s", stats_dir / "resting_movie_coupling_overall_isc_interaction_model_rlm_winsor.csv")
    log.info("Figure: %s", fig_path.with_suffix(".png"))
    if excluded_ids:
        log.info("Excluded subjects: %s", ",".join(sorted(excluded_ids)))


if __name__ == "__main__":
    run(parse_args())
