"""EMG/high-frequency power and ICLabel QC balance analyses (reviewer-defense)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.io_utils import (
    attach_usable_epochs,
    exclude_specparam_low_quality,
    load_analysis_participants,
    save_csv,
)
from src.iclabel_sensitivity import run_iclabel_removed_component_sensitivity
from src.stats_utils import (
    compare_groups_on_variable,
    descriptive_table,
    model_results_to_row,
    run_ols,
    spearman_correlation,
)

logger = logging.getLogger(__name__)

COVARIATES = "age_months + C(sex) + IQ_total + usable_epochs"
HF_BANDS = {
    "low_gamma": (30.0, 40.0),
    "hf_20_40": (20.0, 40.0),
    "beta": (13.0, 30.0),
}


def _residualize(y: pd.Series, cov: pd.DataFrame) -> pd.Series:
    x = sm.add_constant(cov, has_constant="add")
    fit = sm.OLS(y, x).fit()
    return pd.Series(fit.resid, index=y.index)


def load_primary_cohort_df(cfg: dict[str, Any]) -> pd.DataFrame:
    """Primary spectral cohort (N≈138) with global/posterior exponent."""
    deriv = Path(cfg["paths"]["derivatives_root"])
    root = Path(cfg["paths"].get("outputs_root", "outputs"))
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root

    participants = load_analysis_participants(cfg)
    roi = pd.read_csv(deriv / "roi" / "specparam_subject_global.csv")
    feat_path = root / "tables" / "resting_features_locked.csv"
    if feat_path.exists():
        feat = pd.read_csv(feat_path)
        feat["subject_id"] = feat["subject_id"].astype(str)
        keep = ["subject_id", "posterior_exponent"]
        if "mean_r_squared" in feat.columns:
            keep.append("mean_r_squared")
        df = participants.merge(roi, on=["subject_id", "group"], how="inner")
        df = df.merge(feat[keep], on="subject_id", how="left")
    else:
        df = participants.merge(roi, on=["subject_id", "group"], how="inner")

    df = attach_usable_epochs(df, deriv)
    df = exclude_specparam_low_quality(df, deriv)
    return df


def subject_level_band_metrics(cfg: dict[str, Any], cohort_ids: set[str]) -> pd.DataFrame:
    """Mean log10 band power (trapz) across channels per subject."""
    deriv = Path(cfg["paths"]["derivatives_root"])
    psd_dir = deriv / "psd"
    bands = {**cfg.get("bands", {}), **HF_BANDS}
    rows: list[dict[str, Any]] = []

    for sid in cohort_ids:
        p = psd_dir / f"{sid}_psd.csv"
        if not p.exists():
            continue
        psd = pd.read_csv(p)
        bp = compute_band_power_from_psd(psd, bands)
        for band in bands:
            sub = bp[bp["band"] == band]
            if sub.empty:
                continue
            vals = sub["band_power"].replace(0, np.nan).dropna()
            if vals.empty:
                continue
            log_p = np.log10(vals.astype(float))
            rows.append({
                "subject_id": sid,
                "group": sub["group"].iloc[0],
                "band": band,
                "mean_log10_band_power": float(log_p.mean()),
                "mean_band_power": float(vals.mean()),
            })
    return pd.DataFrame(rows)


def merge_band_wide(band_long: pd.DataFrame) -> pd.DataFrame:
    if band_long.empty:
        return pd.DataFrame()
    wide = band_long.pivot_table(
        index=["subject_id", "group"],
        columns="band",
        values="mean_log10_band_power",
        aggfunc="first",
    ).reset_index()
    wide.columns = [
        str(c) if c in ("subject_id", "group") else f"log10_{c}"
        for c in wide.columns
    ]
    return wide


def run_hf_group_models(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for band_col, label in [
        ("log10_low_gamma", "low_gamma_30_40"),
        ("log10_hf_20_40", "hf_20_40"),
        ("log10_beta", "beta_13_30"),
    ]:
        if band_col not in df.columns:
            continue
        sub = df.dropna(subset=[band_col, "group", "age_months", "sex", "IQ_total", "usable_epochs"])
        formula = f"{band_col} ~ C(group) + {COVARIATES}"
        model = run_ols(formula, sub)
        rows.extend(model_results_to_row(model, label, band_col, predictors=list(model.params.index)))
    return pd.DataFrame(rows)


def run_exponent_adjusted_for_hf(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome in ("global_exponent", "posterior_exponent"):
        if outcome not in df.columns or "log10_low_gamma" not in df.columns:
            continue
        sub = df.dropna(
            subset=[outcome, "log10_low_gamma", "group", "age_months", "sex", "IQ_total", "usable_epochs"],
        )
        base = f"{outcome} ~ C(group) + {COVARIATES}"
        adj = f"{outcome} ~ C(group) + log10_low_gamma + {COVARIATES}"
        for name, formula in [("without_hf", base), ("with_log10_low_gamma", adj)]:
            model = run_ols(formula, sub)
            rows.extend(
                model_results_to_row(model, f"{outcome}_{name}", outcome, predictors=list(model.params.index)),
            )
    return pd.DataFrame(rows)


def _covariate_frame(df: pd.DataFrame) -> pd.DataFrame:
    cov = df[["age_months", "IQ_total", "usable_epochs"]].copy()
    if "sex" in df.columns:
        if df["sex"].dtype == object or str(df["sex"].dtype) == "category":
            cov["sex"] = pd.Categorical(df["sex"]).codes
        else:
            cov["sex"] = pd.to_numeric(df["sex"], errors="coerce")
    return cov


def run_partial_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = [
        ("log10_low_gamma", "global_exponent"),
        ("log10_low_gamma", "posterior_exponent"),
        ("log10_hf_20_40", "global_exponent"),
        ("log10_beta", "global_exponent"),
    ]
    for x_var, y_var in pairs:
        if x_var not in df.columns or y_var not in df.columns:
            continue
        sub = df.dropna(subset=[x_var, y_var, "age_months", "IQ_total", "usable_epochs"]).copy()
        cov = _covariate_frame(sub)
        x_res = _residualize(sub[x_var], cov)
        y_res = _residualize(sub[y_var], cov)
        sp = spearman_correlation(x_res, y_res)
        rows.append({
            "predictor": x_var,
            "outcome": y_var,
            "covariates": "age, sex, IQ, usable_epochs (residualized)",
            "spearman_rho": sp["rho"],
            "pvalue": sp["pvalue"],
            "n": sp["n"],
        })
    return pd.DataFrame(rows)


def run_iclabel_qc_balance(
    cfg: dict[str, Any],
    cohort_df: pd.DataFrame,
    threshold: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(cfg["paths"].get("outputs_root", "outputs"))
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root

    tag = str(threshold).replace(".", "_")
    ic_path = root / "tables" / "iclabel_sensitivity" / f"iclabel_qc_summary_threshold_{tag}.csv"
    if not ic_path.exists():
        ic_path = root / "tables" / "iclabel_sensitivity" / "iclabel_qc_summary.csv"
        ic = pd.read_csv(ic_path)
        ic = ic[ic["threshold"] == threshold] if "threshold" in ic.columns else ic
    else:
        ic = pd.read_csv(ic_path)

    ic["subject_id"] = ic["subject_id"].astype(str)
    cohort_ids = set(cohort_df["subject_id"].astype(str))
    ic = ic[ic["subject_id"].isin(cohort_ids) & (ic["status"] == "ok")].copy()
    merged = cohort_df.merge(ic, on="subject_id", how="inner", suffixes=("", "_ic"))

    desc_vars = [
        "usable_epochs",
        "usable_epochs_after_iclabel",
        "n_components_removed",
        "percent_components_removed",
        "n_eye_removed",
        "n_muscle_removed",
        "mean_retained_artifact_probability",
        "max_retained_artifact_probability",
    ]
    desc = descriptive_table(merged, "group", desc_vars, continuous=desc_vars)

    tests = []
    for var in desc_vars:
        if var not in merged.columns:
            continue
        tests.append(compare_groups_on_variable(merged, "group", var, "ASD", "TD"))
    return desc, pd.DataFrame(tests)


def run_iclabel_posterior_removed_component_models(
    cfg: dict[str, Any],
    cohort_df: pd.DataFrame,
    threshold: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ICLabel posterior exponent group models ± n_components_removed."""
    try:
        result = run_iclabel_removed_component_sensitivity(cfg, threshold=threshold, cohort_df=cohort_df)
    except FileNotFoundError as exc:
        logger.warning("ICLabel removed-component sensitivity skipped: %s", exc)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return result["analysis_df"], result["detail"], result["summary"]


def format_summary_strings(
    hf_models: pd.DataFrame,
    adj_models: pd.DataFrame,
    partial: pd.DataFrame,
    qc_tests: pd.DataFrame,
    iclabel_removed_summary: pd.DataFrame | None = None,
) -> dict[str, str]:
    def _pick(df: pd.DataFrame, model: str, term: str) -> tuple[float, float]:
        row = df[(df["model"] == model) & (df["term"].astype(str).str.contains(term, regex=False))]
        if row.empty:
            return np.nan, np.nan
        return float(row.iloc[0]["coef"]), float(row.iloc[0]["pvalue"])

    lg_coef, lg_p = _pick(hf_models, "low_gamma_30_40", "T.TD")
    ge_coef, ge_p = _pick(adj_models, "global_exponent_without_hf", "T.TD")
    ge_adj_coef, ge_adj_p = _pick(adj_models, "global_exponent_with_log10_low_gamma", "T.TD")
    pe_coef, pe_adj_p = _pick(adj_models, "posterior_exponent_with_log10_low_gamma", "T.TD")

    pr = partial[(partial["predictor"] == "log10_low_gamma") & (partial["outcome"] == "global_exponent")]
    pr_rho = float(pr["spearman_rho"].iloc[0]) if len(pr) else np.nan
    pr_p = float(pr["pvalue"].iloc[0]) if len(pr) else np.nan

    muscle = qc_tests[qc_tests["variable"] == "n_muscle_removed"]
    m_p = float(muscle["t_pvalue"].iloc[0]) if len(muscle) else np.nan
    m_asd = float(muscle["mean_a"].iloc[0]) if len(muscle) else np.nan
    m_td = float(muscle["mean_b"].iloc[0]) if len(muscle) else np.nan

    ret = qc_tests[qc_tests["variable"] == "mean_retained_artifact_probability"]
    r_p = float(ret["t_pvalue"].iloc[0]) if len(ret) else np.nan

    ic_removed_text = ""
    if iclabel_removed_summary is not None and not iclabel_removed_summary.empty:
        base = iclabel_removed_summary[
            iclabel_removed_summary["model"].astype(str).str.contains("without_n_removed", regex=False)
        ]
        adj = iclabel_removed_summary[
            iclabel_removed_summary["model"].astype(str).str.contains("with_n_components_removed", regex=False)
        ]
        if not base.empty and not adj.empty:
            b = base.iloc[0]
            a = adj.iloc[0]
            ic_removed_text = (
                f"ICLabel posterior exponent TD − ASD: without removed-component covariate "
                f"β = {b['coef_TD_vs_ASD']:.3f} (p = {b['p']:.3f}); "
                f"with n_components_removed covariate β = {a['coef_TD_vs_ASD']:.3f} (p = {a['p']:.3f})."
            )

    return {
        "hf_low_gamma_td_asd": (
            f"TD − ASD on log10 low-gamma (30–40 Hz) power: β = {lg_coef:.3f}, p = {lg_p:.3f}."
            if np.isfinite(lg_coef)
            else ""
        ),
        "exponent_with_hf": (
            f"Global exponent TD − ASD: without HF covariate β = {ge_coef:.3f} (p = {ge_p:.3f}); "
            f"with log10 low-gamma covariate β = {ge_adj_coef:.3f} (p = {ge_adj_p:.3f})."
            if np.isfinite(ge_adj_coef)
            else ""
        ),
        "partial_corr": (
            f"Partial Spearman (age, sex, IQ, usable epochs): log10 low-gamma vs global exponent "
            f"ρ = {pr_rho:.2f}, p = {pr_p:.3f}."
            if np.isfinite(pr_rho)
            else ""
        ),
        "iclabel_balance": (
            f"ICLabel QC (threshold 0.70): mean muscle components removed ASD {m_asd:.2f} vs TD {m_td:.2f} "
            f"(Welch t, p = {m_p:.3f}); mean retained artifact probability p = {r_p:.3f}."
            if np.isfinite(m_p)
            else ""
        ),
        "iclabel_posterior_removed_component": ic_removed_text,
    }


def run_artifact_defense_analysis(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run both defenses; write tables under outputs/tables/artifact_defense/."""
    root = Path(cfg["paths"].get("outputs_root", "outputs"))
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    tables_dir = root / "tables" / "artifact_defense"
    tables_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_primary_cohort_df(cfg)
    band_long = subject_level_band_metrics(cfg, set(cohort["subject_id"].astype(str)))
    band_wide = merge_band_wide(band_long)
    df = cohort.merge(band_wide, on=["subject_id", "group"], how="left")

    hf_models = run_hf_group_models(df)
    adj_models = run_exponent_adjusted_for_hf(df)
    partial = run_partial_correlations(df)
    qc_desc, qc_tests = run_iclabel_qc_balance(cfg, cohort, threshold=0.70)
    ic_df, ic_detail, ic_summary = run_iclabel_posterior_removed_component_models(
        cfg, cohort, threshold=0.70,
    )

    save_csv(df, tables_dir / "primary_cohort_hf_features.csv")
    save_csv(hf_models, tables_dir / "hf_power_group_models.csv")
    save_csv(adj_models, tables_dir / "exponent_models_with_hf_covariate.csv")
    save_csv(partial, tables_dir / "hf_exponent_partial_correlations.csv")
    save_csv(qc_desc, tables_dir / "iclabel_qc_balance_descriptive.csv")
    save_csv(qc_tests, tables_dir / "iclabel_qc_balance_group_tests.csv")
    if not ic_df.empty:
        save_csv(ic_df, tables_dir / "iclabel_posterior_removed_component_cohort.csv")
    if not ic_detail.empty:
        save_csv(ic_detail, tables_dir / "iclabel_posterior_removed_component_models_detail.csv")
    if not ic_summary.empty:
        save_csv(ic_summary, tables_dir / "iclabel_posterior_removed_component_models.csv")

    summaries = format_summary_strings(hf_models, adj_models, partial, qc_tests, ic_summary)
    logger.info("Artifact defense analysis complete (n=%d)", len(df))
    return {
        "cohort_df": df,
        "hf_models": hf_models,
        "adj_models": adj_models,
        "partial": partial,
        "qc_desc": qc_desc,
        "qc_tests": qc_tests,
        "iclabel_removed_cohort": ic_df,
        "iclabel_removed_detail": ic_detail,
        "iclabel_removed_summary": ic_summary,
        "summaries": summaries,
        "tables_dir": tables_dir,
    }
