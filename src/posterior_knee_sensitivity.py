"""
Posterior exponent knee-mode sensitivity for the primary resting cohort.

Re-fits specparam in knee mode on existing PSD, aggregates locked ROI
(E33/E36/E37/E38), and compares group / age / ADOS effects against fixed mode.
Does not modify primary fixed-mode derivatives.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.io_utils import (
    attach_usable_epochs,
    ensure_dir,
    exclude_specparam_low_quality,
    load_analysis_participants,
    save_csv,
)
from src.qc_utils import run_specparam_qc
from src.specparam_utils import run_specparam_batch
from src.spectral_maturation_analysis import POSTERIOR_CORE, _posterior_channel_mean
from src.stats_utils import cohens_d, run_ols, spearman_correlation

logger = logging.getLogger(__name__)

KNEE_SENS_LABEL = "freq_1.0_40.0_mode_knee"
COVARIATE_FORMULA = " + age_months + C(sex) + IQ_total + usable_epochs"
GROUP_TERM = "C(group)[T.TD]"
INTERACTION_TERM = "C(group)[T.TD]:age_months"


def knee_deriv_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    deriv = Path(cfg["paths"]["derivatives_root"]) / "specparam"
    stem = f"sens_{KNEE_SENS_LABEL}"
    return {
        "raw": deriv / f"{stem}.csv",
        "channel_qc": deriv / f"{stem}_qc.csv",
        "subject_qc": deriv / f"{stem}_qc_subject.csv",
    }


def output_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["outputs_root"]) / "tables" / "posterior_knee_sensitivity"


def _group_term(params_index: pd.Index) -> str | None:
    for term in params_index:
        if "[T.TD]" in term and ":age_months" not in term:
            return term
    return None


def _interaction_term(params_index: pd.Index) -> str | None:
    for term in params_index:
        if "[T.TD]:age_months" in term:
            return term
    return None


def _covariate_suffix(outcome: str, mean_r2_col: str) -> str:
    if "exponent" in outcome:
        return f"{COVARIATE_FORMULA} + {mean_r2_col}"
    return COVARIATE_FORMULA


def load_primary_cohort(cfg: dict[str, Any]) -> pd.DataFrame:
    deriv = Path(cfg["paths"]["derivatives_root"])
    df = load_analysis_participants(cfg)
    df = exclude_specparam_low_quality(df, deriv)
    df = attach_usable_epochs(df, deriv)
    min_ep = int(cfg.get("epochs", {}).get("min_usable_epochs", 60))
    if "usable_epochs" in df.columns:
        df = df[df["usable_epochs"] >= min_ep].copy()
    df["subject_id"] = df["subject_id"].astype(str)
    return df.reset_index(drop=True)


def refit_knee_specparam(
    cfg: dict[str, Any],
    participants: pd.DataFrame,
    *,
    overwrite: bool = False,
) -> Path:
    paths = knee_deriv_paths(cfg)
    out_csv = paths["raw"]
    if out_csv.exists() and not overwrite:
        logger.info("Knee specparam 已存在，跳过 refit: %s", out_csv)
        return out_csv

    deriv = Path(cfg["paths"]["derivatives_root"])
    psd_dir = deriv / "psd"
    override = {"aperiodic_mode": "knee", "freq_range": [1.0, 40.0]}
    run_specparam_batch(participants, psd_dir, out_csv, cfg, sp_cfg_override=override)
    return out_csv


def run_knee_qc(cfg: dict[str, Any], *, overwrite: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = knee_deriv_paths(cfg)
    if not paths["raw"].exists():
        raise FileNotFoundError(f"Knee specparam 结果不存在: {paths['raw']}")

    if paths["channel_qc"].exists() and paths["subject_qc"].exists() and not overwrite:
        logger.info("Knee QC 已存在，跳过: %s", paths["channel_qc"])
        ch = pd.read_csv(paths["channel_qc"])
        sub = pd.read_csv(paths["subject_qc"])
        return ch, sub

    raw = pd.read_csv(paths["raw"])
    return run_specparam_qc(raw, cfg, paths["channel_qc"], paths["subject_qc"])


def _posterior_knee_mean(ch_df: pd.DataFrame) -> pd.DataFrame:
    exp = _posterior_channel_mean(ch_df, "aperiodic_exponent").rename(
        columns={"aperiodic_exponent": "posterior_exponent_knee"}
    )
    if "aperiodic_knee" not in ch_df.columns:
        return exp
    knee = _posterior_channel_mean(ch_df, "aperiodic_knee").rename(
        columns={"aperiodic_knee": "aperiodic_knee_mean"}
    )
    if knee.empty or "aperiodic_knee_mean" not in knee.columns:
        exp["aperiodic_knee_mean"] = np.nan
        return exp
    return exp.merge(knee[["subject_id", "aperiodic_knee_mean"]], on="subject_id", how="left")


def build_subject_comparison_table(
    cfg: dict[str, Any],
    cohort: pd.DataFrame,
    knee_ch: pd.DataFrame,
    knee_sub: pd.DataFrame,
) -> pd.DataFrame:
    deriv = Path(cfg["paths"]["derivatives_root"])
    fixed_ch_path = deriv / "specparam" / "specparam_channel_results_qc.csv"
    fixed_sub_path = deriv / "specparam" / "specparam_qc_summary_subject.csv"
    if not fixed_ch_path.exists():
        raise FileNotFoundError(f"Fixed QC 通道文件不存在: {fixed_ch_path}")

    fixed_ch = pd.read_csv(fixed_ch_path)
    fixed_ch["subject_id"] = fixed_ch["subject_id"].astype(str)
    knee_ch = knee_ch.copy()
    knee_ch["subject_id"] = knee_ch["subject_id"].astype(str)

    fixed_post = _posterior_channel_mean(fixed_ch, "aperiodic_exponent").rename(
        columns={"aperiodic_exponent": "posterior_exponent_fixed"}
    )
    knee_post = _posterior_knee_mean(knee_ch)

    keep_cols = [
        "subject_id",
        "group",
        "age_months",
        "sex",
        "IQ_total",
        "usable_epochs",
        "ADOS_total",
        "ADOS_SA",
        "ADOS_RRB",
    ]
    base = cohort[[c for c in keep_cols if c in cohort.columns]].copy()

    fixed_sub = pd.read_csv(fixed_sub_path)[["subject_id", "mean_r_squared"]].rename(
        columns={"mean_r_squared": "mean_r_squared_fixed"}
    )
    fixed_sub["subject_id"] = fixed_sub["subject_id"].astype(str)
    knee_sub = knee_sub[["subject_id", "mean_r_squared"]].rename(
        columns={"mean_r_squared": "mean_r_squared_knee"}
    )
    knee_sub["subject_id"] = knee_sub["subject_id"].astype(str)

    out = base.merge(fixed_post, on=["subject_id", "group"], how="inner")
    knee_cols = [c for c in knee_post.columns if c not in {"group"}]
    out = out.merge(knee_post[knee_cols], on="subject_id", how="left")
    out = out.merge(fixed_sub, on="subject_id", how="left")
    out = out.merge(knee_sub, on="subject_id", how="left")
    out["posterior_channels"] = ",".join(POSTERIOR_CORE)
    out["aperiodic_mode_knee"] = "knee"
    out["aperiodic_mode_fixed"] = "fixed"
    return out


def fit_group_effect_row(
    df: pd.DataFrame,
    outcome: str,
    mean_r2_col: str,
    model_label: str,
) -> dict[str, Any] | None:
    cov_cols = ["group", "age_months", "sex", "IQ_total", "usable_epochs", mean_r2_col, outcome]
    cov_cols = [c for c in cov_cols if c in df.columns]
    sub = df.dropna(subset=cov_cols)
    if len(sub) < 20 or sub["group"].nunique() < 2:
        return None

    formula = (
        f"{outcome} ~ C(group, Treatment(reference='ASD'))"
        f"{_covariate_suffix(outcome, mean_r2_col)}"
    )
    model = run_ols(formula, sub)
    gterm = _group_term(model.params.index)
    if gterm is None:
        return None
    ci = model.conf_int().loc[gterm]
    asd = sub.loc[sub["group"] == "ASD", outcome]
    td = sub.loc[sub["group"] == "TD", outcome]
    return {
        "model": model_label,
        "outcome": outcome,
        "mean_r_squared_covariate": mean_r2_col,
        "formula": formula,
        "n_total": int(model.nobs),
        "n_ASD": int((sub["group"] == "ASD").sum()),
        "n_TD": int((sub["group"] == "TD").sum()),
        "beta_TD_vs_ASD": float(model.params[gterm]),
        "se": float(model.bse[gterm]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p": float(model.pvalues[gterm]),
        "r_squared": float(model.rsquared),
        "cohens_d": cohens_d(asd.values, td.values),
        "direction": "TD > ASD" if float(model.params[gterm]) > 0 else "ASD > TD",
        "significant_p05": float(model.pvalues[gterm]) < 0.05,
    }


def fit_group_models(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("primary_knee", "posterior_exponent_knee", "mean_r_squared_knee"),
        ("robustness_knee_with_fixed_r2", "posterior_exponent_knee", "mean_r_squared_fixed"),
        ("reference_fixed", "posterior_exponent_fixed", "mean_r_squared_fixed"),
    ]
    rows = []
    for label, outcome, r2_col in specs:
        row = fit_group_effect_row(df, outcome, r2_col, label)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def fit_age_interaction_models(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("knee", "posterior_exponent_knee", "mean_r_squared_knee"),
        ("fixed_reference", "posterior_exponent_fixed", "mean_r_squared_fixed"),
        ("knee_fixed_r2", "posterior_exponent_knee", "mean_r_squared_fixed"),
    ]
    rows: list[dict[str, Any]] = []
    for mode_label, outcome, r2_col in specs:
        cov_cols = ["group", "age_months", "sex", "IQ_total", "usable_epochs", r2_col, outcome]
        cov_cols = [c for c in cov_cols if c in df.columns]
        sub = df.dropna(subset=cov_cols)
        if len(sub) < 20:
            continue
        cov = _covariate_suffix(outcome, r2_col)
        formula_int = f"{outcome} ~ C(group) * age_months{cov}"
        formula_red = f"{outcome} ~ C(group, Treatment(reference='ASD')){cov}"
        model_int = run_ols(formula_int, sub)
        model_red = run_ols(formula_red, sub)
        delta_r2 = float(model_int.rsquared - model_red.rsquared)
        for term in (_group_term(model_int.params.index), _interaction_term(model_int.params.index), "age_months"):
            if term is None or term not in model_int.params.index:
                continue
            ci = model_int.conf_int().loc[term]
            rows.append(
                {
                    "mode": mode_label,
                    "outcome": outcome,
                    "mean_r_squared_covariate": r2_col,
                    "term": term,
                    "n_total": int(model_int.nobs),
                    "coef": float(model_int.params[term]),
                    "se": float(model_int.bse[term]),
                    "ci_low": float(ci[0]),
                    "ci_high": float(ci[1]),
                    "p": float(model_int.pvalues[term]),
                    "r_squared_full": float(model_int.rsquared),
                    "delta_r_squared_interaction": delta_r2 if term == _interaction_term(model_int.params.index) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _partial_pearson(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    covariates: list[str],
) -> dict[str, Any]:
    cols = [y_col, x_col] + covariates
    sub = df.dropna(subset=[c for c in cols if c in df.columns]).copy()
    if len(sub) < 8:
        return {"partial_r": np.nan, "p": np.nan, "n": len(sub), "method": "partial_pearson"}
    cov_formula = " + ".join(covariates)
    y_model = run_ols(f"{y_col} ~ {cov_formula}", sub)
    x_model = run_ols(f"{x_col} ~ {cov_formula}", sub)
    y_resid = sub[y_col].values - y_model.predict(sub)
    x_resid = sub[x_col].values - x_model.predict(sub)
    r, p = stats.pearsonr(x_resid, y_resid)
    return {"partial_r": float(r), "p": float(p), "n": int(len(sub)), "method": "partial_pearson"}


def fit_ados_models(df: pd.DataFrame) -> pd.DataFrame:
    asd = df[df["group"] == "ASD"].copy()
    rows: list[dict[str, Any]] = []
    for mode_label, exp_col in [
        ("knee", "posterior_exponent_knee"),
        ("fixed_reference", "posterior_exponent_fixed"),
    ]:
        if exp_col not in asd.columns:
            continue
        for outcome in ["ADOS_SA", "ADOS_RRB", "ADOS_total"]:
            if outcome not in asd.columns:
                continue
            partial = _partial_pearson(asd, outcome, exp_col, ["age_months", "IQ_total"])
            rows.append(
                {
                    "analysis": "partial_correlation",
                    "mode": mode_label,
                    "predictor": exp_col,
                    "outcome": outcome,
                    "covariates": "age_months + IQ_total",
                    **partial,
                }
            )
            sub = asd.dropna(
                subset=[outcome, exp_col, "age_months", "sex", "IQ_total", "usable_epochs"]
            )
            if len(sub) < 8:
                continue
            formula = f"{outcome} ~ {exp_col} + age_months + C(sex) + IQ_total + usable_epochs"
            try:
                model = run_ols(formula, sub)
                rows.append(
                    {
                        "analysis": "ols",
                        "mode": mode_label,
                        "predictor": exp_col,
                        "outcome": outcome,
                        "covariates": "age_months + sex + IQ_total + usable_epochs",
                        "partial_r": float(model.params[exp_col]),
                        "p": float(model.pvalues[exp_col]),
                        "n": int(model.nobs),
                        "method": "ols",
                    }
                )
            except Exception as exc:
                logger.warning("ADOS OLS 失败 %s %s: %s", mode_label, outcome, exc)
    return pd.DataFrame(rows)


def compute_fixed_knee_correlation(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subsets = [("all", df), ("ASD", df[df["group"] == "ASD"]), ("TD", df[df["group"] == "TD"])]
    for label, sub in subsets:
        pair = sub.dropna(subset=["posterior_exponent_fixed", "posterior_exponent_knee"])
        if len(pair) < 5:
            continue
        x = pair["posterior_exponent_fixed"]
        y = pair["posterior_exponent_knee"]
        pr, pp = stats.pearsonr(x, y)
        sr = spearman_correlation(x, y)
        rows.append(
            {
                "subset": label,
                "n": len(pair),
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_rho": float(sr["rho"]),
                "spearman_p": float(sr["pvalue"]),
            }
        )
    return pd.DataFrame(rows)


def _posterior_channel_qc_slice(ch_df: pd.DataFrame) -> pd.DataFrame:
    return ch_df[ch_df["channel"].isin(POSTERIOR_CORE)].copy()


def build_knee_fit_qc_summary(
    cfg: dict[str, Any],
    cohort: pd.DataFrame,
    knee_ch: pd.DataFrame,
    subject_df: pd.DataFrame,
) -> pd.DataFrame:
    qc_cfg = cfg.get("fit_quality", {})
    knee_min = qc_cfg.get("knee_min_hz", 1.0)
    knee_max = qc_cfg.get("knee_max_hz", 40.0)
    cohort_ids = set(cohort["subject_id"].astype(str))
    post = _posterior_channel_qc_slice(knee_ch)
    post = post[post["subject_id"].astype(str).isin(cohort_ids)]

    rows: list[dict[str, Any]] = []
    valid_post = post["fit_valid"].sum() if "fit_valid" in post.columns else np.nan
    total_post = len(post)
    if "aperiodic_knee" in post.columns:
        knee_vals = pd.to_numeric(post["aperiodic_knee"], errors="coerce")
        implausible = int(
            (knee_vals.notna() & ((knee_vals < knee_min) | (knee_vals > knee_max))).sum()
        )
        implausible_denom = int(knee_vals.notna().sum())
    else:
        knee_vals = pd.Series(dtype=float)
        implausible = 0
        implausible_denom = 0

    rows.append(
        {
            "metric": "posterior_channel_fit_valid_rate",
            "value": float(valid_post / total_post) if total_post else np.nan,
            "n": int(total_post),
            "group": "all",
            "note": f"channels {','.join(POSTERIOR_CORE)}",
        }
    )
    rows.append(
        {
            "metric": "posterior_mean_r_squared",
            "value": float(post["r_squared"].mean()) if total_post else np.nan,
            "n": int(total_post),
            "group": "all",
            "note": "",
        }
    )
    rows.append(
        {
            "metric": "posterior_implausible_knee_rate",
            "value": float(implausible / implausible_denom) if implausible_denom else np.nan,
            "n": implausible_denom,
            "group": "all",
            "note": f"knee outside [{knee_min}, {knee_max}] Hz",
        }
    )

    sub_cohort = subject_df[subject_df["subject_id"].astype(str).isin(cohort_ids)].copy()
    valid_knee_post = subject_df.dropna(subset=["posterior_exponent_knee"])
    rows.append(
        {
            "metric": "subjects_with_valid_posterior_knee",
            "value": float(len(valid_knee_post)),
            "n": len(cohort),
            "group": "all",
            "note": "posterior_exponent_knee non-missing in primary cohort",
        }
    )
    for grp in ["ASD", "TD"]:
        gcohort = cohort[cohort["group"] == grp]
        gpost = post[post["group"] == grp]
        gvalid = subject_df[
            (subject_df["group"] == grp) & subject_df["posterior_exponent_knee"].notna()
        ]
        fail_n = len(gcohort) - len(gvalid)
        rows.append(
            {
                "metric": "posterior_knee_subject_fail_n",
                "value": float(fail_n),
                "n": len(gcohort),
                "group": grp,
                "note": "missing posterior_exponent_knee",
            }
        )
        rows.append(
            {
                "metric": "posterior_channel_fit_valid_rate",
                "value": float(gpost["fit_valid"].mean()) if len(gpost) else np.nan,
                "n": len(gpost),
                "group": grp,
                "note": "",
            }
        )

    asd_fail = int(
        (
            (cohort["group"] == "ASD")
            & subject_df["posterior_exponent_knee"].isna()
        ).sum()
    )
    td_fail = int(
        (
            (cohort["group"] == "TD")
            & subject_df["posterior_exponent_knee"].isna()
        ).sum()
    )
    asd_ok = int((cohort["group"] == "ASD").sum()) - asd_fail
    td_ok = int((cohort["group"] == "TD").sum()) - td_fail
    if asd_fail + td_fail > 0:
        table = np.array([[asd_ok, asd_fail], [td_ok, td_fail]])
        if table.min() >= 0 and table.sum() > 0:
            if table.min() < 5:
                _, fisher_p = stats.fisher_exact(table)
                test_name = "fisher_exact"
            else:
                _, fisher_p, _, _ = stats.chi2_contingency(table)
                test_name = "chi2"
            rows.append(
                {
                    "metric": "posterior_knee_fail_rate_group_test",
                    "value": float(fisher_p),
                    "n": int(table.sum()),
                    "group": "ASD_vs_TD",
                    "note": test_name,
                }
            )

    fixed_ch_path = Path(cfg["paths"]["derivatives_root"]) / "specparam" / "specparam_channel_results_qc.csv"
    if fixed_ch_path.exists():
        fixed_post = _posterior_channel_qc_slice(pd.read_csv(fixed_ch_path))
        fixed_post = fixed_post[fixed_post["subject_id"].astype(str).isin(cohort_ids)]
        rows.append(
            {
                "metric": "fixed_posterior_channel_fit_valid_rate",
                "value": float(fixed_post["fit_valid"].mean()) if len(fixed_post) else np.nan,
                "n": len(fixed_post),
                "group": "all",
                "note": "fixed-mode reference",
            }
        )

    if not sub_cohort.empty and "mean_r_squared_knee" in sub_cohort.columns:
        for grp in ["ASD", "TD"]:
            vals = sub_cohort.loc[sub_cohort["group"] == grp, "mean_r_squared_knee"].dropna()
            rows.append(
                {
                    "metric": "subject_mean_r_squared_knee",
                    "value": float(vals.mean()) if len(vals) else np.nan,
                    "n": len(vals),
                    "group": grp,
                    "note": "",
                }
            )

    return pd.DataFrame(rows)


def _classify_scenario(group_models: pd.DataFrame) -> str:
    primary = group_models[group_models["model"] == "primary_knee"]
    reference = group_models[group_models["model"] == "reference_fixed"]
    if primary.empty or reference.empty:
        return "C"
    knee = primary.iloc[0]
    fixed = reference.iloc[0]
    same_direction = (
        np.sign(knee["beta_TD_vs_ASD"]) == np.sign(fixed["beta_TD_vs_ASD"])
        if np.isfinite(knee["beta_TD_vs_ASD"]) and np.isfinite(fixed["beta_TD_vs_ASD"])
        else False
    )
    if not same_direction:
        return "C"
    if knee["significant_p05"] and fixed["significant_p05"]:
        return "A"
    if same_direction:
        return "B"
    return "C"


def write_report(
    out_dir: Path,
    group_models: pd.DataFrame,
    corr_df: pd.DataFrame,
    age_df: pd.DataFrame,
    ados_df: pd.DataFrame,
    qc_df: pd.DataFrame,
) -> Path:
    scenario = _classify_scenario(group_models)
    primary = group_models[group_models["model"] == "primary_knee"]
    reference = group_models[group_models["model"] == "reference_fixed"]

    def _fmt(row: pd.Series) -> str:
        return (
            f"β={row['beta_TD_vs_ASD']:.4f}, p={row['p']:.4g}, "
            f"n={int(row['n_total'])}, {row['direction']}"
        )

    snippets = {
        "A": (
            "Fixed-mode findings were not dependent on the aperiodic model choice.",
            "fixed 模式主要发现在 knee 敏感性分析中方向一致且均达到 p<0.05，"
            "表明主分析结论不依赖 aperiodic 模型设定。",
        ),
        "B": (
            "Knee-mode sensitivity showed directionally consistent but weaker evidence, "
            "supporting cautious interpretation.",
            "knee 模式敏感性分析方向与 fixed 主分析一致，但证据强度较弱，"
            "支持谨慎解读 fixed 模式主发现。",
        ),
        "C": (
            "Knee-mode sensitivity was unstable or directionally inconsistent with fixed-mode "
            "results; interpret HBN convergence as exploratory only.",
            "knee 模式结果不稳定或与 fixed 主分析方向不一致；"
            "HBN EO 收敛结果仅作探索性解读。",
        ),
    }
    en_snip, zh_snip = snippets.get(scenario, snippets["C"])

    lines = [
        "# Posterior Knee-Mode Sensitivity Report",
        "",
        f"**Scenario classification:** {scenario}",
        "",
        "## Group effect (posterior ROI: E33/E36/E37/E38)",
        "",
    ]
    if not reference.empty:
        lines.append(f"- Fixed reference: {_fmt(reference.iloc[0])}")
    if not primary.empty:
        lines.append(f"- Knee primary: {_fmt(primary.iloc[0])}")

    if not corr_df.empty:
        all_row = corr_df[corr_df["subset"] == "all"]
        if not all_row.empty:
            r = all_row.iloc[0]
            lines.extend(
                [
                    "",
                    "## Fixed vs knee correlation",
                    "",
                    f"- Pearson r = {r['pearson_r']:.3f} (p = {r['pearson_p']:.4g}), "
                    f"Spearman ρ = {r['spearman_rho']:.3f} (n = {int(r['n'])})",
                ]
            )

    if not age_df.empty:
        for mode in ["fixed_reference", "knee"]:
            sub = age_df[(age_df["mode"] == mode) & (age_df["term"] == INTERACTION_TERM)]
            if not sub.empty:
                row = sub.iloc[0]
                lines.append(
                    f"- Age×group ({mode}): coef = {row['coef']:.4f}, p = {row['p']:.4g}"
                )

    if not ados_df.empty:
        sa = ados_df[
            (ados_df["outcome"] == "ADOS_SA")
            & (ados_df["analysis"] == "partial_correlation")
        ]
        for mode in ["fixed_reference", "knee"]:
            sub = sa[sa["mode"] == mode]
            if not sub.empty:
                row = sub.iloc[0]
                lines.append(
                    f"- ADOS SA partial r ({mode}): r = {row['partial_r']:.3f}, p = {row['p']:.4g}, n = {int(row['n'])}"
                )

    lines.extend(["", "## Suggested supplementary text (EN)", "", en_snip, "", "## 建议补充材料表述（中文）", "", zh_snip, ""])

    if not qc_df.empty:
        lines.extend(["", "## QC highlights", ""])
        for _, row in qc_df.head(12).iterrows():
            lines.append(
                f"- {row['metric']} ({row['group']}): {row['value']} (n={row['n']}) {row['note']}"
            )

    report_path = out_dir / "posterior_knee_sensitivity_report_zh.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_posterior_knee_sensitivity(
    cfg: dict[str, Any],
    *,
    overwrite: bool = False,
    skip_refit: bool = False,
    limit_subjects: int | None = None,
) -> dict[str, Path]:
    out_dir_path = output_dir(cfg)
    ensure_dir(out_dir_path)

    cohort = load_primary_cohort(cfg)
    if limit_subjects is not None:
        cohort = cohort.head(limit_subjects).copy()
        logger.info("限制被试数: %d", len(cohort))

    if not skip_refit:
        refit_knee_specparam(cfg, cohort, overwrite=overwrite)
    run_knee_qc(cfg, overwrite=overwrite or not skip_refit)

    paths = knee_deriv_paths(cfg)
    knee_ch = pd.read_csv(paths["channel_qc"])
    knee_sub = pd.read_csv(paths["subject_qc"])

    subject_df = build_subject_comparison_table(cfg, cohort, knee_ch, knee_sub)
    group_models = fit_group_models(subject_df)
    corr_df = compute_fixed_knee_correlation(subject_df)
    age_df = fit_age_interaction_models(subject_df)
    ados_df = fit_ados_models(subject_df)
    qc_df = build_knee_fit_qc_summary(cfg, cohort, knee_ch, subject_df)

    outputs = {
        "subject_table": out_dir_path / "subject_level_fixed_knee_comparison.csv",
        "group_ols": out_dir_path / "group_ols_models.csv",
        "correlation": out_dir_path / "fixed_knee_correlation.csv",
        "age_interaction": out_dir_path / "age_interaction_models.csv",
        "ados": out_dir_path / "ados_association.csv",
        "qc_summary": out_dir_path / "knee_fit_qc_summary.csv",
    }
    save_csv(subject_df, outputs["subject_table"])
    save_csv(group_models, outputs["group_ols"])
    save_csv(corr_df, outputs["correlation"])
    save_csv(age_df, outputs["age_interaction"])
    save_csv(ados_df, outputs["ados"])
    save_csv(qc_df, outputs["qc_summary"])
    report_path = write_report(out_dir_path, group_models, corr_df, age_df, ados_df, qc_df)
    outputs["report"] = report_path

    logger.info("Posterior knee sensitivity 完成: %s", out_dir_path)
    return outputs
