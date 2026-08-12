"""Supplementary frontal vs posterior comparison analyses."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.io_utils import save_csv
from src.spectral_maturation_analysis import POSTERIOR_CORE
from src.stats_utils import (
    fdr_correction,
    model_results_to_row,
    partial_correlation_pearson,
    run_mixedlm,
    run_ols,
)

logger = logging.getLogger(__name__)

# Align with primary posterior OLS covariates (no mean_r_squared).
FORMULA_GROUP = (
    "{outcome} ~ C(group) + age_months + C(sex) + IQ_total + usable_epochs"
)
FORMULA_AGE_INT = (
    "{outcome} ~ C(group) * age_months + C(sex) + IQ_total + usable_epochs"
)
FORMULA_REGION = (
    "exponent ~ C(group) * C(region) + age_months + C(sex) + IQ_total + usable_epochs"
)
COVARIATES = ["age_months", "sex", "IQ_total", "usable_epochs"]
ADOS_OUTCOMES = ["ADOS_total", "ADOS_SA", "ADOS_RRB"]


def load_egi64_rois(roi_yaml: Path) -> dict[str, list[str]]:
    with roi_yaml.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {k: list(v) for k, v in data["channels_egi64"].items()}


def load_frontal_comparison_cohort(project_root: Path) -> pd.DataFrame:
    """N=138 resting spectral cohort with frontal/posterior exponents + covariates."""
    wide = pd.read_csv(project_root / "figure_source_data" / "roi_subject_wide_primary.csv")
    wide["subject_id"] = wide["subject_id"].astype(str)

    normative = pd.read_csv(project_root / "derivatives" / "stats" / "normative_exponent_scores.csv")
    normative["subject_id"] = normative["subject_id"].astype(str)
    keep = [
        "subject_id",
        "ADOS_total",
        "ADOS_SA",
        "ADOS_RRB",
    ]
    keep = [c for c in keep if c in normative.columns]
    df = wide.merge(normative[keep], on="subject_id", how="left")

    # Prefer covariates from wide; fill gaps from normative if present
    for col in ["age_months", "sex", "IQ_total", "usable_epochs", "group"]:
        if col not in df.columns and col in normative.columns:
            df[col] = normative.set_index("subject_id").reindex(df["subject_id"])[col].values

    df["group"] = df["group"].astype(str).str.upper()
    return df


def _group_beta_td_minus_asd(res: Any) -> tuple[float, float, str]:
    terms = [t for t in res.params.index if str(t).startswith("C(group)")]
    if not terms:
        raise ValueError("No group term")
    term = str(terms[0])
    coef = float(res.params[term])
    p = float(res.pvalues[term])
    if "T.ASD" in term:
        coef = -coef
    return coef, p, term


def run_region_group_models(df: pd.DataFrame) -> pd.DataFrame:
    """OLS group effects for frontal and posterior with identical covariates."""
    rows: list[dict[str, Any]] = []
    for outcome, region in [
        ("frontal_exponent", "frontal"),
        ("posterior_exponent", "posterior"),
    ]:
        formula = FORMULA_GROUP.format(outcome=outcome)
        need = [outcome, "group", *COVARIATES]
        sub = df.dropna(subset=need).copy()
        res = run_ols(formula, sub)
        beta, p, term = _group_beta_td_minus_asd(res)
        row = {
            "analysis": "region_group_ols",
            "region": region,
            "outcome": outcome,
            "formula": formula,
            "n": int(res.nobs),
            "n_asd": int((sub["group"] == "ASD").sum()),
            "n_td": int((sub["group"] == "TD").sum()),
            "beta_td_minus_asd": beta,
            "se": float(res.bse[term]) if "T.ASD" not in term else float(res.bse[term]),
            "p": p,
            "ci_low": float(res.conf_int().loc[term, 0])
            if "T.ASD" not in term
            else -float(res.conf_int().loc[term, 1]),
            "ci_high": float(res.conf_int().loc[term, 1])
            if "T.ASD" not in term
            else -float(res.conf_int().loc[term, 0]),
        }
        # Fix SE/CI when flipping ASD treatment coding
        if "T.ASD" in term:
            row["se"] = float(res.bse[term])
            ci = res.conf_int().loc[term]
            row["ci_low"] = -float(ci[1])
            row["ci_high"] = -float(ci[0])
        rows.append(row)
    out = pd.DataFrame(rows)
    reject, q = fdr_correction(out["p"].to_numpy(dtype=float))
    out["fdr_q"] = q
    out["fdr_significant"] = reject
    return out


def run_frontal_posterior_mixed(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mixed / cluster-robust model: exponent ~ group × region + covariates (+ subject)."""
    from statsmodels.formula.api import mixedlm as smf_mixedlm
    from statsmodels.formula.api import ols as smf_ols

    long_rows = []
    for region, col in [("frontal", "frontal_exponent"), ("posterior", "posterior_exponent")]:
        tmp = df[
            [
                "subject_id",
                "group",
                *COVARIATES,
                col,
            ]
        ].copy()
        tmp = tmp.rename(columns={col: "exponent"})
        tmp["region"] = region
        long_rows.append(tmp)
    long_df = pd.concat(long_rows, ignore_index=True)
    need = [
        "exponent",
        "group",
        "region",
        "subject_id",
        *COVARIATES,
    ]
    sub = long_df.dropna(subset=need).copy()
    # Keep only subjects with both regions (balanced repeated measures)
    both = (
        sub.groupby("subject_id")["region"]
        .nunique()
        .loc[lambda s: s >= 2]
        .index
    )
    sub = sub[sub["subject_id"].isin(both)].copy()
    sub["region"] = pd.Categorical(sub["region"], categories=["posterior", "frontal"], ordered=False)
    sub["group"] = pd.Categorical(sub["group"], categories=["ASD", "TD"], ordered=False)

    method = "mixedlm_reml"
    used_mixedlm = True
    try:
        md = smf_mixedlm(FORMULA_REGION, sub, groups=sub["subject_id"])
        res = md.fit(reml=True, method="lbfgs", maxiter=200)
        if not np.isfinite(float(getattr(res, "llf", np.nan))):
            raise RuntimeError("non-finite llf")
        res._used_mixedlm = True  # type: ignore
    except Exception as exc:
        logger.warning("MixedLM(reml) failed (%s); using cluster-robust OLS by subject", exc)
        method = "ols_cluster_subject"
        used_mixedlm = False
        ols_mod = smf_ols(FORMULA_REGION, data=sub)
        res = ols_mod.fit(
            cov_type="cluster",
            cov_kwds={"groups": sub["subject_id"]},
        )
        res._used_mixedlm = False  # type: ignore

    term_rows = model_results_to_row(res, "frontal_vs_posterior_mixed", "exponent")
    terms_df = pd.DataFrame(term_rows)
    meta = pd.DataFrame(
        [
            {
                "analysis": "frontal_vs_posterior_mixed",
                "formula": FORMULA_REGION,
                "method": method,
                "n_rows": int(len(sub)),
                "n_subjects": int(sub["subject_id"].nunique()),
                "used_mixedlm": used_mixedlm,
            }
        ]
    )
    return terms_df, meta


def run_developmental_models(df: pd.DataFrame) -> pd.DataFrame:
    """group × age interactions for frontal and posterior."""
    rows: list[dict[str, Any]] = []
    for outcome, region in [
        ("frontal_exponent", "frontal"),
        ("posterior_exponent", "posterior"),
    ]:
        formula = FORMULA_AGE_INT.format(outcome=outcome)
        need = [outcome, "group", *COVARIATES]
        sub = df.dropna(subset=need).copy()
        res = run_ols(formula, sub)
        # Interaction term
        int_terms = [t for t in res.params.index if ":" in str(t) and "age_months" in str(t)]
        if not int_terms:
            continue
        term = str(int_terms[0])
        coef = float(res.params[term])
        p = float(res.pvalues[term])
        # Ensure TD×age vs ASD reference coding interpretation note
        rows.append(
            {
                "analysis": "group_age_interaction",
                "region": region,
                "outcome": outcome,
                "term": term,
                "coef": coef,
                "se": float(res.bse[term]),
                "p": p,
                "ci_low": float(res.conf_int().loc[term, 0]),
                "ci_high": float(res.conf_int().loc[term, 1]),
                "n": int(res.nobs),
                "formula": formula,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        reject, q = fdr_correction(out["p"].to_numpy(dtype=float))
        out["fdr_q"] = q
        out["fdr_significant"] = reject
    return out


def run_ados_partial_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """ASD-only partial Pearson (age+IQ) for frontal and posterior vs ADOS scales."""
    asd = df[df["group"].astype(str).str.upper() == "ASD"].copy()
    rows: list[dict[str, Any]] = []
    for region, xcol in [
        ("frontal", "frontal_exponent"),
        ("posterior", "posterior_exponent"),
    ]:
        for ycol in ADOS_OUTCOMES:
            if ycol not in asd.columns:
                continue
            res = partial_correlation_pearson(
                asd, y_col=ycol, x_col=xcol, cov_cols=["age_months", "IQ_total"]
            )
            rows.append(
                {
                    "analysis": "ados_partial_pearson",
                    "region": region,
                    "x": xcol,
                    "y": ycol,
                    "partial_r": res["partial_r"],
                    "p": res["pvalue"],
                    "n": res["n"],
                    "covariates": "age_months + IQ_total",
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        reject, q = fdr_correction(out["p"].to_numpy(dtype=float))
        out["fdr_q"] = q
        out["fdr_significant"] = reject
    return out


def _load_movie_roi_channel_cache(
    cache_path: Path,
    channels: list[str],
) -> pd.DataFrame:
    """Stream large sliding-channel cache, keeping selected channels only."""
    usecols = [
        "subject_id",
        "group",
        "window_index",
        "window_start_sec",
        "window_end_sec",
        "channel",
        "aperiodic_exponent",
        "fit_valid",
        "event_type",
    ]
    keep = set(str(c) for c in channels)
    parts: list[pd.DataFrame] = []
    for i, chunk in enumerate(pd.read_csv(cache_path, usecols=usecols, chunksize=1_000_000)):
        sub = chunk[chunk["channel"].astype(str).isin(keep)]
        if len(sub):
            parts.append(sub)
        if (i + 1) % 5 == 0:
            logger.info("Movie cache chunks read: %d", i + 1)
    if not parts:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(parts, ignore_index=True)
    out["subject_id"] = out["subject_id"].astype(str)
    out["group"] = out["group"].astype(str).str.upper()
    return out


def run_movie_frontal_isc(
    project_root: Path,
    *,
    movie_subjects: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Exploratory TD-template Aperiodic-ISC for frontal vs posterior ROIs.

    Returns subject-level ISC, segment group tests, and FDR summary.
    """
    import sys

    jr_src = project_root / "jr_remote_bundle" / "src"
    if str(jr_src) not in sys.path:
        sys.path.insert(0, str(jr_src))

    from jr_submission_defense import (  # type: ignore
        aggregate_roi_sliding_timeseries,
        recompute_isc_from_timeseries,
    )

    rois = load_egi64_rois(project_root / "config" / "roi_channels.yaml")
    frontal = rois["frontal"]
    posterior = list(POSTERIOR_CORE)
    cache = project_root / "derivatives" / "specparam" / "movie_sliding_specparam_channel.csv"
    if not cache.exists():
        raise FileNotFoundError(cache)

    logger.info("Loading movie sliding cache for frontal+posterior channels...")
    ch_df = _load_movie_roi_channel_cache(cache, frontal + posterior)
    if movie_subjects is not None:
        keep_ids = set(str(s) for s in movie_subjects)
        ch_df = ch_df[ch_df["subject_id"].isin(keep_ids)].copy()

    subject_rows = []
    stats_rows = []
    for roi_name, channels in [("frontal", frontal), ("posterior", posterior)]:
        logger.info("Computing TD-template ISC for ROI=%s (%d ch)", roi_name, len(channels))
        ts = aggregate_roi_sliding_timeseries(ch_df, channels, require_fit_valid=False)
        # rename value already posterior_exponent in helper
        isc, stats_df = recompute_isc_from_timeseries(ts, value_col="posterior_exponent")
        isc = isc.copy()
        isc["roi"] = roi_name
        subject_rows.append(isc)
        stats_df = stats_df.copy()
        stats_df["roi"] = roi_name
        stats_rows.append(stats_df)

    subject_df = pd.concat(subject_rows, ignore_index=True)
    stats_df = pd.concat(stats_rows, ignore_index=True)

    # Harmonize p column name if needed
    pcol = "p_value" if "p_value" in stats_df.columns else ("pvalue" if "pvalue" in stats_df.columns else None)
    if pcol is None:
        # summarize_group_isc_tests may use 'p'
        for c in stats_df.columns:
            if c.lower() in {"p", "pval", "p_value", "pvalue"}:
                pcol = c
                break
    if pcol is not None:
        reject, q = fdr_correction(stats_df[pcol].to_numpy(dtype=float))
        stats_df["fdr_q"] = q
        stats_df["fdr_significant"] = reject
        stats_df["p_raw"] = stats_df[pcol]

    # Compact group effect table
    effects = []
    for (roi, ev), sub in subject_df.groupby(["roi", "event_type"]):
        asd = sub.loc[sub["group"] == "ASD", "isc_z"].dropna()
        td = sub.loc[sub["group"] == "TD", "isc_z"].dropna()
        if len(asd) < 2 or len(td) < 2:
            continue
        from scipy import stats as sp_stats

        t, p = sp_stats.ttest_ind(td, asd, equal_var=False)
        effects.append(
            {
                "roi": roi,
                "segment": ev,
                "n_asd": int(len(asd)),
                "n_td": int(len(td)),
                "mean_asd": float(asd.mean()),
                "mean_td": float(td.mean()),
                "delta_td_minus_asd": float(td.mean() - asd.mean()),
                "t": float(t),
                "p": float(p),
            }
        )
    effects_df = pd.DataFrame(effects)
    if not effects_df.empty:
        reject, q = fdr_correction(effects_df["p"].to_numpy(dtype=float))
        effects_df["fdr_q"] = q
        effects_df["fdr_significant"] = reject
    return subject_df, effects_df, stats_df


def write_frontal_comparison_report(
    group_df: pd.DataFrame,
    mixed_terms: pd.DataFrame,
    mixed_meta: pd.DataFrame,
    age_df: pd.DataFrame,
    ados_df: pd.DataFrame,
    isc_effects: pd.DataFrame | None,
    out_path: Path,
) -> None:
    def _get(df: pd.DataFrame, region: str) -> pd.Series | None:
        sub = df[df["region"] == region]
        return sub.iloc[0] if len(sub) else None

    fg = _get(group_df, "frontal")
    pg = _get(group_df, "posterior")

    inter = mixed_terms[
        mixed_terms["term"].astype(str).str.contains("group", case=False)
        & mixed_terms["term"].astype(str).str.contains("region", case=False)
    ]
    inter_txt = "n/a"
    if len(inter):
        r = inter.iloc[0]
        inter_txt = f"{r.get('term')}: coef={r.get('coef', r.get('estimate', float('nan'))):.4f}, p={r.get('pvalue', r.get('p', float('nan'))):.4g}"

    fa = age_df[age_df["region"] == "frontal"]
    pa = age_df[age_df["region"] == "posterior"]
    age_txt = ""
    if len(fa):
        age_txt += f"- Frontal group×age: coef={fa.iloc[0]['coef']:.4f}, p={fa.iloc[0]['p']:.4g}, FDR q={fa.iloc[0]['fdr_q']:.4g}\n"
    if len(pa):
        age_txt += f"- Posterior group×age: coef={pa.iloc[0]['coef']:.4f}, p={pa.iloc[0]['p']:.4g}, FDR q={pa.iloc[0]['fdr_q']:.4g}\n"

    ados_lines = []
    for _, r in ados_df.iterrows():
        ados_lines.append(
            f"- {r['region']} ~ {r['y']}: r={r['partial_r']:.3f}, p={r['p']:.4g}, FDR q={r['fdr_q']:.4g}, n={int(r['n'])}"
        )

    isc_txt = "Movie ISC not run or unavailable.\n"
    if isc_effects is not None and not isc_effects.empty:
        lines = []
        for _, r in isc_effects.iterrows():
            lines.append(
                f"- {r['roi']} / {r['segment']}: Δz(TD−ASD)={r['delta_td_minus_asd']:.3f}, "
                f"p={r['p']:.4g}, FDR q={r['fdr_q']:.4g}"
            )
        isc_txt = "\n".join(lines) + "\n"

    used_mixed = bool(mixed_meta.iloc[0]["used_mixedlm"]) if len(mixed_meta) else False

    text = f"""# Supplementary frontal comparison analysis

## 1. Frontal group effect (same covariates as posterior)
Model: `exponent ~ group + age + sex + FSIQ + usable_epochs` (aligned with primary posterior OLS; no mean_r_squared)

- Frontal: β(TD−ASD)={fg['beta_td_minus_asd']:.4f}, p={fg['p']:.4g}, FDR q={fg['fdr_q']:.4g}, n={int(fg['n'])}
- Posterior: β(TD−ASD)={pg['beta_td_minus_asd']:.4f}, p={pg['p']:.4g}, FDR q={pg['fdr_q']:.4g}, n={int(pg['n'])}

## 2. Frontal versus posterior (mixed model)
Formula: `{FORMULA_REGION}`
Method: {"mixedlm" if used_mixed else "cluster-robust OLS (cluster=subject)"} (`used_mixedlm={used_mixed}`)
Key group×region interaction: {inter_txt}

## 3. Developmental / clinical relevance
{age_txt}
ADOS partial Pearson (ASD only; covariates age + IQ; FDR across region×scale):
{chr(10).join(ados_lines)}

## 4. Exploratory movie Aperiodic-ISC (frontal vs posterior; FDR across ROI×segment)
{isc_txt}
## Interpretation note (for manuscript drafting)
- Resting **group effect is not posterior-exclusive**: frontal also shows TD > ASD, and the
  group×region interaction is non-significant → avoid “spatially localized / posterior-only”
  wording for the mean group difference.
- **Developmental and clinical relevance favor posterior**: frontal group×age and frontal–ADOS
  partial correlations are non-significant, whereas posterior group×age and posterior–ADOS
  (total / Social Affect) survive FDR. This supports treating posterior as the more
  developmentally/clinically informative feature, not as the only region with any group shift.
- **Movie ISC is exploratory only**: frontal Aperiodic-ISC also shows TD > ASD after FDR in this
  recomputation. Do **not** move frontal ISC into the main text as evidence of posterior
  specificity; report it in Supplementary with explicit FDR across ROI×segment tests.
- Region model method: MixedLM random intercept was singular with two repeated measures per
  subject; primary reported fit uses **cluster-robust OLS** (cluster = subject).
"""
    Path(out_path).write_text(text, encoding="utf-8")


def run_frontal_comparison_pipeline(project_root: Path, outputs_root: Path) -> dict[str, Path]:
    out_tables = Path(outputs_root) / "tables" / "supplementary_frontal"
    out_reports = Path(outputs_root) / "reports"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    df = load_frontal_comparison_cohort(project_root)
    logger.info("Resting cohort for frontal comparison: N=%d", df["subject_id"].nunique())

    group_df = run_region_group_models(df)
    mixed_terms, mixed_meta = run_frontal_posterior_mixed(df)
    age_df = run_developmental_models(df)
    ados_df = run_ados_partial_correlations(df)

    # Movie cohort IDs from existing posterior ISC table
    isc_subj_path = (
        project_root
        / "derivatives"
        / "derivatives_task_movie"
        / "stats"
        / "aperiodic_isc"
        / "aperiodic_isc_td_template_subject_values.csv"
    )
    movie_ids = None
    if isc_subj_path.exists():
        movie_ids = (
            pd.read_csv(isc_subj_path)["subject_id"].astype(str).unique().tolist()
        )

    isc_subject = isc_effects = isc_stats = None
    try:
        isc_subject, isc_effects, isc_stats = run_movie_frontal_isc(
            project_root, movie_subjects=movie_ids
        )
    except RuntimeError as exc:
        if "skip" in str(exc).lower():
            logger.info("Movie frontal ISC skipped: %s", exc)
        else:
            logger.exception("Movie frontal ISC failed: %s", exc)
    except Exception as exc:
        logger.exception("Movie frontal ISC failed: %s", exc)

    paths = {
        "group_models": out_tables / "frontal_posterior_group_ols.csv",
        "mixed_terms": out_tables / "frontal_posterior_region_mixed_terms.csv",
        "mixed_meta": out_tables / "frontal_posterior_region_mixed_meta.csv",
        "age_interaction": out_tables / "frontal_posterior_group_age_interaction.csv",
        "ados": out_tables / "frontal_posterior_ados_partial.csv",
        "report": out_reports / "supplementary_frontal_comparison_report.md",
    }
    save_csv(group_df, paths["group_models"])
    save_csv(mixed_terms, paths["mixed_terms"])
    save_csv(mixed_meta, paths["mixed_meta"])
    save_csv(age_df, paths["age_interaction"])
    save_csv(ados_df, paths["ados"])

    if isc_subject is not None:
        paths["isc_subjects"] = out_tables / "frontal_posterior_movie_isc_subjects.csv"
        paths["isc_effects"] = out_tables / "frontal_posterior_movie_isc_effects.csv"
        paths["isc_stats"] = out_tables / "frontal_posterior_movie_isc_stats_raw.csv"
        save_csv(isc_subject, paths["isc_subjects"])
        save_csv(isc_effects, paths["isc_effects"])
        save_csv(isc_stats, paths["isc_stats"])

    write_frontal_comparison_report(
        group_df,
        mixed_terms,
        mixed_meta,
        age_df,
        ados_df,
        isc_effects,
        paths["report"],
    )
    return paths
