"""Exploratory supplementary analyses: posterior aperiodic coordination feature."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.clinical_matched_analysis import load_clinical_posterior_cohort
from src.config import PROJECT_ROOT
from src.io_utils import attach_usable_epochs, load_participants, save_csv
from src.spectral_maturation_analysis import POSTERIOR_CORE, _posterior_channel_mean
from src.stats_utils import (
    bootstrap_partial_spearman,
    fdr_correction,
    model_results_to_row,
    partial_correlation_pearson,
    partial_spearman,
    run_mixedlm,
    run_ols,
)

logger = logging.getLogger(__name__)

OUT_DIR = PROJECT_ROOT / "outputs" / "tables" / "coordination_feature"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "coordination_feature_report.md"

SEGMENTS = ["mental", "pain", "neutral"]
COVARIATES_CLINICAL = ["age_months", "IQ_total"]
COVARIATES_GROUP = "age_months + C(sex) + IQ_total + usable_epochs"
COVARIATES_EXPONENT = f"{COVARIATES_GROUP} + mean_r_squared"

ADOS_ALIASES: dict[str, list[str]] = {
    "ADOS_total": ["ADOS_total", "ados_total", "ADOS_total_score", "ADOS-2", "ADOS"],
    "ADOS_SA": ["ADOS_SA", "ados_sa", "ADOS-SA", "SA", "ADOS_social_affect", "social_affect", "Social"],
    "ADOS_RRB": ["ADOS_RRB", "ados_rrb", "ADOS-RRB", "RRB", "rrb"],
}
CLINICAL_ALTERNATIVES = ["SRS_total", "CARS_total", "ABC_total", "language_score"]

MOVIE_ISC_CANDIDATES: list[tuple[str, str]] = [
    (
        "derivatives_task_movie/stats/aperiodic_isc/aperiodic_isc_td_template_subject_values.csv",
        "td_template_global",
    ),
    (
        "derivatives_task_movie/stats/movie_isc_subject_values_with_neutral.csv",
        "legacy_global",
    ),
    (
        "derivatives_task_movie/stats/movie_isc_subject_values.csv",
        "legacy_global",
    ),
    (
        "outputs/jr_modelling/posterior_movie_isc/posterior_isc_subject_values.csv",
        "posterior_sliding_exploratory",
    ),
    (
        "jr_remote_bundle/outputs/jr_modelling/posterior_movie_isc/posterior_isc_subject_values.csv",
        "posterior_sliding_exploratory",
    ),
]

GROUP_REF = "C(group, Treatment(reference='ASD'))"
REGION_REF = "C(region_type, Treatment(reference='nonposterior'))"


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def resolve_ados_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Map canonical ADOS names to actual column names present in df."""
    resolved: dict[str, str | None] = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for canonical, aliases in ADOS_ALIASES.items():
        found = None
        for alias in aliases:
            if alias in df.columns:
                found = alias
                break
            if alias.lower() in cols_lower:
                found = cols_lower[alias.lower()]
                break
        resolved[canonical] = found
    return resolved


def resolve_movie_isc_path(root: Path | None = None) -> tuple[Path | None, str]:
    """Return first existing movie ISC CSV and its source label."""
    root = root or PROJECT_ROOT
    for rel, label in MOVIE_ISC_CANDIDATES:
        path = (root / rel).resolve()
        if path.exists():
            return path, label
    return None, ""


def _isc_metric_label(source: str) -> str:
    if source == "posterior_sliding_exploratory":
        return "isc_z (posterior sliding-window TD-template Fisher z; exploratory fallback)"
    return "isc_z (TD-template Fisher z)"


def _non_posterior_channel_mean(
    ch_df: pd.DataFrame,
    value_col: str = "aperiodic_exponent",
    exclude: list[str] | None = None,
    min_ratio: float = 0.5,
) -> pd.DataFrame:
    """Mean exponent over fit-valid channels excluding posterior core."""
    exclude = exclude or POSTERIOR_CORE
    if "fit_valid" in ch_df.columns:
        ch_df = ch_df[ch_df["fit_valid"]].copy()
    rows: list[dict[str, Any]] = []
    for (sid, grp), sub in ch_df.groupby(["subject_id", "group"]):
        roi = sub[~sub["channel"].isin(exclude)]
        n_valid = roi[value_col].notna().sum() if value_col in roi.columns else 0
        n_channels = roi["channel"].nunique()
        if n_channels == 0 or n_valid < min_ratio * n_channels:
            val = np.nan
        else:
            val = float(roi[value_col].mean())
        rows.append({"subject_id": str(sid), "group": grp, value_col: val})
    return pd.DataFrame(rows)


def _clinical_outcome_list(ados_map: dict[str, str | None]) -> list[tuple[str, str]]:
    labels = {
        "ADOS_total": "ADOS total",
        "ADOS_SA": "ADOS Social Affect",
        "ADOS_RRB": "ADOS Restricted and Repetitive Behavior",
    }
    out: list[tuple[str, str]] = []
    for canonical, label in labels.items():
        col = ados_map.get(canonical)
        if col is not None:
            out.append((col, label))
    return out


def _apply_fdr(df: pd.DataFrame, p_col: str = "raw_p", q_col: str = "fdr_q") -> pd.DataFrame:
    out = df.copy()
    if len(out) and out[p_col].notna().any():
        _, q = fdr_correction(out[p_col].values)
        out[q_col] = q
    else:
        out[q_col] = np.nan
    return out


def _partial_corr_row(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    *,
    analysis: str,
    eeg_metric: str,
    clinical_label: str,
    segment: str | None,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    pear = partial_correlation_pearson(df, y_col, x_col, cov_cols=COVARIATES_CLINICAL)
    boot = bootstrap_partial_spearman(
        df, y_col, x_col, n_boot=n_boot, seed=seed, cov_cols=COVARIATES_CLINICAL,
    )
    return {
        "analysis": analysis,
        "segment": segment or "",
        "eeg_metric": eeg_metric,
        "clinical_outcome": y_col,
        "clinical_label": clinical_label,
        "n": pear["n"],
        "partial_r": pear["partial_r"],
        "partial_rho": boot["partial_rho"],
        "raw_p": pear["pvalue"],
        "spearman_p": boot["pvalue"],
        "boot_ci_low": boot["boot_ci_low"],
        "boot_ci_high": boot["boot_ci_high"],
        "boot_median": boot.get("boot_median"),
        "n_boot": n_boot,
        "covariates": "+".join(COVARIATES_CLINICAL),
        "direction_note": "lower exponent / lower ISC associated with greater ADOS severity (negative partial r expected)",
    }


def build_posterior_nonposterior_long(
    cfg: dict[str, Any],
    deriv: Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Subject-level long table: posterior vs non-posterior exponent."""
    deriv = deriv or Path(cfg["paths"]["derivatives_root"])
    ch_path = deriv / "specparam" / "specparam_channel_results_qc.csv"
    if not ch_path.exists():
        raise FileNotFoundError(f"Missing channel specparam: {ch_path}")

    ch_df = pd.read_csv(ch_path)
    ch_df["subject_id"] = ch_df["subject_id"].astype(str)
    if "fit_valid" in ch_df.columns:
        ch_df = ch_df[ch_df["fit_valid"]].copy()

    post = _posterior_channel_mean(ch_df, "aperiodic_exponent")
    post = post.rename(columns={"aperiodic_exponent": "exponent"})
    post["region_type"] = "posterior"

    nonpost = _non_posterior_channel_mean(ch_df, "aperiodic_exponent")
    nonpost = nonpost.rename(columns={"aperiodic_exponent": "exponent"})
    nonpost["region_type"] = "nonposterior"

    long_df = pd.concat([post, nonpost], ignore_index=True)

    cohort = load_clinical_posterior_cohort(cfg, deriv)
    demo_cols = ["subject_id", "group", "age_months", "sex", "IQ_total", "usable_epochs"]
    if "mean_r_squared" in cohort.columns:
        demo_cols.append("mean_r_squared")
    demo = cohort[demo_cols].drop_duplicates("subject_id")
    demo["subject_id"] = demo["subject_id"].astype(str)

    long_df = long_df.merge(demo, on=["subject_id", "group"], how="inner")
    long_df = long_df.dropna(
        subset=["exponent", "age_months", "sex", "IQ_total", "usable_epochs"],
    )
    if "mean_r_squared" in long_df.columns:
        long_df = long_df.dropna(subset=["mean_r_squared"])
    return long_df.reset_index(drop=True), ch_path


def run_analysis1_ados_subdomain(
    cfg: dict[str, Any],
    movie_cfg: dict[str, Any] | None,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Analysis 1: ADOS subdomain specificity (resting + movie ISC)."""
    deriv = Path(cfg["paths"]["derivatives_root"])
    participants_path = Path(cfg["paths"]["participants_file"])
    participants = load_participants(participants_path, included_only=True)

    ados_map = resolve_ados_columns(participants)
    missing_ados = [k for k, v in ados_map.items() if v is None]
    alternatives = [c for c in CLINICAL_ALTERNATIVES if c in participants.columns]

    cohort = load_clinical_posterior_cohort(cfg, deriv)
    asd = cohort[cohort["group"].astype(str).str.upper() == "ASD"].copy()
    missing_ados_cols = [c for c in ados_map.values() if c is not None and c not in asd.columns]
    if missing_ados_cols:
        asd = asd.merge(
            participants[["subject_id"] + missing_ados_cols].drop_duplicates("subject_id"),
            on="subject_id",
            how="left",
        )

    resting_rows: list[dict[str, Any]] = []
    for col, label in _clinical_outcome_list(ados_map):
        resting_rows.append(
            _partial_corr_row(
                asd, col, "posterior_exponent",
                analysis="resting_posterior_exponent",
                eeg_metric="posterior_exponent",
                clinical_label=label,
                segment=None,
                n_boot=n_boot,
                seed=seed,
            )
        )
    resting_df = _apply_fdr(pd.DataFrame(resting_rows))

    movie_df = pd.DataFrame()
    movie_isc_path: Path | None = None
    movie_isc_source = ""
    movie_n_asd = 0
    movie_skipped_reason = ""

    movie_isc_path, movie_isc_source = resolve_movie_isc_path(PROJECT_ROOT)
    if movie_isc_path is None:
        movie_skipped_reason = "No movie ISC file found at expected paths."
    else:
        movie_df, movie_n_asd, movie_skipped_reason = _run_movie_ados_partial(
            movie_cfg,
            movie_isc_path,
            movie_isc_source,
            participants,
            ados_map,
            n_boot=n_boot,
            seed=seed,
        )

    return {
        "resting_df": resting_df,
        "movie_df": movie_df,
        "ados_map": ados_map,
        "missing_ados": missing_ados,
        "alternatives": alternatives,
        "n_asd_resting": int(len(asd)),
        "n_asd_movie": movie_n_asd,
        "movie_isc_path": str(movie_isc_path) if movie_isc_path else None,
        "movie_isc_source": movie_isc_source,
        "movie_skipped_reason": movie_skipped_reason,
        "participants_path": str(participants_path),
        "cohort_source": str(deriv / "specparam/specparam_channel_results_qc.csv"),
    }


def _run_movie_ados_partial(
    movie_cfg: dict[str, Any] | None,
    isc_path: Path,
    isc_source: str,
    participants: pd.DataFrame,
    ados_map: dict[str, str | None],
    *,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, int, str]:
    movie_deriv = Path(movie_cfg["paths"]["derivatives_root"]) if movie_cfg else None
    isc = pd.read_csv(isc_path)
    required = {"subject_id", "group", "event_type", "isc_z", "isc_r"}
    if not required.issubset(isc.columns):
        return pd.DataFrame(), 0, f"ISC file missing columns: {sorted(required - set(isc.columns))}"

    isc["subject_id"] = isc["subject_id"].astype(str)
    isc["group"] = isc["group"].astype(str).str.upper()
    isc["event_type"] = isc["event_type"].astype(str).str.lower()

    part_path = Path(movie_cfg["paths"]["participants_file"]) if movie_cfg else Path()
    if part_path.exists():
        movie_part = pd.read_csv(part_path)
        movie_part["subject_id"] = movie_part["subject_id"].astype(str)
        if "included_final" in movie_part.columns:
            movie_part = movie_part[movie_part["included_final"].astype(int) == 1]
        allowed = set(movie_part["subject_id"])
        isc = isc[isc["subject_id"].isin(allowed)]

    if movie_deriv is not None:
        qc_path = movie_deriv / "specparam" / "specparam_qc_summary_subject.csv"
        if qc_path.exists():
            qc = pd.read_csv(qc_path)
            qc["subject_id"] = qc["subject_id"].astype(str)
            bad = set(qc.loc[pd.to_numeric(qc["low_quality_subject"], errors="coerce") == 1, "subject_id"])
            isc = isc[~isc["subject_id"].isin(bad)]

        analysis_path = movie_deriv / "participants_analysis.csv"
        if analysis_path.exists():
            ma = pd.read_csv(analysis_path)
            ma["subject_id"] = ma["subject_id"].astype(str)
            allowed_pairs = ma[["subject_id", "group"]].drop_duplicates()
            allowed_pairs["group"] = allowed_pairs["group"].astype(str).str.upper()
            isc = isc.merge(allowed_pairs, on=["subject_id", "group"], how="inner")

    ados_cols = [c for c in ados_map.values() if c is not None]
    ados_df = participants[["subject_id"] + ados_cols].drop_duplicates("subject_id")
    ados_df["subject_id"] = ados_df["subject_id"].astype(str)

    demo_cols = ["subject_id", "age_months", "IQ_total"]
    if "sex" in participants.columns:
        demo_cols.append("sex")
    demo = participants[demo_cols].drop_duplicates("subject_id")
    demo["subject_id"] = demo["subject_id"].astype(str)

    merged = isc.merge(demo, on="subject_id", how="inner")
    merged = merged.merge(ados_df, on="subject_id", how="inner")
    asd = merged[merged["group"] == "ASD"].copy()

    rows: list[dict[str, Any]] = []
    for seg in SEGMENTS:
        sub_seg = asd[asd["event_type"] == seg].copy()
        for col, label in _clinical_outcome_list(ados_map):
            metric = f"aperiodic_isc_{seg}_fisher_z"
            row = _partial_corr_row(
                sub_seg, col, "isc_z",
                analysis="movie_aperiodic_isc",
                eeg_metric=metric,
                clinical_label=label,
                segment=seg,
                n_boot=n_boot,
                seed=seed,
            )
            row["isc_metric_primary"] = _isc_metric_label(isc_source)
            row["isc_source"] = isc_source
            row["isc_r_available"] = True
            rows.append(row)

    out = _apply_fdr(pd.DataFrame(rows))
    note = ""
    if out.empty:
        note = "Movie ISC loaded but no complete ASD rows for partial correlations."
    return out, int(asd["subject_id"].nunique()), note


def run_analysis2_posterior_nonposterior(
    cfg: dict[str, Any],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Analysis 2: posterior vs non-posterior spatial specificity."""
    deriv = Path(cfg["paths"]["derivatives_root"])
    long_df, ch_path = build_posterior_nonposterior_long(cfg, deriv)

    group_formula = (
        f"exponent ~ {GROUP_REF} * {REGION_REF} + {COVARIATES_EXPONENT}"
    )
    age_formula = (
        f"exponent ~ {GROUP_REF} * age_months * {REGION_REF} + {COVARIATES_EXPONENT}"
    )

    group_model = run_mixedlm(group_formula, long_df, groups="subject_id")
    age_model = run_mixedlm(age_formula, long_df, groups="subject_id")

    group_rows = model_results_to_row(group_model, "group_x_region", "exponent")
    for row in group_rows:
        row["formula"] = group_formula
        row["n_subjects"] = long_df["subject_id"].nunique()
        row["n_rows"] = int(group_model.nobs)
        row["used_mixedlm"] = getattr(group_model, "_used_mixedlm", None)

    age_rows = model_results_to_row(age_model, "group_x_age_x_region", "exponent")
    for row in age_rows:
        row["formula"] = age_formula
        row["n_subjects"] = long_df["subject_id"].nunique()
        row["n_rows"] = int(age_model.nobs)
        row["used_mixedlm"] = getattr(age_model, "_used_mixedlm", None)

    stratified_age_rows: list[dict[str, Any]] = []
    for region in ["posterior", "nonposterior"]:
        sub = long_df[long_df["region_type"] == region].copy()
        f = f"exponent ~ {GROUP_REF} * age_months + {COVARIATES_EXPONENT}"
        m = run_ols(f, sub)
        for row in model_results_to_row(m, f"age_group_{region}", "exponent"):
            row["region_type"] = region
            row["formula"] = f
            row["n_subjects"] = sub["subject_id"].nunique()
            stratified_age_rows.append(row)
    age_df = pd.concat([pd.DataFrame(age_rows), pd.DataFrame(stratified_age_rows)], ignore_index=True)

    participants = load_participants(Path(cfg["paths"]["participants_file"]), included_only=True)
    ados_map = resolve_ados_columns(participants)
    cohort = load_clinical_posterior_cohort(cfg, deriv)
    asd = cohort[cohort["group"].astype(str).str.upper() == "ASD"].copy()
    missing_ados_cols = [c for c in ados_map.values() if c is not None and c not in asd.columns]
    if missing_ados_cols:
        asd = asd.merge(
            participants[["subject_id"] + missing_ados_cols].drop_duplicates("subject_id"),
            on="subject_id",
            how="left",
        )

    ch_df = pd.read_csv(ch_path)
    ch_df["subject_id"] = ch_df["subject_id"].astype(str)
    if "fit_valid" in ch_df.columns:
        ch_df = ch_df[ch_df["fit_valid"]].copy()
    nonpost = _non_posterior_channel_mean(ch_df, "aperiodic_exponent")
    nonpost = nonpost.rename(columns={"aperiodic_exponent": "nonposterior_exponent"})
    asd = asd.merge(nonpost[["subject_id", "nonposterior_exponent"]], on="subject_id", how="left")

    clinical_rows: list[dict[str, Any]] = []
    for region_col, region_label in [
        ("posterior_exponent", "posterior"),
        ("nonposterior_exponent", "nonposterior"),
    ]:
        for col, clabel in _clinical_outcome_list(ados_map):
            row = _partial_corr_row(
                asd, col, region_col,
                analysis="clinical_specificity_asd",
                eeg_metric=region_col,
                clinical_label=clabel,
                segment=None,
                n_boot=n_boot,
                seed=seed,
            )
            row["region_type"] = region_label
            clinical_rows.append(row)
    clinical_df = pd.DataFrame(clinical_rows)

    roi_ref_df = _anatomical_roi_reference(cfg, deriv)

    subj = long_df.drop_duplicates("subject_id")
    n_asd = int((subj["group"] == "ASD").sum())
    n_td = int((subj["group"] == "TD").sum())

    return {
        "group_df": pd.DataFrame(group_rows),
        "age_df": age_df,
        "clinical_df": clinical_df,
        "roi_ref_df": roi_ref_df,
        "long_df": long_df,
        "n_subjects": int(long_df["subject_id"].nunique()),
        "n_asd": n_asd,
        "n_td": n_td,
        "n_long_rows": len(long_df),
        "ch_path": str(ch_path),
        "missing_ados": [k for k, v in ados_map.items() if v is None],
    }


def _anatomical_roi_reference(cfg: dict[str, Any], deriv: Path) -> pd.DataFrame:
    """Optional supplementary: marginal TD-ASD effect per anatomical ROI."""
    roi_long_path = deriv / "roi" / "specparam_subject_roi_long.csv"
    if not roi_long_path.exists():
        return pd.DataFrame()
    roi_long = pd.read_csv(roi_long_path)
    roi_long["subject_id"] = roi_long["subject_id"].astype(str)
    cohort = load_clinical_posterior_cohort(cfg, deriv)
    demo_cols = ["subject_id", "group", "age_months", "sex", "IQ_total", "usable_epochs"]
    if "mean_r_squared" in cohort.columns:
        demo_cols.append("mean_r_squared")
    df = roi_long.merge(cohort[demo_cols].drop_duplicates("subject_id"), on=["subject_id", "group"], how="inner")
    df = df.dropna(subset=["exponent", "age_months", "sex", "IQ_total", "usable_epochs"])
    if "mean_r_squared" in df.columns:
        df = df.dropna(subset=["mean_r_squared"])

    rows: list[dict[str, Any]] = []
    for roi in sorted(df["roi"].dropna().unique()):
        sub = df[df["roi"] == roi].copy()
        f = f"exponent ~ {GROUP_REF} + {COVARIATES_EXPONENT}"
        m = run_ols(f, sub)
        for row in model_results_to_row(m, "anatomical_roi_reference", "exponent"):
            row["roi"] = roi
            row["formula"] = f
            row["n_subjects"] = sub["subject_id"].nunique()
            rows.append(row)
    return pd.DataFrame(rows)


def _find_interaction_term(params_index: pd.Index, pattern: str = "posterior_exponent") -> str | None:
    for term in params_index.astype(str):
        if pattern in term and ":C(group" in term:
            return term
    return None


def bootstrap_interaction(
    df: pd.DataFrame,
    formula: str,
    interaction_term: str,
    n_resamples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap CI for OLS interaction term (from script 34)."""
    rng = np.random.default_rng(random_seed)
    n = len(df)
    betas: list[float] = []
    for _ in range(n_resamples):
        sample_idx = rng.integers(0, n, size=n)
        sample = df.iloc[sample_idx].copy()
        if sample["group"].nunique() < 2:
            betas.append(np.nan)
            continue
        try:
            fit = smf.ols(formula, data=sample).fit()
            beta = fit.params.get(interaction_term, np.nan)
            betas.append(float(beta) if pd.notna(beta) else np.nan)
        except Exception:
            betas.append(np.nan)
    dist = pd.DataFrame({"resample_id": np.arange(n_resamples), "beta_interaction": betas})
    valid = dist["beta_interaction"].dropna().to_numpy()
    if len(valid) == 0:
        summary = pd.DataFrame([{
            "n_resamples": n_resamples, "n_valid": 0,
            "beta_mean": np.nan, "beta_median": np.nan,
            "ci95_low": np.nan, "ci95_high": np.nan,
            "p_bootstrap_two_sided": np.nan,
        }])
        return summary, dist
    ci_low, ci_high = np.quantile(valid, [0.025, 0.975])
    p_boot = 2.0 * min(np.mean(valid <= 0), np.mean(valid >= 0))
    summary = pd.DataFrame([{
        "n_resamples": n_resamples,
        "n_valid": int(len(valid)),
        "beta_mean": float(np.mean(valid)),
        "beta_median": float(np.median(valid)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_bootstrap_two_sided": float(min(1.0, p_boot)),
    }])
    return summary, dist


def extract_simple_slopes(result: Any, interaction_term: str | None, x_term: str = "posterior_exponent") -> pd.DataFrame:
    """ASD and TD simple slopes for rest-movie coupling."""
    rows = []
    candidates = [t for t in result.params.index if x_term in str(t) and ":" not in str(t)]
    asd_term = candidates[0] if candidates else x_term
    if asd_term in result.params.index:
        rows.append({
            "group_slope": "ASD",
            "beta": float(result.params[asd_term]),
            "p_value": float(result.pvalues.get(asd_term, np.nan)),
            "contrast": asd_term,
        })
    if interaction_term and asd_term in result.params.index:
        try:
            test = result.t_test(f"{asd_term} + {interaction_term} = 0")
            beta_td = float(result.params[asd_term] + result.params[interaction_term])
            p_td = float(np.asarray(test.pvalue).reshape(-1)[0])
        except Exception:
            beta_td = float(result.params[asd_term] + result.params.get(interaction_term, 0.0))
            p_td = np.nan
        rows.append({
            "group_slope": "TD",
            "beta": beta_td,
            "p_value": p_td,
            "contrast": f"{asd_term} + {interaction_term}",
        })
    return pd.DataFrame(rows)


def build_rest_movie_coupling_cohort(
    cfg: dict[str, Any],
    movie_cfg: dict[str, Any] | None,
    *,
    subject_filter: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge resting posterior exponent with three-segment movie ISC."""
    meta: dict[str, Any] = {"paths_used": {}, "missing": [], "isc_source": ""}

    rest_path = PROJECT_ROOT / "outputs" / "tables" / "resting_features_locked.csv"
    if not rest_path.exists():
        raise FileNotFoundError(f"Missing resting features: {rest_path}")
    meta["paths_used"]["resting"] = str(rest_path)

    isc_path, isc_source = resolve_movie_isc_path(PROJECT_ROOT)
    if isc_path is None:
        meta["missing"].append("movie ISC file")
        return pd.DataFrame(), meta
    meta["paths_used"]["movie_isc"] = str(isc_path)
    meta["isc_source"] = isc_source

    rest = pd.read_csv(rest_path)[["subject_id", "group", "posterior_exponent"]].copy()
    rest["subject_id"] = rest["subject_id"].astype(str)
    rest["group"] = rest["group"].astype(str).str.upper()

    isc = pd.read_csv(isc_path)
    isc["subject_id"] = isc["subject_id"].astype(str)
    isc["group"] = isc["group"].astype(str).str.upper()
    isc["event_type"] = isc["event_type"].astype(str).str.lower()

    wide = isc.pivot_table(
        index=["subject_id", "group"],
        columns="event_type",
        values=["isc_z", "isc_r"],
        aggfunc="first",
    )
    wide.columns = [f"{val}_{seg}" for val, seg in wide.columns]
    wide = wide.reset_index()

    movie_deriv = Path(movie_cfg["paths"]["derivatives_root"]) if movie_cfg else None
    part_path = Path(movie_cfg["paths"]["participants_file"]) if movie_cfg else None
    part = pd.read_csv(part_path) if part_path is not None and part_path.exists() else pd.DataFrame()
    if part.empty:
        part = load_participants(Path(cfg["paths"]["participants_file"]), included_only=True)
        meta["paths_used"]["participants_fallback"] = str(cfg["paths"]["participants_file"])
        meta["missing"].append("movie participants file (using resting participants.csv)")

    part["subject_id"] = part["subject_id"].astype(str)
    part["group"] = part["group"].astype(str).str.upper()
    if "included_final" in part.columns:
        part = part[part["included_final"].astype(int) == 1]
    keep = ["subject_id", "group", "age_months"]
    for c in ("IQ_total", "sex"):
        if c in part.columns:
            keep.append(c)
    merged = wide.merge(part[keep], on=["subject_id", "group"], how="inner")

    preproc_path = (movie_deriv / "qc" / "preproc_summary.csv") if movie_deriv else None
    if preproc_path is not None and preproc_path.exists():
        preproc = pd.read_csv(preproc_path)[["subject_id", "usable_epochs"]]
        preproc["subject_id"] = preproc["subject_id"].astype(str)
        merged = merged.merge(preproc, on="subject_id", how="left")
        meta["paths_used"]["preproc"] = str(preproc_path)
    else:
        resting_deriv = Path(cfg["paths"]["derivatives_root"])
        preproc_rest = resting_deriv / "qc" / "preproc_summary.csv"
        if preproc_rest.exists():
            preproc = pd.read_csv(preproc_rest)[["subject_id", "usable_epochs"]]
            preproc["subject_id"] = preproc["subject_id"].astype(str)
            merged = merged.merge(preproc, on="subject_id", how="left")
            meta["paths_used"]["preproc_fallback"] = str(preproc_rest)
            meta["missing"].append("movie preproc_summary (using resting usable_epochs)")
        else:
            meta["missing"].append("usable_epochs (movie and resting preproc unavailable)")

    if movie_deriv is not None:
        qc_path = movie_deriv / "specparam" / "specparam_qc_summary_subject.csv"
        if qc_path.exists():
            qc = pd.read_csv(qc_path)
            qc["subject_id"] = qc["subject_id"].astype(str)
            bad = set(qc.loc[pd.to_numeric(qc["low_quality_subject"], errors="coerce") == 1, "subject_id"])
            merged = merged[~merged["subject_id"].isin(bad)]
            if "mean_r_squared" in qc.columns:
                merged = merged.merge(qc[["subject_id", "mean_r_squared"]], on="subject_id", how="left")
            meta["paths_used"]["movie_qc"] = str(qc_path)

        analysis_path = movie_deriv / "participants_analysis.csv"
        if analysis_path.exists():
            ma = pd.read_csv(analysis_path)
            ma["subject_id"] = ma["subject_id"].astype(str)
            ma["group"] = ma["group"].astype(str).str.upper()
            merged = merged.merge(ma[["subject_id", "group"]].drop_duplicates(), on=["subject_id", "group"], how="inner")
            meta["paths_used"]["movie_analysis"] = str(analysis_path)

    merged = merged.merge(rest, on=["subject_id", "group"], how="inner")

    if "n_overlap_points" in isc.columns:
        overlap = isc.pivot_table(
            index=["subject_id", "group"], columns="event_type", values="n_overlap_points", aggfunc="first",
        )
        overlap.columns = [f"n_overlap_{c}" for c in overlap.columns]
        merged = merged.merge(overlap.reset_index(), on=["subject_id", "group"], how="left")

    if subject_filter is not None:
        merged = merged[merged["subject_id"].isin(subject_filter)].copy()

    for c in ["age_months", "IQ_total", "posterior_exponent", "usable_epochs"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
    if "sex" in merged.columns:
        merged["sex"] = merged["sex"].astype(str).str.upper().replace({"NAN": np.nan, "": np.nan})

    return merged.reset_index(drop=True), meta


def run_analysis3_rest_movie_coupling(
    cfg: dict[str, Any],
    movie_cfg: dict[str, Any] | None,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Analysis 3: conservative rest-to-movie cross-context coupling."""
    isc_path, isc_source = resolve_movie_isc_path(PROJECT_ROOT)
    if isc_path is None:
        return {
            "full_df": pd.DataFrame(),
            "group_specific_df": pd.DataFrame(),
            "bootstrap_df": pd.DataFrame(),
            "skipped_reason": "No movie ISC file found at expected paths.",
            "n_total": 0,
            "n_asd": 0,
            "n_td": 0,
            "cohorts": {},
            "isc_source": "",
        }

    matched_path = PROJECT_ROOT / "data" / "participants" / "participants_resting_movie_matched_postqc.csv"
    matched_ids: set[str] | None = None
    if matched_path.exists():
        mp = pd.read_csv(matched_path)
        matched_ids = set(mp["subject_id"].astype(str))

    main_df, meta = build_rest_movie_coupling_cohort(cfg, movie_cfg)
    if main_df.empty:
        return {
            "full_df": pd.DataFrame(),
            "group_specific_df": pd.DataFrame(),
            "bootstrap_df": pd.DataFrame(),
            "skipped_reason": "Movie ISC or merge cohort unavailable.",
            "meta": meta,
            "n_total": 0,
            "n_asd": 0,
            "n_td": 0,
            "cohorts": {},
        }

    cohorts: dict[str, pd.DataFrame] = {"main_overlap": main_df}
    if matched_ids:
        sens_df, _ = build_rest_movie_coupling_cohort(cfg, movie_cfg, subject_filter=matched_ids)
        if not sens_df.empty:
            cohorts["dual_paradigm_matched_postqc"] = sens_df

    full_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    interaction_ps: list[tuple[str, str, float]] = []

    for cohort_label, cdf in cohorts.items():
        for seg in SEGMENTS:
            y_z = f"isc_z_{seg}"
            y_r = f"isc_r_{seg}"
            if y_z not in cdf.columns:
                continue

            cov_numeric = ["age_months", "IQ_total"]
            if "usable_epochs" in cdf.columns and cdf["usable_epochs"].notna().any():
                cov_numeric.append("usable_epochs")
            if "mean_r_squared" in cdf.columns and cdf["mean_r_squared"].notna().any():
                cov_numeric.append("mean_r_squared")
            cov_parts = [f"posterior_exponent * {GROUP_REF}", *cov_numeric]
            if "sex" in cdf.columns:
                cov_parts.append("C(sex)")
            formula = f"{y_z} ~ " + " + ".join(cov_parts)

            need = ["posterior_exponent", "group", "age_months", "IQ_total", y_z]
            if "usable_epochs" in cov_numeric:
                need.append("usable_epochs")
            if "sex" in cdf.columns:
                need.append("sex")
            if "mean_r_squared" in cov_numeric:
                need.append("mean_r_squared")
            mdf = cdf.dropna(subset=[c for c in need if c in cdf.columns]).copy()
            if len(mdf) < 15:
                continue

            model = run_ols(formula, mdf)
            inter = _find_interaction_term(model.params.index, "posterior_exponent")
            outcome_label = _isc_metric_label(meta.get("isc_source", isc_source))
            for row in model_results_to_row(model, f"full_{cohort_label}_{seg}", y_z):
                row["cohort"] = cohort_label
                row["segment"] = seg
                row["outcome_metric"] = outcome_label
                row["isc_source"] = meta.get("isc_source", isc_source)
                row["formula"] = formula
                row["interaction_term"] = inter or ""
                full_rows.append(row)
            if inter:
                interaction_ps.append((cohort_label, seg, float(model.pvalues.get(inter, np.nan))))

            if inter:
                boot_sum, _ = bootstrap_interaction(mdf, formula, inter, n_boot, seed)
                boot_sum["cohort"] = cohort_label
                boot_sum["segment"] = seg
                boot_sum["interaction_term"] = inter
                boot_rows.append(boot_sum)

            for grp in ["ASD", "TD"]:
                gdf = mdf[mdf["group"] == grp].copy()
                g_formula = f"{y_z} ~ posterior_exponent + age_months + IQ_total"
                if "sex" in gdf.columns:
                    g_formula += " + C(sex)"
                g_need = ["posterior_exponent", "age_months", "IQ_total", y_z]
                if "sex" in gdf.columns:
                    g_need.append("sex")
                gdf = gdf.dropna(subset=[c for c in g_need if c in gdf.columns])
                if len(gdf) < 10:
                    continue
                gm = run_ols(g_formula, gdf)
                for row in model_results_to_row(gm, f"group_specific_{grp}", y_z):
                    if row["term"] == "posterior_exponent":
                        row["cohort"] = cohort_label
                        row["segment"] = seg
                        row["group"] = grp
                        row["formula"] = g_formula
                        row["exploratory"] = True
                        group_rows.append(row)

            if y_r in cdf.columns:
                r_formula = f"{y_r} ~ " + " + ".join(cov_parts)
                r_need = [c if c != y_z else y_r for c in need]
                rdf = cdf.dropna(subset=[c for c in r_need if c in cdf.columns]).copy()
                if len(rdf) >= 15:
                    rm = run_ols(r_formula, rdf)
                    for row in model_results_to_row(rm, f"full_pearson_r_{cohort_label}_{seg}", y_r):
                        row["cohort"] = cohort_label
                        row["segment"] = seg
                        row["outcome_metric"] = "isc_r (Pearson r)"
                        row["formula"] = r_formula
                        full_rows.append(row)

    full_df = pd.DataFrame(full_rows)
    if interaction_ps:
        main_inter = [(s, p) for c, s, p in interaction_ps if c == "main_overlap"]
        if main_inter:
            segs, ps = zip(*main_inter)
            _, qs = fdr_correction(np.array(ps))
            fdr_map = dict(zip(segs, qs))
            for i, row in full_df.iterrows():
                if row.get("cohort") == "main_overlap" and row.get("term", "").find("posterior_exponent") >= 0 and ":C(group" in str(row.get("term", "")):
                    seg = row.get("segment")
                    if seg in fdr_map:
                        full_df.at[i, "interaction_fdr_q"] = fdr_map[seg]

    boot_df = pd.concat(boot_rows, ignore_index=True) if boot_rows else pd.DataFrame()
    group_df = pd.DataFrame(group_rows)

    subj = main_df.drop_duplicates("subject_id")
    return {
        "full_df": full_df,
        "group_specific_df": group_df,
        "bootstrap_df": boot_df,
        "meta": meta,
        "skipped_reason": "",
        "n_total": len(subj),
        "n_asd": int((subj["group"] == "ASD").sum()),
        "n_td": int((subj["group"] == "TD").sum()),
        "cohorts": {k: len(v.drop_duplicates("subject_id")) for k, v in cohorts.items()},
        "matched_path": str(matched_path) if matched_path.exists() else None,
        "isc_source": meta.get("isc_source", isc_source),
    }


def _one_line_summary(key: str, results: dict[str, Any]) -> str:
    a1 = results.get("analysis1", {})
    a2 = results.get("analysis2", {})
    a3 = results.get("analysis3", {})

    if key == "analysis1_resting":
        df = a1.get("resting_df", pd.DataFrame())
        if df.empty:
            return "Resting ADOS subdomain analysis: no complete ASD rows."
        sa = df[df["clinical_outcome"].str.contains("SA", na=False)]
        if not sa.empty:
            r = sa.iloc[0]
            return (
                f"Resting posterior exponent vs ADOS (ASD n={int(r['n'])}): "
                f"strongest signal for {r['clinical_label']} (partial r={float(r['partial_r']):.2f}, "
                f"raw p={_format_p(float(r['raw_p']))})."
            )
        return f"Resting ADOS subdomain: {len(df)} tests completed (ASD n≈{a1.get('n_asd_resting', '?')})."

    if key == "analysis1_movie":
        if a1.get("movie_skipped_reason"):
            return f"Movie ISC × ADOS: skipped ({a1['movie_skipped_reason']})."
        df = a1.get("movie_df", pd.DataFrame())
        return f"Movie Aperiodic-ISC × ADOS: {len(df)} partial-correlation tests (ASD n={a1.get('n_asd_movie', 0)})."

    if key == "analysis2":
        gdf = a2.get("group_df", pd.DataFrame())
        if gdf.empty:
            return "Posterior vs non-posterior: models not fitted."
        inter = gdf[gdf["term"].astype(str).str.contains("group", case=False) & gdf["term"].astype(str).str.contains("region", case=False)]
        if not inter.empty:
            row = inter.iloc[0]
            return (
                f"Posterior vs non-posterior group×region interaction: β={float(row['coef']):.3f}, "
                f"p={_format_p(float(row['pvalue']))} (n={a2.get('n_subjects', '?')} subjects)."
            )
        return f"Posterior vs non-posterior specificity fitted on n={a2.get('n_subjects', '?')}."

    if key == "analysis3":
        if a3.get("skipped_reason"):
            return f"Rest-movie coupling: skipped ({a3['skipped_reason']})."
        return (
            f"Rest-movie coupling main overlap n={a3.get('n_total', 0)} "
            f"(ASD={a3.get('n_asd', 0)}, TD={a3.get('n_td', 0)}); "
            f"{len(a3.get('full_df', pd.DataFrame()))} model terms saved."
        )
    return ""


def generate_coordination_feature_report(results: dict[str, Any]) -> str:
    """Build markdown report with Methods/Results/Interpretation snippets."""
    a1 = results["analysis1"]
    a2 = results["analysis2"]
    a3 = results["analysis3"]
    lines: list[str] = []

    lines += [
        "# Coordination Feature Supplementary Analyses",
        "",
        "## 1. Overview",
        "",
        "Exploratory supplementary analyses testing whether posterior aperiodic activity",
        "indexes a **cross-context cortical-state coordination feature** rather than a diffuse",
        "spectral shift or a single-paradigm artifact.",
        "",
        "**Direction convention:** group effects are coded TD − ASD (positive β: TD > ASD exponent);",
        "within ASD, **lower exponent is interpreted as associated with greater ADOS severity**",
        "(negative partial r expected). These analyses are **not** confirmatory mediation or causal pathway tests.",
        "",
        "中文备注：本节为补充/探索性分析，用于支持 posterior aperiodic 作为跨情境皮层状态协调特征的叙述，",
        "不应写成确证性中介或因果路径。",
        "",
        "## 2. Data sources and sample sizes",
        "",
        "| Analysis | Sample | Key paths |",
        "|---|---|---|",
        f"| 1A Resting ADOS | ASD n={a1.get('n_asd_resting', 'NA')} | `{a1.get('participants_path', '')}`, `{a1.get('cohort_source', '')}` |",
        f"| 1B Movie ISC × ADOS | ASD n={a1.get('n_asd_movie', 0)} | `{a1.get('movie_isc_path') or 'MISSING'}` ({a1.get('movie_isc_source') or 'n/a'}) |",
        f"| 2 Posterior vs non-posterior | n={a2.get('n_subjects', 'NA')} ({a2.get('n_asd', '?')} ASD / {a2.get('n_td', '?')} TD) | `{a2.get('ch_path', '')}` |",
        f"| 3 Rest-movie coupling | n={a3.get('n_total', 0)} ({a3.get('n_asd', 0)} ASD / {a3.get('n_td', 0)} TD) | see Analysis 3 |",
        "",
    ]

    missing_all: list[str] = []
    if a1.get("missing_ados"):
        missing_all.extend([f"ADOS canonical: {m}" for m in a1["missing_ados"]])
    if a1.get("movie_skipped_reason"):
        missing_all.append(f"Movie ISC: {a1['movie_skipped_reason']}")
    if a3.get("skipped_reason"):
        missing_all.append(f"Rest-movie: {a3['skipped_reason']}")
    missing_all.append("Gaze proportion: unavailable in project movie QC tables")
    meta3 = a3.get("meta", {})
    for m in meta3.get("missing", []):
        missing_all.append(m)

    lines += [
        "### Missing variables & alternatives",
        "",
    ]
    if a1.get("alternatives"):
        lines.append(f"- Available clinical alternatives: {', '.join(a1['alternatives'])}")
    for m in missing_all:
        lines.append(f"- {m}")
    if not missing_all and not a1.get("alternatives"):
        lines.append("- None")

    lines += _report_analysis1(a1)
    lines += _report_analysis2(a2)
    lines += _report_analysis3(a3)
    lines += _report_manuscript_text(a1, a2, a3)
    lines += _report_limitations(a1, a3)

    return "\n".join(lines) + "\n"


def _report_analysis1(a1: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## 3. Analysis 1: ADOS subdomain specificity",
        "",
        "### Methods snippet (EN)",
        "",
        "Within autistic children, we examined whether posterior aperiodic measures related",
        "preferentially to ADOS total, Social Affect, or Restricted and Repetitive Behavior severity.",
        "Partial correlations controlled for age and IQ, with bootstrap partial Spearman analyses",
        "(1,000 resamples, 95% CI) as robustness checks. Resting and movie ISC families were",
        "FDR-corrected separately. These analyses were interpreted as exploratory tests of clinical",
        "specificity rather than confirmatory evidence for symptom-domain selectivity.",
        "",
        "### Results snippet (EN)",
        "",
    ]
    rdf = a1.get("resting_df", pd.DataFrame())
    if not rdf.empty:
        lines.append("**Resting posterior exponent (ASD only):**")
        lines.append("")
        lines.append("| Outcome | n | Partial r | Raw p | FDR q | Partial ρ | Boot 95% CI |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for _, row in rdf.iterrows():
            ci = f"[{row['boot_ci_low']:.3f}, {row['boot_ci_high']:.3f}]" if np.isfinite(row.get("boot_ci_low", np.nan)) else "—"
            lines.append(
                f"| {row['clinical_label']} | {int(row['n'])} | {float(row['partial_r']):.2f} | "
                f"{_format_p(float(row['raw_p']))} | {_format_p(float(row.get('fdr_q', np.nan)))} | "
                f"{float(row['partial_rho']):.2f} | {ci} |"
            )
    mdf = a1.get("movie_df", pd.DataFrame())
    if not mdf.empty:
        lines += ["", "**Movie TD-template Aperiodic-ISC (ASD only; Fisher z):**", ""]
        lines.append("| Segment | Outcome | n | Partial r | Raw p | FDR q |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for _, row in mdf.iterrows():
            lines.append(
                f"| {row['segment']} | {row['clinical_label']} | {int(row['n'])} | "
                f"{float(row['partial_r']):.2f} | {_format_p(float(row['raw_p']))} | "
                f"{_format_p(float(row.get('fdr_q', np.nan)))} |"
            )
        if a1.get("movie_isc_source") == "posterior_sliding_exploratory":
            lines.append("")
            lines.append(
                "*Note: movie ISC uses posterior sliding-window exploratory fallback "
                "(not global TD-template Aperiodic-ISC from `derivatives_task_movie/`).*"
            )
    elif a1.get("movie_skipped_reason"):
        lines.append(f"*Movie ISC × ADOS not run: {a1['movie_skipped_reason']}*")

    lines += [
        "",
        "### Interpretation snippet (EN)",
        "",
        "Exploratory subdomain comparisons ask whether posterior coordination tracks social-affect",
        "versus repetitive-behavior severity beyond total ADOS. Preferential SA association would",
        "support social-cortical-state coupling; comparable RRB associations would suggest broader",
        "symptom linkage. Null subdomain differences do not disprove the main clinical association.",
        "",
        "中文备注：若 SA 与 total 均显著而 RRB 较弱，可谨慎表述为 social-affect 特异性倾向，",
        "但不应过度解读为症状域选择性的确证证据。",
        "",
    ]
    return lines


def _report_analysis2(a2: dict[str, Any]) -> list[str]:
    lines = [
        "## 4. Analysis 2: Posterior vs non-posterior specificity",
        "",
        "### Methods snippet (EN)",
        "",
        "To test whether the posterior effect reflected a spatially specific cortical-state feature",
        "rather than a diffuse aperiodic shift, we compared the predefined posterior ROI (E33, E36,",
        "E37, E38) with the average of remaining non-posterior channels using identical",
        "covariate-adjusted models. Evidence for stronger group, developmental, or clinical effects",
        "in the posterior ROI would support regional specificity; nonsignificant interaction terms",
        "should be interpreted cautiously as indicating that posterior specificity remains partly descriptive.",
        "",
        "### Results snippet (EN)",
        "",
    ]
    gdf = a2.get("group_df", pd.DataFrame())
    if not gdf.empty:
        highlight = gdf[gdf["term"].astype(str).str.contains("group", case=False)]
        lines.append("**Group × region_type interaction model:**")
        lines.append("")
        lines.append("| Term | β (TD−ASD direction for group main) | p | n subjects |")
        lines.append("|---|---:|---:|---:|")
        for _, row in highlight.head(8).iterrows():
            lines.append(
                f"| {row['term']} | {float(row['coef']):.4f} | {_format_p(float(row['pvalue']))} | {int(row.get('n_subjects', 0))} |"
            )
    cdf = a2.get("clinical_df", pd.DataFrame())
    if not cdf.empty:
        lines += ["", "**ASD clinical partial correlations by region:**", ""]
        lines.append("| Region | Outcome | n | Partial r | Raw p |")
        lines.append("|---|---|---:|---:|---:|")
        for _, row in cdf.iterrows():
            lines.append(
                f"| {row['region_type']} | {row['clinical_label']} | {int(row['n'])} | "
                f"{float(row['partial_r']):.2f} | {_format_p(float(row['raw_p']))} |"
            )
    lines += [
        "",
        "### Interpretation snippet (EN)",
        "",
        "A significant group×posterior interaction would indicate that TD−ASD exponent differences",
        "are amplified in the predefined posterior ROI relative to the remaining scalp average.",
        "Parallel ASD clinical associations in posterior but not non-posterior regions would further",
        "support regional cortical-state relevance. Absence of interaction does not negate the",
        "primary posterior finding but limits strong spatial-specificity claims.",
        "",
        "中文备注：交互不显著时，应强调 posterior 效应仍可能部分为描述性区域聚焦，而非全脑弥漫差异。",
        "",
    ]
    return lines


def _report_analysis3(a3: dict[str, Any]) -> list[str]:
    lines = [
        "## 5. Analysis 3: Rest-to-movie cross-context coupling",
        "",
        "### Methods snippet (EN)",
        "",
        "We revisited rest-to-movie coupling using conservative covariate-adjusted models in the",
        "overlapping resting-plus-movie cohort. For each movie segment (mentalizing, pain, neutral),",
        "TD-template Aperiodic-ISC (Fisher z) was regressed on resting posterior exponent, diagnostic",
        "group, age, IQ, sex, and data-quality covariates where available, including a",
        "resting-exponent×group interaction. Interaction p values across segments were FDR-corrected.",
        "ASD-only and TD-only slopes were exploratory. Bootstrap (1,000 resamples) supported interaction inference.",
        "",
        "### Results snippet (EN)",
        "",
    ]
    if a3.get("skipped_reason"):
        lines.append(f"*{a3['skipped_reason']}*")
    else:
        fdf = a3.get("full_df", pd.DataFrame())
        if not fdf.empty:
            inter = fdf[fdf["term"].astype(str).str.contains("posterior_exponent") & fdf["term"].astype(str).str.contains("group")]
            if not inter.empty:
                lines.append("| Cohort | Segment | Interaction term | β | p | FDR q |")
                lines.append("|---|---|---|---:|---:|---:|")
                for _, row in inter.iterrows():
                    lines.append(
                        f"| {row.get('cohort', '')} | {row.get('segment', '')} | {row['term']} | "
                        f"{float(row['coef']):.4f} | {_format_p(float(row['pvalue']))} | "
                        f"{_format_p(float(row.get('interaction_fdr_q', np.nan)))} |"
                    )
        cohorts = a3.get("cohorts", {})
        if cohorts:
            lines.append("")
            lines.append("Cohort sizes: " + "; ".join(f"{k} n={v}" for k, v in cohorts.items()))
    lines += [
        "",
        "### Interpretation snippet (EN)",
        "",
        "Because this analysis depends on the smaller dual-paradigm sample and tests cross-context",
        "coupling rather than the primary group effect, findings were interpreted as exploratory.",
        "Sensitivity-dependent or nonsignificant results suggest that resting posterior exponent and",
        "naturalistic Aperiodic-ISC index related but partially dissociable levels of posterior",
        "cortical-state coordination.",
        "",
        "中文备注：rest 与 movie 指标相关但不完全一致，符合“协调特征的多层次表达”框架。",
        "",
    ]
    return lines


def _report_manuscript_text(a1: dict[str, Any], a2: dict[str, Any], a3: dict[str, Any]) -> list[str]:
    return [
        "## 6. Suggested manuscript text",
        "",
        "Within autistic children, we examined whether posterior aperiodic measures related",
        "preferentially to ADOS total, Social Affect, or Restricted and Repetitive Behavior severity.",
        "Partial correlations controlled for age and IQ, with bootstrap partial Spearman analyses used",
        "as robustness checks. These analyses were interpreted as exploratory tests of clinical specificity",
        "rather than confirmatory evidence for symptom-domain selectivity.",
        "",
        "To test whether the posterior effect reflected a spatially specific cortical-state feature",
        "rather than a diffuse aperiodic shift, we compared the predefined posterior ROI with the average",
        "of remaining non-posterior channels using identical covariate-adjusted models. Evidence for stronger",
        "group, developmental, or clinical effects in the posterior ROI would support regional specificity;",
        "nonsignificant interaction terms should be interpreted cautiously as indicating that posterior",
        "specificity remains partly descriptive.",
        "",
        "We revisited rest-to-movie coupling using conservative covariate-adjusted models in the",
        "overlapping resting-plus-movie cohort. These models tested whether resting posterior exponent",
        "predicted TD-template Aperiodic-ISC and whether this association differed by diagnostic group.",
        "Because the analysis depends on the smaller dual-paradigm sample and tests cross-context coupling",
        "rather than the primary group effect, findings were interpreted as exploratory. Sensitivity-dependent",
        "or nonsignificant results would suggest that resting posterior exponent and naturalistic Aperiodic-ISC",
        "index related but partially dissociable levels of posterior cortical-state coordination.",
        "",
    ]


def _report_limitations(a1: dict[str, Any], a3: dict[str, Any]) -> list[str]:
    return [
        "## 7. Limitations / interpretation boundaries",
        "",
        "- Exploratory supplementary scope; not preregistered confirmatory tests.",
        "- Multiple comparisons controlled within analysis families (FDR), not globally across all supplementary tests.",
        "- Movie analyses require locally generated `derivatives_task_movie/` outputs; absent files are reported, not imputed.",
        "- Gaze/behavioral movie QC covariates are unavailable in current project tables.",
        "- Cross-context coupling does not imply causal mediation between resting state and naturalistic synchrony.",
        "- Lower exponent ↔ greater severity direction is correlational within ASD only.",
        "",
    ]


def run_all_coordination_analyses(
    cfg: dict[str, Any],
    movie_cfg: dict[str, Any] | None = None,
    *,
    n_boot: int = 1000,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run all three analyses, save CSVs, write report, return results bundle."""
    seed = seed if seed is not None else int(cfg.get("project", {}).get("random_seed", 42))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Running Analysis 1: ADOS subdomain specificity")
    a1 = run_analysis1_ados_subdomain(cfg, movie_cfg, n_boot=n_boot, seed=seed)
    save_csv(a1["resting_df"], OUT_DIR / "ados_subdomain_posterior_exponent.csv")
    if not a1["movie_df"].empty:
        save_csv(a1["movie_df"], OUT_DIR / "ados_subdomain_movie_isc.csv")

    logger.info("Running Analysis 2: Posterior vs non-posterior specificity")
    a2 = run_analysis2_posterior_nonposterior(cfg, n_boot=n_boot, seed=seed)
    save_csv(a2["group_df"], OUT_DIR / "posterior_nonposterior_group_models.csv")
    save_csv(a2["age_df"], OUT_DIR / "posterior_nonposterior_age_models.csv")
    save_csv(a2["clinical_df"], OUT_DIR / "posterior_nonposterior_clinical_models.csv")
    if not a2["roi_ref_df"].empty:
        save_csv(a2["roi_ref_df"], OUT_DIR / "posterior_nonposterior_anatomical_roi_reference.csv")

    logger.info("Running Analysis 3: Rest-to-movie coupling")
    a3 = run_analysis3_rest_movie_coupling(cfg, movie_cfg, n_boot=n_boot, seed=seed)
    if not a3["full_df"].empty:
        save_csv(a3["full_df"], OUT_DIR / "rest_movie_coupling_full_models.csv")
    if not a3["group_specific_df"].empty:
        save_csv(a3["group_specific_df"], OUT_DIR / "rest_movie_coupling_group_specific.csv")
    if not a3["bootstrap_df"].empty:
        save_csv(a3["bootstrap_df"], OUT_DIR / "rest_movie_coupling_bootstrap.csv")

    results = {
        "analysis1": a1,
        "analysis2": a2,
        "analysis3": a3,
        "out_dir": str(OUT_DIR),
        "report_path": str(REPORT_PATH),
        "summaries": {
            "analysis1_resting": _one_line_summary("analysis1_resting", {"analysis1": a1}),
            "analysis1_movie": _one_line_summary("analysis1_movie", {"analysis1": a1}),
            "analysis2": _one_line_summary("analysis2", {"analysis2": a2}),
            "analysis3": _one_line_summary("analysis3", {"analysis3": a3}),
        },
    }

    report_md = generate_coordination_feature_report(results)
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    results["report_md"] = report_md
    return results
