# Ideal release name: export_supplementary_source_data.py
# Original path: scripts/export_supplementary_source_data.py
# Note: Export supplementary source tables
# This file is a copy for the public github_release/ bundle.

"""Assemble Supplementary Figures S1–S9 source-data CSVs from existing pipeline outputs."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "figure_source_data" / "supplementary"
JR_SRC = PROJECT_ROOT / "jr_remote_bundle" / "src"
if str(JR_SRC) not in sys.path:
    sys.path.insert(0, str(JR_SRC))

POSTERIOR_CHANNELS = {"E33", "E36", "E37", "E38"}
SEG_MAP_TC = {"mental": "Mentalizing", "pain": "Pain-related", "neutral": "Neutral"}


def _safe_corr(x: np.ndarray, y: np.ndarray, min_points: int = 5) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < min_points:
        return np.nan
    xv, yv = x[mask], y[mask]
    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def build_movie_timecourse(ts_path: Path) -> pd.DataFrame:
    """Within-group LOO Aperiodic-ISC timecourse from sliding exponent trajectories."""
    ts = pd.read_csv(ts_path)
    val_col = "posterior_exponent" if "posterior_exponent" in ts.columns else "exponent_mean"
    ts["center_sec"] = (ts["window_start_sec"] + ts["window_end_sec"]) / 2.0
    ts["event_type"] = ts["event_type"].astype(str).str.lower()
    times = np.sort(ts["center_sec"].unique())
    half = 2
    rows = []
    for t_idx, center in enumerate(times):
        lo_t, hi_t = times[max(0, t_idx - half)], times[min(len(times) - 1, t_idx + half)]
        sub_t = ts[(ts["center_sec"] >= lo_t) & (ts["center_sec"] <= hi_t)]
        for grp in ("TD", "ASD"):
            g = sub_t[sub_t["group"] == grp]
            pivot = g.pivot_table(index="subject_id", columns="center_sec", values=val_col, aggfunc="mean")
            if pivot.shape[0] < 3 or pivot.shape[1] < 5:
                continue
            local_rs = []
            mat = pivot.to_numpy(dtype=float)
            t_local = int(np.argmin(np.abs(pivot.columns.to_numpy(dtype=float) - center)))
            lo = max(0, t_local - half)
            hi = min(mat.shape[1], t_local + half + 1)
            for i in range(mat.shape[0]):
                x = mat[i, lo:hi]
                others = np.delete(mat, i, axis=0)[:, lo:hi]
                if others.size == 0:
                    continue
                tmpl = np.nanmean(others, axis=0)
                r = _safe_corr(x, tmpl, min_points=5)
                if np.isfinite(r):
                    local_rs.append(r)
            if not local_rs:
                continue
            seg_row = ts.loc[np.isclose(ts["center_sec"], center), "event_type"]
            seg = SEG_MAP_TC.get(seg_row.iloc[0] if len(seg_row) else "neutral", "Neutral")
            mean_r = float(np.mean(local_rs))
            sem_r = float(np.std(local_rs, ddof=1) / np.sqrt(len(local_rs))) if len(local_rs) > 1 else 0.0
            rows.append(
                {
                    "time_sec": float(center),
                    "segment_label": seg,
                    "group": grp,
                    "mean_aperiodic_isc": mean_r,
                    "sem_aperiodic_isc": sem_r,
                }
            )
    return pd.DataFrame(rows)


SEGMENT_MAP = {
    "mental": "mentalizing",
    "mentalizing": "mentalizing",
    "pain": "pain",
    "neutral": "neutral",
}
MANIFEST_ROWS: list[dict[str, Any]] = []
GENERATION_DATE = date.today().isoformat()


def _p(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _save(df: pd.DataFrame, name: str, figure: str, sources: list[str], script: str, notes: str = "") -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / name
    df.to_csv(out, index=False)
    n_asd = n_td = n_total = ""
    if "n_asd" in df.columns and "n_td" in df.columns and len(df) == 1:
        n_asd = int(df["n_asd"].iloc[0]) if pd.notna(df["n_asd"].iloc[0]) else ""
        n_td = int(df["n_td"].iloc[0]) if pd.notna(df["n_td"].iloc[0]) else ""
        n_total = int(df["n_total"].iloc[0]) if "n_total" in df.columns and pd.notna(df["n_total"].iloc[0]) else n_asd + n_td
    elif "group" in df.columns and "subject_id" in df.columns:
        g = df["group"].astype(str).str.upper()
        n_asd = int((g == "ASD").sum())
        n_td = int((g == "TD").sum())
        n_total = len(df)
    MANIFEST_ROWS.append(
        {
            "figure": figure,
            "output_file": str(out.relative_to(PROJECT_ROOT)),
            "original_source": "; ".join(sources),
            "generating_script": script,
            "analysis_definition": notes,
            "n_total": n_total,
            "n_asd": n_asd,
            "n_td": n_td,
            "notes": notes,
            "generation_date": GENERATION_DATE,
        }
    )
    return out


def _cohort_row(
    branch: str,
    order: int,
    label: str,
    n_asd: int,
    n_td: int,
    reason: str,
    source: str,
) -> dict[str, Any]:
    return {
        "analysis_branch": branch,
        "stage_order": order,
        "stage_label": label,
        "n_total": n_asd + n_td,
        "n_asd": n_asd,
        "n_td": n_td,
        "exclusion_reason": reason,
        "source_file": source,
    }


def export_s1() -> None:
    rows = [
        _cohort_row("resting", 1, "Registration/effective resting sample", 80, 88, "", "Supplementary Figure S1"),
        _cohort_row("resting", 2, "Primary resting-state spectral cohort", 61, 77, "", "Supplementary Figure S1"),
        _cohort_row("movie", 3, "Paired rest-to-movie exponent", 61, 75, "", "Supplementary Figure S1"),
        _cohort_row("movie", 4, "Movie Aperiodic-ISC cohort", 58, 78, "", "Supplementary Figure S1"),
        _cohort_row("cross_context", 5, "Resting + movie matched", 46, 46, "", "Supplementary Figure S1"),
        _cohort_row("resting", 6, "IQ-balanced subset", 38, 38, "", "Supplementary Figure S1"),
        _cohort_row("resting", 7, "Strict specparam-QC", 44, 46, "", "Supplementary Figure S1"),
        _cohort_row("hbn", 8, "HBN The Present matched cohort", 119, 119, "", "Supplementary Figure S1"),
        _cohort_row("hbn", 9, "The Present movie Aperiodic-ISC", 119, 119, "", "Supplementary Figure S1"),
    ]

    flow = pd.DataFrame(rows).sort_values("stage_order")
    _save(
        flow,
        "s1_cohort_flow.csv",
        "S1",
        ["Supplementary_20260812(q).docx"],
        "scripts/export_supplementary_source_data.py::export_s1",
        "Cohort flow matching Supplementary Figure S1",
    )

    characteristics_rows = [
        ("Registration/effective resting sample", "N", "80", "88", "", ""),
        ("Registration/effective resting sample", "Age months mean (SD)", "85.0 (16.9)", "88.8 (19.9)", "0.184", ""),
        ("Registration/effective resting sample", "Age range months", "46-131", "40-130", "", ""),
        ("Registration/effective resting sample", "Sex M/F", "71/9", "58/30", "", ""),
        ("Registration/effective resting sample", "Full-scale IQ mean (SD)", "93.4 (17.0)", "112.8 (14.3)", "<0.001", ""),
        ("Registration/effective resting sample", "ADOS total mean (SD)", "14.6 (3.4)", "NA", "", ""),
        ("Registration/effective resting sample", "ADOS social affect mean (SD)", "9.7 (2.2)", "NA", "", ""),
        ("Registration/effective resting sample", "ADOS restricted/repetitive behavior mean (SD)", "5.1 (1.4)", "NA", "", ""),
        ("Primary resting-state spectral cohort", "N", "61", "77", "", ""),
        ("Primary resting-state spectral cohort", "Age months mean (SD)", "85.7 (16.9)", "88.8 (19.6)", "0.319", ""),
        ("Primary resting-state spectral cohort", "Age range months", "47-131", "40-130", "", ""),
        ("Primary resting-state spectral cohort", "Sex M/F", "56/5", "49/28", "", ""),
        ("Primary resting-state spectral cohort", "Full-scale IQ mean (SD)", "95.0 (15.2)", "113.2 (14.6)", "<0.001", ""),
        ("Primary resting-state spectral cohort", "ADOS total mean (SD)", "14.1 (3.1)", "NA", "", ""),
        ("Primary resting-state spectral cohort", "ADOS social affect mean (SD)", "9.3 (2.0)", "NA", "", ""),
        ("Primary resting-state spectral cohort", "ADOS restricted/repetitive behavior mean (SD)", "4.9 (1.3)", "NA", "", ""),
        ("Resting 1:1 matched cohort", "N", "55", "55", "", ""),
        ("Resting 1:1 matched cohort", "Age months mean (SD)", "84.1 (16.3)", "84.8 (17.6)", "0.823", ""),
        ("Resting 1:1 matched cohort", "Age range months", "46-131", "46-119", "", ""),
        ("Resting 1:1 matched cohort", "Sex M/F", "51/4", "43/12", "", "Corrected against current supplementary table."),
        ("Resting 1:1 matched cohort", "Full-scale IQ mean (SD)", "101.3 (13.6)", "106.2 (12.3)", "0.053", ""),
        ("Resting 1:1 matched cohort", "ADOS total mean (SD)", "14.0 (3.2)", "NA", "", ""),
        ("Resting 1:1 matched cohort", "ADOS social affect mean (SD)", "9.4 (2.1)", "NA", "", ""),
        ("Resting 1:1 matched cohort", "ADOS restricted/repetitive behavior mean (SD)", "4.9 (1.3)", "NA", "", ""),
        ("IQ-balanced subset", "N", "38", "38", "", ""),
        ("IQ-balanced subset", "Age months mean (SD)", "83.2 (13.5)", "86.6 (14.8)", "0.308", ""),
        ("IQ-balanced subset", "Age range months", "47-108", "48-112", "", ""),
        ("IQ-balanced subset", "Sex M/F", "34/4", "34/4", "", ""),
        ("IQ-balanced subset", "Full-scale IQ mean (SD)", "102.9 (12.7)", "105.1 (12.5)", "0.452", ""),
        ("IQ-balanced subset", "ADOS total mean (SD)", "14.0 (2.9)", "NA", "", ""),
        ("IQ-balanced subset", "ADOS social affect mean (SD)", "9.2 (2.1)", "NA", "", ""),
        ("IQ-balanced subset", "ADOS restricted/repetitive behavior mean (SD)", "4.8 (1.3)", "NA", "", ""),
        ("Strict specparam-QC cohort", "N", "44", "46", "", ""),
        ("Strict specparam-QC cohort", "Age months mean (SD)", "85.4 (16.3)", "84.4 (16.4)", "0.788", ""),
        ("Strict specparam-QC cohort", "Age range months", "47-131", "48-119", "", ""),
        ("Strict specparam-QC cohort", "Sex M/F", "42/2", "35/11", "", ""),
        ("Strict specparam-QC cohort", "Full-scale IQ mean (SD)", "101.8 (12.3)", "106.0 (12.5)", "0.109", ""),
        ("Strict specparam-QC cohort", "ADOS total mean (SD)", "13.8 (3.2)", "NA", "", ""),
        ("Strict specparam-QC cohort", "ADOS social affect mean (SD)", "9.2 (2.0)", "NA", "", ""),
        ("Strict specparam-QC cohort", "ADOS restricted/repetitive behavior mean (SD)", "4.8 (1.3)", "NA", "", ""),
        ("Movie Aperiodic-ISC cohort", "N", "58", "78", "", ""),
        ("Movie Aperiodic-ISC cohort", "Age months mean (SD)", "83.9 (15.7)", "88.1 (19.9)", "0.192", ""),
        ("Movie Aperiodic-ISC cohort", "Age range months", "46-121", "40-121", "", ""),
        ("Movie Aperiodic-ISC cohort", "Sex M/F", "53/5", "58/20", "0.013", "Corrected against current supplementary table; ASD movie cohort has 53 male and 5 female participants."),
        ("Movie Aperiodic-ISC cohort", "Full-scale IQ mean (SD)", "93.5 (17.2)", "112.2 (14.7)", "<0.001", ""),
        ("Movie Aperiodic-ISC cohort", "ADOS total mean (SD)", "14.4 (2.9)", "NA", "", ""),
        ("Resting + movie matched cohort", "N", "46", "46", "", ""),
        ("Resting + movie matched cohort", "Age months mean (SD)", "80.4 (13.6)", "82.5 (15.7)", "0.498", ""),
        ("Resting + movie matched cohort", "Age range months", "46-108", "40-112", "", ""),
        ("Resting + movie matched cohort", "Sex M/F", "39/7", "39/7", "", "Corrected against current supplementary table."),
        ("Resting + movie matched cohort", "Full-scale IQ mean (SD)", "102.7 (15.0)", "108.9 (14.8)", "0.05", ""),
        ("Dual-paradigm post-QC matched cohort", "N", "34", "34", "", ""),
        ("Dual-paradigm post-QC matched cohort", "Age months mean (SD)", "82.2 (11.9)", "86.1 (15.8)", "0.245", ""),
        ("Dual-paradigm post-QC matched cohort", "Age range months", "47-108", "48-117", "", ""),
        ("Dual-paradigm post-QC matched cohort", "Sex M/F", "31/3", "26/8", "", "Corrected against current supplementary table."),
        ("Dual-paradigm post-QC matched cohort", "Full-scale IQ mean (SD)", "98.8 (14.9)", "104.9 (12.2)", "0.069", ""),
        ("HBN The Present matched cohort", "N", "119", "119", "", ""),
        ("HBN The Present matched cohort", "Age months mean (SD)", "99.7 (15.2)", "99.1 (16.5)", "0.755", ""),
        ("HBN The Present matched cohort", "Age range months", "72-131", "70-129", "", ""),
        ("HBN The Present matched cohort", "Sex M/F", "18/101", "18/101", "", "Retained as manuscript-facing value; the sex-matched HBN subset is female-predominant in both groups."),
        ("HBN The Present matched cohort", "Full-scale IQ mean (SD)", "96.0 (20.6)", "103.8 (15.2)", "0.001", ""),
    ]
    characteristics = pd.DataFrame(
        characteristics_rows,
        columns=["cohort", "measure", "asd", "td", "p_value", "notes"],
    )
    _save(
        characteristics,
        "s1_participant_characteristics.csv",
        "S1",
        ["Supplementary_20260812(q).docx"],
        "scripts/export_supplementary_source_data.py::export_s1",
        "Supplementary Table S1 participant characteristics matching current manuscript",
    )


def export_s2() -> None:
    folds = _read(_p("outputs", "tables", "robustness", "posterior_roi_loocv_fdr_folds.csv"))
    summary = _read(_p("outputs", "tables", "robustness", "posterior_roi_loocv_fdr_summary.csv"))

    n_folds = len(folds)
    elec_rows = []
    for elec in ["E33", "E36", "E37", "E38"]:
        fdr_col = f"{elec}_fdr"
        n_fdr = int(folds[fdr_col].astype(bool).sum()) if fdr_col in folds.columns else np.nan
        raw_key = f"{elec}_raw_p_lt_005"
        if raw_key in summary["metric"].values:
            n_raw = int(summary.loc[summary["metric"] == raw_key, "n_survived"].iloc[0])
        else:
            p_col = f"{elec}_pvalue"
            n_raw = int((folds[p_col] < 0.05).sum()) if p_col in folds.columns else n_folds
        elec_rows.append(
            {
                "electrode": elec,
                "n_folds": n_folds,
                "n_fdr_significant": n_fdr,
                "survival_percent": 100.0 * n_fdr / n_folds,
                "n_uncorrected_significant": n_raw,
                "uncorrected_survival_percent": 100.0 * n_raw / n_folds,
            }
        )
    elec_df = pd.DataFrame(elec_rows)
    _save(
        elec_df,
        "s2_loocv_electrode_survival.csv",
        "S2",
        ["outputs/tables/robustness/posterior_roi_loocv_fdr_folds.csv"],
        "scripts/export_supplementary_source_data.py::export_s2",
        "LOOCV electrode FDR survival N=138",
    )

    crit_map = {
        "all_four_posterior_fdr": "all four electrodes",
        "at_least_three_posterior_fdr": "at least three",
        "any_posterior_fdr": "at least one",
        "E33_fdr": "E33",
        "E36_fdr": "E36",
        "E37_fdr": "E37",
        "E38_fdr": "E38",
    }
    crit_rows = []
    for metric, label in crit_map.items():
        sub = summary.loc[summary["metric"] == metric]
        if sub.empty:
            continue
        r = sub.iloc[0]
        crit_rows.append(
            {
                "criterion": label,
                "n_folds_satisfied": int(r["n_survived"]),
                "total_folds": int(r["n_folds"]),
                "survival_percent": float(r["survival_rate"]) * 100.0,
            }
        )
    crit_df = pd.DataFrame(crit_rows)
    _save(
        crit_df,
        "s2_loocv_criteria_summary.csv",
        "S2",
        ["outputs/tables/robustness/posterior_roi_loocv_fdr_summary.csv"],
        "scripts/export_supplementary_source_data.py::export_s2",
        "LOOCV criteria summary",
    )


def export_s3() -> None:
    primary = _read(_p("outputs", "tables", "posterior_roi_sensitivity", "group_ols_models.csv"))
    pr = primary.loc[primary["model"] == "posterior_core_4"].iloc[0]
    robust = _read(_p("figure_source_data", "posterior_robustness_models.csv"))
    hf = _read(_p("outputs", "tables", "artifact_defense", "exponent_models_with_hf_covariate.csv"))
    hf_post = hf.loc[
        (hf["model"] == "posterior_exponent_with_log10_low_gamma")
        & (hf["term"] == "C(group)[T.TD]")
    ].iloc[0]
    iclabel = _read(_p("outputs", "tables", "iclabel_sensitivity", "iclabel_local_posterior_exponent_fdr_summary.csv"))
    iclabel_row = iclabel.loc[(iclabel["threshold"] == 0.7) & (iclabel["subset"] == "all")].iloc[0]
    ic_removed = _read(_p("outputs", "tables", "iclabel_sensitivity", "iclabel_posterior_removed_component_models.csv"))
    ic_rem_row = ic_removed.loc[ic_removed["model"] == "posterior_exponent_with_n_components_removed"].iloc[0]
    knee = _read(_p("outputs", "tables", "posterior_knee_sensitivity", "group_ols_models.csv"))
    knee_row = knee.loc[knee["model"] == "primary_knee"].iloc[0]

    iq = robust.loc[robust["analysis"] == "IQ-balanced matched"].iloc[0]
    strict = robust.loc[robust["analysis"] == "strict-QC"].iloc[0]

    model_rows = [
        {
            "model": "Primary fixed",
            "pipeline": "fixed",
            "estimate_td_minus_asd": float(pr["coef"]),
            "se": float(pr["std_err"]),
            "ci_low": float(pr["ci_low"]),
            "ci_high": float(pr["ci_high"]),
            "p": float(pr["pvalue"]),
            "n_total": int(pr["n_obs"]),
            "n_asd": 61,
            "n_td": 77,
            "covariates": "age_months + sex + IQ_total + usable_epochs + mean_r_squared_fixed",
        },
        {
            "model": "Low-gamma adjusted",
            "pipeline": "fixed",
            "estimate_td_minus_asd": float(hf_post["coef"]),
            "se": float(hf_post["std_err"]),
            "ci_low": float(hf_post["ci_low"]),
            "ci_high": float(hf_post["ci_high"]),
            "p": float(hf_post["pvalue"]),
            "n_total": int(hf_post["n_obs"]),
            "n_asd": 61,
            "n_td": 74,
            "covariates": "primary + log10_low_gamma",
        },
        {
            "model": "IQ-balanced",
            "pipeline": "fixed",
            "estimate_td_minus_asd": float(iq["beta_TD_minus_ASD"]),
            "se": np.nan,
            "ci_low": float(iq["ci_low"]),
            "ci_high": float(iq["ci_high"]),
            "p": float(iq["p"]),
            "n_total": int(iq["n"]),
            "n_asd": 38,
            "n_td": 38,
            "covariates": "age_months + sex + IQ_total + usable_epochs + mean_r_squared",
        },
        {
            "model": "Strict specparam-QC",
            "pipeline": "fixed",
            "estimate_td_minus_asd": float(strict["beta_TD_minus_ASD"]),
            "se": float(strict["SE"]) if pd.notna(strict["SE"]) else np.nan,
            "ci_low": float(strict["ci_low"]),
            "ci_high": float(strict["ci_high"]),
            "p": float(strict["p"]),
            "n_total": int(strict["n"]),
            "n_asd": 44,
            "n_td": 46,
            "covariates": "matched strict specparam-QC cohort",
        },
        {
            "model": "ICLabel",
            "pipeline": "ICLabel 0.70",
            "estimate_td_minus_asd": float(iclabel_row["coef_TD_vs_ASD"]),
            "se": float(iclabel_row["se"]),
            "ci_low": float(iclabel_row["ci_low"]),
            "ci_high": float(iclabel_row["ci_high"]),
            "p": float(iclabel_row["p"]),
            "n_total": int(iclabel_row["n"]),
            "n_asd": 60,
            "n_td": 77,
            "covariates": "ICLabel threshold 0.70 branch",
        },
        {
            "model": "ICLabel + n_components_removed",
            "pipeline": "ICLabel 0.70",
            "estimate_td_minus_asd": float(ic_rem_row["coef_TD_vs_ASD"]),
            "se": float(ic_rem_row["se"]),
            "ci_low": float(ic_rem_row["ci_low"]),
            "ci_high": float(ic_rem_row["ci_high"]),
            "p": float(ic_rem_row["p"]),
            "n_total": int(ic_rem_row["n_total"]),
            "n_asd": int(ic_rem_row["n_ASD"]),
            "n_td": int(ic_rem_row["n_TD"]),
            "covariates": "ICLabel 0.70 + n_components_removed",
        },
        {
            "model": "Knee-mode complete-case",
            "pipeline": "knee",
            "estimate_td_minus_asd": float(knee_row["beta_TD_vs_ASD"]),
            "se": float(knee_row["se"]),
            "ci_low": float(knee_row["ci_low"]),
            "ci_high": float(knee_row["ci_high"]),
            "p": float(knee_row["p"]),
            "n_total": int(knee_row["n_total"]),
            "n_asd": int(knee_row["n_ASD"]),
            "n_td": int(knee_row["n_TD"]),
            "covariates": "knee-mode posterior exponent + mean_r_squared_knee",
        },
    ]
    models_df = pd.DataFrame(model_rows)
    _save(
        models_df,
        "s3_sensitivity_models.csv",
        "S3",
        [
            "outputs/tables/posterior_roi_sensitivity/group_ols_models.csv",
            "figure_source_data/posterior_robustness_models.csv",
        ],
        "scripts/export_supplementary_source_data.py::export_s3",
        "Posterior exponent sensitivity forest models",
    )

    subj = _read(_p("outputs", "tables", "posterior_knee_sensitivity", "subject_level_fixed_knee_comparison.csv"))
    subj_out = subj[
        [
            "subject_id",
            "group",
            "posterior_exponent_fixed",
            "posterior_exponent_knee",
            "age_months",
            "sex",
            "IQ_total",
        ]
    ].rename(
        columns={
            "posterior_exponent_fixed": "fixed_posterior_exponent",
            "posterior_exponent_knee": "knee_posterior_exponent",
            "IQ_total": "iq",
        }
    )
    subj_out["knee_valid"] = subj_out["knee_posterior_exponent"].notna().astype(int)
    _save(
        subj_out,
        "s3_fixed_knee_subjects.csv",
        "S3",
        ["outputs/tables/posterior_knee_sensitivity/subject_level_fixed_knee_comparison.csv"],
        "scripts/export_supplementary_source_data.py::export_s3",
        "Subject-level fixed vs knee posterior exponent",
    )

    qc = _read(_p("outputs", "tables", "posterior_knee_sensitivity", "knee_fit_qc_summary.csv"))
    valid_subj = qc.loc[qc["metric"] == "subjects_with_valid_posterior_knee", "value"].iloc[0]
    impl_rate = qc.loc[qc["metric"] == "posterior_implausible_knee_rate", "value"].iloc[0]
    ch_valid_all = qc.loc[
        (qc["metric"] == "posterior_channel_fit_valid_rate") & (qc["group"] == "all"), "value"
    ].iloc[0]
    fail_asd = int(qc.loc[(qc["metric"] == "posterior_knee_subject_fail_n") & (qc["group"] == "ASD"), "value"].iloc[0])
    fail_td = int(qc.loc[(qc["metric"] == "posterior_knee_subject_fail_n") & (qc["group"] == "TD"), "value"].iloc[0])
    qc_df = pd.DataFrame(
        [
            {
                "group": "all",
                "n_subjects": 138,
                "n_valid_knee": int(valid_subj),
                "valid_percent": 100.0 * float(valid_subj) / 138.0,
                "n_implausible_knee": np.nan,
                "implausible_percent": float(impl_rate) * 100.0,
                "channel_valid_percent": float(ch_valid_all) * 100.0,
            },
            {
                "group": "ASD",
                "n_subjects": 61,
                "n_valid_knee": 61 - fail_asd,
                "valid_percent": 100.0 * (61 - fail_asd) / 61.0,
                "n_implausible_knee": fail_asd,
                "implausible_percent": np.nan,
                "channel_valid_percent": float(
                    qc.loc[
                        (qc["metric"] == "posterior_channel_fit_valid_rate") & (qc["group"] == "ASD"), "value"
                    ].iloc[0]
                )
                * 100.0,
            },
            {
                "group": "TD",
                "n_subjects": 77,
                "n_valid_knee": 77 - fail_td,
                "valid_percent": 100.0 * (77 - fail_td) / 77.0,
                "n_implausible_knee": fail_td,
                "implausible_percent": np.nan,
                "channel_valid_percent": float(
                    qc.loc[
                        (qc["metric"] == "posterior_channel_fit_valid_rate") & (qc["group"] == "TD"), "value"
                    ].iloc[0]
                )
                * 100.0,
            },
        ]
    )
    _save(
        qc_df,
        "s3_knee_qc.csv",
        "S3",
        ["outputs/tables/posterior_knee_sensitivity/knee_fit_qc_summary.csv"],
        "scripts/export_supplementary_source_data.py::export_s3",
        "Knee-mode fit QC summary",
    )


def export_s4() -> None:
    age_grid = _read(_p("outputs", "tables", "nonlinear_age_sensitivity", "group_difference_by_age.csv"))
    post = age_grid.loc[age_grid["outcome"] == "posterior_exponent"].copy()
    pred_rows = []
    for _, r in post.iterrows():
        for grp, col in [("ASD", "ASD_pred"), ("TD", "TD_pred")]:
            pred_rows.append(
                {
                    "model": r["model"],
                    "group": grp,
                    "age_months": float(r["age_months"]),
                    "predicted_exponent": float(r[col]),
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "cohort": "primary_resting_N138",
                }
            )
    pred_df = pd.DataFrame(pred_rows)
    _save(
        pred_df,
        "s4_development_predictions.csv",
        "S4",
        ["outputs/tables/nonlinear_age_sensitivity/group_difference_by_age.csv"],
        "scripts/export_supplementary_source_data.py::export_s4",
        "Linear/spline age predictions for posterior exponent",
    )

    import importlib.util

    ci_path = PROJECT_ROOT / "scripts" / "export_s4_development_predictions_with_ci.py"
    spec = importlib.util.spec_from_file_location("export_s4_ci", ci_path)
    ci_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(ci_mod)
    ci_df, _ci_meta = ci_mod.export_predictions_with_ci()
    _save(
        ci_df,
        "s4_development_predictions_with_ci.csv",
        "S4",
        [
            "src/nonlinear_age_sensitivity.py",
            "outputs/tables/nonlinear_age_sensitivity/model_comparison.csv",
        ],
        "scripts/export_s4_development_predictions_with_ci.py",
        "Covariate-adjusted posterior exponent trajectories with 95% mean CI",
    )

    post_age = _read(_p("outputs", "tables", "coordination_feature", "posterior_nonposterior_age_models.csv"))
    regional = _read(_p("outputs", "tables", "supplementary_roi_clinical", "regional_age_group_interaction.csv"))
    int_rows = []

    pa = post_age.loc[post_age["model"] == "age_group_posterior"]
    ix = pa.loc[pa["term"].str.contains("C\\(group.*TD.*:age_months", regex=True)].iloc[0]
    int_rows.append(
        {
            "model": "Primary",
            "region": "posterior",
            "estimate_group_by_age": float(ix["coef"]),
            "se": float(ix["std_err"]),
            "ci_low": float(ix["ci_low"]),
            "ci_high": float(ix["ci_high"]),
            "p": float(ix["pvalue"]),
            "n_total": int(ix["n_obs"]),
            "n_asd": 61,
            "n_td": 77,
        }
    )

    for region in ["frontal_exponent", "occipital_exponent", "posterior_exponent"]:
        sub = regional.loc[regional["outcome"] == region]
        ix = sub.loc[sub["term"].str.contains("C\\(group.*TD.*:age_months", regex=True)]
        if ix.empty:
            continue
        r = ix.iloc[0]
        int_rows.append(
            {
                "model": "Primary regional",
                "region": region.replace("_exponent", ""),
                "estimate_group_by_age": float(r["coef"]),
                "se": float(r["std_err"]),
                "ci_low": float(r["ci_low"]),
                "ci_high": float(r["ci_high"]),
                "p": float(r["pvalue"]),
                "n_total": int(r["n_obs"]),
                "n_asd": 61 if int(r["n_obs"]) == 138 else 60,
                "n_td": 77 if int(r["n_obs"]) == 138 else 75,
            }
        )

    int_df = pd.DataFrame(int_rows)

    sens_path = _p("outputs", "tables", "spectral_maturation", "age_group_interaction_sensitivity_cohorts.csv")
    if sens_path.exists():
        sens = pd.read_csv(sens_path)
        for _, r in sens.iterrows():
            if r["term"] != "C(group)[T.TD]:age_months":
                continue
            label = str(r["cohort"])
            if r["model_type"] == "RLM":
                label = f"{label} (robust)"
            int_rows.append(
                {
                    "model": label,
                    "region": "posterior",
                    "estimate_group_by_age": float(r["estimate"]),
                    "se": float(r["se"]),
                    "ci_low": float(r["ci_low"]),
                    "ci_high": float(r["ci_high"]),
                    "p": float(r["p"]),
                    "n_total": int(r["n_total"]),
                    "n_asd": int(r["n_asd"]),
                    "n_td": int(r["n_td"]),
                }
            )
        int_df = pd.DataFrame(int_rows)

    _save(
        int_df,
        "s4_development_interactions.csv",
        "S4",
        [
            "outputs/tables/coordination_feature/posterior_nonposterior_age_models.csv",
            "outputs/tables/supplementary_roi_clinical/regional_age_group_interaction.csv",
            "outputs/tables/spectral_maturation/age_group_interaction_sensitivity_cohorts.csv",
        ],
        "scripts/export_supplementary_source_data.py::export_s4",
        "Age x group interactions including IQ-balanced and strict-QC sensitivity cohorts",
    )

    norm = _read(_p("derivatives", "stats", "normative_exponent_scores.csv"))
    diag_rows = []
    for _, r in norm.iterrows():
        if pd.notna(r.get("deviation")):
            diag_rows.append(
                {
                    "model": "TD_reference",
                    "diagnostic_type": "deviation_vs_age",
                    "x": float(r["age_months"]),
                    "y": float(r["deviation"]),
                    "subject_id": r["subject_id"],
                }
            )
        if pd.notna(r.get("global_exponent")) and pd.notna(r.get("predicted")):
            diag_rows.append(
                {
                    "model": "TD_reference",
                    "diagnostic_type": "observed_vs_predicted",
                    "x": float(r["predicted"]),
                    "y": float(r["global_exponent"]),
                    "subject_id": r["subject_id"],
                }
            )
    fit = _read(_p("outputs", "tables", "normative_exponent", "normative_model_fit_comparison.csv"))
    for _, r in fit.iterrows():
        diag_rows.append(
            {
                "model": r["model_name"],
                "diagnostic_type": "model_fit_metric",
                "x": r["model_name"],
                "y": float(r["r_squared"]),
                "subject_id": "",
            }
        )
    diag_df = pd.DataFrame(diag_rows)
    _save(
        diag_df,
        "s4_development_diagnostics.csv",
        "S4",
        [
            "derivatives/stats/normative_exponent_scores.csv",
            "outputs/tables/normative_exponent/normative_model_fit_comparison.csv",
        ],
        "scripts/export_supplementary_source_data.py::export_s4",
        "TD-reference diagnostics long table",
    )


def _posterior_iaf_from_channels() -> pd.Series:
    ch = _read(_p("derivatives", "specparam", "specparam_channel_results_qc.csv"))
    ch = ch.loc[ch["channel"].isin(POSTERIOR_CHANNELS) & (ch["fit_valid"] == 1)]
    return ch.groupby("subject_id")["alpha_cf"].mean()


def export_s5() -> None:
    rest = _read(_p("outputs", "tables", "resting_features_locked.csv"))
    dev = _read(_p("derivatives", "stats", "spectral_maturation_deviation_scores.csv"))
    post_iaf = _posterior_iaf_from_channels()
    subj = rest[["subject_id", "group", "alpha_cf", "posterior_exponent"]].merge(
        dev[
            [
                "subject_id",
                "age_months",
                "alpha_cf_deviation_z",
                "posterior_exponent_deviation_z",
            ]
        ],
        on="subject_id",
        how="left",
    )
    subj = subj.merge(post_iaf.rename("posterior_iaf"), on="subject_id", how="left")
    subj_out = subj.rename(
        columns={
            "alpha_cf": "global_iaf",
            "posterior_exponent_deviation_z": "posterior_exponent_deviation_z",
            "alpha_cf_deviation_z": "iaf_deviation_z",
        }
    )[
        [
            "subject_id",
            "group",
            "age_months",
            "posterior_iaf",
            "global_iaf",
            "posterior_exponent",
            "posterior_exponent_deviation_z",
            "iaf_deviation_z",
        ]
    ]
    _save(
        subj_out,
        "s5_iaf_subjects.csv",
        "S5",
        [
            "outputs/tables/resting_features_locked.csv",
            "derivatives/stats/spectral_maturation_deviation_scores.csv",
            "derivatives/specparam/specparam_channel_results_qc.csv",
        ],
        "scripts/export_supplementary_source_data.py::export_s5",
        "Subject-level IAF and exponent deviation scores",
    )

    age_iaf = _read(_p("outputs", "tables", "spectral_maturation", "age_group_interaction_models.csv"))
    indep = _read(_p("outputs", "tables", "spectral_maturation", "independence_models.csv"))
    corr = _read(_p("outputs", "tables", "spectral_maturation", "deviation_correlations_asd.csv"))
    model_rows = []

    def _add_from_df(df: pd.DataFrame, outcome: str, model: str, term_filter: str) -> None:
        sub = df.loc[df["outcome"] == outcome] if "outcome" in df.columns else df.loc[df["model"] == model]
        if term_filter:
            sub = sub.loc[sub["term"].astype(str).str.contains(term_filter, regex=True, na=False)]
        if sub.empty:
            return
        r = sub.iloc[0]
        model_rows.append(
            {
                "outcome": outcome,
                "model": model,
                "term": r["term"],
                "estimate": float(r["coef"]),
                "se": float(r["std_err"]),
                "ci_low": float(r["ci_low"]),
                "ci_high": float(r["ci_high"]),
                "p": float(r["pvalue"]),
                "n": int(r["n_obs"]),
            }
        )

    _add_from_df(age_iaf, "alpha_cf", "global IAF group effect", "C\\(group\\)\\[T\\.TD\\]$")
    _add_from_df(age_iaf, "alpha_cf", "global IAF group x age", "C\\(group\\)\\[T\\.TD\\]:age_months")
    _add_from_df(age_iaf, "posterior_alpha_cf", "posterior IAF group x age", "C\\(group\\)\\[T\\.TD\\]:age_months")
    _add_from_df(
        indep,
        "posterior_exponent",
        "posterior exponent group x age adjusted for posterior IAF",
        "C\\(group\\)\\[T\\.TD\\]:age_months",
    )
    for _, r in corr.iterrows():
        if r["var_x"] == "alpha_cf_deviation_z" and r["var_y"] == "posterior_exponent_deviation_z":
            model_rows.append(
                {
                    "outcome": "posterior_exponent_deviation_z",
                    "model": "exponent-deviation vs IAF-deviation correlation",
                    "term": "Spearman_rho",
                    "estimate": float(r["rho"]),
                    "se": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p": float(r["pvalue"]),
                    "n": int(r["n"]),
                }
            )
    models_df = pd.DataFrame(model_rows)
    _save(
        models_df,
        "s5_iaf_models.csv",
        "S5",
        ["outputs/tables/spectral_maturation/age_group_interaction_models.csv"],
        "scripts/export_supplementary_source_data.py::export_s5",
        "IAF control models",
    )


def export_s6() -> None:
    ts_path = _p(
        "jr_remote_bundle",
        "outputs",
        "jr_modelling",
        "posterior_movie_isc",
        "posterior_sliding_exponent_timeseries.csv",
    )
    tc = build_movie_timecourse(ts_path)
    tc_out = tc.rename(
        columns={
            "time_sec": "time_seconds",
            "mean_aperiodic_isc": "isc_mean",
            "sem_aperiodic_isc": "isc_sem",
        }
    )
    tc_out["ci_low"] = tc_out["isc_mean"] - 1.96 * tc_out["isc_sem"]
    tc_out["ci_high"] = tc_out["isc_mean"] + 1.96 * tc_out["isc_sem"]
    tc_out["n"] = np.nan
    tc_out["isc_definition"] = "within_group_loo"
    tc_out = tc_out[
        ["time_seconds", "group", "isc_mean", "ci_low", "ci_high", "n", "isc_definition", "segment_label"]
    ]
    _save(
        tc_out,
        "s6_isc_timecourse.csv",
        "S6",
        [str(ts_path.relative_to(PROJECT_ROOT))],
        "jr_remote_bundle/src/bp_figures/io.py::build_movie_timecourse",
        "Within-group LOO Aperiodic-ISC timecourse",
    )

    mech = _read(_p("outputs", "tables", "followup_exploration", "isc_mechanism_group_tests.csv"))

    def _segment_summary(prefix: str, isc_def: str, fname: str) -> None:
        rows = []
        for analysis, seg in [
            ("td_template_mental", "mentalizing"),
            ("td_template_pain", "pain"),
            ("td_template_neutral", "neutral"),
            ("within_group_mental", "mentalizing"),
            ("within_group_pain", "pain"),
            ("within_group_neutral", "neutral"),
        ]:
            if not analysis.startswith(prefix):
                continue
            r = mech.loc[mech["analysis"] == analysis].iloc[0]
            n = int(r["n_asd"] + r["n_td"])
            se = abs(float(r["mean_diff_asd_minus_td"]) / float(r["t_stat"])) if r["t_stat"] else np.nan
            rows.append(
                {
                    "segment": seg,
                    "group": "ASD",
                    "mean_isc": float(r["asd_mean"]),
                    "se": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n": n,
                    "comparison_p": float(r["p_value"]),
                    "fdr_p": float(r["fdr_p"]),
                    "isc_definition": isc_def,
                    "td_mean": float(r["td_mean"]),
                    "asd_mean": float(r["asd_mean"]),
                }
            )
            rows.append(
                {
                    "segment": seg,
                    "group": "TD",
                    "mean_isc": float(r["td_mean"]),
                    "se": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n": n,
                    "comparison_p": float(r["p_value"]),
                    "fdr_p": float(r["fdr_p"]),
                    "isc_definition": isc_def,
                    "td_mean": float(r["td_mean"]),
                    "asd_mean": float(r["asd_mean"]),
                }
            )
        df = pd.DataFrame(rows)
        _save(
            df.drop(columns=["td_mean", "asd_mean"]),
            fname,
            "S6",
            ["outputs/tables/followup_exploration/isc_mechanism_group_tests.csv"],
            "scripts/export_supplementary_source_data.py::export_s6",
            isc_def,
        )

    _segment_summary("td_template", "td_template", "s6_td_template_segment_summary.csv")
    _segment_summary("within_group", "within_group_loo", "s6_within_group_segment_summary.csv")

    ts = _read(ts_path)
    bounds = (
        ts.groupby("event_type", as_index=False)
        .agg(start_seconds=("window_start_sec", "min"), end_seconds=("window_end_sec", "max"))
        .rename(columns={"event_type": "segment"})
    )
    bounds["event_label"] = bounds["segment"].map(SEGMENT_MAP).fillna(bounds["segment"])
    _save(
        bounds,
        "s6_event_boundaries.csv",
        "S6",
        [str(ts_path.relative_to(PROJECT_ROOT))],
        "scripts/export_supplementary_source_data.py::export_s6",
        "Event boundaries from sliding-window timeseries",
    )

    import importlib.util

    s6_isc_path = PROJECT_ROOT / "scripts" / "export_s6_isc_subject_level.py"
    spec = importlib.util.spec_from_file_location("export_s6_isc", s6_isc_path)
    s6_isc_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(s6_isc_mod)
    _subj, _ci, ok = s6_isc_mod.export_s6_isc_subject_data()
    if not ok:
        raise RuntimeError(
            "S6 subject-level ISC validation failed; see "
            "figure_source_data/supplementary/s6_isc_export_validation_report.md"
        )
    _save(
        _subj,
        "s6_isc_subject_level.csv",
        "S6",
        [
            "derivatives/derivatives_task_movie/stats/aperiodic_isc/aperiodic_isc_td_template_subject_values.csv",
            "derivatives/derivatives_task_movie/stats/aperiodic_isc/aperiodic_isc_within_group_subject_values.csv",
        ],
        "scripts/export_s6_isc_subject_level.py",
        "Subject-level Fisher-z Aperiodic-ISC for panels D/E",
    )
    _save(
        _ci,
        "s6_isc_group_ci.csv",
        "S6",
        ["figure_source_data/supplementary/s6_isc_subject_level.csv"],
        "scripts/export_s6_isc_subject_level.py",
        "Group mean 95% CI from subject-level Fisher-z ISC",
    )


def export_s7() -> None:
    mech = _read(_p("outputs", "tables", "followup_exploration", "isc_mechanism_group_tests.csv"))
    metric_map = {
        "td_template": ("Aperiodic-ISC", "td_template"),
        "within_group": ("Aperiodic-ISC", "within_group_loo"),
        "envelope": ("Envelope ISC", "within_group_loo"),
    }
    sync_rows = []
    for _, r in mech.iterrows():
        analysis = str(r["analysis"])
        matched = None
        for prefix, (metric, isc_def) in metric_map.items():
            if analysis.startswith(prefix + "_"):
                matched = (prefix, metric, isc_def, analysis[len(prefix) + 1 :])
                break
        if matched is None:
            continue
        _, metric, isc_def, seg = matched
        n = int(r["n_asd"] + r["n_td"])
        ge = float(r["td_mean"] - r["asd_mean"])
        ge_se = abs(ge / float(r["t_stat"])) if r["t_stat"] else np.nan
        for grp, mean in [("ASD", float(r["asd_mean"])), ("TD", float(r["td_mean"]))]:
            sync_rows.append(
                {
                    "segment": SEGMENT_MAP.get(seg, seg),
                    "metric": metric,
                    "group": grp,
                    "mean": mean,
                    "se": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n": n,
                    "group_effect": ge,
                    "group_effect_se": ge_se,
                    "p": float(r["p_value"]),
                    "fdr_p": float(r["fdr_p"]),
                    "isc_definition": isc_def,
                }
            )
    sync_df = pd.DataFrame(sync_rows)

    classic_path = _p(
        "derivatives",
        "derivatives_task_movie",
        "stats",
        "classic_isc",
        "classic_vs_aperiodic_within_group_comparison.csv",
    )
    if classic_path.exists():
        classic = pd.read_csv(classic_path)
        alpha = classic.loc[classic["isc_type"].astype(str).str.contains("Alpha", case=False, na=False)]
        for _, r in alpha.iterrows():
            seg = SEGMENT_MAP.get(str(r["event_type"]), str(r["event_type"]))
            n = int(r["n_asd"] + r["n_td"])
            ge = float(r["td_effect"] - r["asd_effect"])
            for grp, mean in [("ASD", float(r["asd_effect"])), ("TD", float(r["td_effect"]))]:
                sync_rows.append(
                    {
                        "segment": seg,
                        "metric": "Alpha PLV ISC",
                        "group": grp,
                        "mean": mean,
                        "se": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n": n,
                        "group_effect": ge,
                        "group_effect_se": np.nan,
                        "p": float(r["p_value"]),
                        "fdr_p": np.nan,
                        "isc_definition": "within_group_loo",
                    }
                )
        sync_df = pd.DataFrame(sync_rows)

    sync_sources = ["outputs/tables/followup_exploration/isc_mechanism_group_tests.csv"]
    if classic_path.exists():
        sync_sources.append(str(classic_path.relative_to(PROJECT_ROOT)))
    _save(
        sync_df,
        "s7_synchrony_controls.csv",
        "S7",
        sync_sources,
        "scripts/export_supplementary_source_data.py::export_s7",
        "Group-level synchrony controls including Alpha PLV ISC",
    )

    partial_path = _p(
        "derivatives",
        "derivatives_task_movie",
        "stats",
        "classic_isc",
        "aperiodic_envelope_partial_analysis.csv",
    )
    if partial_path.exists():
        partial = pd.read_csv(partial_path)
        env_out = partial.rename(
            columns={
                "ancova_group_beta_z": "envelope_adjusted_group_beta_z",
                "ancova_group_se": "envelope_adjusted_group_se",
                "ancova_group_p": "envelope_adjusted_group_p",
                "ancova_envelope_beta": "envelope_covariate_beta",
                "ancova_envelope_p": "envelope_covariate_p",
                "partial_cohen_d_asd_minus_td": "partial_cohen_d",
                "partial_effect_retained_pct": "effect_retained_pct",
                "ancova_group_fdr_p": "envelope_adjusted_group_fdr_p",
            }
        )
        _save(
            env_out,
            "s7_envelope_adjusted.csv",
            "S7",
            [str(partial_path.relative_to(PROJECT_ROOT))],
            "scripts/93_aperiodic_envelope_partial_isc.py; scripts/export_supplementary_source_data.py::export_s7",
            "Envelope-adjusted ANCOVA per event",
        )

    gaze = _read(_p("outputs", "tables", "gaze_sensitivity_ancova.csv"))
    gaze_grp = _read(_p("outputs", "tables", "gaze_sensitivity_group_tests.csv"))
    gaze_rows = []
    for _, r in gaze.iterrows():
        seg = r["segment"]
        g = gaze_grp.loc[gaze_grp["segment"] == seg]
        gaze_rows.append(
            {
                "segment": seg,
                "model": r["model_type"],
                "group_beta": float(r["group_beta_TD_vs_ASD"]),
                "se": float(r["group_se"]),
                "ci_low": float(r["group_ci_low"]),
                "ci_high": float(r["group_ci_high"]),
                "p": float(r["group_p"]),
                "n": int(r["n"]),
                "mean_gaze_asd": float(g["ASD_mean"].iloc[0]) if not g.empty else np.nan,
                "sd_gaze_asd": float(g["ASD_sd"].iloc[0]) if not g.empty else np.nan,
                "mean_gaze_td": float(g["TD_mean"].iloc[0]) if not g.empty else np.nan,
                "sd_gaze_td": float(g["TD_sd"].iloc[0]) if not g.empty else np.nan,
                "gaze_group_p": float(g["primary_p"].iloc[0]) if not g.empty else np.nan,
            }
        )
    gaze_df = pd.DataFrame(gaze_rows)
    _save(
        gaze_df,
        "s7_gaze_sensitivity.csv",
        "S7",
        [
            "outputs/tables/gaze_sensitivity_ancova.csv",
            "outputs/tables/gaze_sensitivity_group_tests.csv",
        ],
        "scripts/export_supplementary_source_data.py::export_s7",
        "Gaze-adjusted movie ISC models",
    )


def export_s8() -> None:
    movie_sum = _read(_p("jr_remote_bundle", "outputs", "hbn_external_movie", "tables", "isc_group_stats_matched.csv"))
    movie_out = []
    for _, r in movie_sum.iterrows():
        analysis = "sliding_window" if r["isc_mode"] == "sliding" else "nonoverlapping_2s_epoch"
        n = int(r["n_asd"])
        movie_out.append(
            {
                "analysis": analysis,
                "group": "ASD",
                "n": n,
                "mean": float(r["asd_mean_z"]),
                "sd": np.nan,
                "se": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "welch_t": float(r["t_stat"]),
                "p": float(r["p_value"]),
            }
        )
        movie_out.append(
            {
                "analysis": analysis,
                "group": "TD",
                "n": int(r["n_td"]),
                "mean": float(r["td_mean_z"]),
                "sd": np.nan,
                "se": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "welch_t": float(r["t_stat"]),
                "p": float(r["p_value"]),
            }
        )
    _save(
        pd.DataFrame(movie_out),
        "s8_hbn_movie_summary.csv",
        "S8",
        ["jr_remote_bundle/outputs/hbn_external_movie/tables/isc_group_stats_matched.csv"],
        "scripts/export_supplementary_source_data.py::export_s8",
        "HBN movie ISC matched cohort summary",
    )

    hbn_subj_paths = [
        _p(
            "derivatives",
            "hbn_external_movie",
            "replication",
            "isc_matched",
            "isc_subject_values_thepresent_matched_sliding_0p5s.csv",
        ),
        _p(
            "derivatives",
            "hbn_external_movie",
            "replication",
            "isc_matched",
            "isc_subject_values_thepresent_matched_epoch_2s.csv",
        ),
    ]
    hbn_subj_parts: list[pd.DataFrame] = []
    for path in hbn_subj_paths:
        if not path.exists():
            continue
        sub = pd.read_csv(path)
        analysis = "sliding_window" if "sliding" in path.name else "nonoverlapping_2s_epoch"
        sub = sub.assign(analysis=analysis, cohort="hbn_matched")
        hbn_subj_parts.append(sub)
    if hbn_subj_parts:
        hbn_subj = pd.concat(hbn_subj_parts, ignore_index=True)
        _save(
            hbn_subj,
            "s8_hbn_movie_subjects.csv",
            "S8",
            [str(p.relative_to(PROJECT_ROOT)) for p in hbn_subj_paths if p.exists()],
            "scripts/export_supplementary_source_data.py::export_s8",
            "HBN matched movie ISC subject-level values (pseudonymous HBN_#### IDs)",
        )


def export_s9() -> None:
    cross = _read(_p("figure_source_data", "posterior_cross_state_subject_metrics.csv"))
    supp = _read(_p("outputs", "tables", "manuscript0621", "supp_table_s1_participant_characteristics.csv"))
    dual_ids = set(str(supp.loc[supp["cohort"] == "Dual-paradigm matched", "subject_ids_asd"].iloc[0]).split(";"))
    dual_ids |= set(str(supp.loc[supp["cohort"] == "Dual-paradigm matched", "subject_ids_td"].iloc[0]).split(";"))

    subj = cross[
        [
            "subject_id",
            "group",
            "rest_posterior_exponent",
            "movie_isc_neutral_z",
            "age_months",
            "sex",
            "IQ_total",
            "usable_epochs",
        ]
    ].rename(
        columns={
            "rest_posterior_exponent": "resting_posterior_exponent",
            "movie_isc_neutral_z": "neutral_aperiodic_isc",
            "IQ_total": "iq",
        }
    )
    subj["cohort"] = np.where(subj["subject_id"].isin(dual_ids), "dual_paradigm_matched", "overlapping")
    _save(
        subj,
        "s9_coupling_subjects.csv",
        "S9",
        ["figure_source_data/posterior_cross_state_subject_metrics.csv"],
        "scripts/export_supplementary_source_data.py::export_s9",
        "Rest-movie coupling subjects",
    )

    auth = _read(_p("outputs", "tables", "manuscript0621", "rest_movie_coupling_authoritative.csv"))
    model_rows = []
    for _, r in auth.iterrows():
        cohort = "overlapping" if r["cohort"] == "main_overlap" else "dual_paradigm_matched"
        n_obs = int(r["n_obs"])
        model_rows.append(
            {
                "cohort": cohort,
                "model": f"rest_x_group_x_{r['segment']}",
                "interaction_beta": float(r["interaction_beta"]),
                "se": float(r["interaction_se"]),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "raw_p": float(r["raw_p"]),
                "fdr_p": float(r["fdr_q"]) if pd.notna(r["fdr_q"]) else np.nan,
                "n_total": n_obs,
                "n_asd": np.nan,
                "n_td": np.nan,
            }
        )
    _save(
        pd.DataFrame(model_rows),
        "s9_coupling_models.csv",
        "S9",
        ["outputs/tables/manuscript0621/rest_movie_coupling_authoritative.csv"],
        "scripts/export_supplementary_source_data.py::export_s9",
        "Authoritative rest-movie coupling interaction models",
    )

    boot = _read(_p("outputs", "tables", "coordination_feature", "rest_movie_coupling_bootstrap.csv"))
    boot_out = []
    for _, r in boot.iterrows():
        cohort = "overlapping" if r["cohort"] == "main_overlap" else "dual_paradigm_matched"
        boot_out.append(
            {
                "cohort": cohort,
                "segment": r["segment"],
                "median_beta": float(r["beta_median"]),
                "ci_low": float(r["ci95_low"]),
                "ci_high": float(r["ci95_high"]),
                "bootstrap_p": float(r["p_bootstrap_two_sided"]),
                "n_resamples": int(r["n_resamples"]),
            }
        )
    _save(
        pd.DataFrame(boot_out),
        "s9_coupling_bootstrap_summary.csv",
        "S9",
        ["outputs/tables/coordination_feature/rest_movie_coupling_bootstrap.csv"],
        "scripts/export_supplementary_source_data.py::export_s9",
        "Bootstrap interaction summary",
    )

    boot_dist_paths = [
        (
            "mental",
            _p(
                "derivatives",
                "derivatives_task_movie",
                "stats",
                "resting_movie_coupling_interaction_bootstrap_dist_mental.csv",
            ),
        ),
        (
            "pain",
            _p(
                "derivatives",
                "derivatives_task_movie",
                "stats",
                "resting_movie_coupling_interaction_bootstrap_dist_pain.csv",
            ),
        ),
        (
            "neutral",
            _p(
                "derivatives",
                "derivatives_task_movie",
                "stats",
                "resting_movie_coupling_interaction_bootstrap_dist_neutral.csv",
            ),
        ),
    ]
    boot_parts: list[pd.DataFrame] = []
    boot_sources: list[str] = []
    for segment, path in boot_dist_paths:
        if not path.exists():
            continue
        part = pd.read_csv(path)
        part = part.rename(columns={"beta_interaction": "beta_interaction"})
        part["segment"] = segment
        part["cohort"] = "main_overlap"
        part["interaction_term"] = "posterior_exponent × group"
        boot_parts.append(part[["resample_id", "beta_interaction", "segment", "cohort", "interaction_term"]])
        boot_sources.append(str(path.relative_to(PROJECT_ROOT)))
    if boot_parts:
        boot_full = pd.concat(boot_parts, ignore_index=True)
        _save(
            boot_full,
            "s9_coupling_bootstrap.csv",
            "S9",
            boot_sources,
            "scripts/export_supplementary_source_data.py::export_s9",
            "Bootstrap interaction iteration table (main_overlap cohort)",
        )


def write_manifest() -> None:
    manifest = pd.DataFrame(MANIFEST_ROWS)
    out = OUT_DIR / "source_data_manifest.csv"
    manifest.to_csv(out, index=False)


def main() -> None:
    import importlib.util

    recompute_path = PROJECT_ROOT / "scripts" / "recompute_s4_age_interaction_sensitivity.py"
    spec = importlib.util.spec_from_file_location("recompute_s4", recompute_path)
    recompute_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(recompute_mod)
    recompute_mod.main()

    export_s1()
    export_s2()
    export_s3()
    export_s4()
    export_s5()
    export_s6()
    export_s7()
    export_s8()
    export_s9()
    write_manifest()
    print(f"Wrote supplementary source data to {OUT_DIR}")


if __name__ == "__main__":
    main()
