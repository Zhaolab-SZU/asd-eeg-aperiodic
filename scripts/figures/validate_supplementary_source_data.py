# Ideal release name: validate_supplementary_source_data.py
# Original path: scripts/validate_supplementary_source_data.py
# Note: Validate supplementary source bundle
# This file is a copy for the public github_release/ bundle.

"""Validate supplementary figure source-data exports against manuscript tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUP_DIR = PROJECT_ROOT / "figure_source_data" / "supplementary"
REPORT_PATH = SUP_DIR / "source_data_validation_report.md"

EXPECTED_FILES: dict[str, list[str]] = {
    "s1_cohort_flow.csv": [
        "analysis_branch",
        "stage_order",
        "stage_label",
        "n_total",
        "n_asd",
        "n_td",
        "exclusion_reason",
        "source_file",
    ],
    "s2_loocv_electrode_survival.csv": [
        "electrode",
        "n_folds",
        "n_fdr_significant",
        "survival_percent",
        "n_uncorrected_significant",
        "uncorrected_survival_percent",
    ],
    "s2_loocv_criteria_summary.csv": ["criterion", "n_folds_satisfied", "total_folds", "survival_percent"],
    "s3_sensitivity_models.csv": [
        "model",
        "pipeline",
        "estimate_td_minus_asd",
        "se",
        "ci_low",
        "ci_high",
        "p",
        "n_total",
        "n_asd",
        "n_td",
        "covariates",
    ],
    "s3_fixed_knee_subjects.csv": [
        "subject_id",
        "group",
        "fixed_posterior_exponent",
        "knee_posterior_exponent",
        "knee_valid",
        "age_months",
        "sex",
        "iq",
    ],
    "s3_knee_qc.csv": [
        "group",
        "n_subjects",
        "n_valid_knee",
        "valid_percent",
        "n_implausible_knee",
        "implausible_percent",
        "channel_valid_percent",
    ],
    "s4_development_predictions.csv": ["model", "group", "age_months", "predicted_exponent", "ci_low", "ci_high", "cohort"],
    "s4_development_predictions_with_ci.csv": [
        "model",
        "group",
        "age_months",
        "predicted_exponent",
        "ci_low",
        "ci_high",
        "ci_method",
        "n_total",
        "n_asd",
        "n_td",
        "covariate_setting",
        "model_formula",
    ],
    "s4_development_interactions.csv": [
        "model",
        "region",
        "estimate_group_by_age",
        "se",
        "ci_low",
        "ci_high",
        "p",
        "n_total",
        "n_asd",
        "n_td",
    ],
    "s4_development_diagnostics.csv": ["model", "diagnostic_type", "x", "y", "subject_id"],
    "s5_iaf_subjects.csv": [
        "subject_id",
        "group",
        "age_months",
        "posterior_iaf",
        "global_iaf",
        "posterior_exponent",
        "posterior_exponent_deviation_z",
        "iaf_deviation_z",
    ],
    "s5_iaf_models.csv": ["outcome", "model", "term", "estimate", "se", "ci_low", "ci_high", "p", "n"],
    "s6_isc_timecourse.csv": ["time_seconds", "group", "isc_mean", "ci_low", "ci_high", "n", "isc_definition"],
    "s6_td_template_segment_summary.csv": [
        "segment",
        "group",
        "mean_isc",
        "se",
        "ci_low",
        "ci_high",
        "n",
        "comparison_p",
        "fdr_p",
    ],
    "s6_within_group_segment_summary.csv": [
        "segment",
        "group",
        "mean_isc",
        "se",
        "ci_low",
        "ci_high",
        "n",
        "comparison_p",
        "fdr_p",
    ],
    "s6_event_boundaries.csv": ["segment", "start_seconds", "end_seconds", "event_label"],
    "s6_isc_subject_level.csv": ["subject_id", "group", "segment", "isc_definition", "isc_value"],
    "s6_isc_group_ci.csv": [
        "segment",
        "group",
        "isc_definition",
        "n",
        "mean_isc",
        "se",
        "ci_low",
        "ci_high",
        "comparison_p",
        "fdr_p",
    ],
    "s7_synchrony_controls.csv": [
        "segment",
        "metric",
        "group",
        "mean",
        "se",
        "ci_low",
        "ci_high",
        "n",
        "group_effect",
        "group_effect_se",
        "p",
        "fdr_p",
        "isc_definition",
    ],
    "s7_envelope_adjusted.csv": [
        "event_type",
        "n_total",
        "envelope_adjusted_group_p",
        "envelope_adjusted_group_fdr_p",
        "partial_cohen_d",
    ],
    "s7_gaze_sensitivity.csv": [
        "segment",
        "model",
        "group_beta",
        "se",
        "ci_low",
        "ci_high",
        "p",
        "n",
        "mean_gaze_asd",
        "sd_gaze_asd",
        "mean_gaze_td",
        "sd_gaze_td",
        "gaze_group_p",
    ],
    "s8_hbn_movie_subjects.csv": ["subject_id", "group", "isc_r", "isc_z", "analysis", "cohort"],
    "s8_hbn_movie_summary.csv": ["analysis", "group", "n", "mean", "sd", "se", "ci_low", "ci_high", "welch_t", "p"],
    "s8_hbn_resting_models.csv": [
        "eye_state",
        "pipeline",
        "model_type",
        "estimate_td_minus_asd",
        "se",
        "ci_low",
        "ci_high",
        "p",
        "n",
        "n_pairs",
    ],
    "s8_hbn_resting_subjects.csv": [
        "subject_id",
        "pair_id",
        "group",
        "eye_state",
        "pipeline",
        "posterior_exponent",
        "age_months",
        "sex",
        "iq",
    ],
    "s9_coupling_subjects.csv": [
        "subject_id",
        "group",
        "resting_posterior_exponent",
        "neutral_aperiodic_isc",
        "age_months",
        "sex",
        "iq",
        "usable_epochs",
        "cohort",
    ],
    "s9_coupling_models.csv": [
        "cohort",
        "model",
        "interaction_beta",
        "se",
        "ci_low",
        "ci_high",
        "raw_p",
        "fdr_p",
        "n_total",
        "n_asd",
        "n_td",
    ],
    "s9_coupling_bootstrap_summary.csv": [
        "cohort",
        "segment",
        "median_beta",
        "ci_low",
        "ci_high",
        "bootstrap_p",
        "n_resamples",
    ],
    "s9_coupling_bootstrap.csv": ["resample_id", "beta_interaction", "segment", "cohort", "interaction_term"],
    "source_data_manifest.csv": [
        "figure",
        "output_file",
        "original_source",
        "generating_script",
        "analysis_definition",
        "generation_date",
    ],
}

COHORT_EXPECT = {
    "Registration/effective resting sample": (168, 80, 88),
    "Primary resting-state spectral cohort": (138, 61, 77),
    "Posterior resting cohort (primary posterior exponent)": (138, 61, 77),
    "ADOS complete-case subset": (60, 60, 0),
    "Movie spectral-QC / ISC cohort": (136, 58, 78),
    "Rest–movie paired cohort": (104, 46, 58),
    "Dual-paradigm post-QC matched cohort": (68, 34, 34),
    "HBN matched cohort": (238, 119, 119),
    "HBN eyes-open matched subset": (224, 112, 112),
}


def _load(name: str) -> pd.DataFrame | None:
    path = SUP_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def _check_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [c for c in required if c not in df.columns]


def main() -> None:
    lines: list[str] = [
        "# Supplementary source data validation report",
        "",
        f"Directory: `{SUP_DIR.relative_to(PROJECT_ROOT)}`",
        "",
        "## 1. File existence",
        "",
    ]
    all_ok = True

    for fname, req in EXPECTED_FILES.items():
        df = _load(fname)
        if df is None:
            lines.append(f"- **MISSING** `{fname}`")
            all_ok = False
            continue
        missing = _check_columns(df, req)
        status = "PASS" if not missing else "FAIL"
        if missing:
            all_ok = False
        lines.append(f"- `{fname}`: **{status}** ({len(df)} rows)")
        if missing:
            lines.append(f"  - Missing columns: {', '.join(missing)}")

    lines.extend(["", "## 2. Sample size checks (vs supp_table_s1)", ""])
    s1 = _load("s1_cohort_flow.csv")
    if s1 is not None:
        for label, (nt, na, nd) in COHORT_EXPECT.items():
            row = s1.loc[s1["stage_label"] == label]
            if row.empty:
                lines.append(f"- **MISSING ROW** `{label}`")
                all_ok = False
                continue
            r = row.iloc[0]
            ok = int(r["n_total"]) == nt and int(r["n_asd"]) == na and int(r["n_td"]) == nd
            mark = "PASS" if ok else "MISMATCH"
            if not ok:
                all_ok = False
            lines.append(
                f"- {label}: **{mark}** "
                f"(file {int(r['n_total'])}/{int(r['n_asd'])}/{int(r['n_td'])} vs expected {nt}/{na}/{nd})"
            )

    lines.extend(["", "## 3. Key result consistency", ""])
    s2c = _load("s2_loocv_criteria_summary.csv")
    if s2c is not None:
        all4 = s2c.loc[s2c["criterion"] == "all four electrodes"].iloc[0]
        ok = int(all4["n_folds_satisfied"]) == 125 and int(all4["total_folds"]) == 138
        lines.append(
            f"- LOOCV all-four FDR: **{'PASS' if ok else 'MISMATCH'}** "
            f"({int(all4['n_folds_satisfied'])}/138, expected 125/138)"
        )
        if not ok:
            all_ok = False

    s3 = _load("s3_sensitivity_models.csv")
    if s3 is not None:
        strict = s3.loc[s3["model"] == "Strict specparam-QC"].iloc[0]
        iq = s3.loc[s3["model"] == "IQ-balanced"].iloc[0]
        ok_strict = abs(float(strict["estimate_td_minus_asd"]) - 0.139) < 0.002 and int(strict["n_total"]) == 90
        ok_iq = abs(float(iq["estimate_td_minus_asd"]) - 0.123) < 0.002 and int(iq["n_total"]) == 76
        lines.append(
            f"- Strict-QC posterior β: **{'PASS' if ok_strict else 'MISMATCH'}** "
            f"(β={float(strict['estimate_td_minus_asd']):.3f}, n={int(strict['n_total'])})"
        )
        lines.append(
            f"- IQ-balanced posterior β: **{'PASS' if ok_iq else 'MISMATCH'}** "
            f"(β={float(iq['estimate_td_minus_asd']):.3f}, n={int(iq['n_total'])})"
        )
        if not (ok_strict and ok_iq):
            all_ok = False

    lines.extend(["", "## 4. Gap-fill checks", ""])
    env = _load("s7_envelope_adjusted.csv")
    if env is not None:
        pain = env.loc[env["event_type"] == "pain"].iloc[0]
        ok = float(pain["envelope_adjusted_group_p"]) < 0.001
        lines.append(f"- s7 envelope-adjusted pain p: **{'PASS' if ok else 'MISMATCH'}** (p={float(pain['envelope_adjusted_group_p']):.2e})")
        if not ok:
            all_ok = False

    sync = _load("s7_synchrony_controls.csv")
    if sync is not None:
        pain_alpha = sync.loc[(sync["segment"] == "pain") & (sync["metric"] == "Alpha PLV ISC")]
        ok = not pain_alpha.empty and abs(float(pain_alpha["p"].iloc[0]) - 0.97) < 0.02
        lines.append(f"- s7 Alpha PLV pain p≈0.97: **{'PASS' if ok else 'MISMATCH'}**")
        if not ok:
            all_ok = False

    hbn_subj = _load("s8_hbn_movie_subjects.csv")
    if hbn_subj is not None:
        ok = hbn_subj["subject_id"].astype(str).str.startswith("HBN_").all() and len(hbn_subj) >= 200
        lines.append(f"- s8 HBN pseudonymous subjects: **{'PASS' if ok else 'MISMATCH'}** (n={len(hbn_subj)})")
        if not ok:
            all_ok = False

    boot = _load("s9_coupling_bootstrap.csv")
    if boot is not None:
        ok = len(boot) >= 10000
        lines.append(f"- s9 bootstrap iterations: **{'PASS' if ok else 'MISMATCH'}** (n={len(boot)})")
        if not ok:
            all_ok = False

    s4i = _load("s4_development_interactions.csv")
    if s4i is not None:
        models = set(s4i["model"].astype(str))
        ok = "IQ-balanced matched" in models and any("Strict-QC" in m for m in models)
        lines.append(f"- s4 sensitivity cohort interactions: **{'PASS' if ok else 'MISMATCH'}**")
        if not ok:
            all_ok = False

    s4ci = _load("s4_development_predictions_with_ci.csv")
    if s4ci is not None:
        grid_ok = True
        for model_name in ("linear_interaction", "spline_interaction"):
            for grp in ("ASD", "TD"):
                sub = s4ci[(s4ci["model"] == model_name) & (s4ci["group"] == grp)]
                if len(sub) != 120:
                    grid_ok = False
        ci_ok = (
            (s4ci["ci_low"] <= s4ci["predicted_exponent"])
            & (s4ci["predicted_exponent"] <= s4ci["ci_high"])
        ).all()
        finite_ok = np.isfinite(s4ci.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all()
        method_ok = s4ci["ci_method"].astype(str).str.contains("get_prediction", case=False, na=False).all()
        lines.append(f"- s4 predictions with CI grid complete: **{'PASS' if grid_ok else 'MISMATCH'}**")
        lines.append(f"- s4 predictions CI bounds valid: **{'PASS' if ci_ok else 'MISMATCH'}**")
        lines.append(f"- s4 predictions finite values: **{'PASS' if finite_ok else 'MISMATCH'}**")
        lines.append(f"- s4 predictions mean CI (not pred interval): **{'PASS' if method_ok else 'MISMATCH'}**")
        if not (grid_ok and ci_ok and finite_ok and method_ok):
            all_ok = False

        ref = _load("s4_development_predictions.csv")
        if ref is not None:
            merged = ref.merge(
                s4ci[["model", "group", "age_months", "predicted_exponent"]],
                on=["model", "group", "age_months"],
                how="inner",
                suffixes=("_ref", "_new"),
            )
            if not merged.empty:
                max_diff = float((merged["predicted_exponent_ref"] - merged["predicted_exponent_new"]).abs().max())
                ok = max_diff < 1e-6
                lines.append(
                    f"- s4 CI file vs reference predictions: **{'PASS' if ok else 'MISMATCH'}** "
                    f"(max |diff|={max_diff:.3e})"
                )
                if not ok:
                    all_ok = False

    s6_subj = _load("s6_isc_subject_level.csv")
    s6_ci = _load("s6_isc_group_ci.csv")
    sum_td = _load("s6_td_template_segment_summary.csv")
    sum_wg = _load("s6_within_group_segment_summary.csv")
    if s6_subj is not None and sum_td is not None and sum_wg is not None:
        summary = pd.concat([sum_td, sum_wg], ignore_index=True)
        mean_ok = True
        for _, srow in summary.iterrows():
            sub = s6_subj[
                (s6_subj["segment"] == srow["segment"])
                & (s6_subj["group"] == srow["group"])
                & (s6_subj["isc_definition"] == srow["isc_definition"])
            ]
            if sub.empty or abs(float(sub["isc_value"].mean()) - float(srow["mean_isc"])) > 1e-12:
                mean_ok = False
                break
        row_ok = len(s6_subj) == 816 and s6_subj["subject_id"].nunique() == 136
        lines.append(f"- s6 subject-level row count (816) and n subjects (136): **{'PASS' if row_ok else 'MISMATCH'}**")
        lines.append(f"- s6 subject means vs summary tables: **{'PASS' if mean_ok else 'MISMATCH'}**")
        if not (mean_ok and row_ok):
            all_ok = False
    if s6_ci is not None:
        ci_bounds_ok = ((s6_ci["ci_low"] <= s6_ci["mean_isc"]) & (s6_ci["mean_isc"] <= s6_ci["ci_high"])).all()
        lines.append(f"- s6 group CI bounds valid: **{'PASS' if ci_bounds_ok else 'MISMATCH'}**")
        if not ci_bounds_ok:
            all_ok = False

    lines.extend(["", f"## Overall verdict: **{'PASS' if all_ok else 'PASS WITH GAPS'}**", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
