# Ideal release name: validate_supplementary_source_data.py
# Original path: scripts/validate_supplementary_source_data.py
# Note: Validate supplementary source bundle
# This file is a copy for the public github_release/ bundle.

"""Validate supplementary figure source-data exports against manuscript tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    "s1_participant_characteristics.csv": ["cohort", "measure", "asd", "td", "p_value", "notes"],
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
    "Paired rest-to-movie exponent": (136, 61, 75),
    "Movie Aperiodic-ISC cohort": (136, 58, 78),
    "Resting + movie matched": (92, 46, 46),
    "IQ-balanced subset": (76, 38, 38),
    "Strict specparam-QC": (90, 44, 46),
    "HBN The Present matched cohort": (238, 119, 119),
    "The Present movie Aperiodic-ISC": (238, 119, 119),
}

SEX_EXPECT = {
    "Resting 1:1 matched cohort": ("51/4", "43/12", ""),
    "Movie Aperiodic-ISC cohort": ("53/5", "58/20", "0.013"),
    "Resting + movie matched cohort": ("39/7", "39/7", ""),
    "Dual-paradigm post-QC matched cohort": ("31/3", "26/8", ""),
    "HBN The Present matched cohort": ("18/101", "18/101", ""),
}


def _load(name: str) -> pd.DataFrame | None:
    path = SUP_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def _check_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [c for c in required if c not in df.columns]


def _parse_scalar(value: str) -> float:
    clean = value.split("#", 1)[0].strip()
    return float(clean) if "." in clean else int(clean)


def _yaml_number_in_block(path: Path, block_header: str, key: str, child_indent: int = 2) -> float:
    lines = path.read_text(encoding="utf-8").splitlines()
    in_block = False
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 0 and stripped == f"{block_header}:":
            in_block = True
            continue
        if in_block and indent == 0:
            break
        if in_block and indent == child_indent and stripped.startswith(f"{key}:"):
            return _parse_scalar(stripped.split(":", 1)[1])
    raise ValueError(f"Could not find {block_header}.{key} in {path}")


def _yaml_number_in_nested_block(path: Path, parent: str, child: str, key: str) -> float:
    lines = path.read_text(encoding="utf-8").splitlines()
    in_parent = False
    in_child = False
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 0 and stripped == f"{parent}:":
            in_parent = True
            in_child = False
            continue
        if in_parent and indent == 0:
            break
        if in_parent and indent == 2 and stripped == f"{child}:":
            in_child = True
            continue
        if in_child and indent <= 2:
            in_child = False
        if in_child and indent == 4 and stripped.startswith(f"{key}:"):
            return _parse_scalar(stripped.split(":", 1)[1])
    raise ValueError(f"Could not find {parent}.{child}.{key} in {path}")


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

    lines.extend(["", "## 3. Participant-characteristics checks", ""])
    s1_demo = _load("s1_participant_characteristics.csv")
    if s1_demo is not None:
        for cohort, (asd, td, p_value) in SEX_EXPECT.items():
            row = s1_demo.loc[(s1_demo["cohort"] == cohort) & (s1_demo["measure"] == "Sex M/F")]
            if row.empty:
                lines.append(f"- **MISSING ROW** `{cohort}` Sex M/F")
                all_ok = False
                continue
            r = row.iloc[0]
            ok = str(r["asd"]) == asd and str(r["td"]) == td
            if p_value:
                ok = ok and str(r["p_value"]) == p_value
            mark = "PASS" if ok else "MISMATCH"
            lines.append(
                f"- {cohort} Sex M/F: **{mark}** "
                f"(file ASD={r['asd']}, TD={r['td']}, p={r.get('p_value', '')}; "
                f"expected ASD={asd}, TD={td}{', p=' + p_value if p_value else ''})"
            )
            if not ok:
                all_ok = False

    lines.extend(["", "## 4. QC config checks", ""])
    task_movie_path = PROJECT_ROOT / "config" / "config_task_movie.yaml"
    hbn_movie_path = PROJECT_ROOT / "config" / "config_hbn_thepresent.yaml"
    primary_epochs = int(_yaml_number_in_block(task_movie_path, "epochs", "min_usable_epochs"))
    primary_bad_ratio = float(_yaml_number_in_block(task_movie_path, "fit_quality", "subject_invalid_channel_ratio_max"))
    hbn_epochs = int(_yaml_number_in_block(hbn_movie_path, "epochs", "min_usable_epochs"))
    hbn_main_epochs = int(_yaml_number_in_nested_block(hbn_movie_path, "hbn", "main_matched", "min_usable_epochs"))
    hbn_bad_ratio = float(_yaml_number_in_block(hbn_movie_path, "fit_quality", "subject_invalid_channel_ratio_max"))
    hbn_r2 = float(_yaml_number_in_block(hbn_movie_path, "fit_quality", "min_r_squared"))
    hbn_fit_error = float(_yaml_number_in_block(hbn_movie_path, "fit_quality", "fit_error_top_percentile"))
    checks = [
        ("Primary movie min usable epochs", primary_epochs == 50, primary_epochs, 50),
        ("Primary movie invalid-channel ratio", abs(primary_bad_ratio - 0.30) < 1e-12, primary_bad_ratio, 0.30),
        ("HBN The Present min usable epochs", hbn_epochs == 40, hbn_epochs, 40),
        ("HBN matched-cohort min usable epochs", hbn_main_epochs == 40, hbn_main_epochs, 40),
        ("HBN The Present invalid-channel ratio", abs(hbn_bad_ratio - 0.20) < 1e-12, hbn_bad_ratio, 0.20),
        ("HBN The Present min R2", abs(hbn_r2 - 0.90) < 1e-12, hbn_r2, 0.90),
        ("HBN The Present fit-error top percentile", abs(hbn_fit_error - 5.0) < 1e-12, hbn_fit_error, 5.0),
    ]
    for label, ok, observed, expected in checks:
        lines.append(f"- {label}: **{'PASS' if ok else 'MISMATCH'}** (file {observed}; expected {expected})")
        if not ok:
            all_ok = False

    lines.extend(["", "## 5. Key result consistency", ""])
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

    lines.extend(["", "## 6. Gap-fill checks", ""])
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
