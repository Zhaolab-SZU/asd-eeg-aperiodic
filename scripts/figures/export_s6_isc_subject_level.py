# Ideal release name: export_s6_isc_subject_level.py
# Original path: scripts/export_s6_isc_subject_level.py
# Note: S6 subject-level ISC
# This file is a copy for the public github_release/ bundle.

"""Export S6 panel D/E subject-level Aperiodic-ISC and group mean 95% CIs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "figure_source_data" / "supplementary"

TD_TEMPLATE_PATH = (
    PROJECT_ROOT
    / "derivatives"
    / "derivatives_task_movie"
    / "stats"
    / "aperiodic_isc"
    / "aperiodic_isc_td_template_subject_values.csv"
)
WITHIN_GROUP_PATH = (
    PROJECT_ROOT
    / "derivatives"
    / "derivatives_task_movie"
    / "stats"
    / "aperiodic_isc"
    / "aperiodic_isc_within_group_subject_values.csv"
)
MECH_TESTS_PATH = PROJECT_ROOT / "outputs" / "tables" / "followup_exploration" / "isc_mechanism_group_tests.csv"
SUM_TD_PATH = OUT_DIR / "s6_td_template_segment_summary.csv"
SUM_WG_PATH = OUT_DIR / "s6_within_group_segment_summary.csv"

SUBJECT_OUT = OUT_DIR / "s6_isc_subject_level.csv"
GROUP_CI_OUT = OUT_DIR / "s6_isc_group_ci.csv"
REPORT_OUT = OUT_DIR / "s6_isc_export_validation_report.md"

SEGMENT_MAP = {"mental": "mentalizing", "pain": "pain", "neutral": "neutral"}
ISC_DEF_MAP = {
    "td_template": "td_template",
    "within_group": "within_group_loo",
}
MECH_PREFIX = {
    "td_template": "td_template",
    "within_group_loo": "within_group",
}
EVENT_FROM_SEGMENT = {v: k for k, v in SEGMENT_MAP.items()}


def _load_subject_file(path: Path, isc_definition: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"subject_id", "group", "event_type", "isc_z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    out = df[["subject_id", "group", "event_type", "isc_z"]].copy()
    out["subject_id"] = out["subject_id"].astype(str)
    out["group"] = out["group"].astype(str).str.upper()
    out["segment"] = out["event_type"].map(SEGMENT_MAP)
    if out["segment"].isna().any():
        bad = sorted(out.loc[out["segment"].isna(), "event_type"].unique())
        raise ValueError(f"Unmapped event_type values: {bad}")
    out["isc_definition"] = isc_definition
    out["isc_value"] = out["isc_z"].astype(float)
    return out[["subject_id", "group", "segment", "isc_definition", "isc_value"]]


def build_subject_level() -> pd.DataFrame:
    td = _load_subject_file(TD_TEMPLATE_PATH, "td_template")
    wg = _load_subject_file(WITHIN_GROUP_PATH, "within_group_loo")
    combined = pd.concat([td, wg], ignore_index=True)
    combined = combined.sort_values(["isc_definition", "segment", "group", "subject_id"]).reset_index(drop=True)
    return combined


def _summary_reference() -> pd.DataFrame:
    td = pd.read_csv(SUM_TD_PATH)
    wg = pd.read_csv(SUM_WG_PATH)
    return pd.concat([td, wg], ignore_index=True)


def _mech_lookup() -> pd.DataFrame:
    mech = pd.read_csv(MECH_TESTS_PATH)
    rows = []
    for _, r in mech.iterrows():
        analysis = str(r["analysis"])
        for isc_def, prefix in MECH_PREFIX.items():
            if analysis.startswith(prefix + "_"):
                seg_key = analysis[len(prefix) + 1 :]
                segment = SEGMENT_MAP.get(seg_key, seg_key)
                rows.append(
                    {
                        "segment": segment,
                        "isc_definition": isc_def,
                        "comparison_p": float(r["p_value"]),
                        "fdr_p": float(r["fdr_p"]),
                        "n_asd_ref": int(r["n_asd"]),
                        "n_td_ref": int(r["n_td"]),
                        "asd_mean_ref": float(r["asd_mean"]),
                        "td_mean_ref": float(r["td_mean"]),
                    }
                )
                break
    return pd.DataFrame(rows)


def validate_means(subject_df: pd.DataFrame, summary_df: pd.DataFrame) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for _, srow in summary_df.iterrows():
        seg = str(srow["segment"])
        grp = str(srow["group"])
        isc_def = str(srow["isc_definition"])
        expected = float(srow["mean_isc"])
        sub = subject_df[
            (subject_df["segment"] == seg)
            & (subject_df["group"] == grp)
            & (subject_df["isc_definition"] == isc_def)
        ]
        if sub.empty:
            issues.append(
                {
                    "segment": seg,
                    "group": grp,
                    "isc_definition": isc_def,
                    "issue": "no subject rows",
                    "expected": expected,
                    "computed": np.nan,
                    "abs_diff": np.nan,
                }
            )
            continue
        computed = float(sub["isc_value"].mean())
        diff = abs(computed - expected)
        if diff > 1e-12:
            issues.append(
                {
                    "segment": seg,
                    "group": grp,
                    "isc_definition": isc_def,
                    "issue": "mean mismatch",
                    "expected": expected,
                    "computed": computed,
                    "abs_diff": diff,
                    "n_subjects": int(len(sub)),
                }
            )
    return issues


def build_group_ci(subject_df: pd.DataFrame, mech_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (seg, grp, isc_def), sub in subject_df.groupby(["segment", "group", "isc_definition"]):
        values = sub["isc_value"].to_numpy(dtype=float)
        n = int(len(values))
        mean_isc = float(np.mean(values))
        se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        if n > 1:
            t_crit = float(stats.t.ppf(0.975, n - 1))
            ci_low = mean_isc - t_crit * se
            ci_high = mean_isc + t_crit * se
        else:
            ci_low = ci_high = float("nan")

        mech = mech_df[(mech_df["segment"] == seg) & (mech_df["isc_definition"] == isc_def)]
        comparison_p = float(mech["comparison_p"].iloc[0]) if not mech.empty else np.nan
        fdr_p = float(mech["fdr_p"].iloc[0]) if not mech.empty else np.nan

        rows.append(
            {
                "segment": seg,
                "group": grp,
                "isc_definition": isc_def,
                "n": n,
                "mean_isc": mean_isc,
                "se": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "comparison_p": comparison_p,
                "fdr_p": fdr_p,
            }
        )
    return pd.DataFrame(rows).sort_values(["isc_definition", "segment", "group"]).reset_index(drop=True)


def write_report(
    subject_df: pd.DataFrame,
    group_ci_df: pd.DataFrame,
    mean_issues: list[dict[str, Any]],
    passed: bool,
) -> None:
    lines = [
        "# S6 Aperiodic-ISC subject-level export validation",
        "",
        f"Subject file: `{SUBJECT_OUT.relative_to(PROJECT_ROOT)}`",
        f"Group CI file: `{GROUP_CI_OUT.relative_to(PROJECT_ROOT)}`",
        "",
        "## Upstream sources",
        "",
        f"- TD-template: `{TD_TEMPLATE_PATH.relative_to(PROJECT_ROOT)}` (isc_z scale)",
        f"- Within-group LOO: `{WITHIN_GROUP_PATH.relative_to(PROJECT_ROOT)}` (isc_z scale)",
        f"- Summary reference: `{SUM_TD_PATH.name}`, `{SUM_WG_PATH.name}`",
        f"- Group tests: `{MECH_TESTS_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "## Cohort",
        "",
        f"- Subject-level rows: {len(subject_df)} (expected 816 = 136 subjects × 3 segments × 2 definitions)",
        f"- Unique subjects: {subject_df['subject_id'].nunique()}",
        "",
        "## Mean reproduction vs summary tables",
        "",
    ]
    if passed:
        lines.append("All segment × group × isc_definition means **match** summary `mean_isc` (tolerance 1e-12).")
    else:
        lines.append("**FAILED** — mean mismatches detected; export aborted.")
        lines.append("")
        lines.append("| segment | group | isc_definition | expected | computed | |diff| | n |")
        lines.append("|---------|-------|----------------|----------|----------|-------|---|")
        for issue in mean_issues:
            lines.append(
                f"| {issue['segment']} | {issue['group']} | {issue['isc_definition']} | "
                f"{issue.get('expected', 'NA')} | {issue.get('computed', 'NA')} | "
                f"{issue.get('abs_diff', 'NA')} | {issue.get('n_subjects', 'NA')} |"
            )

    lines.extend(
        [
            "",
            "## Group 95% CI method",
            "",
            "- mean_isc: subject-level mean of Fisher-z ISC (`isc_z`)",
            "- se: SD / sqrt(n) with ddof=1",
            "- ci_low/ci_high: mean ± t_(0.975, n−1) × SE (within-group mean CI)",
            "- comparison_p / fdr_p: from `isc_mechanism_group_tests.csv` (Welch t-test on isc_z)",
            "",
            f"## Overall verdict: **{'PASS' if passed else 'FAIL'}**",
        ]
    )
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


def export_s6_isc_subject_data() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    summary_ref = _summary_reference()
    subject_df = build_subject_level()
    mean_issues = validate_means(subject_df, summary_ref)
    passed = len(mean_issues) == 0

    write_report(subject_df, pd.DataFrame(), mean_issues, passed)
    if not passed:
        print(f"Validation FAILED — see {REPORT_OUT}")
        for issue in mean_issues:
            print(issue)
        return subject_df, pd.DataFrame(), False

    mech_df = _mech_lookup()
    group_ci_df = build_group_ci(subject_df, mech_df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subject_df.to_csv(SUBJECT_OUT, index=False)
    group_ci_df.to_csv(GROUP_CI_OUT, index=False)
    write_report(subject_df, group_ci_df, mean_issues, passed=True)
    return subject_df, group_ci_df, True


def main() -> None:
    subject_df, group_ci_df, ok = export_s6_isc_subject_data()
    if not ok:
        raise SystemExit(1)
    print(f"Wrote {SUBJECT_OUT} ({len(subject_df)} rows)")
    print(f"Wrote {GROUP_CI_OUT} ({len(group_ci_df)} rows)")
    print(f"Wrote {REPORT_OUT}")
    print("Validation: PASS")


if __name__ == "__main__":
    main()
