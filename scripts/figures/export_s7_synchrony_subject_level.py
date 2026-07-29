# Ideal release name: export_s7_synchrony_subject_level.py
# Original path: scripts/export_s7_synchrony_subject_level.py
# Note: S7 synchrony subject-level
# This file is a copy for the public github_release/ bundle.

"""Export S7 synchrony-control subject-level ISC and group-effect summary tables."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "figure_source_data"

AP_PATH = (
    PROJECT_ROOT
    / "derivatives"
    / "derivatives_task_movie"
    / "stats"
    / "aperiodic_isc"
    / "aperiodic_isc_within_group_subject_values.csv"
)
ENV_PATH = (
    PROJECT_ROOT
    / "derivatives"
    / "derivatives_task_movie"
    / "stats"
    / "classic_isc"
    / "envelope_within_group_subject_values.csv"
)
PLV_PATH = (
    PROJECT_ROOT
    / "derivatives"
    / "derivatives_task_movie"
    / "stats"
    / "classic_isc"
    / "alpha_plv_within_group_subject_values.csv"
)
MECH_TESTS_PATH = PROJECT_ROOT / "outputs" / "tables" / "followup_exploration" / "isc_mechanism_group_tests.csv"
OLD_SYNCH_PATH = PROJECT_ROOT / "figure_source_data" / "supplementary" / "s7_synchrony_controls.csv"
ENVELOPE_ADJ_PATH = PROJECT_ROOT / "figure_source_data" / "supplementary" / "s7_envelope_adjusted.csv"
ENVELOPE_SRC_PATH = (
    PROJECT_ROOT
    / "derivatives"
    / "derivatives_task_movie"
    / "stats"
    / "classic_isc"
    / "aperiodic_envelope_partial_analysis.csv"
)

SUBJECT_OUT = OUT_DIR / "s7_synchrony_subject_level.csv"
GROUP_OUT = OUT_DIR / "s7_synchrony_group_effects.csv"
REPORT_OUT = OUT_DIR / "s7_synchrony_validation_report.md"

SEGMENT_MAP = {"mental": "mentalizing", "pain": "pain", "neutral": "neutral"}
METRIC_SOURCES = {
    "aperiodic_isc": (AP_PATH, "isc_z"),
    "envelope_isc": (ENV_PATH, "isc_z"),
    "alpha_plv_isc": (PLV_PATH, "isc_r"),
}
MECH_KEY = {
    "aperiodic_isc": "within_group",
    "envelope_isc": "envelope",
    "alpha_plv_isc": None,
}
OLD_METRIC_MAP = {
    "Aperiodic-ISC": "aperiodic_isc",
    "Envelope ISC": "envelope_isc",
    "Alpha PLV ISC": "alpha_plv_isc",
}
ENVELOPE_COL_MAP = {
    "pearson_r": "pearson_r",
    "shared_variance_pct": "shared_variance_pct",
    "envelope_adjusted_group_beta_z": "ancova_group_beta_z",
    "envelope_adjusted_group_se": "ancova_group_se",
    "envelope_adjusted_group_p": "ancova_group_p",
    "envelope_adjusted_group_fdr_p": "ancova_group_fdr_p",
    "partial_cohen_d": "partial_cohen_d_asd_minus_td",
    "effect_retained_pct": "partial_effect_retained_pct",
}
TOL = 1e-12


def _welch_diff_stats(td_vals: np.ndarray, asd_vals: np.ndarray) -> dict[str, float]:
    td_vals = np.asarray(td_vals, dtype=float)
    asd_vals = np.asarray(asd_vals, dtype=float)
    mean_td = float(np.mean(td_vals))
    mean_asd = float(np.mean(asd_vals))
    group_effect = mean_td - mean_asd
    t_stat, p_value = stats.ttest_ind(td_vals, asd_vals, equal_var=False)
    se = float(np.sqrt(asd_vals.var(ddof=1) / len(asd_vals) + td_vals.var(ddof=1) / len(td_vals)))
    va, vb = asd_vals.var(ddof=1), td_vals.var(ddof=1)
    na, nb = len(asd_vals), len(td_vals)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t_crit = float(stats.t.ppf(0.975, df))
    return {
        "n_asd": na,
        "n_td": nb,
        "mean_asd": mean_asd,
        "mean_td": mean_td,
        "group_effect_td_minus_asd": group_effect,
        "se": se,
        "ci_low": group_effect - t_crit * se,
        "ci_high": group_effect + t_crit * se,
        "p": float(p_value),
        "t_stat_td_minus_asd": float(t_stat),
    }


def _load_metric(path: Path, value_col: str, metric: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"subject_id", "group", "event_type", value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    out = df[["subject_id", "group", "event_type", value_col]].copy()
    out["subject_id"] = out["subject_id"].astype(str)
    out["group"] = out["group"].astype(str).str.upper()
    out["segment"] = out["event_type"].map(SEGMENT_MAP)
    if out["segment"].isna().any():
        bad = sorted(out.loc[out["segment"].isna(), "event_type"].unique())
        raise ValueError(f"{path.name}: unmapped event_type values {bad}")
    out["metric"] = metric
    out["isc_definition"] = "within_group_loo"
    out["isc_value"] = out[value_col].astype(float)
    return out[["subject_id", "group", "segment", "metric", "isc_definition", "isc_value"]]


def build_subject_level() -> pd.DataFrame:
    frames = [_load_metric(path, col, metric) for metric, (path, col) in METRIC_SOURCES.items()]
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["metric", "segment", "group", "subject_id"]).reset_index(drop=True)


def _mech_row(metric: str, segment: str) -> pd.Series | None:
    seg_key = {v: k for k, v in SEGMENT_MAP.items()}[segment]
    prefix = MECH_KEY[metric]
    if prefix is None:
        return None
    mech = pd.read_csv(MECH_TESTS_PATH)
    key = f"{prefix}_{seg_key}"
    rows = mech.loc[mech["analysis"] == key]
    if rows.empty:
        raise ValueError(f"Missing mechanism test row: {key}")
    return rows.iloc[0]


def build_group_effects(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (segment, metric), sub in subject_df.groupby(["segment", "metric"]):
        asd = sub.loc[sub["group"] == "ASD", "isc_value"].to_numpy(dtype=float)
        td = sub.loc[sub["group"] == "TD", "isc_value"].to_numpy(dtype=float)
        if len(asd) == 0 or len(td) == 0:
            raise ValueError(f"No ASD/TD rows for {segment} × {metric}")
        stats_row = _welch_diff_stats(td, asd)
        fdr_p = np.nan
        mech = _mech_row(str(metric), str(segment))
        if mech is not None:
            fdr_p = float(mech["fdr_p"])
        rows.append(
            {
                "segment": segment,
                "metric": metric,
                "isc_definition": "within_group_loo",
                "fdr_p": fdr_p,
                **stats_row,
            }
        )
    out = pd.DataFrame(rows)
    keep = [
        "segment",
        "metric",
        "isc_definition",
        "n_asd",
        "n_td",
        "mean_asd",
        "mean_td",
        "group_effect_td_minus_asd",
        "se",
        "ci_low",
        "ci_high",
        "p",
        "fdr_p",
    ]
    return out[keep].sort_values(["segment", "metric"]).reset_index(drop=True)


def _load_old_within_group_loo() -> pd.DataFrame:
    old = pd.read_csv(OLD_SYNCH_PATH)
    old = old.loc[old["isc_definition"] == "within_group_loo"].copy()
    old["metric"] = old["metric"].map(OLD_METRIC_MAP)
    if old["metric"].isna().any():
        raise ValueError("Unmapped metric labels in old synchrony controls table")
    return old


def validate_against_old(subject_df: pd.DataFrame, group_df: pd.DataFrame) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    old = _load_old_within_group_loo()

    for _, orow in old.iterrows():
        seg = str(orow["segment"])
        metric = str(orow["metric"])
        grp = str(orow["group"])
        expected_mean = float(orow["mean"])
        sub = subject_df[
            (subject_df["segment"] == seg)
            & (subject_df["metric"] == metric)
            & (subject_df["group"] == grp)
        ]
        if sub.empty:
            issues.append({"check": "subject_mean", "segment": seg, "metric": metric, "group": grp, "issue": "no rows"})
            continue
        computed_mean = float(sub["isc_value"].mean())
        if abs(computed_mean - expected_mean) > TOL:
            issues.append(
                {
                    "check": "subject_mean",
                    "segment": seg,
                    "metric": metric,
                    "group": grp,
                    "expected": expected_mean,
                    "computed": computed_mean,
                    "abs_diff": abs(computed_mean - expected_mean),
                }
            )

    for _, orow in old.groupby(["segment", "metric"]).first().reset_index().iterrows():
        seg = str(orow["segment"])
        metric = str(orow["metric"])
        grow = group_df[(group_df["segment"] == seg) & (group_df["metric"] == metric)].iloc[0]
        expected_ge = float(orow["group_effect"])
        expected_p = float(orow["p"])
        expected_fdr = orow["fdr_p"]
        expected_se = orow["group_effect_se"]

        checks = [
            ("group_effect_td_minus_asd", expected_ge, float(grow["group_effect_td_minus_asd"])),
            ("p", expected_p, float(grow["p"])),
        ]
        if pd.notna(expected_se):
            checks.append(("se", float(expected_se), float(grow["se"])))
        if pd.notna(expected_fdr):
            checks.append(("fdr_p", float(expected_fdr), float(grow["fdr_p"])))

        for field, expected, computed in checks:
            if abs(expected - computed) > TOL:
                issues.append(
                    {
                        "check": "group_effect",
                        "segment": seg,
                        "metric": metric,
                        "field": field,
                        "expected": expected,
                        "computed": computed,
                        "abs_diff": abs(expected - computed),
                    }
                )

        if int(grow["n_asd"]) != 58 or int(grow["n_td"]) != 78:
            issues.append(
                {
                    "check": "sample_size",
                    "segment": seg,
                    "metric": metric,
                    "expected_n_asd": 58,
                    "expected_n_td": 78,
                    "computed_n_asd": int(grow["n_asd"]),
                    "computed_n_td": int(grow["n_td"]),
                }
            )

    return issues


def validate_envelope_adjusted() -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not ENVELOPE_ADJ_PATH.exists():
        issues.append({"check": "envelope_adjusted", "issue": f"missing file {ENVELOPE_ADJ_PATH}"})
        return issues
    if not ENVELOPE_SRC_PATH.exists():
        issues.append({"check": "envelope_adjusted", "issue": f"missing upstream {ENVELOPE_SRC_PATH}"})
        return issues

    adj = pd.read_csv(ENVELOPE_ADJ_PATH)
    src = pd.read_csv(ENVELOPE_SRC_PATH)
    required = ["event_type", *ENVELOPE_COL_MAP.keys()]
    missing = set(required) - set(adj.columns)
    if missing:
        issues.append({"check": "envelope_adjusted", "issue": f"missing columns {sorted(missing)}"})
        return issues

    for _, row in adj.iterrows():
        ev = row["event_type"]
        up = src.loc[src["event_type"] == ev]
        if up.empty:
            issues.append({"check": "envelope_adjusted", "event_type": ev, "issue": "no upstream row"})
            continue
        up = up.iloc[0]
        for out_col, src_col in ENVELOPE_COL_MAP.items():
            diff = abs(float(row[out_col]) - float(up[src_col]))
            if diff > TOL:
                issues.append(
                    {
                        "check": "envelope_adjusted",
                        "event_type": ev,
                        "field": out_col,
                        "expected": float(up[src_col]),
                        "computed": float(row[out_col]),
                        "abs_diff": diff,
                    }
                )
    return issues


def write_report(
    subject_df: pd.DataFrame,
    group_df: pd.DataFrame,
    sync_issues: list[dict[str, Any]],
    envelope_issues: list[dict[str, Any]],
    passed: bool,
) -> None:
    alpha_rows = group_df[group_df["metric"] == "alpha_plv_isc"]
    lines = [
        "# S7 synchrony-control export validation",
        "",
        f"Subject file: `{SUBJECT_OUT.relative_to(PROJECT_ROOT)}`",
        f"Group effects file: `{GROUP_OUT.relative_to(PROJECT_ROOT)}`",
        f"Reference table: `{OLD_SYNCH_PATH.relative_to(PROJECT_ROOT)}` (within_group_loo rows only)",
        "",
        "## Upstream sources",
        "",
        f"- Aperiodic within-group LOO: `{AP_PATH.relative_to(PROJECT_ROOT)}` (`isc_z`)",
        f"- Envelope within-group LOO: `{ENV_PATH.relative_to(PROJECT_ROOT)}` (`isc_z`)",
        f"- Alpha PLV within-group LOO: `{PLV_PATH.relative_to(PROJECT_ROOT)}` (`isc_r`)",
        f"- Mechanism group tests: `{MECH_TESTS_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "## Cohort and scale",
        "",
        f"- Subject-level rows: {len(subject_df)} (expected 1224 = 136 subjects × 3 segments × 3 metrics)",
        f"- Unique subjects: {subject_df['subject_id'].nunique()} (expected 136)",
        f"- All metrics use `isc_definition = within_group_loo`",
        f"- Aperiodic-ISC & Envelope ISC on Fisher-z (`isc_z`); Alpha PLV ISC on PLV correlation (`isc_r`)",
        "",
        "## 1. Group means and TD−ASD effects vs legacy `s7_synchrony_controls.csv`",
        "",
    ]
    if not sync_issues:
        lines.append("All within_group_loo means, group effects, SE (where present), p, and fdr_p **match** legacy table (tol 1e-12).")
    else:
        lines.append("**FAILED** — mismatches detected:")
        lines.append("")
        lines.append("| check | segment | metric | field | expected | computed | |diff| |")
        lines.append("|-------|---------|--------|-------|----------|----------|-------|")
        for issue in sync_issues:
            lines.append(
                f"| {issue.get('check','')} | {issue.get('segment','')} | {issue.get('metric','')} | "
                f"{issue.get('field', issue.get('group', issue.get('issue','')))} | "
                f"{issue.get('expected','NA')} | {issue.get('computed','NA')} | {issue.get('abs_diff','NA')} |"
            )

    lines.extend(
        [
            "",
            "## 2. Sample sizes",
            "",
            "- Per segment × metric: n_asd = 58, n_td = 78 (movie synchrony-control cohort, n = 136)",
            "- Legacy group-level table stored n = 136 on each row; new tables report group-specific counts.",
            "",
            "## 3. Alpha PLV SE and 95% CI (Welch TD−ASD difference)",
            "",
            "| segment | group_effect | se | ci_low | ci_high | p |",
            "|---------|--------------|----|--------|---------|---|",
        ]
    )
    for _, r in alpha_rows.iterrows():
        lines.append(
            f"| {r['segment']} | {r['group_effect_td_minus_asd']:.12g} | {r['se']:.12g} | "
            f"{r['ci_low']:.12g} | {r['ci_high']:.12g} | {r['p']:.12g} |"
        )

    lines.extend(
        [
            "",
            "- Method: Welch unequal-variance t-test on subject-level `isc_r`; SE = sqrt(s²_ASD/n_ASD + s²_TD/n_TD);",
            "  95% CI = group_effect ± t_(0.975, df_Welch–Satterthwaite) × SE.",
            "",
            "## 4. FDR correction family",
            "",
            "- **Aperiodic-ISC** (`within_group_*`) and **Envelope ISC** (`envelope_*`):",
            "  `fdr_p` taken from `isc_mechanism_group_tests.csv`, Benjamini–Hochberg across **12** mechanism",
            "  comparisons (td_template, within_group, template_gap, envelope × mental/pain/neutral).",
            "- **Alpha PLV ISC**: not included in that 12-test family; `fdr_p` left empty (NaN), matching legacy",
            "  `s7_synchrony_controls.csv`. Raw two-sided Welch p-values reported.",
            "",
            "## 5. Envelope-adjusted analysis (`s7_envelope_adjusted.csv`)",
            "",
            f"- Validated file: `{ENVELOPE_ADJ_PATH.relative_to(PROJECT_ROOT)}`",
            f"- Upstream: `{ENVELOPE_SRC_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    if not envelope_issues:
        lines.append("Required fields present; all numeric values **match** upstream (tol 1e-12).")
        lines.append("")
        lines.append("Fields checked: event_type, pearson_r, shared_variance_pct, envelope_adjusted_group_beta_z,")
        lines.append("envelope_adjusted_group_se, envelope_adjusted_group_p, envelope_adjusted_group_fdr_p,")
        lines.append("partial_cohen_d, effect_retained_pct.")
    else:
        lines.append("**FAILED** — envelope-adjusted mismatches:")
        for issue in envelope_issues:
            lines.append(f"- {issue}")

    lines.extend(["", f"## Overall verdict: **{'PASS' if passed else 'FAIL'}**"])
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


def export_s7_synchrony_data() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    subject_df = build_subject_level()
    group_df = build_group_effects(subject_df)
    sync_issues = validate_against_old(subject_df, group_df)
    envelope_issues = validate_envelope_adjusted()
    passed = len(sync_issues) == 0 and len(envelope_issues) == 0

    write_report(subject_df, group_df, sync_issues, envelope_issues, passed)
    if not passed:
        print(f"Validation FAILED — see {REPORT_OUT}")
        for issue in sync_issues + envelope_issues:
            print(issue)
        return subject_df, group_df, False

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subject_df.to_csv(SUBJECT_OUT, index=False)
    group_df.to_csv(GROUP_OUT, index=False)
    write_report(subject_df, group_df, sync_issues, envelope_issues, True)
    return subject_df, group_df, True


def main() -> None:
    subject_df, group_df, ok = export_s7_synchrony_data()
    if not ok:
        raise SystemExit(1)
    print(f"Wrote {SUBJECT_OUT} ({len(subject_df)} rows)")
    print(f"Wrote {GROUP_OUT} ({len(group_df)} rows)")
    print(f"Wrote {REPORT_OUT}")
    print("Validation: PASS")


if __name__ == "__main__":
    main()
