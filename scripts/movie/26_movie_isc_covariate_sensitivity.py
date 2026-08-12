#!/usr/bin/env python
"""
154_movie_isc_covariate_sensitivity.py
--------------------------------------
Pre-submission sensitivity for Partly Cloudy TD-template Aperiodic-ISC:

Sensitivity A (priority):
  isc_z ~ C(group) + age_months + C(sex) + IQ_total
  separately for mental / pain / neutral; BH-FDR on Group p values.

Sensitivity B:
  + one movie-quality covariate (prefer n_overlap_points / valid windows).

Sensitivity B-extended (optional):
  + n_overlap_points + mean posterior spectral-fit R²
  (no invalid-channel proportion).

Outputs:
  outputs/tables/movie_isc_covariate_sensitivity/
  outputs/reports/movie_isc_covariate_sensitivity_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import save_csv  # noqa: E402

SEGMENTS = ("mental", "pain", "neutral")
SEGMENT_LABEL = {
    "mental": "mentalizing",
    "pain": "pain-related",
    "neutral": "neutral",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Movie Aperiodic-ISC covariate sensitivity")
    p.add_argument(
        "--isc-csv",
        type=str,
        default="derivatives/derivatives_task_movie/stats/aperiodic_isc/aperiodic_isc_td_template_subject_values.csv",
    )
    return p.parse_args()


def _load_master_demographics() -> pd.DataFrame:
    """Prefer registration master sex; fall back to flipped task_movie sex for movie-only IDs."""
    master = pd.read_csv(PROJECT_ROOT / "data" / "participants" / "participants.csv")
    master["subject_id"] = master["subject_id"].astype(str)
    master["sex"] = master["sex"].astype(str).str.upper().str.strip()
    master["group"] = master["group"].astype(str).str.upper()

    tm = pd.read_csv(PROJECT_ROOT / "data" / "participants" / "participants_task_movie.csv")
    tm["subject_id"] = tm["subject_id"].astype(str)
    tm["sex"] = tm["sex"].astype(str).str.upper().str.strip()
    tm["group"] = tm["group"].astype(str).str.upper()

    # task_movie sex is inverted vs master where both exist; flip for fallback IDs only.
    flip = {"M": "F", "F": "M"}
    rows = []
    for sid, row in tm.set_index("subject_id").iterrows():
        if sid in set(master["subject_id"]):
            m = master.set_index("subject_id").loc[sid]
            sex = str(m["sex"]).upper()
            age = m["age_months"]
            iq = m["IQ_total"]
            group = str(m["group"]).upper()
        else:
            sex = flip.get(str(row["sex"]).upper(), np.nan)
            age = row.get("age_months", np.nan)
            iq = row.get("IQ_total", np.nan)
            group = str(row["group"]).upper()
        rows.append(
            {
                "subject_id": sid,
                "group": group,
                "age_months": pd.to_numeric(age, errors="coerce"),
                "sex": sex,
                "IQ_total": pd.to_numeric(iq, errors="coerce"),
            }
        )
    demo = pd.DataFrame(rows)
    # Prefer master values for IDs present there
    m2 = master[["subject_id", "group", "age_months", "sex", "IQ_total"]].copy()
    demo = demo.set_index("subject_id")
    m2 = m2.set_index("subject_id")
    for col in ["group", "age_months", "sex", "IQ_total"]:
        demo.loc[m2.index.intersection(demo.index), col] = m2.loc[
            m2.index.intersection(demo.index), col
        ]
    return demo.reset_index()


def _load_movie_quality() -> pd.DataFrame:
    pre = pd.read_csv(
        PROJECT_ROOT / "derivatives" / "derivatives_task_movie" / "qc" / "preproc_summary.csv"
    )
    pre["subject_id"] = pre["subject_id"].astype(str)
    pre = pre.rename(columns={"usable_epochs": "movie_usable_epochs"})[
        ["subject_id", "movie_usable_epochs"]
    ]

    qc = pd.read_csv(
        PROJECT_ROOT
        / "derivatives"
        / "derivatives_task_movie"
        / "specparam"
        / "specparam_qc_summary_subject.csv"
    )
    qc["subject_id"] = qc["subject_id"].astype(str)
    qc = qc[
        [
            "subject_id",
            "mean_r_squared",
            "invalid_channel_ratio",
            "low_quality_subject",
        ]
    ].rename(columns={"mean_r_squared": "mean_movie_r_squared"})

    return pre.merge(qc, on="subject_id", how="outer")


def _group_term(res: Any) -> str:
    terms = [t for t in res.params.index if str(t).startswith("C(group)")]
    if not terms:
        raise ValueError("No group term in model")
    return str(terms[0])


def _td_minus_asd(res: Any) -> tuple[float, float, float, float, float, str]:
    term = _group_term(res)
    coef = float(res.params[term])
    se = float(res.bse[term])
    p = float(res.pvalues[term])
    ci = res.conf_int().loc[term]
    lo, hi = float(ci[0]), float(ci[1])
    if "T.ASD" in term:
        coef, lo, hi = -coef, -hi, -lo
    return coef, se, lo, hi, p, term


def _fit_segment_models(df: pd.DataFrame, formula: str, model_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seg in SEGMENTS:
        sub = df[df["event_type"] == seg].copy()
        # drop incomplete rows for formula vars
        needed = ["isc_z", "group", "age_months", "sex", "IQ_total"]
        if "n_overlap_points" in formula:
            needed.append("n_overlap_points")
        if "mean_movie_r_squared" in formula:
            needed.append("mean_movie_r_squared")
        if "movie_usable_epochs" in formula:
            needed.append("movie_usable_epochs")
        sub = sub.dropna(subset=needed)
        sub = sub[sub["sex"].isin(["M", "F"])].copy()
        if sub["group"].nunique() < 2 or len(sub) < 20:
            continue
        res = smf.ols(formula, data=sub).fit()
        beta, se, lo, hi, p, term = _td_minus_asd(res)
        # unadjusted Welch for reference
        asd = sub.loc[sub["group"] == "ASD", "isc_z"]
        td = sub.loc[sub["group"] == "TD", "isc_z"]
        welch_t, welch_p = stats.ttest_ind(td, asd, equal_var=False, nan_policy="omit")
        rows.append(
            {
                "model": model_name,
                "segment": seg,
                "segment_label": SEGMENT_LABEL[seg],
                "formula": formula,
                "n": int(res.nobs),
                "n_asd": int((sub["group"] == "ASD").sum()),
                "n_td": int((sub["group"] == "TD").sum()),
                "beta_td_minus_asd": beta,
                "se": se,
                "ci_low": lo,
                "ci_high": hi,
                "p_group": p,
                "term": term,
                "r_squared": float(res.rsquared),
                "asd_mean_isc_z": float(asd.mean()),
                "td_mean_isc_z": float(td.mean()),
                "welch_t": float(welch_t),
                "welch_p": float(welch_p),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    reject, q, _, _ = multipletests(out["p_group"].to_numpy(float), method="fdr_bh")
    out["fdr_q_group"] = q
    out["fdr_significant"] = reject
    return out


def _write_report(models: dict[str, pd.DataFrame], out_path: Path) -> None:
    lines = [
        "# Movie Aperiodic-ISC covariate sensitivity",
        "",
        "TD-template posterior Aperiodic-ISC (Partly Cloudy). "
        "Demographics from master `participants.csv` (sex corrected for movie-only IDs).",
        "",
        "## Models",
        "",
        "- **A (priority):** `isc_z ~ C(group) + age_months + C(sex) + IQ_total`",
        "- **B:** A + `n_overlap_points` (segment-wise valid ISC windows)",
        "- **B-extended:** A + `n_overlap_points` + `mean_movie_r_squared`",
        "- BH-FDR applied across the three segment Group p values within each model family.",
        "",
    ]
    for name, df in models.items():
        lines.append(f"## {name}")
        lines.append("")
        if df.empty:
            lines.append("_No results._")
            lines.append("")
            continue
        for _, r in df.iterrows():
            lines.append(
                f"- **{r['segment_label']}**: β(TD−ASD)={r['beta_td_minus_asd']:.4f}, "
                f"SE={r['se']:.4f}, 95% CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}], "
                f"p={r['p_group']:.4g}, FDR q={r['fdr_q_group']:.4g}, "
                f"n={int(r['n'])} ({int(r['n_asd'])}/{int(r['n_td'])}); "
                f"unadjusted Welch p={r['welch_p']:.4g}"
            )
        lines.append("")
        sig = df.loc[df["fdr_significant"], "segment_label"].tolist()
        lines.append(
            f"FDR-significant segments: {', '.join(sig) if sig else 'none'}."
        )
        lines.append("")

    a = models.get("Sensitivity_A_demographics")
    if a is not None and not a.empty:
        pain_n = a.loc[a["segment"].isin(["pain", "neutral"])]
        all_pos = bool((a["beta_td_minus_asd"] > 0).all())
        pain_n_sig = bool(pain_n["fdr_significant"].all()) if len(pain_n) == 2 else False
        mental = a.loc[a["segment"] == "mental"].iloc[0]
        lines.extend(
            [
                "## Interpretation for manuscript",
                "",
                f"- Direction TD > ASD across all three segments after demographics: **{all_pos}**.",
                f"- Pain + neutral FDR-significant after demographics: **{pain_n_sig}**.",
                f"- Mentalizing after demographics: β={mental['beta_td_minus_asd']:.4f}, "
                f"p={mental['p_group']:.4g}, FDR q={mental['fdr_q_group']:.4g} "
                f"({'significant' if mental['fdr_significant'] else 'not FDR-significant'}).",
                "- If mentalizing weakens after adjustment, interpret as broad reduced "
                "naturalistic alignment (consistent with non-significant group×segment).",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    isc_path = PROJECT_ROOT / args.isc_csv
    isc = pd.read_csv(isc_path)
    isc["subject_id"] = isc["subject_id"].astype(str)
    isc["group"] = isc["group"].astype(str).str.upper()
    isc["event_type"] = isc["event_type"].astype(str).str.lower().str.strip()
    isc = isc[isc["event_type"].isin(SEGMENTS)].copy()

    demo = _load_master_demographics()
    quality = _load_movie_quality()
    isc_group = isc[["subject_id", "group"]].drop_duplicates()
    demo_cov = demo.drop(columns=["group"], errors="ignore")
    df = isc.merge(demo_cov, on="subject_id", how="left")
    df = df.merge(quality, on="subject_id", how="left")
    # restore ISC analysis group labels
    df = df.drop(columns=["group"], errors="ignore").merge(isc_group, on="subject_id", how="left")
    df["group"] = df["group"].astype(str).str.upper()

    out_dir = PROJECT_ROOT / "outputs" / "tables" / "movie_isc_covariate_sensitivity"
    report_dir = PROJECT_ROOT / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # save analysis cohort snapshot
    snap_cols = [
        "subject_id",
        "group",
        "event_type",
        "isc_z",
        "n_overlap_points",
        "age_months",
        "sex",
        "IQ_total",
        "movie_usable_epochs",
        "mean_movie_r_squared",
        "invalid_channel_ratio",
    ]
    snap_cols = [c for c in snap_cols if c in df.columns]
    save_csv(df[snap_cols], out_dir / "analysis_cohort_long.csv")

    models = {
        "Sensitivity_A_demographics": _fit_segment_models(
            df,
            "isc_z ~ C(group) + age_months + C(sex) + IQ_total",
            "Sensitivity_A_demographics",
        ),
        "Sensitivity_B_plus_valid_windows": _fit_segment_models(
            df,
            "isc_z ~ C(group) + age_months + C(sex) + IQ_total + n_overlap_points",
            "Sensitivity_B_plus_valid_windows",
        ),
        "Sensitivity_B_extended_windows_r2": _fit_segment_models(
            df,
            "isc_z ~ C(group) + age_months + C(sex) + IQ_total + n_overlap_points + mean_movie_r_squared",
            "Sensitivity_B_extended_windows_r2",
        ),
    }

    for name, tab in models.items():
        save_csv(tab, out_dir / f"{name}.csv")

    combined = pd.concat(models.values(), ignore_index=True)
    save_csv(combined, out_dir / "all_models_summary.csv")

    report_path = report_dir / "movie_isc_covariate_sensitivity_report.md"
    _write_report(models, report_path)
    print(f"Wrote tables to {out_dir}")
    print(f"Wrote report to {report_path}")
    print(combined[["model", "segment_label", "beta_td_minus_asd", "p_group", "fdr_q_group", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()
