# Ideal release name: hbn_isc_fsiq_adjusted.py
# Original path: scripts/hbn/hbn_isc_fsiq_adjusted.py
# Note: Matched-HBN FSIQ-adjusted Aperiodic-ISC OLS (statistical_tables)
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
hbn_isc_fsiq_adjusted.py
------------------------
Residual-FSIQ-adjusted sensitivity for matched-HBN posterior Aperiodic-ISC.

Fits OLS:  isc_z ~ C(group) + IQ_total
for sliding-window and non-overlapping 2-s epoch analyses on the age/sex/IQ
caliper-matched cohort (n = 119 ASD + 119 TD).

Outputs:
  outputs/tables/hbn_fsiq_adjusted/hbn_isc_fsiq_adjusted_ols.csv
  github_release/statistical_tables/hbn_fsiq_adjusted_isc.csv  (if present)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols

def _default_project_root() -> Path:
    """Prefer the full analysis repo when this file lives under github_release/."""
    here = Path(__file__).resolve()
    # scripts/hbn/ → parents[2] is repo or github_release root
    bundle_or_repo = here.parents[2]
    parent = bundle_or_repo.parent
    for cand in (parent, bundle_or_repo):
        if (cand / "derivatives" / "hbn_external_movie").exists() or (
            cand / "figure_source_data" / "supplementary" / "s8_hbn_movie_subjects.csv"
        ).exists():
            return cand
    return bundle_or_repo


PROJECT_ROOT = _default_project_root()
sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _resolve_isc_path(root: Path) -> Path:
    candidates = [
        root / "figure_source_data" / "supplementary" / "s8_hbn_movie_subjects.csv",
        root / "github_release" / "figure_source_data" / "supplementary" / "s8_hbn_movie_subjects.csv",
        root / "outputs" / "figure_source_data" / "supplementary" / "s8_hbn_movie_subjects.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Missing s8_hbn_movie_subjects.csv (tried figure_source_data/supplementary/)."
    )


def _resolve_participants_path(root: Path) -> Path:
    p = (
        root
        / "derivatives"
        / "hbn_external_movie"
        / "replication"
        / "matched"
        / "participants_matched.csv"
    )
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _td_minus_asd_from_ols(res) -> tuple[float, float, float, float, float]:
    """Return beta(TD−ASD), SE, CI low, CI high, p."""
    term = [t for t in res.params.index if str(t).startswith("C(group)")][0]
    coef = float(res.params[term])
    se = float(res.bse[term])
    p = float(res.pvalues[term])
    ci = res.conf_int().loc[term]
    lo, hi = float(ci[0]), float(ci[1])
    if "T.ASD" in str(term):
        coef = -coef
        lo, hi = -hi, -lo
    return coef, se, lo, hi, p


def fit_fsiq_adjusted(
    isc: pd.DataFrame,
    demo: pd.DataFrame,
    *,
    analyses: list[str] | None = None,
) -> pd.DataFrame:
    analyses = analyses or ["sliding_window", "nonoverlapping_2s_epoch"]
    demo = demo.copy()
    demo["subject_id"] = demo["subject_id"].astype(str)
    keep = ["subject_id", "IQ_total", "age_months", "sex"]
    keep = [c for c in keep if c in demo.columns]
    demo = demo[keep].drop_duplicates("subject_id")

    rows: list[dict] = []
    for analysis in analyses:
        sub = isc[isc["analysis"] == analysis].merge(demo, on="subject_id", how="left")
        sub = sub.dropna(subset=["isc_z", "group", "IQ_total"]).copy()
        sub["group"] = sub["group"].astype(str).str.upper()
        if sub["group"].nunique() < 2:
            raise RuntimeError(f"Need both ASD and TD for analysis={analysis}")

        m = ols("isc_z ~ C(group) + IQ_total", data=sub).fit()
        beta, se, lo, hi, p = _td_minus_asd_from_ols(m)

        # Optional fully covariate-adjusted note (not primary)
        beta_full = p_full = float("nan")
        if {"age_months", "sex"}.issubset(sub.columns):
            sub2 = sub.dropna(subset=["age_months", "sex"])
            if sub2["group"].nunique() == 2 and len(sub2) >= 20:
                m2 = ols(
                    "isc_z ~ C(group) + IQ_total + age_months + C(sex)", data=sub2
                ).fit()
                beta_full, _, _, _, p_full = _td_minus_asd_from_ols(m2)

        asd = sub.loc[sub["group"] == "ASD", "isc_z"]
        td = sub.loc[sub["group"] == "TD", "isc_z"]
        _, unadj_p = stats.ttest_ind(td, asd, equal_var=False)

        rows.append(
            {
                "analysis": analysis,
                "n": int(len(sub)),
                "n_asd": int((sub["group"] == "ASD").sum()),
                "n_td": int((sub["group"] == "TD").sum()),
                "beta_td_minus_asd": beta,
                "se": se,
                "ci_low": lo,
                "ci_high": hi,
                "p": p,
                "iq_missing": 0,
                "beta_full_cov": beta_full,
                "p_full_cov": p_full,
                "unadj_delta": float(td.mean() - asd.mean()),
                "unadj_p": float(unadj_p),
                "mean_iq_asd": float(sub.loc[sub["group"] == "ASD", "IQ_total"].mean()),
                "mean_iq_td": float(sub.loc[sub["group"] == "TD", "IQ_total"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_public_summary(ols_df: pd.DataFrame, out_path: Path) -> None:
    out = ols_df[
        [
            "analysis",
            "n",
            "n_asd",
            "n_td",
            "beta_td_minus_asd",
            "se",
            "ci_low",
            "ci_high",
            "p",
            "unadj_delta",
            "unadj_p",
            "mean_iq_asd",
            "mean_iq_td",
        ]
    ].copy()
    out.insert(0, "cohort", "matched_HBN")
    out.insert(1, "roi", "posterior")
    out.insert(2, "model", "isc_z ~ C(group) + IQ_total")
    out["note"] = "Residual-FSIQ-adjusted sensitivity; beta = TD - ASD"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HBN matched-cohort FSIQ-adjusted Aperiodic-ISC OLS"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Analysis repository root (default: inferred)",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()

    isc = pd.read_csv(_resolve_isc_path(root))
    isc["subject_id"] = isc["subject_id"].astype(str)
    part = pd.read_csv(_resolve_participants_path(root))

    ols_df = fit_fsiq_adjusted(isc, part)

    out_dir = root / "outputs" / "tables" / "hbn_fsiq_adjusted"
    out_dir.mkdir(parents=True, exist_ok=True)
    ols_path = out_dir / "hbn_isc_fsiq_adjusted_ols.csv"
    ols_df.to_csv(ols_path, index=False)

    public_paths = [
        root / "github_release" / "statistical_tables" / "hbn_fsiq_adjusted_isc.csv",
        root / "statistical_tables" / "hbn_fsiq_adjusted_isc.csv",
        root / "outputs" / "tables" / "statistical_summary" / "hbn_fsiq_adjusted_isc.csv",
    ]
    for p in public_paths:
        # Write when parent exists or is the canonical github_release folder
        if p.parent.name in {"statistical_tables", "statistical_summary"}:
            if p.parent.exists() or "github_release" in p.parts:
                write_public_summary(ols_df, p)

    print(f"Wrote {ols_path}")
    for _, r in ols_df.iterrows():
        print(
            f"  {r['analysis']}: β={r['beta_td_minus_asd']:.3f}, "
            f"SE={r['se']:.3f}, 95% CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}], "
            f"p={r['p']:.3g} (n_asd={int(r['n_asd'])}, n_td={int(r['n_td'])})"
        )


if __name__ == "__main__":
    main()
