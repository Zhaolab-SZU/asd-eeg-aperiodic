# Ideal release name: recompute_s4_age_interaction_sensitivity.py
# Original path: scripts/recompute_s4_age_interaction_sensitivity.py
# Note: Recompute S4 age×group sensitivities
# This file is a copy for the public github_release/ bundle.

"""Recompute posterior exponent age × group interactions for sensitivity cohorts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols, rlm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.io_utils import save_csv  # noqa: E402
from src.spectral_maturation_analysis import load_spectral_maturation_cohort  # noqa: E402

OUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "spectral_maturation"
    / "age_group_interaction_sensitivity_cohorts.csv"
)

INTERACTION_TERM = "C(group)[T.TD]:age_months"
FORMULA = (
    "posterior_exponent ~ C(group) * age_months + C(sex) + IQ_total + usable_epochs + mean_r_squared"
)

FALLBACK = {
    "IQ-balanced matched": {"beta": 0.0048, "p": 0.046},
    "Strict-QC OLS": {"beta": 0.0024, "p": 0.287},
    "Strict-QC robust": {"beta": 0.0037, "p": 0.040},
}


def _parse_cohort_ids(supp: pd.DataFrame, cohort_key: str) -> set[str]:
    row = supp.loc[supp["cohort"] == cohort_key].iloc[0]
    asd = {s.strip() for s in str(row["subject_ids_asd"]).split(";") if s.strip()}
    td = {s.strip() for s in str(row["subject_ids_td"]).split(";") if s.strip()}
    return asd | td


def _fit_interaction(sub: pd.DataFrame, cohort_label: str, model_type: str) -> dict:
    req = ["posterior_exponent", "group", "age_months", "sex", "IQ_total", "usable_epochs", "mean_r_squared"]
    data = sub.dropna(subset=req).copy()
    if len(data) < 20:
        raise ValueError(f"{cohort_label}: insufficient n={len(data)}")

    if model_type == "OLS":
        res = ols(FORMULA, data=data).fit()
    elif model_type == "RLM":
        res = rlm(FORMULA, data=data, M=sm.robust.norms.HuberT()).fit()
    else:
        raise ValueError(model_type)

    if INTERACTION_TERM not in res.params.index:
        raise KeyError(f"Missing interaction term in {cohort_label} {model_type}")

    ci = res.conf_int().loc[INTERACTION_TERM]
    g = data["group"].astype(str).str.upper()
    return {
        "cohort": cohort_label,
        "model_type": model_type,
        "outcome": "posterior_exponent",
        "term": INTERACTION_TERM,
        "estimate": float(res.params[INTERACTION_TERM]),
        "se": float(res.bse[INTERACTION_TERM]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p": float(res.pvalues[INTERACTION_TERM]),
        "n_total": int(res.nobs),
        "n_asd": int((g == "ASD").sum()),
        "n_td": int((g == "TD").sum()),
    }


def main() -> None:
    cfg = load_config()
    deriv = Path(cfg["paths"]["derivatives_root"])
    df = load_spectral_maturation_cohort(cfg, deriv)

    supp = pd.read_csv(
        PROJECT_ROOT / "outputs" / "tables" / "manuscript0621" / "supp_table_s1_participant_characteristics.csv"
    )
    cohort_specs = [
        ("IQ-balanced matched", "IQ-balanced matched", ["OLS"]),
        ("Matched strict specparam-QC", "Strict-QC OLS", ["OLS"]),
        ("Matched strict specparam-QC", "Strict-QC robust", ["RLM"]),
    ]

    rows: list[dict] = []
    for supp_key, label, model_types in cohort_specs:
        ids = _parse_cohort_ids(supp, supp_key)
        sub = df[df["subject_id"].astype(str).isin(ids)].copy()
        for mt in model_types:
            rows.append(_fit_interaction(sub, label, mt))

    out = pd.DataFrame(rows)
    save_csv(out, OUT_PATH)

    print(f"Wrote {OUT_PATH}")
    print("\nComparison vs manuscript fallback (bp_figures/io.py):")
    for _, r in out.iterrows():
        fb = FALLBACK.get(r["cohort"], {})
        beta_ok = abs(float(r["estimate"]) - fb.get("beta", np.nan)) < 0.002 if fb else True
        p_ok = abs(float(r["p"]) - fb.get("p", np.nan)) < 0.02 if fb else True
        mark = "OK" if beta_ok and p_ok else "CHECK"
        print(
            f"  [{mark}] {r['cohort']} ({r['model_type']}): "
            f"beta={r['estimate']:.4f} p={r['p']:.3f} "
            f"(fallback beta={fb.get('beta', 'NA')}, p={fb.get('p', 'NA')})"
        )


if __name__ == "__main__":
    main()
