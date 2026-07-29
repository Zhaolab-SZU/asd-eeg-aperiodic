# Ideal release name: export_s4_development_predictions_with_ci.py
# Original path: scripts/export_s4_development_predictions_with_ci.py
# Note: S4 development CI trajectories
# This file is a copy for the public github_release/ bundle.

"""Export covariate-adjusted posterior exponent age trajectories with 95% mean CIs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.io_utils import save_csv  # noqa: E402
from src.nonlinear_age_sensitivity import (  # noqa: E402
    SPLINE_DF,
    _linear_formula,
    _median_covariates,
    _required_cols,
    _spline_interaction_formula,
)
from src.spectral_maturation_analysis import load_spectral_maturation_cohort  # noqa: E402

OUTCOME = "posterior_exponent"
N_AGE_GRID = 120
OUT_PATH = PROJECT_ROOT / "figure_source_data" / "supplementary" / "s4_development_predictions_with_ci.csv"
REFERENCE_PATH = PROJECT_ROOT / "figure_source_data" / "supplementary" / "s4_development_predictions.csv"


def _covariate_setting_label(med: dict[str, Any]) -> str:
    parts = [
        f"sex=mode({med['sex']})",
        f"IQ_total=median({med['IQ_total']:.6g})",
        f"usable_epochs=median({med['usable_epochs']:.6g})",
    ]
    if "mean_r_squared" in med:
        parts.append(f"mean_r_squared=median({med['mean_r_squared']:.6g})")
    return "; ".join(parts)


def _interaction_term(res) -> str:
    hits = [p for p in res.params.index if ":age_months" in p]
    if not hits:
        raise KeyError("No group × age_months interaction term found")
    return hits[0]


def export_predictions_with_ci() -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = load_config()
    deriv = Path(cfg["paths"]["derivatives_root"])
    df = load_spectral_maturation_cohort(cfg, deriv)

    req = [OUTCOME, *_required_cols(OUTCOME)]
    sub = df.dropna(subset=req).copy()
    age_min = float(sub["age_months"].min())
    age_max = float(sub["age_months"].max())
    n_total = int(len(sub))
    n_asd = int((sub["group"] == "ASD").sum())
    n_td = int((sub["group"] == "TD").sum())

    med = _median_covariates(sub, OUTCOME)
    cov_setting = _covariate_setting_label(med)
    ages = np.linspace(age_min, age_max, N_AGE_GRID)

    linear_formula = _linear_formula(OUTCOME)
    spline_formula = _spline_interaction_formula(OUTCOME, age_min, age_max, df=SPLINE_DF)

    model_specs = [
        ("linear_interaction", linear_formula),
        ("spline_interaction", spline_formula),
    ]

    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "n_total": n_total,
        "n_asd": n_asd,
        "n_td": n_td,
        "age_min": age_min,
        "age_max": age_max,
        "n_age_grid": N_AGE_GRID,
        "covariate_setting": cov_setting,
        "formulas": {},
        "interaction_beta": {},
        "ci_methods": {},
        "max_abs_diff_vs_reference": {},
    }

    for model_name, formula in model_specs:
        res = ols(formula, data=sub).fit()
        meta["formulas"][model_name] = formula
        if model_name == "linear_interaction":
            iterm = _interaction_term(res)
            meta["interaction_beta"][model_name] = float(res.params[iterm])
        meta["ci_methods"][model_name] = "statsmodels_get_prediction_mean_ci_alpha_0.05"

        for grp in ("ASD", "TD"):
            pred_df = pd.DataFrame({"age_months": ages, "group": grp, OUTCOME: 0.0, **med})
            sf = res.get_prediction(pred_df).summary_frame(alpha=0.05)
            for i, age in enumerate(ages):
                pred = float(sf["mean"].iloc[i])
                lo = float(sf["mean_ci_lower"].iloc[i])
                hi = float(sf["mean_ci_upper"].iloc[i])
                rows.append(
                    {
                        "model": model_name,
                        "group": grp,
                        "age_months": float(age),
                        "predicted_exponent": pred,
                        "ci_low": lo,
                        "ci_high": hi,
                        "ci_method": meta["ci_methods"][model_name],
                        "n_total": n_total,
                        "n_asd": n_asd,
                        "n_td": n_td,
                        "covariate_setting": cov_setting,
                        "model_formula": formula,
                    }
                )

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_csv(out, OUT_PATH)

    if REFERENCE_PATH.exists():
        ref = pd.read_csv(REFERENCE_PATH)
        ref_post = ref[ref["model"].isin(["linear_interaction", "spline_interaction"])].copy()
        merged = ref_post.merge(
            out[["model", "group", "age_months", "predicted_exponent"]],
            on=["model", "group", "age_months"],
            how="inner",
            suffixes=("_ref", "_new"),
        )
        if not merged.empty:
            diff = (merged["predicted_exponent_ref"] - merged["predicted_exponent_new"]).abs()
            for model_name in ("linear_interaction", "spline_interaction"):
                sub_diff = merged.loc[merged["model"] == model_name]
                if not sub_diff.empty:
                    d = (sub_diff["predicted_exponent_ref"] - sub_diff["predicted_exponent_new"]).abs()
                    meta["max_abs_diff_vs_reference"][model_name] = float(d.max())

    return out, meta


def validate_output(df: pd.DataFrame, meta: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    expected_n = meta["n_age_grid"] * 2 * 2
    if len(df) != expected_n:
        issues.append(f"row count {len(df)} != expected {expected_n}")

    for model_name in ("linear_interaction", "spline_interaction"):
        for grp in ("ASD", "TD"):
            sub = df[(df["model"] == model_name) & (df["group"] == grp)]
            if len(sub) != meta["n_age_grid"]:
                issues.append(f"incomplete grid for {model_name}/{grp}: n={len(sub)}")

    if not ((df["ci_low"] <= df["predicted_exponent"]) & (df["predicted_exponent"] <= df["ci_high"])).all():
        issues.append("CI bounds violated (ci_low > pred or pred > ci_high)")

    num = df.select_dtypes(include=[np.number])
    if not np.isfinite(num.to_numpy(dtype=float)).all():
        issues.append("NaN or infinite values present")

    beta = meta["interaction_beta"].get("linear_interaction")
    if beta is None or abs(beta - 0.0048) > 0.0005:
        issues.append(f"linear interaction beta={beta} not ≈ 0.0048/month")

    for model_name, max_diff in meta.get("max_abs_diff_vs_reference", {}).items():
        if max_diff > 1e-6:
            issues.append(f"{model_name} max abs diff vs reference = {max_diff:.6g} (>1e-6)")

    return issues


def main() -> None:
    out, meta = export_predictions_with_ci()
    issues = validate_output(out, meta)

    print(f"Wrote {OUT_PATH} ({len(out)} rows)")
    print("\nModel formulas:")
    for name, formula in meta["formulas"].items():
        print(f"  {name}: {formula}")
    print(f"\nCovariate setting: {meta['covariate_setting']}")
    print("\nCI method: statsmodels OLS get_prediction().summary_frame(alpha=0.05)")
    print("  -> mean_ci_lower / mean_ci_upper (fitted mean 95% CI, not prediction interval)")
    print("Bootstrap: not used (analytic mean CI available for both models)")
    print(f"\nLinear interaction beta(group×age): {meta['interaction_beta'].get('linear_interaction'):.6f} /month")
    if meta.get("max_abs_diff_vs_reference"):
        print("\nMax |diff| vs s4_development_predictions.csv:")
        for k, v in meta["max_abs_diff_vs_reference"].items():
            print(f"  {k}: {v:.3e}")

    if issues:
        print("\nValidation issues:")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)
    print("\nValidation: PASS")


if __name__ == "__main__":
    main()
