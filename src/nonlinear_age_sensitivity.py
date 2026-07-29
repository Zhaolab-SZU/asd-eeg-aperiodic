"""Nonlinear age sensitivity: spline vs linear group × age interaction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

from src.extension_analysis import COL_ASD, COL_GRAY, COL_TD, save_extension_figure
from src.io_utils import save_csv
from src.normative_analysis import _spline_age_term
from src.spectral_maturation_analysis import (
    COVARIATES_EXPONENT,
    OUTCOME_LABELS,
    load_spectral_maturation_cohort,
)

logger = logging.getLogger(__name__)

OUTCOMES = ["global_exponent", "posterior_exponent"]
SPLINE_DF = 3


def _covariates(outcome: str) -> str:
    return COVARIATES_EXPONENT if "exponent" in outcome else "C(sex) + IQ_total + usable_epochs"


def _required_cols(outcome: str) -> list[str]:
    cols = ["group", "age_months", "sex", "IQ_total", "usable_epochs"]
    if "exponent" in outcome:
        cols.append("mean_r_squared")
    return cols


def _linear_formula(outcome: str) -> str:
    return f"{outcome} ~ C(group) * age_months + {_covariates(outcome)}"


def _spline_interaction_formula(outcome: str, age_min: float, age_max: float, df: int = SPLINE_DF) -> str:
    spline = _spline_age_term(age_min, age_max, df=df)
    return f"{outcome} ~ C(group) * {spline} + {_covariates(outcome)}"


def _spline_main_formula(outcome: str, age_min: float, age_max: float, df: int = SPLINE_DF) -> str:
    spline = _spline_age_term(age_min, age_max, df=df)
    return f"{outcome} ~ C(group) + {spline} + {_covariates(outcome)}"


def _median_covariates(sub: pd.DataFrame, outcome: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sex": sub["sex"].mode().iloc[0],
        "IQ_total": float(sub["IQ_total"].median()),
        "usable_epochs": float(sub["usable_epochs"].median()),
    }
    if "exponent" in outcome:
        row["mean_r_squared"] = float(sub["mean_r_squared"].median())
    return row


def fit_model_comparison(df: pd.DataFrame, age_min: float, age_max: float) -> pd.DataFrame:
    """Linear vs spline-main vs spline-interaction model comparison."""
    rows: list[dict[str, Any]] = []
    specs = [
        ("linear_interaction", lambda o: _linear_formula(o)),
        ("spline_main_only", lambda o: _spline_main_formula(o, age_min, age_max)),
        ("spline_interaction", lambda o: _spline_interaction_formula(o, age_min, age_max)),
    ]

    for outcome in OUTCOMES:
        req = [outcome, *_required_cols(outcome)]
        sub = df.dropna(subset=req)
        if len(sub) < 30:
            logger.warning("样本不足，跳过 %s (n=%d)", outcome, len(sub))
            continue

        fitted: dict[str, Any] = {}
        for model_name, formula_fn in specs:
            formula = formula_fn(outcome)
            try:
                res = ols(formula, data=sub).fit()
                fitted[model_name] = res
                rows.append({
                    "outcome": outcome,
                    "model": model_name,
                    "formula": formula,
                    "n": int(res.nobs),
                    "n_ASD": int((sub["group"] == "ASD").sum()),
                    "n_TD": int((sub["group"] == "TD").sum()),
                    "r_squared": float(res.rsquared),
                    "aic": float(res.aic),
                    "bic": float(res.bic),
                })
            except Exception as exc:
                logger.warning("模型 %s / %s 拟合失败: %s", outcome, model_name, exc)

        lin = fitted.get("linear_interaction")
        spl_main = fitted.get("spline_main_only")
        spl_int = fitted.get("spline_interaction")

        if lin is not None and spl_main is not None:
            f_stat, p_val, _ = spl_main.compare_f_test(lin)
            rows.append({
                "outcome": outcome,
                "model": "compare_spline_main_vs_linear",
                "n": int(lin.nobs),
                "delta_aic": float(spl_main.aic - lin.aic),
                "compare_F": float(f_stat),
                "compare_p": float(p_val),
            })
        if lin is not None and spl_int is not None:
            f_stat, p_val, _ = spl_int.compare_f_test(lin)
            rows.append({
                "outcome": outcome,
                "model": "compare_spline_interaction_vs_linear",
                "n": int(lin.nobs),
                "delta_aic": float(spl_int.aic - lin.aic),
                "compare_F": float(f_stat),
                "compare_p": float(p_val),
            })
        if spl_main is not None and spl_int is not None:
            f_stat, p_val, _ = spl_int.compare_f_test(spl_main)
            rows.append({
                "outcome": outcome,
                "model": "compare_spline_interaction_vs_spline_main",
                "n": int(spl_main.nobs),
                "delta_aic": float(spl_int.aic - spl_main.aic),
                "compare_F": float(f_stat),
                "compare_p": float(p_val),
            })

    return pd.DataFrame(rows)


def compute_group_difference_curve(
    df: pd.DataFrame,
    outcome: str,
    age_min: float,
    age_max: float,
    model_name: str = "spline_interaction",
) -> pd.DataFrame:
    """Predicted TD − ASD difference across age grid."""
    req = [outcome, *_required_cols(outcome)]
    sub = df.dropna(subset=req)
    if model_name == "linear_interaction":
        formula = _linear_formula(outcome)
    else:
        formula = _spline_interaction_formula(outcome, age_min, age_max)

    res = ols(formula, data=sub).fit()
    med = _median_covariates(sub, outcome)
    ages = np.linspace(age_min, age_max, 120)
    rows: list[dict[str, Any]] = []

    for age in ages:
        base = {"age_months": age, **med}
        asd_df = pd.DataFrame([{**base, "group": "ASD", outcome: 0.0}])
        td_df = pd.DataFrame([{**base, "group": "TD", outcome: 0.0}])
        asd_pred = float(res.predict(asd_df).iloc[0])
        td_pred = float(res.predict(td_df).iloc[0])
        diff_frame = res.get_prediction(
            pd.concat([asd_df.assign(group="ASD"), td_df.assign(group="TD")], ignore_index=True)
        ).summary_frame(alpha=0.05)
        rows.append({
            "outcome": outcome,
            "model": model_name,
            "age_months": float(age),
            "ASD_pred": asd_pred,
            "TD_pred": td_pred,
            "TD_minus_ASD": td_pred - asd_pred,
        })

    return pd.DataFrame(rows)


def plot_nonlinear_age_trends(
    df: pd.DataFrame,
    age_min: float,
    age_max: float,
    out_base: Path,
) -> None:
    """Figure S: nonlinear age trends of global/posterior exponent by group."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))

    for ax, outcome in zip(axes, OUTCOMES):
        req = [outcome, *_required_cols(outcome)]
        sub = df.dropna(subset=req)
        if len(sub) < 30:
            ax.set_visible(False)
            continue

        lin_formula = _linear_formula(outcome)
        spl_formula = _spline_interaction_formula(outcome, age_min, age_max)
        lin_res = ols(lin_formula, data=sub).fit()
        spl_res = ols(spl_formula, data=sub).fit()
        med = _median_covariates(sub, outcome)
        ages = np.linspace(age_min, age_max, 120)

        for grp, color in [("ASD", COL_ASD), ("TD", COL_TD)]:
            for res, ls, lw, alpha in [
                (lin_res, "--", 1.2, 0.55),
                (spl_res, "-", 2.0, 1.0),
            ]:
                pred_df = pd.DataFrame({"age_months": ages, "group": grp, **med})
                pred = res.get_prediction(pred_df).summary_frame(alpha=0.05)
                label = grp if ls == "-" else None
                ax.plot(ages, pred["mean"], color=color, ls=ls, lw=lw, alpha=alpha, label=label)
                if ls == "-":
                    ax.fill_between(
                        ages,
                        pred["mean_ci_lower"],
                        pred["mean_ci_upper"],
                        color=color,
                        alpha=0.15,
                        linewidth=0,
                    )
            pts = sub[sub["group"] == grp]
            ax.scatter(
                pts["age_months"],
                pts[outcome],
                s=16,
                alpha=0.45,
                color=color,
                edgecolors="none",
            )

        _, p_cmp, _ = spl_res.compare_f_test(lin_res)
        label = OUTCOME_LABELS.get(outcome, outcome)
        ax.set_title(f"{label}\nspline vs linear p = {p_cmp:.3f}")
        ax.set_xlabel("Age (months)")
        ax.set_ylabel(label.split("(")[0].strip())
        ax.axvline(72, color=COL_GRAY, ls=":", lw=1.0, alpha=0.75)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.text(
        0.5,
        0.01,
        "Solid = spline interaction (df=3); dashed = linear interaction. "
        "Covariates at median/mode. Cross-sectional sensitivity analysis.",
        ha="center",
        fontsize=7.5,
        color=COL_GRAY,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    save_extension_figure(fig, out_base)


def plot_group_difference_by_age(
    diff_curves: pd.DataFrame,
    comparisons: pd.DataFrame,
    out_base: Path,
) -> None:
    """TD − ASD adjusted difference across age (linear vs spline)."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for ax, outcome in zip(axes, OUTCOMES):
        sub = diff_curves[diff_curves["outcome"] == outcome]
        if sub.empty:
            ax.set_visible(False)
            continue
        for model_name, ls, color in [
            ("linear_interaction", "--", COL_GRAY),
            ("spline_interaction", "-", "#2B2B2B"),
        ]:
            msub = sub[sub["model"] == model_name]
            if msub.empty:
                continue
            ax.plot(msub["age_months"], msub["TD_minus_ASD"], ls=ls, lw=1.8, color=color, label=model_name)

        cmp_row = comparisons[
            (comparisons["outcome"] == outcome)
            & (comparisons["model"] == "compare_spline_interaction_vs_linear")
        ]
        p_cmp = float(cmp_row.iloc[0]["compare_p"]) if not cmp_row.empty else np.nan
        label = OUTCOME_LABELS.get(outcome, outcome)
        ax.axhline(0, color=COL_GRAY, ls="-", lw=0.6, alpha=0.5)
        ax.axvline(72, color=COL_GRAY, ls=":", lw=1.0, alpha=0.75)
        ax.set_title(f"Adjusted TD − ASD: {label.split('(')[0].strip()}\nspline vs linear p = {p_cmp:.3f}")
        ax.set_xlabel("Age (months)")
        ax.set_ylabel("TD − ASD (adjusted)")

    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save_extension_figure(fig, out_base)


def build_report_markdown(
    comparisons: pd.DataFrame,
    diff_curves: pd.DataFrame,
    n_total: int,
    n_asd: int,
    n_td: int,
    age_min: float,
    age_max: float,
) -> str:
    lines = [
        "# Nonlinear age sensitivity (spline vs linear interaction)",
        "",
        f"**Cohort:** main QC-passed, *N* = {n_total} (ASD = {n_asd}, TD = {n_td}).",
        f"**Age range:** {age_min:.0f}–{age_max:.0f} months; spline df = {SPLINE_DF}.",
        "",
        "> Supplementary sensitivity analysis. Cross-sectional; does not imply longitudinal development.",
        "",
        "## Model comparison",
        "",
        "| Outcome | Comparison | ΔAIC | *F* | *p* |",
        "|---------|------------|------|-----|-----|",
    ]

    for outcome in OUTCOMES:
        for comp in [
            "compare_spline_interaction_vs_linear",
            "compare_spline_interaction_vs_spline_main",
        ]:
            hit = comparisons[(comparisons["outcome"] == outcome) & (comparisons["model"] == comp)]
            if hit.empty:
                continue
            r = hit.iloc[0]
            lines.append(
                f"| {OUTCOME_LABELS.get(outcome, outcome)} | {comp} | "
                f"{r.get('delta_aic', float('nan')):.2f} | "
                f"{r.get('compare_F', float('nan')):.2f} | "
                f"{r.get('compare_p', float('nan')):.4f} |"
            )

    lines.extend(["", "## Interpretation notes", ""])

    for outcome in OUTCOMES:
        sub = diff_curves[
            (diff_curves["outcome"] == outcome) & (diff_curves["model"] == "spline_interaction")
        ]
        if sub.empty:
            continue
        peak_idx = sub["TD_minus_ASD"].idxmax()
        peak = sub.loc[peak_idx]
        low = sub[sub["age_months"] <= 72]["TD_minus_ASD"].mean()
        high = sub[sub["age_months"] > 72]["TD_minus_ASD"].mean()
        lines.append(
            f"- **{OUTCOME_LABELS.get(outcome, outcome)}:** "
            f"mean adjusted TD−ASD ≤72 mo = {low:.3f}; >72 mo = {high:.3f}; "
            f"spline peak ≈ {peak['TD_minus_ASD']:.3f} at {peak['age_months']:.0f} mo."
        )

    lines.extend([
        "",
        "**Outputs**",
        "",
        "- `outputs/tables/nonlinear_age_sensitivity/model_comparison.csv`",
        "- `outputs/tables/nonlinear_age_sensitivity/group_difference_by_age.csv`",
        "- `outputs/figures/nonlinear_age_sensitivity/fig_nonlinear_age_trends_by_group`",
        "- `outputs/figures/nonlinear_age_sensitivity/fig_adjusted_group_difference_by_age`",
    ])
    return "\n".join(lines)


def run_nonlinear_age_sensitivity(cfg: dict[str, Any]) -> dict[str, Path]:
    deriv = Path(cfg["paths"]["derivatives_root"])
    outputs = Path(cfg["paths"]["outputs_root"])
    tables_dir = outputs / "tables" / "nonlinear_age_sensitivity"
    fig_dir = outputs / "figures" / "nonlinear_age_sensitivity"
    reports_dir = outputs / "reports"
    for d in (tables_dir, fig_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = load_spectral_maturation_cohort(cfg, deriv)
    age_min = float(df["age_months"].min())
    age_max = float(df["age_months"].max())
    n_total = len(df)
    n_asd = int((df["group"] == "ASD").sum())
    n_td = int((df["group"] == "TD").sum())
    logger.info("Nonlinear age 队列 n=%d (ASD=%d, TD=%d), age %.0f–%.0f", n_total, n_asd, n_td, age_min, age_max)

    comparisons = fit_model_comparison(df, age_min, age_max)
    save_csv(comparisons, tables_dir / "model_comparison.csv")

    diff_parts = []
    for model_name in ("linear_interaction", "spline_interaction"):
        for outcome in OUTCOMES:
            diff_parts.append(
                compute_group_difference_curve(df, outcome, age_min, age_max, model_name=model_name)
            )
    diff_curves = pd.concat(diff_parts, ignore_index=True)
    save_csv(diff_curves, tables_dir / "group_difference_by_age.csv")

    plot_nonlinear_age_trends(df, age_min, age_max, fig_dir / "fig_nonlinear_age_trends_by_group")
    plot_group_difference_by_age(diff_curves, comparisons, fig_dir / "fig_adjusted_group_difference_by_age")

    report = build_report_markdown(comparisons, diff_curves, n_total, n_asd, n_td, age_min, age_max)
    report_path = reports_dir / "nonlinear_age_sensitivity_report.md"
    report_path.write_text(report, encoding="utf-8")

    return {"tables": tables_dir, "figures": fig_dir, "report": report_path}
