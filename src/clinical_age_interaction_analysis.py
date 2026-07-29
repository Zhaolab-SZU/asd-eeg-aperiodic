"""ASD 内：posterior exponent 与症状 × 年龄交互分析。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols

from src.extension_analysis import COL_ASD, COL_GRAY, PLOT_RC, save_extension_figure
from src.io_utils import save_csv
from src.spectral_maturation_analysis import (
    COVARIATES_EXPONENT,
    load_spectral_maturation_cohort,
)
from src.stats_utils import model_results_to_row, run_ols, spearman_correlation

logger = logging.getLogger(__name__)

OUTCOME = "posterior_exponent"
CLINICAL_VARS = [
    "ADOS_total",
    "ADOS_SA",
    "ADOS_RRB",
    "SRS_total",
]
STRATUM_COV = "C(sex) + IQ_total + usable_epochs + mean_r_squared"
AGE_STRATA = [
    ("tertile_young", "young", lambda s: s["age_bin"] == "young"),
    ("tertile_mid", "mid", lambda s: s["age_bin"] == "mid"),
    ("tertile_old", "old", lambda s: s["age_bin"] == "old"),
    ("below_72mo", "<72 months", lambda s: s["age_months"] < 72),
    ("at_or_above_72mo", "≥72 months", lambda s: s["age_months"] >= 72),
]


def _residualize(y: pd.Series, cov_df: pd.DataFrame, formula_rhs: str) -> np.ndarray:
    data = cov_df.copy()
    data["_y"] = pd.to_numeric(y, errors="coerce")
    sub = data.dropna(subset=["_y"])
    if len(sub) < 8:
        return np.full(len(y), np.nan)
    model = ols(f"_y ~ {formula_rhs}", data=sub).fit()
    resid = pd.Series(np.nan, index=y.index, dtype=float)
    resid.loc[sub.index] = model.resid
    return resid.to_numpy(dtype=float)


def partial_spearman(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    cov_formula: str,
) -> dict[str, Any]:
    req = [x_col, y_col, "sex", "IQ_total", "usable_epochs", "mean_r_squared"]
    sub = df.dropna(subset=req).copy()
    if len(sub) < 8:
        return {"partial_rho": np.nan, "pvalue": np.nan, "n": len(sub)}
    rx = _residualize(sub[x_col], sub, cov_formula)
    ry = _residualize(sub[y_col], sub, cov_formula)
    mask = np.isfinite(rx) & np.isfinite(ry)
    if mask.sum() < 8:
        return {"partial_rho": np.nan, "pvalue": np.nan, "n": int(mask.sum())}
    rho, p = stats.spearmanr(rx[mask], ry[mask])
    return {"partial_rho": float(rho), "pvalue": float(p), "n": int(mask.sum())}


def load_asd_posterior_cohort(cfg: dict[str, Any]) -> pd.DataFrame:
    deriv = Path(cfg["paths"]["derivatives_root"])
    df = load_spectral_maturation_cohort(cfg, deriv)
    asd = df[df["group"] == "ASD"].copy()
    asd["age_bin"] = pd.qcut(
        asd["age_months"],
        3,
        labels=["young", "mid", "old"],
        duplicates="drop",
    )
    return asd.reset_index(drop=True)


def fit_symptom_age_interactions(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    req_base = [OUTCOME, "age_months", "sex", "IQ_total", "usable_epochs", "mean_r_squared"]
    for symptom in CLINICAL_VARS:
        if symptom not in df.columns:
            continue
        sub = df.dropna(subset=[symptom, *req_base])
        if len(sub) < 15:
            logger.warning("跳过 %s：完整样本 n=%d", symptom, len(sub))
            continue
        formula = f"{OUTCOME} ~ {symptom} * age_months + {COVARIATES_EXPONENT}"
        res = run_ols(formula, sub)
        for row in model_results_to_row(res, "symptom_x_age", OUTCOME):
            row["symptom"] = symptom
            row["formula"] = formula
            row["n_asd"] = int(res.nobs)
            rows.append(row)
    return pd.DataFrame(rows)


def compute_simple_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """在样本年龄 25/50/75 分位，提取 symptom 对 posterior exponent 的简单斜率。"""
    rows: list[dict[str, Any]] = []
    req_base = [OUTCOME, "age_months", "sex", "IQ_total", "usable_epochs", "mean_r_squared"]
    age_mean = float(df["age_months"].mean())
    age_points = {
        "p25": float(df["age_months"].quantile(0.25)),
        "p50": float(df["age_months"].quantile(0.50)),
        "p75": float(df["age_months"].quantile(0.75)),
    }

    for symptom in CLINICAL_VARS:
        if symptom not in df.columns:
            continue
        sub = df.dropna(subset=[symptom, *req_base]).copy()
        if len(sub) < 15:
            continue
        sub["age_c"] = sub["age_months"] - age_mean
        formula = f"{OUTCOME} ~ {symptom} * age_c + {COVARIATES_EXPONENT}"
        res = run_ols(formula, sub)
        b_sym = float(res.params.get(symptom, np.nan))
        b_int = float(res.params.get(f"{symptom}:age_c", np.nan))
        int_p = float(res.pvalues.get(f"{symptom}:age_c", np.nan))
        for label, age_val in age_points.items():
            age_c = age_val - age_mean
            slope = b_sym + b_int * age_c if np.isfinite(b_sym) and np.isfinite(b_int) else np.nan
            rows.append({
                "symptom": symptom,
                "age_label": label,
                "age_months": age_val,
                "symptom_slope": slope,
                "interaction_coef": b_int,
                "interaction_p": int_p,
                "n_asd": int(res.nobs),
            })
    return pd.DataFrame(rows)


def fit_age_stratified_partial_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    req_base = [OUTCOME, "sex", "IQ_total", "usable_epochs", "mean_r_squared"]
    for symptom in CLINICAL_VARS:
        if symptom not in df.columns:
            continue
        for stratum_id, stratum_label, mask_fn in AGE_STRATA:
            sub = df.dropna(subset=[symptom, *req_base]).copy()
            if stratum_id.startswith("tertile"):
                sub = sub.dropna(subset=["age_bin"])
            sub = sub[mask_fn(sub)].copy()
            if len(sub) < 10:
                continue
            pc = partial_spearman(sub, symptom, OUTCOME, STRATUM_COV)
            raw = spearman_correlation(sub[symptom], sub[OUTCOME])
            rows.append({
                "symptom": symptom,
                "stratum_id": stratum_id,
                "stratum_label": stratum_label,
                "partial_rho": pc["partial_rho"],
                "partial_p": pc["pvalue"],
                "raw_rho": raw["rho"],
                "raw_p": raw["pvalue"],
                "n_asd": pc["n"],
                "age_min": float(sub["age_months"].min()),
                "age_max": float(sub["age_months"].max()),
            })
    return pd.DataFrame(rows)


def _plot_ados_strata(df: pd.DataFrame, out_base: Path) -> None:
    symptom = "ADOS_total"
    sub = df.dropna(subset=[symptom, OUTCOME, "age_bin"])
    if len(sub) < 15:
        logger.warning("ADOS 分层散点图样本不足，跳过")
        return

    plt.rcParams.update(PLOT_RC)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    colors = {"young": "#3182ce", "mid": "#805ad5", "old": "#d69e2e"}
    for lab, g in sub.groupby("age_bin", observed=True):
        if len(g) < 5:
            continue
        color = colors.get(str(lab), COL_GRAY)
        ax.scatter(g[symptom], g[OUTCOME], alpha=0.55, s=24, color=color, label=str(lab))
        sl, ic, _, _, _ = stats.linregress(g[symptom], g[OUTCOME])
        xs = np.linspace(g[symptom].min(), g[symptom].max(), 50)
        ax.plot(xs, ic + sl * xs, color=color, lw=2)
    ax.set_xlabel("ADOS total")
    ax.set_ylabel("Posterior aperiodic exponent")
    ax.legend(title="Age tertile", frameon=False)
    ax.set_title("ADOS–posterior exponent by age stratum (ASD)")
    save_extension_figure(fig, out_base)


def _plot_interaction_panel(df: pd.DataFrame, slopes_df: pd.DataFrame, out_base: Path) -> None:
    symptom = "ADOS_total"
    req = [symptom, OUTCOME, "age_months", "sex", "IQ_total", "usable_epochs", "mean_r_squared"]
    sub = df.dropna(subset=req)
    if len(sub) < 15:
        logger.warning("ADOS 交互面板样本不足，跳过")
        return

    age_mean = float(sub["age_months"].mean())
    sub = sub.copy()
    sub["age_c"] = sub["age_months"] - age_mean
    formula = f"{OUTCOME} ~ {symptom} * age_c + {COVARIATES_EXPONENT}"
    res = run_ols(formula, sub)

    mode_sex = sub["sex"].mode().iloc[0]
    iq_m = float(sub["IQ_total"].median())
    ep_m = float(sub["usable_epochs"].median())
    rsq_m = float(sub["mean_r_squared"].median())
    ados_grid = np.linspace(sub[symptom].min(), sub[symptom].max(), 80)

    slope_sub = slopes_df[slopes_df["symptom"] == symptom]
    if slope_sub.empty:
        age_vals = [
            ("p25", float(sub["age_months"].quantile(0.25))),
            ("p75", float(sub["age_months"].quantile(0.75))),
        ]
    else:
        age_vals = [
            ("p25", float(slope_sub.loc[slope_sub["age_label"] == "p25", "age_months"].iloc[0])),
            ("p75", float(slope_sub.loc[slope_sub["age_label"] == "p75", "age_months"].iloc[0])),
        ]
    int_row = slope_sub.dropna(subset=["interaction_p"]).head(1)
    int_p = float(int_row["interaction_p"].iloc[0]) if not int_row.empty else np.nan

    plt.rcParams.update(PLOT_RC)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    line_colors = {"p25": "#3182ce", "p75": "#d69e2e"}
    for label, age_val in age_vals:
        pred_df = pd.DataFrame({
            symptom: ados_grid,
            "age_c": age_val - age_mean,
            "sex": mode_sex,
            "IQ_total": iq_m,
            "usable_epochs": ep_m,
            "mean_r_squared": rsq_m,
        })
        pred = res.get_prediction(pred_df).summary_frame(alpha=0.05)
        ax.plot(
            ados_grid,
            pred["mean"],
            color=line_colors.get(label, COL_GRAY),
            lw=2,
            label=f"Age {label} ({age_val:.0f} mo)",
        )
        ax.fill_between(
            ados_grid,
            pred["mean_ci_lower"],
            pred["mean_ci_upper"],
            color=line_colors.get(label, COL_GRAY),
            alpha=0.15,
            linewidth=0,
        )

    ax.scatter(sub[symptom], sub[OUTCOME], s=20, alpha=0.45, color=COL_ASD)
    ax.set_xlabel("ADOS total")
    ax.set_ylabel("Posterior aperiodic exponent")
    title = "Predicted ADOS × age interaction (ASD)"
    if np.isfinite(int_p):
        title += f"\ninteraction p = {int_p:.3f}"
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    save_extension_figure(fig, out_base)


def _p_fmt(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def build_report(
    interaction_df: pd.DataFrame,
    slopes_df: pd.DataFrame,
    strat_df: pd.DataFrame,
    n_asd: int,
) -> str:
    ados_ix = interaction_df[
        (interaction_df["symptom"] == "ADOS_total")
        & (interaction_df["term"] == "ADOS_total:age_months")
    ]
    ados_main = interaction_df[
        (interaction_df["symptom"] == "ADOS_total") & (interaction_df["term"] == "ADOS_total")
    ]

    lines = [
        "# Posterior exponent × 症状 × 年龄（ASD 内探索性分析）",
        "",
        f"**队列**：主分析 ASD 子集，n = {n_asd}；结局 = 后枕四导 posterior exponent（E33/E36/E37/E38）。",
        "",
        "## 1. 交互 OLS",
        "",
        "模型：`posterior_exponent ~ symptom * age_months + sex + IQ + usable_epochs + mean_r_squared`",
        "",
    ]

    if not ados_ix.empty:
        r = ados_ix.iloc[0]
        main_r = ados_main.iloc[0] if not ados_main.empty else None
        lines.extend([
            "### ADOS total",
            "",
            f"- **ADOS × age 交互**：β = {float(r['coef']):.4f}, p = {_p_fmt(float(r['pvalue']))}, n = {int(r['n_asd'])}",
        ])
        if main_r is not None:
            lines.append(
                f"- ADOS 主效应（同模型）：β = {float(main_r['coef']):.4f}, p = {_p_fmt(float(main_r['pvalue']))}"
            )
        lines.append("")
    else:
        lines.append("ADOS total 交互模型未收敛或样本不足。\n")

    if not slopes_df.empty:
        ados_sl = slopes_df[slopes_df["symptom"] == "ADOS_total"]
        if not ados_sl.empty:
            lines.extend(["## 2. 简单斜率（ADOS → posterior exponent）", ""])
            for _, r in ados_sl.iterrows():
                lines.append(
                    f"- {r['age_label']} ({r['age_months']:.0f} mo): slope = {float(r['symptom_slope']):.4f}"
                )
            lines.append("")

    if not strat_df.empty:
        ados_st = strat_df[strat_df["symptom"] == "ADOS_total"].copy()
        if not ados_st.empty:
            lines.extend(["## 3. 年龄分层偏 Spearman（控制 sex/IQ/epochs/R²）", ""])
            for _, r in ados_st.iterrows():
                lines.append(
                    f"- **{r['stratum_label']}** (n={int(r['n_asd'])}; "
                    f"{r['age_min']:.0f}–{r['age_max']:.0f} mo): "
                    f"partial ρ = {float(r['partial_rho']):.3f}, p = {_p_fmt(float(r['partial_p']))}"
                )
            lines.append("")

    lines.extend([
        "## 解读提示",
        "",
        "- 显著交互表示 **ADOS 与 posterior exponent 的关联随年龄变化**；不显著则不支持「年长 ASD 关联更强」的假说（功效受 n≈61 限制）。",
        "- 分层偏相关为 **exploratory**；未做多重比较校正。",
        "",
        "**输出**",
        "- `outputs/tables/clinical_age_interaction/symptom_age_interaction_ols.csv`",
        "- `outputs/tables/clinical_age_interaction/symptom_age_simple_slopes.csv`",
        "- `outputs/tables/clinical_age_interaction/symptom_age_stratified_partial_spearman.csv`",
    ])
    return "\n".join(lines)


def run_clinical_age_interaction_analysis(cfg: dict[str, Any]) -> dict[str, Path]:
    out_root = Path(cfg["paths"].get("outputs_root", "outputs"))
    tables_dir = out_root / "tables" / "clinical_age_interaction"
    fig_dir = out_root / "figures" / "clinical_age_interaction"
    report_dir = out_root / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    asd = load_asd_posterior_cohort(cfg)
    interaction_df = fit_symptom_age_interactions(asd)
    slopes_df = compute_simple_slopes(asd)
    strat_df = fit_age_stratified_partial_correlations(asd)

    ix_path = tables_dir / "symptom_age_interaction_ols.csv"
    slope_path = tables_dir / "symptom_age_simple_slopes.csv"
    strat_path = tables_dir / "symptom_age_stratified_partial_spearman.csv"
    save_csv(interaction_df, ix_path)
    save_csv(slopes_df, slope_path)
    save_csv(strat_df, strat_path)

    _plot_ados_strata(asd, fig_dir / "fig_ados_age_strata_slopes")
    _plot_interaction_panel(asd, slopes_df, fig_dir / "fig_ados_age_exponent_interaction")

    report_path = report_dir / "clinical_age_exponent_interaction_report_zh.md"
    report_path.write_text(
        build_report(interaction_df, slopes_df, strat_df, len(asd)),
        encoding="utf-8",
    )

    logger.info("临床 age×symptom 交互分析完成 (ASD n=%d)", len(asd))
    return {
        "interaction": ix_path,
        "slopes": slope_path,
        "stratified": strat_path,
        "fig_strata": fig_dir / "fig_ados_age_strata_slopes.png",
        "fig_interaction": fig_dir / "fig_ados_age_exponent_interaction.png",
        "report": report_path,
    }
