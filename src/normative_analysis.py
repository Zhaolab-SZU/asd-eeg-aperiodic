"""Normative modeling: TD-only age reference for global aperiodic exponent."""

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

from src.extension_analysis import (
    COL_ASD,
    COL_GRAY,
    COL_TD,
    load_main_qc_cohort,
    save_extension_figure,
)
from src.io_utils import save_csv
from src.stats_utils import cohens_d, model_results_to_row, run_ols, spearman_correlation

logger = logging.getLogger(__name__)

OUTCOME = "global_exponent"
COVARIATES = "C(sex) + IQ_total + usable_epochs + mean_r_squared"


def _spline_age_term(age_min: float, age_max: float, df: int = 4) -> str:
    """固定 age 边界，避免 ASD 预测时超出 TD 拟合 knots。"""
    return (
        f"bs(age_months, df={df}, include_intercept=False, "
        f"lower_bound={age_min}, upper_bound={age_max})"
    )


def build_normative_models(age_min: float, age_max: float) -> dict[str, str]:
    spline = _spline_age_term(age_min, age_max)
    return {
        "spline_primary": f"{OUTCOME} ~ {spline} + {COVARIATES}",
        "linear_age": f"{OUTCOME} ~ age_months + {COVARIATES}",
        "quadratic_age": f"{OUTCOME} ~ age_months + I(age_months ** 2) + {COVARIATES}",
        "no_iq": f"{OUTCOME} ~ {spline} + C(sex) + usable_epochs + mean_r_squared",
    }

REQUIRED_COLS = [
    OUTCOME,
    "group",
    "age_months",
    "sex",
    "IQ_total",
    "usable_epochs",
    "mean_r_squared",
]

CLINICAL_VARS = [
    "ADOS_total",
    "ADOS_SA",
    "ADOS_RRB",
    "language_score",
]


def load_normative_cohort(cfg: dict[str, Any], deriv: Path) -> pd.DataFrame:
    """主分析队列 + specparam / 预处理 QC 协变量。"""
    df = load_main_qc_cohort(cfg, deriv)

    preproc = deriv / "qc" / "preproc_summary.csv"
    if preproc.exists():
        pre = pd.read_csv(preproc)
        pre["subject_id"] = pre["subject_id"].astype(str)
        cols = ["subject_id", "bad_channel_count"]
        df = df.merge(pre[[c for c in cols if c in pre.columns]], on="subject_id", how="left")

    sp_qc = deriv / "specparam" / "specparam_qc_summary_subject.csv"
    if sp_qc.exists():
        sp = pd.read_csv(sp_qc)
        sp["subject_id"] = sp["subject_id"].astype(str)
        cols = ["subject_id", "mean_r_squared"]
        df = df.merge(sp[[c for c in cols if c in sp.columns]], on="subject_id", how="left")

    return df.reset_index(drop=True)


def _analysis_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=REQUIRED_COLS).copy()


def fit_td_normative_model(formula: str, td_df: pd.DataFrame) -> Any:
    """仅在 TD 组拟合 normative 模型。"""
    sub = _analysis_subset(td_df)
    if len(sub) < 15:
        raise ValueError(f"TD 样本不足 (n={len(sub)})，无法拟合 normative 模型")
    return run_ols(formula, sub)


def compute_deviation_scores(model: Any, df: pd.DataFrame, residual_sd: float) -> pd.DataFrame:
    """计算 observed / predicted / deviation / z-score。"""
    sub = _analysis_subset(df).copy()
    sub["predicted"] = model.predict(sub)
    sub["deviation"] = sub[OUTCOME] - sub["predicted"]
    sub["deviation_z"] = sub["deviation"] / residual_sd
    sub["deviation_percentile"] = stats.norm.cdf(sub["deviation_z"]) * 100.0
    return sub


def _one_sample_summary(z: pd.Series, label: str) -> dict[str, Any]:
    z = z.dropna()
    n = len(z)
    if n < 2:
        return {
            "stratum": label,
            "n": n,
            "mean_z": np.nan,
            "sd_z": np.nan,
            "median_z": np.nan,
            "t_stat": np.nan,
            "p_one_sided_lt0": np.nan,
            "p_two_sided": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "wilcoxon_p_one_sided_lt0": np.nan,
        }
    t_stat, p_two = stats.ttest_1samp(z, 0.0)
    p_one = float(stats.t.cdf(t_stat, df=n - 1))  # H1: mean < 0
    ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=float(z.mean()), scale=float(stats.sem(z)))
    try:
        w_stat, w_p_two = stats.wilcoxon(z, alternative="two-sided")
        _, w_p_one = stats.wilcoxon(z, alternative="less")
    except ValueError:
        w_stat, w_p_two, w_p_one = np.nan, np.nan, np.nan
    return {
        "stratum": label,
        "n": n,
        "mean_z": float(z.mean()),
        "sd_z": float(z.std(ddof=1)),
        "median_z": float(z.median()),
        "t_stat": float(t_stat),
        "p_one_sided_lt0": p_one,
        "p_two_sided": float(p_two),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else np.nan,
        "wilcoxon_p_one_sided_lt0": float(w_p_one) if not np.isnan(w_p_one) else np.nan,
        "wilcoxon_p_two_sided": float(w_p_two) if not np.isnan(w_p_two) else np.nan,
    }


def run_deviation_tests(scores: pd.DataFrame) -> pd.DataFrame:
    """ASD 偏离检验 + 分层 + ASD vs TD 比较。"""
    rows: list[dict[str, Any]] = []

    asd = scores[scores["group"] == "ASD"]
    td = scores[scores["group"] == "TD"]

    for label, sub in [
        ("ASD_all", asd),
        ("ASD_age_le_72mo", asd[asd["age_months"] <= 72]),
        ("ASD_age_gt_72mo", asd[asd["age_months"] > 72]),
        ("TD_all", td),
    ]:
        rows.append(_one_sample_summary(sub["deviation_z"], label))

    if len(asd) >= 2 and len(td) >= 2:
        t_stat, p_two = stats.ttest_ind(asd["deviation_z"], td["deviation_z"], equal_var=False)
        u_stat, p_mw = stats.mannwhitneyu(asd["deviation_z"], td["deviation_z"], alternative="less")
        rows.append({
            "stratum": "ASD_vs_TD_comparison",
            "n": len(scores),
            "n_ASD": len(asd),
            "n_TD": len(td),
            "mean_z_ASD": float(asd["deviation_z"].mean()),
            "mean_z_TD": float(td["deviation_z"].mean()),
            "mean_diff_ASD_minus_TD": float(asd["deviation_z"].mean() - td["deviation_z"].mean()),
            "cohens_d": cohens_d(asd["deviation_z"].values, td["deviation_z"].values),
            "t_stat": float(t_stat),
            "p_two_sided": float(p_two),
            "p_one_sided_ASD_lt_TD": float(p_mw),
            "mannwhitney_u": float(u_stat),
        })

    return pd.DataFrame(rows)


def run_age_association(scores: pd.DataFrame) -> pd.DataFrame:
    """组内：deviation_z ~ age + sex + IQ。"""
    rows: list[dict[str, Any]] = []
    specs = [
        ("ASD", "deviation_z ~ age_months + C(sex) + IQ_total"),
        ("TD", "deviation_z ~ age_months + C(sex) + IQ_total"),
        ("ASD_no_cov", "deviation_z ~ age_months"),
    ]
    for group, formula in specs:
        if group.startswith("ASD"):
            sub = scores[scores["group"] == "ASD"]
            model_name = group
        else:
            sub = scores[scores["group"] == "TD"]
            model_name = group
        sub = sub.dropna(subset=["deviation_z", "age_months", "sex", "IQ_total"])
        if len(sub) < 10:
            continue
        res = run_ols(formula, sub)
        rows.extend(model_results_to_row(res, model_name, "deviation_z"))
    return pd.DataFrame(rows)


def run_clinical_associations(asd_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ASD 组内：deviation_z 与临床变量（探索性）。"""
    ols_rows: list[dict[str, Any]] = []
    spearman_rows: list[dict[str, Any]] = []

    for clin in CLINICAL_VARS:
        if clin not in asd_scores.columns:
            continue
        formula = f"deviation_z ~ {clin} + age_months + IQ_total"
        sub = asd_scores.dropna(subset=["deviation_z", clin, "age_months", "IQ_total"])
        if len(sub) >= 10:
            res = run_ols(formula, sub)
            ols_rows.extend(model_results_to_row(res, f"clinical_{clin}", "deviation_z"))

        sp = spearman_correlation(asd_scores[clin], asd_scores["deviation_z"])
        spearman_rows.append({
            "clinical_var": clin,
            "outcome": "deviation_z",
            **sp,
        })

    return pd.DataFrame(ols_rows), pd.DataFrame(spearman_rows)


def fit_all_sensitivity_models(td_df: pd.DataFrame, models: dict[str, str]) -> pd.DataFrame:
    """各 normative 模型在 TD 上的拟合摘要。"""
    rows: list[dict[str, Any]] = []
    for name, formula in models.items():
        try:
            res = fit_td_normative_model(formula, td_df)
            rows.append({
                "model_name": name,
                "formula": formula,
                "n_TD": int(res.nobs),
                "r_squared": float(res.rsquared),
                "residual_sd": float(np.sqrt(res.mse_resid)),
                "aic": float(res.aic),
                "bic": float(res.bic),
            })
        except Exception as exc:
            logger.warning("模型 %s 拟合失败: %s", name, exc)
    return pd.DataFrame(rows)


def run_sensitivity_deviation_tests(
    df: pd.DataFrame,
    models: dict[str, str],
    primary_model: str = "spline_primary",
) -> pd.DataFrame:
    """各 normative 设定下 ASD mean z 与 one-sample p。"""
    td = df[df["group"] == "TD"]
    asd = df[df["group"] == "ASD"]
    rows: list[dict[str, Any]] = []

    for name, formula in models.items():
        try:
            model = fit_td_normative_model(formula, td)
            resid_sd = float(np.sqrt(model.mse_resid))
            asd_scores = compute_deviation_scores(model, asd, resid_sd)
            summ = _one_sample_summary(asd_scores["deviation_z"], name)
            summ["model_name"] = name
            summ["residual_sd_TD"] = resid_sd
            summ["is_primary"] = name == primary_model
            rows.append(summ)
        except Exception as exc:
            logger.warning("敏感性 %s 失败: %s", name, exc)
    return pd.DataFrame(rows)


def _median_covariates(df: pd.DataFrame) -> dict[str, float]:
    return {
        "IQ_total": float(df["IQ_total"].median()),
        "usable_epochs": float(df["usable_epochs"].median()),
        "mean_r_squared": float(df["mean_r_squared"].median()),
    }


def plot_normative_trajectory(
    scores: pd.DataFrame,
    td_df: pd.DataFrame,
    model: Any,
    out_base: Path,
) -> None:
    """TD 典型发育曲线 + ASD 散点（按 z 着色）。"""
    med = _median_covariates(td_df)
    age_grid = np.linspace(
        float(scores["age_months"].min()),
        float(scores["age_months"].max()),
        200,
    )

    pred_rows = []
    for sex in ["F", "M"]:
        grid_df = pd.DataFrame({
            "age_months": age_grid,
            "sex": sex,
            "IQ_total": med["IQ_total"],
            "usable_epochs": med["usable_epochs"],
            "mean_r_squared": med["mean_r_squared"],
        })
        grid_df["predicted"] = model.predict(grid_df)
        grid_df["sex"] = sex
        pred_rows.append(grid_df)
    pred = pd.concat(pred_rows, ignore_index=True)

    resid_sd = float(np.sqrt(model.mse_resid))
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    td = scores[scores["group"] == "TD"]
    asd = scores[scores["group"] == "ASD"]
    ax.scatter(
        td["age_months"], td[OUTCOME],
        c=COL_TD, s=36, alpha=0.75, edgecolors="white", linewidths=0.4,
        label=f"TD (n={len(td)})", zorder=2,
    )
    sc = ax.scatter(
        asd["age_months"], asd[OUTCOME],
        c=asd["deviation_z"], cmap="RdYlBu_r", vmin=-2.5, vmax=2.5,
        s=42, edgecolors="0.3", linewidths=0.4,
        label=f"ASD (n={len(asd)})", zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Reference deviation z (vs TD trajectory)")

    for sex, ls in [("F", "-"), ("M", "--")]:
        sub = pred[pred["sex"] == sex]
        ax.plot(sub["age_months"], sub["predicted"], color=COL_GRAY, ls=ls, lw=2.0,
                label=f"TD reference fit ({sex}, med. covariates)")

    ax.fill_between(
        age_grid,
        pred.groupby("age_months")["predicted"].mean() - 1.96 * resid_sd,
        pred.groupby("age_months")["predicted"].mean() + 1.96 * resid_sd,
        color=COL_GRAY, alpha=0.15, label="±1.96 SD (TD residual)",
    )
    ax.axvline(72, color=COL_GRAY, ls=":", lw=1.2, alpha=0.8)
    ax.text(72.5, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            "72 mo", fontsize=8, color=COL_GRAY)

    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Global aperiodic exponent")
    ax.legend(frameon=False, loc="best", fontsize=7)
    fig.tight_layout()
    save_extension_figure(fig, out_base)


def plot_deviation_by_age(scores: pd.DataFrame, out_base: Path) -> None:
    """ASD deviation z 随年龄变化 + 分层箱线图。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))

    asd = scores[scores["group"] == "ASD"].copy()
    td = scores[scores["group"] == "TD"]

    ax = axes[0]
    ax.axhline(0, color=COL_GRAY, ls="--", lw=1.0)
    ax.scatter(td["age_months"], td["deviation_z"], c=COL_TD, alpha=0.45, s=28,
               label="TD", edgecolors="white", linewidths=0.3)
    ax.scatter(asd["age_months"], asd["deviation_z"], c=COL_ASD, alpha=0.85, s=36,
               label="ASD", edgecolors="white", linewidths=0.3)
    if len(asd) >= 3:
        z = np.polyfit(asd["age_months"], asd["deviation_z"], 1)
        x_line = np.linspace(asd["age_months"].min(), asd["age_months"].max(), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), color=COL_ASD, lw=2.0, ls="-")
    ax.axvline(72, color=COL_GRAY, ls=":", lw=1.0)
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Reference deviation z")
    ax.set_title("A. Deviation from TD reference trajectory")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    asd["age_stratum"] = np.where(asd["age_months"] <= 72, "≤72 mo", ">72 mo")
    strata = ["≤72 mo", ">72 mo"]
    data = [asd.loc[asd["age_stratum"] == s, "deviation_z"].dropna().values for s in strata]
    bp = ax.boxplot(data, tick_labels=[f"{s}\n(n={len(d)})" for s, d in zip(strata, data)],
                    patch_artist=True, widths=0.55)
    for patch, color in zip(bp["boxes"], [COL_ASD, COL_ASD]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    ax.axhline(0, color=COL_GRAY, ls="--", lw=1.0)
    ax.set_ylabel("Reference deviation z")
    ax.set_title("B. ASD age-stratified deviation")

    fig.tight_layout()
    save_extension_figure(fig, out_base)


def run_normative_analysis(cfg: dict[str, Any]) -> dict[str, Path]:
    """完整 normative 分析流水线。"""
    deriv = Path(cfg["paths"]["derivatives_root"])
    outputs = Path(cfg["paths"]["outputs_root"])
    tables_dir = outputs / "tables" / "normative_exponent"
    fig_dir = outputs / "figures" / "normative_exponent"
    stats_dir = deriv / "stats"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    df = load_normative_cohort(cfg, deriv)
    df = _analysis_subset(df)
    logger.info("Normative 队列 n=%d (ASD=%d, TD=%d)", len(df),
                (df["group"] == "ASD").sum(), (df["group"] == "TD").sum())

    age_min = float(df["age_months"].min())
    age_max = float(df["age_months"].max())
    models = build_normative_models(age_min, age_max)

    td = df[df["group"] == "TD"]
    primary_formula = models["spline_primary"]
    model = fit_td_normative_model(primary_formula, td)
    resid_sd = float(np.sqrt(model.mse_resid))

    scores = compute_deviation_scores(model, df, resid_sd)
    scores_path = stats_dir / "normative_exponent_scores.csv"
    save_csv(scores, scores_path)

    model_fit_rows = model_results_to_row(model, "spline_primary_TD", OUTCOME)
    model_fit_df = pd.DataFrame(model_fit_rows)
    model_fit_df["residual_sd"] = resid_sd
    model_fit_df["n_TD"] = int(model.nobs)
    save_csv(model_fit_df, tables_dir / "normative_model_coefficients.csv")

    save_csv(fit_all_sensitivity_models(td, models), tables_dir / "normative_model_fit_comparison.csv")
    save_csv(run_deviation_tests(scores), tables_dir / "normative_deviation_tests.csv")
    save_csv(run_age_association(scores), tables_dir / "normative_age_association.csv")
    save_csv(run_sensitivity_deviation_tests(df, models), tables_dir / "normative_sensitivity_deviation.csv")

    asd = scores[scores["group"] == "ASD"]
    clin_ols, clin_sp = run_clinical_associations(asd)
    save_csv(clin_ols, tables_dir / "normative_clinical_ols.csv")
    save_csv(clin_sp, tables_dir / "normative_clinical_spearman.csv")

    plot_normative_trajectory(scores, td, model, fig_dir / "fig_normative_trajectory")
    plot_deviation_by_age(scores, fig_dir / "fig_deviation_by_age")

    summary = {
        "n_total": len(scores),
        "n_ASD": int((scores["group"] == "ASD").sum()),
        "n_TD": int((scores["group"] == "TD").sum()),
        "primary_model": "spline_primary",
        "primary_formula": primary_formula,
        "TD_residual_sd": resid_sd,
        "TD_r_squared": float(model.rsquared),
        "ASD_mean_z": float(asd["deviation_z"].mean()),
        "ASD_median_z": float(asd["deviation_z"].median()),
        "ASD_p_one_sided": float(_one_sample_summary(asd["deviation_z"], "ASD")["p_one_sided_lt0"]),
    }
    save_csv(pd.DataFrame([summary]), tables_dir / "normative_summary.csv")

    return {
        "scores": scores_path,
        "tables": tables_dir,
        "figures": fig_dir,
    }
