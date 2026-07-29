"""IAF + aperiodic exponent 联合发育模型分析。"""

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
    save_extension_figure,
)
from src.io_utils import save_csv
from src.normative_analysis import (
    _one_sample_summary,
    _spline_age_term,
    fit_td_normative_model,
    load_normative_cohort,
)
from src.stats_utils import cohens_d, model_results_to_row, run_ols, spearman_correlation

logger = logging.getLogger(__name__)

POSTERIOR_CORE = ["E33", "E36", "E37", "E38"]

OUTCOMES = [
    "alpha_cf",
    "posterior_alpha_cf",
    "global_exponent",
    "posterior_exponent",
]

COVARIATES = "C(sex) + IQ_total + usable_epochs"
COVARIATES_EXPONENT = f"{COVARIATES} + mean_r_squared"

OUTCOME_LABELS = {
    "alpha_cf": "Alpha peak frequency (Hz, global mean)",
    "posterior_alpha_cf": "Alpha peak frequency (Hz, posterior)",
    "global_exponent": "Global aperiodic exponent",
    "posterior_exponent": "Posterior aperiodic exponent",
}


def _interaction_formula(outcome: str) -> str:
    cov = COVARIATES_EXPONENT if "exponent" in outcome else COVARIATES
    return f"{outcome} ~ C(group) * age_months + {cov}"


def _posterior_channel_mean(
    ch_df: pd.DataFrame,
    value_col: str,
    channels: list[str] | None = None,
    min_ratio: float = 0.5,
) -> pd.DataFrame:
    """后部核心通道（E33/E36/E37/E38）被试层均值。"""
    channels = channels or POSTERIOR_CORE
    if "fit_valid" in ch_df.columns:
        ch_df = ch_df[ch_df["fit_valid"]].copy()

    rows: list[dict[str, Any]] = []
    for (sid, grp), sub in ch_df.groupby(["subject_id", "group"]):
        roi = sub[sub["channel"].isin(channels)]
        n_req = len(channels)
        n_valid = roi[value_col].notna().sum() if value_col in roi.columns else 0
        if n_valid < min_ratio * n_req:
            val = np.nan
        else:
            val = float(roi[value_col].mean())
        rows.append({"subject_id": str(sid), "group": grp, value_col: val})
    return pd.DataFrame(rows)


def load_spectral_maturation_cohort(cfg: dict[str, Any], deriv: Path) -> pd.DataFrame:
    """主分析队列 + 周期峰 + 后部指标。"""
    df = load_normative_cohort(cfg, deriv)
    ch_path = deriv / "specparam" / "specparam_channel_results_qc.csv"
    ch_df = pd.read_csv(ch_path)
    ch_df["subject_id"] = ch_df["subject_id"].astype(str)
    if "fit_valid" in ch_df.columns:
        valid = ch_df[ch_df["fit_valid"]].copy()
    else:
        valid = ch_df.copy()

    peak_cols = ["alpha_cf", "alpha_pw", "alpha_bw", "theta_pw", "beta_pw"]
    subject_peak = (
        valid.groupby(["subject_id", "group"], as_index=False)[peak_cols].mean()
    )
    subject_peak["subject_id"] = subject_peak["subject_id"].astype(str)

    post_exp = _posterior_channel_mean(valid, "aperiodic_exponent")
    post_exp = post_exp.rename(columns={"aperiodic_exponent": "posterior_exponent"})
    post_cf = _posterior_channel_mean(valid, "alpha_cf")
    post_cf = post_cf.rename(columns={"alpha_cf": "posterior_alpha_cf"})

    df["subject_id"] = df["subject_id"].astype(str)
    df = df.merge(subject_peak, on=["subject_id", "group"], how="left")
    df = df.merge(post_exp[["subject_id", "posterior_exponent"]], on="subject_id", how="left")
    df = df.merge(post_cf[["subject_id", "posterior_alpha_cf"]], on="subject_id", how="left")
    return df.reset_index(drop=True)


def fit_age_group_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """并行 age × group 交互模型。"""
    rows: list[dict[str, Any]] = []
    highlight = {"C(group)[T.TD]", "C(group)[T.TD]:age_months", "age_months"}
    for outcome in OUTCOMES:
        formula = _interaction_formula(outcome)
        cov_cols = ["group", "age_months", "sex", "IQ_total", "usable_epochs"]
        if "exponent" in outcome:
            cov_cols.append("mean_r_squared")
        sub = df.dropna(subset=[outcome] + cov_cols)
        if len(sub) < 20:
            logger.warning("样本不足，跳过 %s (n=%d)", outcome, len(sub))
            continue
        res = run_ols(formula, sub)
        for row in model_results_to_row(res, "age_group_interaction", outcome):
            row["formula"] = formula
            row["n_ASD"] = int((sub["group"] == "ASD").sum())
            row["n_TD"] = int((sub["group"] == "TD").sum())
            row["highlight"] = row["term"] in highlight
            rows.append(row)
    return pd.DataFrame(rows)


def fit_independence_models(df: pd.DataFrame) -> pd.DataFrame:
    """控制另一轴后，检验 age × group 是否仍显著。"""
    specs = [
        (
            "global_exponent_adjusted_iaf",
            "global_exponent",
            ["alpha_cf"],
            f"global_exponent ~ C(group) * age_months + alpha_cf + {COVARIATES_EXPONENT}",
        ),
        (
            "alpha_cf_adjusted_exponent",
            "alpha_cf",
            ["global_exponent"],
            f"alpha_cf ~ C(group) * age_months + global_exponent + {COVARIATES}",
        ),
        (
            "posterior_exponent_adjusted_iaf",
            "posterior_exponent",
            ["posterior_alpha_cf"],
            f"posterior_exponent ~ C(group) * age_months + posterior_alpha_cf + {COVARIATES_EXPONENT}",
        ),
        (
            "posterior_alpha_cf_adjusted_exponent",
            "posterior_alpha_cf",
            ["posterior_exponent"],
            f"posterior_alpha_cf ~ C(group) * age_months + posterior_exponent + {COVARIATES}",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for model_name, outcome, adjusters, formula in specs:
        cov_cols = ["group", "age_months", "sex", "IQ_total", "usable_epochs"]
        if "exponent" in outcome:
            cov_cols.append("mean_r_squared")
        sub = df.dropna(subset=[outcome, *adjusters, *cov_cols])
        if len(sub) < 20:
            continue
        res = run_ols(formula, sub)
        for row in model_results_to_row(res, model_name, outcome):
            row["formula"] = formula
            row["n"] = int(res.nobs)
            rows.append(row)
    return pd.DataFrame(rows)


def compute_simple_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """在样本平均年龄处，按组提取 age 简单斜率（中心化 age）。"""
    rows: list[dict[str, Any]] = []
    age_mean = float(df["age_months"].mean())
    df_c = df.copy()
    df_c["age_c"] = df_c["age_months"] - age_mean

    for outcome in OUTCOMES:
        cov = COVARIATES_EXPONENT if "exponent" in outcome else COVARIATES
        formula = f"{outcome} ~ C(group) * age_c + {cov}"
        cov_cols = ["group", "age_months", "sex", "IQ_total", "usable_epochs"]
        if "exponent" in outcome:
            cov_cols.append("mean_r_squared")
        sub = df_c.dropna(subset=[outcome, "age_c"] + cov_cols)
        if len(sub) < 20:
            continue
        res = run_ols(formula, sub)
        for grp in ["ASD", "TD"]:
            if grp == "ASD":
                slope = float(res.params.get("age_c", np.nan))
            else:
                slope = float(res.params.get("age_c", np.nan) + res.params.get("C(group)[T.TD]:age_c", 0.0))
            rows.append({
                "outcome": outcome,
                "group": grp,
                "age_slope_per_month": slope,
                "age_mean_center": age_mean,
                "n": int(res.nobs),
            })
        inter_p = float(res.pvalues.get("C(group)[T.TD]:age_c", np.nan))
        rows.append({
            "outcome": outcome,
            "group": "interaction",
            "age_slope_per_month": float(res.params.get("C(group)[T.TD]:age_c", np.nan)),
            "interaction_p": inter_p,
            "n": int(res.nobs),
        })
    return pd.DataFrame(rows)


def _normative_formula(outcome: str, age_min: float, age_max: float) -> str:
    spline = _spline_age_term(age_min, age_max)
    cov = COVARIATES_EXPONENT if "exponent" in outcome else COVARIATES
    return f"{outcome} ~ {spline} + {cov}"


def run_normative_deviations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """TD 参照 deviation z：IAF 与 posterior exponent。"""
    age_min = float(df["age_months"].min())
    age_max = float(df["age_months"].max())
    td = df[df["group"] == "TD"]

    score_frames: list[pd.DataFrame] = []
    test_rows: list[dict[str, Any]] = []

    for outcome in ["alpha_cf", "posterior_exponent", "global_exponent"]:
        req = [outcome, "group", "age_months", "sex", "IQ_total", "usable_epochs"]
        if "exponent" in outcome:
            req.append("mean_r_squared")
        sub_all = df.dropna(subset=req).copy()
        sub_td = sub_all[sub_all["group"] == "TD"]
        if len(sub_td) < 15:
            continue

        formula = _normative_formula(outcome, age_min, age_max)
        model = fit_td_normative_model(formula, sub_td)
        resid_sd = float(np.sqrt(model.mse_resid))
        scored = sub_all.copy()
        scored["predicted"] = model.predict(scored)
        scored["deviation"] = scored[outcome] - scored["predicted"]
        scored["deviation_z"] = scored["deviation"] / resid_sd

        z_col = f"{outcome}_deviation_z"
        scored = scored.rename(columns={"deviation_z": z_col, "predicted": f"{outcome}_predicted"})
        keep = ["subject_id", "group", "age_months", z_col, f"{outcome}_predicted"]
        score_frames.append(scored[keep])

        asd_z = scored.loc[scored["group"] == "ASD", z_col]
        summ = _one_sample_summary(asd_z, f"ASD_{outcome}")
        summ["outcome"] = outcome
        summ["TD_residual_sd"] = resid_sd
        test_rows.append(summ)

    if not score_frames:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    merged = score_frames[0]
    for extra in score_frames[1:]:
        merged = merged.merge(extra, on=["subject_id", "group", "age_months"], how="outer")

    corr_rows: list[dict[str, Any]] = []
    asd = merged[merged["group"] == "ASD"]
    pairs = [
        ("alpha_cf_deviation_z", "global_exponent_deviation_z"),
        ("alpha_cf_deviation_z", "posterior_exponent_deviation_z"),
        ("posterior_exponent_deviation_z", "global_exponent_deviation_z"),
    ]
    for x, y in pairs:
        if x not in asd.columns or y not in asd.columns:
            continue
        sub = asd.dropna(subset=[x, y])
        if len(sub) < 5:
            continue
        sp = spearman_correlation(sub[x], sub[y])
        corr_rows.append({"var_x": x, "var_y": y, "subset": "ASD", **sp})

    return merged, pd.DataFrame(test_rows), pd.DataFrame(corr_rows)


def plot_interaction_panels(df: pd.DataFrame, out_base: Path) -> None:
    """2×2 age × group 预测线。"""
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5))
    axes = axes.ravel()

    for ax, outcome in zip(axes, OUTCOMES):
        formula = _interaction_formula(outcome)
        cov_cols = ["group", "age_months", "sex", "IQ_total", "usable_epochs"]
        if "exponent" in outcome:
            cov_cols.append("mean_r_squared")
        sub = df.dropna(subset=[outcome] + cov_cols)
        if len(sub) < 20:
            ax.set_visible(False)
            continue

        res = ols(formula, data=sub).fit()
        mode_sex = sub["sex"].mode().iloc[0]
        iq_m = float(sub["IQ_total"].median())
        ep_m = float(sub["usable_epochs"].median())
        rsq_m = float(sub["mean_r_squared"].median()) if "exponent" in outcome else np.nan
        ages = np.linspace(sub["age_months"].min(), sub["age_months"].max(), 80)

        for grp, color in [("ASD", COL_ASD), ("TD", COL_TD)]:
            pred_df = pd.DataFrame({
                "age_months": ages,
                "group": grp,
                "sex": mode_sex,
                "IQ_total": iq_m,
                "usable_epochs": ep_m,
            })
            if "exponent" in outcome:
                pred_df["mean_r_squared"] = rsq_m
            pred = res.get_prediction(pred_df).summary_frame(alpha=0.05)
            ax.plot(ages, pred["mean"], color=color, lw=1.6, label=grp)
            ax.fill_between(
                ages, pred["mean_ci_lower"], pred["mean_ci_upper"],
                color=color, alpha=0.18, linewidth=0,
            )
            pts = sub[sub["group"] == grp]
            ax.scatter(
                pts["age_months"], pts[outcome],
                s=18, alpha=0.55, color=color, edgecolors="none",
            )

        inter_term = "C(group)[T.TD]:age_months"
        p_int = float(res.pvalues.get(inter_term, np.nan))
        ax.set_title(f"{OUTCOME_LABELS[outcome]}\ninteraction p = {p_int:.3f}")
        ax.set_xlabel("Age (months)")
        ax.set_ylabel(OUTCOME_LABELS[outcome].split("(")[0].strip())
        ax.axvline(72, color=COL_GRAY, ls=":", lw=1.0, alpha=0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_extension_figure(fig, out_base)


def plot_deviation_scatter(deviations: pd.DataFrame, out_base: Path) -> None:
    """ASD 双轴 deviation 散点。"""
    if deviations.empty:
        return
    x = "alpha_cf_deviation_z"
    y = "posterior_exponent_deviation_z"
    if x not in deviations.columns or y not in deviations.columns:
        return

    asd = deviations[deviations["group"] == "ASD"].dropna(subset=[x, y])
    td = deviations[deviations["group"] == "TD"].dropna(subset=[x, y])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.axhline(0, color=COL_GRAY, ls="--", lw=0.8)
    ax.axvline(0, color=COL_GRAY, ls="--", lw=0.8)
    ax.scatter(td[x], td[y], c=COL_TD, s=30, alpha=0.5, label=f"TD (n={len(td)})", edgecolors="white", linewidths=0.3)
    sc = ax.scatter(
        asd[x], asd[y], c=asd["age_months"], cmap="viridis", s=42, alpha=0.9,
        label=f"ASD (n={len(asd)})", edgecolors="0.25", linewidths=0.35,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Age (months)")

    if len(asd) >= 5:
        sp = spearman_correlation(asd[x], asd[y])
        ax.text(
            0.03, 0.97,
            f"ASD Spearman ρ = {sp['rho']:.3f}, p = {sp['pvalue']:.3f}",
            transform=ax.transAxes, va="top", fontsize=8,
        )

    ax.set_xlabel("IAF reference deviation z")
    ax.set_ylabel("Posterior exponent reference deviation z")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    save_extension_figure(fig, out_base)


def build_report_markdown(
    interactions: pd.DataFrame,
    independence: pd.DataFrame,
    simple_slopes: pd.DataFrame,
    deviation_tests: pd.DataFrame,
    deviation_corr: pd.DataFrame,
    n_total: int,
    n_asd: int,
    n_td: int,
) -> str:
    """生成中文结果汇报 Markdown。"""

    def _pick(df: pd.DataFrame, outcome: str, term: str) -> dict[str, Any]:
        if df.empty:
            return {}
        hit = df[(df["outcome"] == outcome) & (df["term"] == term)]
        if hit.empty:
            return {}
        r = hit.iloc[0]
        return {"coef": r.get("coef"), "p": r.get("pvalue", r.get("p")), "ci_low": r.get("ci_low"), "ci_high": r.get("ci_high")}

    lines = [
        "# IAF + Aperiodic Exponent 联合发育模型 — 结果汇报",
        "",
        f"**队列：** 主分析 QC 通过，*N* = {n_total}（ASD = {n_asd}，TD = {n_td}）。",
        "",
        "**模型：** `{outcome} ~ C(group) × age_months + sex + IQ_total + usable_epochs`",
        "（exponent 结局额外纳入 `mean_r_squared`）。",
        "",
        "> 横断面 age × group 交互表示组间差异随年龄变化，**不能**等同于纵向成熟延迟的因果证据。",
        "",
        "---",
        "",
        "## 1. 平行 age × group 交互",
        "",
        "| Outcome | Group (TD−ASD) β | *p* | Age×Group β | *p* | Age β | *p* |",
        "|---------|------------------|-----|-------------|-----|-------|-----|",
    ]

    for outcome in OUTCOMES:
        g = _pick(interactions, outcome, "C(group)[T.TD]")
        ia = _pick(interactions, outcome, "C(group)[T.TD]:age_months")
        a = _pick(interactions, outcome, "age_months")
        label = OUTCOME_LABELS[outcome]
        lines.append(
            f"| {label} | "
            f"{g.get('coef', float('nan')):.3f} | {g.get('p', float('nan')):.3f} | "
            f"{ia.get('coef', float('nan')):.4f} | {ia.get('p', float('nan')):.3f} | "
            f"{a.get('coef', float('nan')):.4f} | {a.get('p', float('nan')):.3f} |"
        )

    lines.extend([
        "",
        "### 解读要点",
        "",
    ])

    iaf_int = _pick(interactions, "alpha_cf", "C(group)[T.TD]:age_months")
    post_iaf_int = _pick(interactions, "posterior_alpha_cf", "C(group)[T.TD]:age_months")
    glob_int = _pick(interactions, "global_exponent", "C(group)[T.TD]:age_months")
    post_exp_int = _pick(interactions, "posterior_exponent", "C(group)[T.TD]:age_months")

    if iaf_int.get("p", 1) >= 0.05 and post_iaf_int.get("p", 1) >= 0.05:
        lines.append("- **IAF maturation delay：** global / posterior alpha_cf 的 age×group 交互均不显著，不支持 ASD 存在明确的 IAF 成熟斜率延迟。")
    else:
        lines.append("- **IAF maturation delay：** alpha_cf age×group 交互达到显著，提示 IAF 成熟轨迹可能存在组间差异（需结合简单斜率解读）。")

    if glob_int.get("p", 1) < 0.05 or post_exp_int.get("p", 1) < 0.05:
        lines.append("- **Aperiodic maturation deviation：** global 和/或 posterior exponent 的 age×group 交互显著，组间差异随年龄变化，符合非周期背景成熟轨迹偏离。")
    else:
        lines.append("- **Aperiodic maturation deviation：** exponent age×group 交互未达显著，组间差异不强烈依赖年龄。")

    lines.extend(["", "## 2. 简单斜率（age 中心化）", "", "| Outcome | ASD slope/month | TD slope/month | Interaction *p* |", "|---------|-----------------|----------------|-----------------|"])
    for outcome in OUTCOMES:
        sub = simple_slopes[simple_slopes["outcome"] == outcome]
        asd_s = sub[sub["group"] == "ASD"]
        td_s = sub[sub["group"] == "TD"]
        int_s = sub[sub["group"] == "interaction"]
        if asd_s.empty:
            continue
        label = OUTCOME_LABELS[outcome]
        lines.append(
            f"| {label} | {asd_s.iloc[0]['age_slope_per_month']:.4f} | "
            f"{td_s.iloc[0]['age_slope_per_month']:.4f} | "
            f"{int_s.iloc[0].get('interaction_p', float('nan')):.3f} |"
        )

    lines.extend(["", "## 3. Normative deviation（TD 样条参照）", ""])
    if not deviation_tests.empty:
        lines.append("| Outcome | ASD mean z | ASD one-sided *p* (z<0) |")
        lines.append("|---------|------------|-------------------------|")
        for _, r in deviation_tests.iterrows():
            lines.append(
                f"| {r['outcome']} | {r['mean_z']:.3f} | {r['p_one_sided_lt0']:.4f} |"
            )
    else:
        lines.append("_未生成 deviation 结果。_")

    lines.extend(["", "## 4. IAF 与 exponent 是否独立？", ""])
    if not deviation_corr.empty:
        for _, r in deviation_corr.iterrows():
            lines.append(f"- ASD 内 `{r['var_x']}` vs `{r['var_y']}`：Spearman ρ = {r['rho']:.3f}, *p* = {r['pvalue']:.3f}")
    if not independence.empty:
        lines.append("")
        lines.append("偏相关模型（控制另一轴后 age×group）：")
        lines.append("")
        lines.append("| Model | Term | β | *p* |")
        lines.append("|-------|------|---|-----|")
        for _, r in independence.iterrows():
            if r["term"] in {"C(group)[T.TD]:age_months", "alpha_cf", "global_exponent", "posterior_alpha_cf", "posterior_exponent"}:
                lines.append(f"| {r['model']} | {r['term']} | {r['coef']:.4f} | {r['pvalue']:.3f} |")

    lines.extend([
        "",
        "## 5. 叙事升级（Discussion 建议）",
        "",
        "本研究由「ASD exponent 更低」扩展为 **ASD EEG spectral maturation profile**：",
        "",
        "1. **周期轴（IAF）**：检验 ASD 是否存在 alpha 峰频率成熟延迟；",
        "2. **非周期轴（exponent）**：检验 aperiodic 背景是否相对 TD 规范轨迹系统性偏低；",
        "3. **独立性**：两轴 deviation 相关 + 互相调整后的 age×group 检验。",
        "",
        "若 IAF 交互不显著而 exponent 交互显著，则支持 **选择性非周期成熟偏离**，而非全面频谱成熟延迟。",
        "",
        "---",
        "",
        "**输出文件**",
        "",
        "- `outputs/tables/spectral_maturation/age_group_interaction_models.csv`",
        "- `outputs/tables/spectral_maturation/independence_models.csv`",
        "- `outputs/tables/spectral_maturation/normative_deviation_tests.csv`",
        "- `outputs/figures/spectral_maturation/fig_age_group_interaction_panels.png`",
        "- `outputs/figures/spectral_maturation/fig_iaf_exponent_deviation_scatter.png`",
    ])
    return "\n".join(lines)


def run_spectral_maturation_analysis(cfg: dict[str, Any]) -> dict[str, Path]:
    """完整联合发育分析。"""
    deriv = Path(cfg["paths"]["derivatives_root"])
    outputs = Path(cfg["paths"]["outputs_root"])
    tables_dir = outputs / "tables" / "spectral_maturation"
    fig_dir = outputs / "figures" / "spectral_maturation"
    stats_dir = deriv / "stats"
    reports_dir = outputs / "reports"
    for d in (tables_dir, fig_dir, stats_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = load_spectral_maturation_cohort(cfg, deriv)
    n_total = len(df)
    n_asd = int((df["group"] == "ASD").sum())
    n_td = int((df["group"] == "TD").sum())
    logger.info("Spectral maturation 队列 n=%d (ASD=%d, TD=%d)", n_total, n_asd, n_td)

    interactions = fit_age_group_interactions(df)
    save_csv(interactions, tables_dir / "age_group_interaction_models.csv")

    independence = fit_independence_models(df)
    save_csv(independence, tables_dir / "independence_models.csv")

    simple_slopes = compute_simple_slopes(df)
    save_csv(simple_slopes, tables_dir / "simple_slopes_by_group.csv")

    deviations, dev_tests, dev_corr = run_normative_deviations(df)
    if not deviations.empty:
        save_csv(deviations, stats_dir / "spectral_maturation_deviation_scores.csv")
    if not dev_tests.empty:
        save_csv(dev_tests, tables_dir / "normative_deviation_tests.csv")
    if not dev_corr.empty:
        save_csv(dev_corr, tables_dir / "deviation_correlations_asd.csv")

    plot_interaction_panels(df, fig_dir / "fig_age_group_interaction_panels")
    plot_deviation_scatter(deviations, fig_dir / "fig_iaf_exponent_deviation_scatter")

    report = build_report_markdown(
        interactions, independence, simple_slopes, dev_tests, dev_corr,
        n_total, n_asd, n_td,
    )
    report_path = reports_dir / "spectral_maturation_joint_model_report_zh.md"
    report_path.write_text(report, encoding="utf-8")

    return {
        "tables": tables_dir,
        "figures": fig_dir,
        "report": report_path,
    }
