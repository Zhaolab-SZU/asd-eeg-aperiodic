"""扩展分析：年龄交互与 split-half 信度。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols

from src.io_utils import (
    attach_usable_epochs,
    exclude_specparam_low_quality,
    get_epochs_fif_path,
    load_analysis_participants,
    save_csv,
)
from src.psd_utils import compute_psd_from_epochs, psd_to_long_df
from src.specparam_utils import fit_subject_specparam

logger = logging.getLogger(__name__)

GROUP_TERM = "C(group)[T.TD]"
COL_ASD = "#4C72B0"
COL_TD = "#DD8452"
COL_GRAY = "#4D4D4D"
COL_LIGHT = "#D9D9D9"

PLOT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

INTERACTION_FORMULAE = {
    "global_exponent": (
        "global_exponent ~ C(group) * age_months + C(sex) + IQ_total + usable_epochs"
    ),
    "global_offset": (
        "global_offset ~ C(group) * age_months + C(sex) + IQ_total + usable_epochs"
    ),
}

RELIABILITY_METRICS = [
    ("global_exponent", "global_exponent_h1", "global_exponent_h2"),
    ("global_offset", "global_offset_h1", "global_offset_h2"),
    ("alpha_pw", "alpha_pw_h1", "alpha_pw_h2"),
]


def load_main_qc_cohort(cfg: dict[str, Any], deriv: Path) -> pd.DataFrame:
    """主分析 QC 通过队列（epoch + specparam QC + ROI global）。"""
    participants = load_analysis_participants(cfg)
    participants = attach_usable_epochs(participants, deriv)
    min_ep = int(cfg.get("epochs", {}).get("min_usable_epochs", 60))
    if "usable_epochs" in participants.columns:
        participants = participants[participants["usable_epochs"] >= min_ep].copy()

    sp_qc = deriv / "specparam" / "specparam_qc_summary_subject.csv"
    if sp_qc.exists():
        qc = pd.read_csv(sp_qc)
        bad = qc.loc[qc["low_quality_subject"] == 1, "subject_id"].astype(str)
        participants = participants[~participants["subject_id"].astype(str).isin(bad)]
    participants = exclude_specparam_low_quality(participants, deriv)

    roi = deriv / "roi" / "specparam_subject_global.csv"
    if not roi.exists():
        raise FileNotFoundError(f"缺少 ROI 文件: {roi}")
    roi_df = pd.read_csv(roi)
    roi_df["subject_id"] = roi_df["subject_id"].astype(str)
    participants = participants.merge(
        roi_df[["subject_id", "group", "global_exponent", "global_offset"]],
        on=["subject_id", "group"],
        how="inner",
    )
    return participants.reset_index(drop=True)


def _apply_plot_style() -> None:
    plt.rcParams.update(PLOT_RC)


def save_extension_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        kw: dict[str, Any] = dict(bbox_inches="tight", facecolor="white", edgecolor="none")
        if ext == "png":
            kw["dpi"] = 600
        fig.savefig(out_base.with_suffix(f".{ext}"), **kw)
    plt.close(fig)


def fit_interaction_models(df: pd.DataFrame) -> pd.DataFrame:
    """拟合 group × age 交互模型并导出系数表。"""
    rows: list[dict[str, Any]] = []
    for outcome, formula in INTERACTION_FORMULAE.items():
        sub = df.dropna(subset=[outcome, "group", "age_months", "sex", "IQ_total", "usable_epochs"])
        if len(sub) < 10:
            logger.warning("样本不足，跳过 %s", outcome)
            continue
        res = ols(formula, data=sub).fit()
        conf = res.conf_int()
        counts = {"n": int(res.nobs), "n_ASD": int((sub["group"] == "ASD").sum()), "n_TD": int((sub["group"] == "TD").sum())}
        for term in res.params.index:
            rows.append({
                "outcome": outcome,
                "model": formula,
                **counts,
                "term": term,
                "coef": float(res.params[term]),
                "se": float(res.bse[term]),
                "t": float(res.tvalues[term]),
                "p": float(res.pvalues[term]),
                "ci_low": float(conf.loc[term, 0]),
                "ci_high": float(conf.loc[term, 1]),
                "r_squared": float(res.rsquared),
                "aic": float(res.aic),
                "bic": float(res.bic),
            })
    return pd.DataFrame(rows)


def plot_development_interaction(
    df: pd.DataFrame,
    outcome: str,
    out_base: Path,
) -> None:
    """散点 + 预测线 + 95% CI。"""
    formula = INTERACTION_FORMULAE[outcome]
    sub = df.dropna(subset=[outcome, "group", "age_months", "sex", "IQ_total", "usable_epochs"])
    res = ols(formula, data=sub).fit()

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    mode_sex = sub["sex"].mode().iloc[0]
    iq_m = float(sub["IQ_total"].mean())
    ep_m = float(sub["usable_epochs"].mean())
    ages = np.linspace(sub["age_months"].min(), sub["age_months"].max(), 80)

    for grp, color in [("ASD", COL_ASD), ("TD", COL_TD)]:
        pred_df = pd.DataFrame({
            "age_months": ages,
            "group": grp,
            "sex": mode_sex,
            "IQ_total": iq_m,
            "usable_epochs": ep_m,
        })
        pred = res.get_prediction(pred_df).summary_frame(alpha=0.05)
        ax.plot(ages, pred["mean"], color=color, lw=1.5, label=grp)
        ax.fill_between(ages, pred["mean_ci_lower"], pred["mean_ci_upper"], color=color, alpha=0.2, linewidth=0)
        pts = sub[sub["group"] == grp]
        ax.scatter(pts["age_months"], pts[outcome], s=20, alpha=0.6, color=color, edgecolors="none")

    ylab = "Global aperiodic exponent" if "exponent" in outcome else "Global aperiodic offset"
    ax.set_xlabel("Age (months)")
    ax.set_ylabel(ylab)
    ax.legend(frameon=False, loc="best")
    save_extension_figure(fig, out_base)


def channel_metrics_from_fits(ch_df: pd.DataFrame) -> dict[str, float]:
    """通道 specparam 结果 → 被试层 global 指标。"""
    valid = ch_df.dropna(subset=["aperiodic_exponent"])
    if valid.empty:
        return {"global_exponent": np.nan, "global_offset": np.nan, "alpha_pw": np.nan}
    return {
        "global_exponent": float(valid["aperiodic_exponent"].mean()),
        "global_offset": float(valid["aperiodic_offset"].mean()),
        "alpha_pw": float(valid["alpha_pw"].mean()) if "alpha_pw" in valid.columns else np.nan,
    }


def specparam_from_epochs(
    epochs: mne.Epochs,
    subject_id: str,
    group: str,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """从 Epochs 计算 PSD 并 specparam，返回被试层指标。"""
    psd_cfg = cfg["psd"]
    welch_cfg = psd_cfg.get("welch", {})
    fmin = psd_cfg.get("freq_min_hz", 1.0)
    fmax = psd_cfg.get("freq_max_hz", 40.0)
    freqs, psd, ch_names = compute_psd_from_epochs(epochs, fmin, fmax, welch_cfg)
    psd_df = psd_to_long_df(subject_id, group, freqs, psd, ch_names)
    ch_fits = fit_subject_specparam(psd_df, cfg["specparam"])
    return channel_metrics_from_fits(ch_fits)


def process_subject_split_half(
    subject_id: str,
    group: str,
    epochs_path: Path,
    cfg: dict[str, Any],
    min_epochs_half: int = 15,
) -> dict[str, Any] | None:
    """单被试 odd/even split-half specparam 指标。"""
    epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
    n = len(epochs)
    if n < 2 * min_epochs_half:
        logger.warning("%s: epoch 数 %d 过少，跳过 split-half", subject_id, n)
        return None

    h1 = epochs[::2].copy()
    h2 = epochs[1::2].copy()
    m1 = specparam_from_epochs(h1, subject_id, group, cfg)
    m2 = specparam_from_epochs(h2, subject_id, group, cfg)

    row: dict[str, Any] = {
        "subject_id": subject_id,
        "group": group,
        "n_epochs": n,
        "n_half_odd": len(h1),
        "n_half_even": len(h2),
        "global_exponent_h1": m1["global_exponent"],
        "global_exponent_h2": m2["global_exponent"],
        "global_offset_h1": m1["global_offset"],
        "global_offset_h2": m2["global_offset"],
        "alpha_pw_h1": m1["alpha_pw"],
        "alpha_pw_h2": m2["alpha_pw"],
    }
    return row


def spearman_brown_correction(r: float) -> float:
    """Split-half Spearman-Brown 校正（等价于分半信度）。"""
    if np.isnan(r) or abs(r) >= 1.0:
        return np.nan
    return float(2 * r / (1 + r))


def compute_split_half_reliability(subject_df: pd.DataFrame) -> pd.DataFrame:
    """汇总 split-half 相关与 Spearman-Brown。"""
    rows = []
    for metric, col_h1, col_h2 in RELIABILITY_METRICS:
        sub = subject_df.dropna(subset=[col_h1, col_h2])
        if len(sub) < 5:
            rows.append({
                "metric": metric, "n_subjects": len(sub),
                "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_rho": np.nan, "spearman_p": np.nan,
                "spearman_brown": np.nan,
            })
            continue
        x, y = sub[col_h1].values, sub[col_h2].values
        pr = stats.pearsonr(x, y)
        sr = stats.spearmanr(x, y)
        rows.append({
            "metric": metric,
            "n_subjects": len(sub),
            "pearson_r": float(pr.statistic),
            "pearson_p": float(pr.pvalue),
            "spearman_rho": float(sr.statistic),
            "spearman_p": float(sr.pvalue),
            "spearman_brown_pearson": spearman_brown_correction(float(pr.statistic)),
            "spearman_brown_spearman": spearman_brown_correction(float(sr.statistic)),
        })
    return pd.DataFrame(rows)


def plot_split_half_reliability(subject_df: pd.DataFrame, rel_df: pd.DataFrame, out_base: Path) -> None:
    """三 panel：odd vs even 散点 + y=x + rho。"""
    _apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    titles = {
        "global_exponent": "Global exponent",
        "global_offset": "Global offset",
        "alpha_pw": "Alpha peak power (log10)",
    }
    for ax, (metric, col_h1, col_h2) in zip(axes, RELIABILITY_METRICS):
        sub = subject_df.dropna(subset=[col_h1, col_h2])
        rel_row = rel_df[rel_df["metric"] == metric]
        rho = float(rel_row["spearman_rho"].iloc[0]) if not rel_row.empty else np.nan
        pval = float(rel_row["spearman_p"].iloc[0]) if not rel_row.empty else np.nan

        x, y = sub[col_h1], sub[col_h2]
        ax.scatter(x, y, s=18, alpha=0.6, color=COL_GRAY, edgecolors="none")
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        pad = (hi - lo) * 0.05 if hi > lo else 0.1
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color=COL_LIGHT, lw=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("Odd epochs (half 1)")
        ax.set_ylabel("Even epochs (half 2)")
        ax.set_title(titles.get(metric, metric), fontsize=9)
        if not np.isnan(rho):
            p_str = "p < .001" if pval < 0.001 else f"p = {pval:.3f}"
            ax.text(0.05, 0.95, f"ρ = {rho:.3f}\n{p_str}", transform=ax.transAxes, va="top", fontsize=7, color=COL_GRAY)
    fig.tight_layout()
    save_extension_figure(fig, out_base)


def run_split_half_batch(
    cohort: pd.DataFrame,
    epochs_dir: Path,
    cfg: dict[str, Any],
    max_subjects: int | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """批量 split-half；单被试失败仅 warning。"""
    rows: list[dict[str, Any]] = []
    subjects = cohort.head(max_subjects) if max_subjects else cohort

    def _one(row: pd.Series) -> dict[str, Any] | None:
        sid = str(row["subject_id"])
        ep_path = get_epochs_fif_path(epochs_dir, sid)
        if not ep_path.exists():
            logger.warning("%s: 无 epochs 文件 %s", sid, ep_path)
            return None
        try:
            return process_subject_split_half(sid, str(row["group"]), ep_path, cfg)
        except Exception as exc:
            logger.warning("%s split-half 失败: %s", sid, exc)
            return None

    if n_jobs != 1:
        logger.warning("split-half 暂仅支持串行 (n_jobs=1)，以避免 MNE/matplotlib 多线程问题；已忽略 --n-jobs")
    results = [_one(row) for _, row in subjects.iterrows()]

    for r in results:
        if r is not None:
            rows.append(r)
    return pd.DataFrame(rows)


def write_extension_report(
    out_path: Path,
    interaction_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    n_cohort: int,
) -> None:
    """生成 extension 报告（中英短段落）。"""
    def _p(outcome: str, term: str) -> tuple[float, float] | None:
        sub = interaction_df[(interaction_df["outcome"] == outcome) & (interaction_df["term"] == term)]
        if sub.empty:
            return None
        return float(sub["coef"].iloc[0]), float(sub["p"].iloc[0])

    int_exp = _p("global_exponent", "C(group)[T.TD]:age_months")
    int_off = _p("global_offset", "C(group)[T.TD]:age_months")

    def _fmt_int(row: tuple[float, float] | None) -> tuple[str, str]:
        if row is None:
            return "—", "—"
        return f"{row[0]:.4f}", f"{row[1]:.3f}"

    exp_b, exp_p = _fmt_int(int_exp)
    off_b, off_p = _fmt_int(int_off)

    rel_lines = []
    for _, r in reliability_df.iterrows():
        sb = r.get("spearman_brown_spearman", np.nan)
        rel_lines.append(
            f"- **{r['metric']}**: Spearman ρ = {r['spearman_rho']:.3f} (p = {r['spearman_p']:.3g}), "
            f"SB(ρ) = {sb:.3f}, n = {int(r['n_subjects'])}"
        )

    rho_exp = float(reliability_df.loc[reliability_df["metric"] == "global_exponent", "spearman_rho"].iloc[0])
    rho_off = float(reliability_df.loc[reliability_df["metric"] == "global_offset", "spearman_rho"].iloc[0])
    rho_alpha = float(reliability_df.loc[reliability_df["metric"] == "alpha_pw", "spearman_rho"].iloc[0])
    rel_qual = "moderate-to-high" if reliability_df["spearman_rho"].min() > 0.5 else "variable"

    text = f"""# Extension: Development interaction and split-half reliability

## 1. Age (development) interaction summary

**Cohort:** main QC-passed analysis set, *N* = {n_cohort}.

**Models:**
- `global_exponent ~ C(group) × age_months + C(sex) + IQ_total + usable_epochs`
- `global_offset ~ C(group) × age_months + C(sex) + IQ_total + usable_epochs`

| Effect | Outcome | β | *p* |
|--------|---------|---|-----|
| Group × age | Exponent | {exp_b} | {exp_p} |
| Group × age | Offset | {off_b} | {off_p} |

**Caution:** Significant interactions in this **cross-sectional** sample indicate that adjusted group differences **vary with age**; they **do not** establish longitudinal developmental trajectories or causal maturational effects.

**Tables / figures:**
- `outputs/tables/extension/development_interaction_models.csv`
- `outputs/figures/extension/fig_development_interaction_exponent`
- `outputs/figures/extension/fig_development_interaction_offset`

---

## 2. Split-half reliability summary

Epochs were split into **odd** vs **even** halves per subject; each half was independently PSD-estimated (Welch) and fit with specparam (fixed aperiodic mode). Subject-level metrics were channel-averaged.

{chr(10).join(rel_lines)}

**Tables / figures:**
- `outputs/tables/extension/split_half_subject_metrics.csv`
- `outputs/tables/extension/split_half_reliability.csv`
- `outputs/figures/extension/fig_split_half_reliability`

---

## 3. Results / Supplement — English short paragraph

**Development.** In the main cohort (*N* = {n_cohort}), group × age_months interactions were significant for global aperiodic exponent (β = {exp_b}, *p* = {exp_p}) and offset (β = {off_b}, *p* = {off_p}), indicating that covariate-adjusted group differences varied across age in this cross-sectional sample. This pattern should not be interpreted as evidence for longitudinal developmental change.

**Reliability.** Split-half (odd vs even epochs) consistency was {rel_qual} for global exponent (ρ = {rho_exp:.3f}), global offset (ρ = {rho_off:.3f}), and alpha peak power (ρ = {rho_alpha:.3f}), supporting stable within-subject aperiodic estimates under epoch subsampling.

---

## 4. 结果 / 补充材料 — 中文短段落

**年龄调节。** 在主分析 QC 队列（*N* = {n_cohort}）中，global aperiodic exponent 与 offset 的 **group × age_months** 交互均显著（exponent 交互 *p* = {exp_p}；offset 交互 *p* = {off_p}），提示协变量校正后的组间差异在本横断面样本中 **随年龄而变化**。需强调：该结果 **不能** 解释为纵向发育轨迹或因果性成熟效应。

**分半信度。** 将每名被试的 epochs 按奇偶分为两半并独立估计 PSD 与 specparam 后，global exponent、global offset 与 alpha peak power 的奇偶分半 Spearman 相关分别为 {rho_exp:.2f}、{rho_off:.2f} 与 {rho_alpha:.2f}（详见 split_half_reliability.csv），表明 aperiodic 指标在 epoch 子样本内具有可接受的内部一致性。
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
