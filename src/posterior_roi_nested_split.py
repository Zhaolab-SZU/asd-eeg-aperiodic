"""Nested / repeated split-sample validation for data-driven posterior ROI selection.

Addresses circularity: select FDR-significant channels on a training split, then
estimate the ROI mean group effect only on the held-out test split.

Also reports:
- selection frequency of each channel across splits
- hold-out effect of the fixed full-sample posterior core (E33/E36/E37/E38)
- in-sample (train) effect of the train-selected ROI (for winner's-curse contrast)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.io_utils import save_csv
from src.posterior_roi_loocv import FORMULA, fit_channel_level_fdr, load_resting_channel_cohort
from src.spectral_maturation_analysis import POSTERIOR_CORE
from src.stats_utils import run_ols

logger = logging.getLogger(__name__)

COVARS = ["age_months", "sex", "IQ_total", "usable_epochs"]


def _group_effect_td_minus_asd(res: Any) -> tuple[float, float, str]:
    """Return (beta_TD_minus_ASD, p, term_name)."""
    terms = [t for t in res.params.index if str(t).startswith("C(group)")]
    if not terms:
        raise ValueError("No C(group) term in model")
    term = str(terms[0])
    coef = float(res.params[term])
    p = float(res.pvalues[term])
    # statsmodels Treatment coding: reference = first level alphabetically (ASD)
    # → C(group)[T.TD] is already TD − ASD. If ASD is the treatment level, flip.
    if "T.ASD" in term:
        coef = -coef
    return coef, p, term


def subject_roi_mean(
    channel_df: pd.DataFrame,
    participants: pd.DataFrame,
    channels: list[str],
    *,
    min_frac: float = 0.5,
) -> pd.DataFrame:
    """Subject-level mean aperiodic exponent over selected channels."""
    ch = channel_df.copy()
    if "fit_valid" in ch.columns:
        ch = ch[ch["fit_valid"]]
    ch["subject_id"] = ch["subject_id"].astype(str)
    ch = ch[ch["channel"].astype(str).isin([str(c) for c in channels])]
    if ch.empty:
        return pd.DataFrame()

    n_req = len(channels)
    rows: list[dict[str, Any]] = []
    for sid, sub in ch.groupby("subject_id"):
        vals = sub["aperiodic_exponent"].dropna()
        if len(vals) < max(1, int(np.ceil(min_frac * n_req))):
            continue
        rows.append({"subject_id": str(sid), "roi_exponent": float(vals.mean())})
    roi = pd.DataFrame(rows)
    part = participants.copy()
    part["subject_id"] = part["subject_id"].astype(str)
    return part.merge(roi, on="subject_id", how="inner")


def estimate_roi_group_effect(
    channel_df: pd.DataFrame,
    participants: pd.DataFrame,
    channels: list[str],
) -> dict[str, Any]:
    """OLS group effect for ROI mean on the provided participant subset."""
    out: dict[str, Any] = {
        "n_channels_roi": len(channels),
        "roi_channels": ",".join(channels),
        "n_obs": 0,
        "beta_td_minus_asd": np.nan,
        "p": np.nan,
        "n_asd": 0,
        "n_td": 0,
        "ok": False,
    }
    if not channels:
        return out
    df = subject_roi_mean(channel_df, participants, channels)
    need = ["roi_exponent", "group"] + [c for c in COVARS if c in participants.columns]
    df = df.dropna(subset=[c for c in need if c in df.columns])
    if len(df) < 20 or df["group"].nunique() < 2:
        return out
    formula = "roi_exponent ~ C(group) + age_months + C(sex) + IQ_total + usable_epochs"
    try:
        res = run_ols(formula, df)
        beta, p, term = _group_effect_td_minus_asd(res)
    except Exception as exc:
        logger.debug("ROI OLS failed (%s): %s", channels, exc)
        return out
    out.update(
        {
            "n_obs": int(res.nobs),
            "beta_td_minus_asd": beta,
            "p": p,
            "n_asd": int((df["group"] == "ASD").sum()),
            "n_td": int((df["group"] == "TD").sum()),
            "ok": True,
            "group_term": term,
        }
    )
    return out


def select_roi_from_train(
    channel_df: pd.DataFrame,
    train_participants: pd.DataFrame,
    *,
    alpha: float = 0.05,
    fallback_topk: int = 4,
) -> tuple[list[str], str, pd.DataFrame]:
    """
    Select ROI channels on the training split.

    Primary rule: BH-FDR significant channels (same as script 10).
    Fallback if none: top-|β| channels with uncorrected p < 0.05 (up to fallback_topk);
    if still none: top-|β| fallback_topk channels overall.
    """
    ch_stats = fit_channel_level_fdr(channel_df, train_participants, alpha=alpha)
    if ch_stats.empty:
        return [], "failed_no_channel_models", ch_stats

    sig = ch_stats.loc[ch_stats["significant_fdr"], "channel"].astype(str).tolist()
    if sig:
        # Stable order by |coef| descending among FDR hits
        sub = ch_stats[ch_stats["channel"].astype(str).isin(sig)].copy()
        sub["abs_coef"] = sub["coef"].abs()
        sub = sub.sort_values("abs_coef", ascending=False)
        return sub["channel"].astype(str).tolist(), "fdr_significant", ch_stats

    unc = ch_stats[ch_stats["pvalue"] < 0.05].copy()
    if not unc.empty:
        unc["abs_coef"] = unc["coef"].abs()
        unc = unc.sort_values("abs_coef", ascending=False).head(fallback_topk)
        return unc["channel"].astype(str).tolist(), "fallback_uncorrected_topk", ch_stats

    top = ch_stats.copy()
    top["abs_coef"] = top["coef"].abs()
    top = top.sort_values("abs_coef", ascending=False).head(fallback_topk)
    return top["channel"].astype(str).tolist(), "fallback_abscoef_topk", ch_stats


def _split_participants(
    participants: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = participants.iloc[train_idx].reset_index(drop=True)
    test = participants.iloc[test_idx].reset_index(drop=True)
    return train, test


def run_repeated_split_validation(
    channel_df: pd.DataFrame,
    participants: pd.DataFrame,
    *,
    n_splits: int = 200,
    test_size: float = 0.30,
    random_state: int = 42,
    alpha: float = 0.05,
    fallback_topk: int = 4,
) -> pd.DataFrame:
    """Repeated stratified train/test splits with train-select / test-estimate."""
    part = participants.reset_index(drop=True).copy()
    part["subject_id"] = part["subject_id"].astype(str)
    y = part["group"].astype(str).values
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )
    rows: list[dict[str, Any]] = []
    for i, (tr, te) in enumerate(splitter.split(np.zeros(len(part)), y)):
        train, test = _split_participants(part, tr, te)
        roi, rule, ch_stats = select_roi_from_train(
            channel_df, train, alpha=alpha, fallback_topk=fallback_topk
        )
        n_fdr = int(ch_stats["significant_fdr"].sum()) if not ch_stats.empty else 0
        hold = estimate_roi_group_effect(channel_df, test, roi)
        train_eff = estimate_roi_group_effect(channel_df, train, roi)
        fixed = estimate_roi_group_effect(channel_df, test, list(POSTERIOR_CORE))
        posterior_overlap = sorted(set(roi) & set(POSTERIOR_CORE))
        rows.append(
            {
                "mode": "repeated_split",
                "split_id": i,
                "n_train": int(train["subject_id"].nunique()),
                "n_test": int(test["subject_id"].nunique()),
                "selection_rule": rule,
                "n_fdr_significant_train": n_fdr,
                "n_roi_channels": len(roi),
                "roi_channels": ",".join(roi),
                "n_overlap_posterior_core": len(posterior_overlap),
                "overlap_posterior_core": ",".join(posterior_overlap),
                "includes_all_posterior_core": int(set(POSTERIOR_CORE).issubset(set(roi))),
                "holdout_beta": hold["beta_td_minus_asd"],
                "holdout_p": hold["p"],
                "holdout_n": hold["n_obs"],
                "holdout_ok": int(hold["ok"]),
                "train_beta": train_eff["beta_td_minus_asd"],
                "train_p": train_eff["p"],
                "fixed_core_holdout_beta": fixed["beta_td_minus_asd"],
                "fixed_core_holdout_p": fixed["p"],
            }
        )
        if (i + 1) % 25 == 0:
            logger.info("Repeated split progress: %d / %d", i + 1, n_splits)
    return pd.DataFrame(rows)


def run_kfold_nested_validation(
    channel_df: pd.DataFrame,
    participants: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    alpha: float = 0.05,
    fallback_topk: int = 4,
) -> pd.DataFrame:
    """Stratified K-fold: select ROI on K-1 folds, estimate on held-out fold."""
    part = participants.reset_index(drop=True).copy()
    part["subject_id"] = part["subject_id"].astype(str)
    y = part["group"].astype(str).values
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows: list[dict[str, Any]] = []
    for i, (tr, te) in enumerate(kf.split(np.zeros(len(part)), y)):
        train, test = _split_participants(part, tr, te)
        roi, rule, ch_stats = select_roi_from_train(
            channel_df, train, alpha=alpha, fallback_topk=fallback_topk
        )
        n_fdr = int(ch_stats["significant_fdr"].sum()) if not ch_stats.empty else 0
        hold = estimate_roi_group_effect(channel_df, test, roi)
        train_eff = estimate_roi_group_effect(channel_df, train, roi)
        fixed = estimate_roi_group_effect(channel_df, test, list(POSTERIOR_CORE))
        posterior_overlap = sorted(set(roi) & set(POSTERIOR_CORE))
        rows.append(
            {
                "mode": "kfold",
                "split_id": i,
                "n_train": int(train["subject_id"].nunique()),
                "n_test": int(test["subject_id"].nunique()),
                "selection_rule": rule,
                "n_fdr_significant_train": n_fdr,
                "n_roi_channels": len(roi),
                "roi_channels": ",".join(roi),
                "n_overlap_posterior_core": len(posterior_overlap),
                "overlap_posterior_core": ",".join(posterior_overlap),
                "includes_all_posterior_core": int(set(POSTERIOR_CORE).issubset(set(roi))),
                "holdout_beta": hold["beta_td_minus_asd"],
                "holdout_p": hold["p"],
                "holdout_n": hold["n_obs"],
                "holdout_ok": int(hold["ok"]),
                "train_beta": train_eff["beta_td_minus_asd"],
                "train_p": train_eff["p"],
                "fixed_core_holdout_beta": fixed["beta_td_minus_asd"],
                "fixed_core_holdout_p": fixed["p"],
            }
        )
    return pd.DataFrame(rows)


def run_single_split_discovery_validation(
    channel_df: pd.DataFrame,
    participants: pd.DataFrame,
    *,
    test_size: float = 0.50,
    random_state: int = 42,
    alpha: float = 0.05,
    fallback_topk: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One 50/50 discovery/validation split with channel stats on discovery."""
    part = participants.reset_index(drop=True).copy()
    part["subject_id"] = part["subject_id"].astype(str)
    y = part["group"].astype(str).values
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr, te = next(splitter.split(np.zeros(len(part)), y))
    train, test = _split_participants(part, tr, te)
    roi, rule, ch_stats = select_roi_from_train(
        channel_df, train, alpha=alpha, fallback_topk=fallback_topk
    )
    hold = estimate_roi_group_effect(channel_df, test, roi)
    train_eff = estimate_roi_group_effect(channel_df, train, roi)
    fixed = estimate_roi_group_effect(channel_df, test, list(POSTERIOR_CORE))
    summary = pd.DataFrame(
        [
            {
                "mode": "single_split_50_50",
                "selection_rule": rule,
                "discovery_n": int(train["subject_id"].nunique()),
                "validation_n": int(test["subject_id"].nunique()),
                "roi_channels": ",".join(roi),
                "n_fdr_significant_discovery": int(ch_stats["significant_fdr"].sum())
                if not ch_stats.empty
                else 0,
                "discovery_beta": train_eff["beta_td_minus_asd"],
                "discovery_p": train_eff["p"],
                "validation_beta": hold["beta_td_minus_asd"],
                "validation_p": hold["p"],
                "fixed_core_validation_beta": fixed["beta_td_minus_asd"],
                "fixed_core_validation_p": fixed["p"],
            }
        ]
    )
    if not ch_stats.empty:
        ch_stats = ch_stats.copy()
        ch_stats["selected_in_roi"] = ch_stats["channel"].astype(str).isin(roi)
        ch_stats["split_role"] = "discovery"
    return summary, ch_stats


def channel_selection_frequency(splits_df: pd.DataFrame) -> pd.DataFrame:
    """How often each channel enters the train-selected ROI."""
    if splits_df.empty:
        return pd.DataFrame()
    counts: dict[str, int] = {}
    n = 0
    for channels in splits_df["roi_channels"].fillna(""):
        n += 1
        for ch in str(channels).split(","):
            ch = ch.strip()
            if not ch:
                continue
            counts[ch] = counts.get(ch, 0) + 1
    rows = [
        {
            "channel": ch,
            "n_selected": c,
            "n_splits": n,
            "selection_rate": c / n if n else np.nan,
            "is_posterior_core": ch in POSTERIOR_CORE,
        }
        for ch, c in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    ]
    return pd.DataFrame(rows)


def summarize_holdout_effects(splits_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hold-out / train betas across splits."""
    if splits_df.empty:
        return pd.DataFrame()
    ok = splits_df[splits_df["holdout_ok"] == 1].copy()
    if ok.empty:
        ok = splits_df.copy()

    rows: list[dict[str, Any]] = []
    for beta_col, p_col, label in [
        ("holdout_beta", "holdout_p", "train_selected_ROI_holdout"),
        ("train_beta", "train_p", "train_selected_ROI_insample"),
        ("fixed_core_holdout_beta", "fixed_core_holdout_p", "fixed_posterior_core_holdout"),
    ]:
        mask = pd.to_numeric(ok[beta_col], errors="coerce").notna()
        x = pd.to_numeric(ok.loc[mask, beta_col], errors="coerce")
        p = pd.to_numeric(ok.loc[mask, p_col], errors="coerce") if p_col in ok.columns else None
        rows.append(
            {
                "estimate": label,
                "n_splits": int(len(x)),
                "mean_beta": float(x.mean()) if len(x) else np.nan,
                "median_beta": float(x.median()) if len(x) else np.nan,
                "std_beta": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                "ci95_low": float(np.quantile(x, 0.025)) if len(x) else np.nan,
                "ci95_high": float(np.quantile(x, 0.975)) if len(x) else np.nan,
                "prop_beta_positive": float((x > 0).mean()) if len(x) else np.nan,
                "prop_p_lt_05": float((p < 0.05).mean()) if p is not None and len(x) else np.nan,
            }
        )

    # Selection stability extras
    rows.append(
        {
            "estimate": "selection_includes_all_posterior_core",
            "n_splits": int(len(ok)),
            "mean_beta": float(ok["includes_all_posterior_core"].mean()),
            "median_beta": np.nan,
            "std_beta": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "prop_beta_positive": np.nan,
            "prop_p_lt_05": np.nan,
        }
    )
    rows.append(
        {
            "estimate": "mean_n_overlap_posterior_core",
            "n_splits": int(len(ok)),
            "mean_beta": float(ok["n_overlap_posterior_core"].mean()),
            "median_beta": float(ok["n_overlap_posterior_core"].median()),
            "std_beta": float(ok["n_overlap_posterior_core"].std(ddof=1)) if len(ok) > 1 else np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "prop_beta_positive": np.nan,
            "prop_p_lt_05": np.nan,
        }
    )
    rows.append(
        {
            "estimate": "prop_selection_rule_fdr",
            "n_splits": int(len(ok)),
            "mean_beta": float((ok["selection_rule"] == "fdr_significant").mean()),
            "median_beta": np.nan,
            "std_beta": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "prop_beta_positive": np.nan,
            "prop_p_lt_05": np.nan,
        }
    )
    return pd.DataFrame(rows)


def plot_nested_split_summary(
    splits_df: pd.DataFrame,
    freq_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Two-panel figure: hold-out beta distribution + channel selection rates."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    ax = axes[0]
    hold = pd.to_numeric(splits_df["holdout_beta"], errors="coerce").dropna()
    train = pd.to_numeric(splits_df["train_beta"], errors="coerce").dropna()
    fixed = pd.to_numeric(splits_df["fixed_core_holdout_beta"], errors="coerce").dropna()
    ax.hist(hold, bins=20, color="#333333", alpha=0.75, label="Train-selected ROI (hold-out)")
    if len(train):
        ax.axvline(train.mean(), color="#888888", ls="--", lw=1.5, label=f"In-sample mean={train.mean():.3f}")
    if len(hold):
        ax.axvline(hold.mean(), color="#D23538", lw=1.8, label=f"Hold-out mean={hold.mean():.3f}")
    if len(fixed):
        ax.axvline(fixed.mean(), color="#FDB933", lw=1.5, label=f"Fixed core hold-out mean={fixed.mean():.3f}")
    ax.axvline(0, color="#666666", lw=0.8)
    ax.set_xlabel("β (TD − ASD)")
    ax.set_ylabel("Count")
    ax.set_title("Hold-out group effect across splits", fontsize=9)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    top = freq_df.head(12).copy()
    colors = ["#D23538" if bool(v) else "#9A9A9A" for v in top["is_posterior_core"]]
    ax.barh(top["channel"][::-1], (top["selection_rate"] * 100)[::-1], color=colors[::-1])
    ax.set_xlabel("Selection rate (%)")
    ax.set_title("Channel inclusion in train-selected ROI", fontsize=9)
    ax.set_xlim(0, 105)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_nested_split_report(
    summary_df: pd.DataFrame,
    freq_df: pd.DataFrame,
    single_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    out_path: Path,
    *,
    n_subjects: int,
) -> None:
    """Markdown report with English/Chinese draft snippets."""
    def _row(name: str) -> pd.Series | None:
        sub = summary_df[summary_df["estimate"] == name]
        return sub.iloc[0] if len(sub) else None

    hold = _row("train_selected_ROI_holdout")
    ins = _row("train_selected_ROI_insample")
    fixed = _row("fixed_posterior_core_holdout")
    all_four = _row("selection_includes_all_posterior_core")
    prop_fdr = _row("prop_selection_rule_fdr")

    top_lines = []
    for _, r in freq_df.head(8).iterrows():
        mark = " [posterior core]" if r["is_posterior_core"] else ""
        top_lines.append(
            f"- {r['channel']}: {100 * r['selection_rate']:.1f}% ({int(r['n_selected'])}/{int(r['n_splits'])}){mark}"
        )

    single_txt = ""
    if single_df is not None and not single_df.empty:
        s = single_df.iloc[0]
        single_txt = (
            f"- Discovery/validation 50/50 (seed=42): ROI={s['roi_channels']}; "
            f"discovery β={s['discovery_beta']:.3f} (p={s['discovery_p']:.4g}); "
            f"validation β={s['validation_beta']:.3f} (p={s['validation_p']:.4g}); "
            f"fixed-core validation β={s['fixed_core_validation_beta']:.3f} "
            f"(p={s['fixed_core_validation_p']:.4g})\n"
        )

    def fmt(r: pd.Series | None) -> str:
        if r is None or pd.isna(r.get("mean_beta", np.nan)):
            return "n/a"
        return (
            f"mean β={r['mean_beta']:.3f} "
            f"(median={r['median_beta']:.3f}; 95% split-CI [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]; "
            f"P(β>0)={100 * r['prop_beta_positive']:.1f}%; "
            f"P(p<.05)={100 * r['prop_p_lt_05']:.1f}%; n_splits={int(r['n_splits'])})"
        )

    atten = ""
    if hold is not None and ins is not None and pd.notna(hold["mean_beta"]) and pd.notna(ins["mean_beta"]) and abs(ins["mean_beta"]) > 1e-8:
        atten = f"{100 * (1 - hold['mean_beta'] / ins['mean_beta']):.1f}%"

    n_rep = int(splits_df["split_id"].nunique()) if not splits_df.empty else 0
    text = f"""# Posterior ROI nested / repeated split-sample validation

## Purpose
Quantify the group effect of a **data-driven ROI** when channel selection and effect
estimation are separated across train/test splits (mitigates same-sample selection–
inference circularity and winner's curse inflation). This does **not** convert the ROI
into an a priori anatomical ROI.

## Design
- Cohort: N = {n_subjects} resting spectral participants (same as channel-level FDR)
- Model: `{FORMULA}`
- Selection (train): BH-FDR significant channels; fallback = top-|β| uncorrected / top-|β|
- Estimation (test only): mean exponent across selected channels → OLS group β (TD − ASD)
- Modes: repeated stratified 70/30 splits + stratified K-fold + one 50/50 discovery/validation split

## Key quantitative results (repeated splits)
- Train-selected ROI **hold-out**: {fmt(hold)}
- Train-selected ROI **in-sample** (same split, optimistic): {fmt(ins)}
- Fixed posterior core (E33/E36/E37/E38) **hold-out**: {fmt(fixed)}
- Mean attenuation (in-sample → hold-out) for train-selected ROI: {atten or "n/a"}
- Splits where selection used FDR (not fallback): {100 * float(prop_fdr['mean_beta']):.1f}% ({n_rep} splits)
- Splits where all four posterior-core electrodes were included: {100 * float(all_four['mean_beta']):.1f}%

{single_txt}
## Channel selection frequency (top)
{chr(10).join(top_lines)}

## English Results snippet
To address circularity between data-driven posterior channel selection and subsequent
inference, we performed repeated stratified train/test splits (and complementary K-fold
and 50/50 discovery/validation analyses). In each split, scalp-wide channel models with
BH-FDR correction were fit in the training subset only; the selected channels were then
averaged and the TD−ASD group effect was estimated exclusively in the held-out test
subset. Across repeated splits, the hold-out effect for the train-selected ROI was
{fmt(hold)}. Relative to the corresponding in-sample estimate ({fmt(ins)}), this indicates
attenuation consistent with winner's-curse bias. The fixed four-electrode posterior core
evaluated on the same held-out subsets yielded {fmt(fixed)}. Channel inclusion frequencies
were highest for posterior electrodes (see selection table), supporting spatial
concentration without claiming an independently pre-specified topographic dissociation.

## 中文结果草稿
为降低「同一数据选 ROI、同一数据做推断」的循环偏倚，我们在分层划分的训练集上重新进行
全头皮通道 FDR 筛选，仅在测试集上估计所选通道均值的组间效应（TD−ASD）。重复划分下，
训练选 ROI 的 hold-out 效应为：{fmt(hold)}；同折的样本内估计为：{fmt(ins)}（可见衰减）。
固定后枕四导（E33/E36/E37/E38）在相同 hold-out 子集上的效应为：{fmt(fixed)}。
通道入选频率以后部电极最高。该分析缓解选择偏差带来的效应膨胀，但并不把 ROI 变成
先验独立定义的解剖 ROI；正文对 “posterior localization” 的表述宜保持审慎。

## Interpretation bounds
- Hold-out significance rate is not a single confirmatory p-value; report the β distribution.
- Region × group formal interactions may remain non-significant; do not overclaim strict localization.
- Prefer pairing this analysis with the a priori / larger occipital ROI sensitivity already reported.
"""
    Path(out_path).write_text(text, encoding="utf-8")


def run_posterior_nested_split_pipeline(
    channel_df: pd.DataFrame,
    participants: pd.DataFrame,
    outputs_root: Path,
    *,
    n_repeats: int = 200,
    test_size: float = 0.30,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Path]:
    """Run all split modes and write tables / figure / report."""
    tables = Path(outputs_root) / "tables" / "robustness"
    reports = Path(outputs_root) / "reports"
    figs = Path(outputs_root) / "figures" / "robustness"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    logger.info("Running repeated stratified splits (n=%d)...", n_repeats)
    rep = run_repeated_split_validation(
        channel_df,
        participants,
        n_splits=n_repeats,
        test_size=test_size,
        random_state=random_state,
    )
    logger.info("Running stratified %d-fold nested validation...", n_folds)
    kf = run_kfold_nested_validation(
        channel_df,
        participants,
        n_splits=n_folds,
        random_state=random_state,
    )
    logger.info("Running single 50/50 discovery/validation split...")
    single, disc_ch = run_single_split_discovery_validation(
        channel_df,
        participants,
        test_size=0.50,
        random_state=random_state,
    )

    splits_all = pd.concat([rep, kf], ignore_index=True)
    freq = channel_selection_frequency(rep)
    summary = summarize_holdout_effects(rep)
    freq_kf = channel_selection_frequency(kf)
    summary_kf = summarize_holdout_effects(kf)

    paths = {
        "repeated_splits": tables / "posterior_roi_nested_split_repeats.csv",
        "kfold_splits": tables / "posterior_roi_nested_split_kfold.csv",
        "summary_repeats": tables / "posterior_roi_nested_split_summary_repeats.csv",
        "summary_kfold": tables / "posterior_roi_nested_split_summary_kfold.csv",
        "selection_freq_repeats": tables / "posterior_roi_nested_split_selection_freq_repeats.csv",
        "selection_freq_kfold": tables / "posterior_roi_nested_split_selection_freq_kfold.csv",
        "single_split": tables / "posterior_roi_nested_split_single50.csv",
        "single_discovery_channels": tables / "posterior_roi_nested_split_single50_discovery_channels.csv",
        "figure": figs / "fig_posterior_roi_nested_split_validation.png",
        "report": reports / "posterior_roi_nested_split_validation_report.md",
    }

    save_csv(rep, paths["repeated_splits"])
    save_csv(kf, paths["kfold_splits"])
    save_csv(summary, paths["summary_repeats"])
    save_csv(summary_kf, paths["summary_kfold"])
    save_csv(freq, paths["selection_freq_repeats"])
    save_csv(freq_kf, paths["selection_freq_kfold"])
    save_csv(single, paths["single_split"])
    if disc_ch is not None and not disc_ch.empty:
        save_csv(disc_ch, paths["single_discovery_channels"])
    plot_nested_split_summary(rep, freq, paths["figure"])
    write_nested_split_report(
        summary,
        freq,
        single,
        rep,
        paths["report"],
        n_subjects=int(participants["subject_id"].nunique()),
    )
    return paths
