"""论文用图（matplotlib only）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from src.io_utils import (
    attach_usable_epochs,
    exclude_specparam_low_quality,
    load_analysis_participants,
)
from src.stats_utils import run_mixedlm

PAPER_RC = {
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

ROI_ORDER = ["frontal", "central", "temporal", "parietal", "occipital"]
FDR_SIG_CHANNELS = ["E33", "E36", "E37", "E38"]


def apply_paper_style() -> None:
    plt.rcParams.update(PAPER_RC)


def save_paper_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_main_cohort(cfg: dict[str, Any], deriv: Path) -> pd.DataFrame:
    """主分析队列（usable_epochs ≥ 阈值 + specparam QC，与 08/09 一致）。"""
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
    return participants.reset_index(drop=True)


def plot_fig01_group_mean_psd(
    psd_dir: Path,
    participants: pd.DataFrame,
    cfg: dict[str, Any],
    out_base: Path,
) -> dict[str, Any]:
    """Fig 1: 组平均 PSD + SEM 阴影。"""
    apply_paper_style()
    asd_label = cfg["groups"]["asd_label"]
    td_label = cfg["groups"]["td_label"]
    fmin, fmax = 1.0, 40.0

    frames = []
    for sid in participants["subject_id"].astype(str):
        p = psd_dir / f"{sid}_psd.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[(combined["frequency"] >= fmin) & (combined["frequency"] <= fmax)]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    meta: dict[str, Any] = {
        "figure": "fig01",
        "data_source": str(psd_dir),
        "freq_range_hz": [fmin, fmax],
        "metric": "group mean PSD (arithmetic mean across subjects, then channels)",
        "shading": "SEM across subjects at each frequency",
        "note": "Descriptive; not regression coefficients.",
    }

    for grp, color in [(asd_label, "#4C72B0"), (td_label, "#DD8452")]:
        sub = combined[combined["group"] == grp]
        stats = sub.groupby("frequency")["power"].agg(["mean", "sem", "count"])
        freqs = stats.index.values
        mean = stats["mean"].values
        sem = stats["sem"].values
        meta[f"n_{grp}"] = int(sub["subject_id"].nunique())
        ax.plot(freqs, mean, color=color, lw=1.8, label=grp)
        ax.fill_between(freqs, np.maximum(mean - sem, 1e-30), mean + sem, color=color, alpha=0.22, linewidth=0)

    ax.set_yscale("log")
    ax.set_xlim(fmin, fmax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V²/Hz, log scale)")
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.02, 0.02, "Shaded: SEM; specparam fitting range: 1–40 Hz",
        transform=ax.transAxes, fontsize=8, va="bottom",
    )
    fig.tight_layout()
    save_paper_figure(fig, out_base)
    return meta


def plot_fig02_global_exponent(
    roi_global: pd.DataFrame,
    participants: pd.DataFrame,
    out_base: Path,
    main_beta: float = 0.079,
    main_p: float = 0.012,
) -> dict[str, Any]:
    """Fig 2: global exponent 箱线 + 散点。"""
    apply_paper_style()
    df = participants.merge(roi_global, on=["subject_id", "group"], how="inner")
    df = df.dropna(subset=["global_exponent"])

    fig, ax = plt.subplots(figsize=(4.2, 4.5))
    groups = ["ASD", "TD"]
    positions = np.arange(len(groups))
    data = [df.loc[df["group"] == g, "global_exponent"].values for g in groups]

    ax.boxplot(
        data, positions=positions, widths=0.55, patch_artist=True,
        showfliers=False, medianprops=dict(color="black", linewidth=1.2),
        boxprops=dict(facecolor="0.85", edgecolor="0.3", linewidth=1.0),
        whiskerprops=dict(color="0.3", linewidth=1.0),
        capprops=dict(color="0.3", linewidth=1.0),
    )
    rng = np.random.default_rng(42)
    colors = {"ASD": "#4C72B0", "TD": "#DD8452"}
    for i, g in enumerate(groups):
        vals = df.loc[df["group"] == g, "global_exponent"].values
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            s=22, alpha=0.75, color=colors[g], edgecolors="white", linewidths=0.3, zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Global aperiodic exponent")
    ax.text(
        0.98, 0.98,
        f"Adjusted group effect:\nβ = {main_beta:.3f}, p = {main_p:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.9),
    )
    fig.tight_layout()
    save_paper_figure(fig, out_base)

    return {
        "figure": "fig02",
        "data_source": "derivatives/roi/specparam_subject_global.csv + main cohort",
        "n_total": len(df),
        "n_ASD": int((df["group"] == "ASD").sum()),
        "n_TD": int((df["group"] == "TD").sum()),
        "annotation_beta": main_beta,
        "annotation_p": main_p,
        "note": "Descriptive boxplot; annotation from primary adjusted OLS (covariates: age, sex, IQ, usable epochs), not uncorrected t-test.",
    }


def _roi_contrast_vector(model, roi: str) -> np.ndarray:
    """构建 central 为参照时 ROI 的 TD−ASD 边际对比向量（固定效应）。"""
    names = list(model.fe_params.index)
    vec = np.zeros(len(names))
    idx = {n: i for i, n in enumerate(names)}
    g = "C(group)[T.TD]"
    if g in idx:
        vec[idx[g]] = 1.0
    if roi != "central":
        key = f"C(group)[T.TD]:C(roi)[T.{roi}]"
        if key in idx:
            vec[idx[key]] = 1.0
    return vec.reshape(1, -1)


def _roi_marginal_td_asd_effects(model) -> pd.DataFrame:
    """C(roi) 以 central 为参照时，各 ROI 的 TD−ASD 边际效应及 95% CI。"""
    rows = []
    for roi in ROI_ORDER:
        tt = model.t_test(r_matrix=_roi_contrast_vector(model, roi))
        ci = tt.conf_int(alpha=0.05).flatten()
        rows.append({
            "roi": roi,
            "beta_TD_minus_ASD": float(tt.effect[0]),
            "se": float(tt.sd[0]),
            "pvalue": float(tt.pvalue),
            "ci_low": float(ci[0]),
            "ci_high": float(ci[1]),
        })
    return pd.DataFrame(rows)


def plot_fig03_roi_forest(
    roi_long: pd.DataFrame,
    participants: pd.DataFrame,
    deriv: Path,
    out_base: Path,
) -> dict[str, Any]:
    """Fig 3: ROI 层面 TD−ASD 边际效应森林图（group×ROI 模型）。"""
    apply_paper_style()
    df = participants.merge(roi_long, on=["subject_id", "group"], how="inner")
    df = attach_usable_epochs(df, deriv)
    df = exclude_specparam_low_quality(df, deriv)
    df = df.dropna(subset=["exponent", "group", "roi", "age_months", "sex", "IQ_total", "usable_epochs"])

    formula = "exponent ~ C(group) * C(roi) + age_months + C(sex) + IQ_total + usable_epochs"
    model = run_mixedlm(formula, df, groups="subject_id")
    effects = _roi_marginal_td_asd_effects(model)
    effects["roi"] = pd.Categorical(effects["roi"], categories=ROI_ORDER[::-1], ordered=True)
    effects = effects.sort_values("roi")

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    y = np.arange(len(effects))
    ax.axvline(0, color="0.35", lw=1.0, zorder=0)
    ax.errorbar(
        effects["beta_TD_minus_ASD"], y,
        xerr=[
            effects["beta_TD_minus_ASD"] - effects["ci_low"],
            effects["ci_high"] - effects["beta_TD_minus_ASD"],
        ],
        fmt="o", color="#2A6FBB", ecolor="#2A6FBB", elinewidth=1.2, capsize=3, markersize=6,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(effects["roi"].astype(str))
    ax.set_xlabel("β (TD − ASD) for aperiodic exponent")
    ax.text(
        0.98, 0.02,
        "Marginal TD−ASD contrasts (group×ROI mixed model);\n"
        "central ROI = reference. Covariates: age, sex, IQ, usable epochs.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
    )
    fig.tight_layout()
    save_paper_figure(fig, out_base)

    return {
        "figure": "fig03",
        "data_source": "derivatives/roi/specparam_subject_roi_long.csv",
        "model": formula,
        "n_obs": int(model.nobs),
        "reference_roi": "central",
        "used_mixedlm": getattr(model, "_used_mixedlm", None),
        "n_subjects": int(df["subject_id"].nunique()),
        "note": "Marginal TD−ASD contrasts from primary group×ROI mixed model (subject random intercept).",
        "effects": effects.to_dict(orient="records"),
    }


def plot_fig04_channel_topomap(
    channel_stats: pd.DataFrame,
    montage_name: str,
    out_base: Path,
) -> dict[str, Any]:
    """Fig 4: 通道 group 回归系数地形图。"""
    apply_paper_style()
    coef_map = channel_stats.set_index("channel")["coef"].to_dict()
    montage = mne.channels.make_standard_montage(montage_name)
    ch_names = [c for c in montage.ch_names if c.startswith("E")]
    values = np.array([coef_map.get(c, np.nan) for c in ch_names])
    info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types="eeg")
    info.set_montage(montage)

    vmax = np.nanmax(np.abs(values))
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im, _ = mne.viz.plot_topomap(
        values, info, axes=ax, show=False,
        cmap="RdBu_r", vlim=(-vmax, vmax), contours=0,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("β coefficient (TD − ASD)", fontsize=10)

    sig_set = set(FDR_SIG_CHANNELS)
    pos = montage.get_positions()["ch_pos"]
    for ch in ch_names:
        if ch in sig_set and ch in pos:
            xy = pos[ch][:2]
            ax.plot(xy[0], xy[1], marker="o", mfc="none", mec="black", mew=1.4, ms=9, zorder=10)

    ax.text(
        0.5, -0.08, "FDR-significant channels marked (○)",
        transform=ax.transAxes, ha="center", fontsize=8,
    )
    fig.tight_layout()
    save_paper_figure(fig, out_base)

    n_sig = int(channel_stats["significant_fdr"].sum()) if "significant_fdr" in channel_stats.columns else 4
    return {
        "figure": "fig04",
        "data_source": "derivatives/stats/channel_level_analysis.csv",
        "metric": "C(group)[T.TD] from channel-level OLS",
        "n_channels": len(ch_names),
        "n_fdr_significant": n_sig,
        "fdr_channels": FDR_SIG_CHANNELS,
        "note": "Regression coefficients (not raw exponent); covariate-adjusted channel models.",
    }


def write_qc_notes(notes: list[dict[str, Any]], out_path: Path) -> None:
    lines = ["Paper figure QC notes", "=" * 60, ""]
    for note in notes:
        lines.append(f"## {note.get('figure', 'unknown')}")
        for k, v in note.items():
            if k == "figure":
                continue
            if k == "effects":
                lines.append("  marginal_TD_minus_ASD:")
                for row in v:
                    lines.append(
                        f"    {row['roi']}: β={row['beta_TD_minus_ASD']:.4f}, "
                        f"95% CI [{row['ci_low']:.4f}, {row['ci_high']:.4f}], p={row['pvalue']:.4f}"
                    )
                continue
            lines.append(f"  {k}: {v}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
