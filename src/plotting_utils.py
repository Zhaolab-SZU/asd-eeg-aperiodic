"""matplotlib 画图工具。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from src.io_utils import ensure_dir

logger = logging.getLogger(__name__)

# 默认样式
plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save_figure(fig: plt.Figure, out_path: Path, dpi: int = 150) -> None:
    """同时保存 PNG 和 PDF。"""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("已保存图: %s", out_path)


def plot_box_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    out_path: Path,
    title: str = "",
    ylabel: str = "",
) -> None:
    """箱线图 + 个体散点（抖动）。"""
    fig, ax = plt.subplots(figsize=(5, 5))
    groups = df[x].dropna().unique()
    positions = np.arange(len(groups))
    data = [df.loc[df[x] == g, y].dropna().values for g in groups]
    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_alpha(0.4)
    for i, g in enumerate(groups):
        vals = df.loc[df[x] == g, y].dropna().values
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, alpha=0.7, s=30, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title)
    save_figure(fig, out_path)


def plot_roi_exponent(
    roi_long: pd.DataFrame,
    out_path: Path,
    group_col: str = "group",
) -> None:
    """ROI exponent 组间对比（点 + 箱线）。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    rois = roi_long["roi"].unique()
    groups = roi_long[group_col].unique()
    width = 0.35
    for gi, grp in enumerate(groups):
        sub = roi_long[roi_long[group_col] == grp]
        means = [sub.loc[sub["roi"] == r, "exponent"].mean() for r in rois]
        sems = [sub.loc[sub["roi"] == r, "exponent"].sem() for r in rois]
        x = np.arange(len(rois)) + (gi - 0.5) * width
        ax.bar(x, means, width=width, yerr=sems, label=grp, alpha=0.6, capsize=3)
        for ri, r in enumerate(rois):
            vals = sub.loc[sub["roi"] == r, "exponent"].dropna()
            jitter = np.random.default_rng(42).uniform(-0.05, 0.05, len(vals))
            ax.scatter(x[ri] + jitter, vals, s=20, alpha=0.6, zorder=3)
    ax.set_xticks(np.arange(len(rois)))
    ax.set_xticklabels(rois)
    ax.set_ylabel("Aperiodic exponent")
    ax.legend()
    ax.set_title("ROI aperiodic exponent by group")
    save_figure(fig, out_path)


def plot_clinical_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    out_path: Path,
    title: str = "",
) -> None:
    """临床变量散点图 + 拟合线。"""
    sub = df[[x, y]].dropna()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(sub[x], sub[y], alpha=0.7)
    if len(sub) >= 3:
        z = np.polyfit(sub[x], sub[y], 1)
        xs = np.linspace(sub[x].min(), sub[x].max(), 50)
        ax.plot(xs, np.poly1d(z)(xs), "r--", alpha=0.8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    save_figure(fig, out_path)


def plot_exponent_topomap(
    channel_df: pd.DataFrame,
    effect_col: str,
    out_path: Path,
    montage: str = "GSN-HydroCel-64_1.0",
) -> None:
    """通道水平效应 topomap（需 MNE）。"""
    ch_df = channel_df.dropna(subset=[effect_col])
    if ch_df.empty:
        logger.warning("无有效通道数据，跳过 topomap")
        return
    info = mne.create_info(ch_df["channel"].tolist(), sfreq=250.0, ch_types="eeg")
    try:
        info.set_montage(montage)
    except Exception as exc:
        logger.warning("topomap montage 设置失败: %s", exc)
        return
    data = ch_df[effect_col].values[np.newaxis, :]
    fig, ax = plt.subplots(figsize=(6, 5))
    im, _ = mne.viz.plot_topomap(
        data[0],
        info,
        axes=ax,
        show=False,
        cmap="RdBu_r",
    )
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"Channel-level {effect_col}")
    save_figure(fig, out_path)


def plot_specparam_example(
    freqs: np.ndarray,
    power: np.ndarray,
    model_fit: np.ndarray | None,
    out_path: Path,
    title: str = "specparam fit example",
) -> None:
    """specparam 拟合示例图。"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(freqs, power, "k", alpha=0.6, label="PSD")
    if model_fit is not None:
        ax.loglog(freqs, model_fit, "r--", linewidth=2, label="Model fit")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(title)
    ax.legend()
    save_figure(fig, out_path)


def plot_band_power_comparison(
    traditional: pd.DataFrame,
    specparam_peak: pd.DataFrame,
    band: str,
    out_path: Path,
) -> None:
    """传统频段功率 vs specparam peak power 对照。"""
    trad = traditional[traditional["band"] == band].groupby("subject_id")["band_power"].mean()
    spec = specparam_peak.groupby("subject_id")[f"{band}_pw"].mean()
    common = trad.index.intersection(spec.index)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(trad[common], spec[common], alpha=0.7)
    ax.set_xlabel(f"Traditional {band} power")
    ax.set_ylabel(f"specparam {band} peak power")
    ax.set_title(f"{band}: traditional vs specparam")
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.3)
    save_figure(fig, out_path)
