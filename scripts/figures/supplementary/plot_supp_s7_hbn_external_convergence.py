from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]  # repo root
SRC = ROOT / "data" / "supplementary_source"
OUT = ROOT / "outputs" / "figures" / "supplementary"
BACKUP = OUT.parent / "backup"; BACKUP.mkdir(parents=True, exist_ok=True)
BACKUP.mkdir(parents=True, exist_ok=True)

STEM = "Archived_HBN_movie_external_convergence"

ASD = "#C25450"
TD = "#4A6FA5"
ASD_LIGHT = "#E9A6A2"
TD_LIGHT = "#B8C9E2"
INK = "#333333"
MUTED = "#666666"
GREY = "#9A9A9A"
LIGHT_GREY = "#DADADA"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def backup_existing() -> None:
    for ext in ("png", "pdf", "svg", "tiff"):
        src = OUT / f"{STEM}.{ext}"
        if src.exists():
            dst = BACKUP / f"{STEM}.{ext}"
            if not dst.exists():
                shutil.copy2(src, dst)


def style_axes(ax):
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(labelsize=7, colors=INK, width=0.9, length=3.2)
    ax.grid(False)


def panel_label(ax, label, x=-0.16, y=1.06):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )


def p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def draw_movie_panel(ax, subjects: pd.DataFrame, summary: pd.DataFrame, analysis: str, title: str, label: str):
    panel_label(ax, label)
    rng = np.random.default_rng(20260713 + len(analysis))
    for i, (group, color, light) in enumerate([("ASD", ASD, ASD_LIGHT), ("TD", TD, TD_LIGHT)]):
        vals = subjects[(subjects["analysis"] == analysis) & (subjects["group"] == group)]["isc_z"].dropna().to_numpy()
        parts = ax.violinplot(vals, positions=[i], widths=0.62, showmeans=False, showextrema=False, showmedians=False)
        for body in parts["bodies"]:
            body.set_facecolor(light)
            body.set_edgecolor(color)
            body.set_alpha(0.62)
            body.set_linewidth(0.9)
        ax.scatter(
            np.full(len(vals), i) + rng.normal(0, 0.045, len(vals)),
            vals,
            s=11,
            color=color,
            alpha=0.50,
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )
        mean = np.mean(vals)
        ci = stats.sem(vals) * stats.t.ppf(0.975, len(vals) - 1)
        ax.errorbar(i, mean, yerr=ci, fmt="o", color=INK, ecolor=INK,
                    markersize=3.6, elinewidth=1.0, capsize=3.2, zorder=5)
    p = summary[(summary["analysis"] == analysis) & (summary["group"] == "ASD")]["p"].iloc[0]
    y = 0.66
    ax.plot([0, 0, 1, 1], [y, y + 0.03, y + 0.03, y], color=INK, lw=0.8)
    stars = p_to_stars(float(p))
    ax.text(
        0.5,
        y + 0.036,
        stars,
        ha="center",
        va="bottom",
        fontsize=11.0,
        color=ASD,
        fontweight="bold",
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ASD", "TD"], fontsize=7)
    ax.set_ylim(-0.55, 0.75)
    ax.set_title(title, fontsize=8.8, fontweight="bold", color=INK, pad=6)
    ax.set_ylabel("Aperiodic-ISC (Fisher z)", fontsize=8, fontweight="bold")
    style_axes(ax)


def draw() -> None:
    backup_existing()
    subjects = pd.read_csv(SRC / "s8_hbn_movie_subjects.csv")
    summary = pd.read_csv(SRC / "s8_hbn_movie_summary.csv")

    fig = plt.figure(figsize=(5.0, 3.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.42)
    draw_movie_panel(fig.add_subplot(gs[0, 0]), subjects, summary, "sliding_window", "Movie ISC, sliding windows", "A")
    draw_movie_panel(fig.add_subplot(gs[0, 1]), subjects, summary, "nonoverlapping_2s_epoch", "Movie ISC, 2-s epochs", "B")
    fig.subplots_adjust(left=0.065, right=0.990, top=0.88, bottom=0.18)

    for ext, kw in {
        "png": {"dpi": 600},
        "tiff": {"dpi": 600},
        "pdf": {},
        "svg": {},
    }.items():
        fig.savefig(OUT / f"{STEM}.{ext}", bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)
    print(OUT / f"{STEM}.png")


if __name__ == "__main__":
    draw()
