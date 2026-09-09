"""Shared stylesheet for Supplementary Figures S1–S5 (firebrick/navy).

Outputs go to ``outputs/figures/supplementary/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve()
while not (ROOT / "src" / "config.py").is_file():
    if ROOT.parent == ROOT:
        raise RuntimeError("Cannot locate repository root (missing src/config.py)")
    ROOT = ROOT.parent

MAIN_SCRIPTS = ROOT / "scripts" / "figures" / "main"
if str(MAIN_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MAIN_SCRIPTS))

from _grammar_style import (  # noqa: E402
    ASD, AX, DATA, INK, LABEL, LABEL_GRID, LABEL_ROW, LEGEND, LETTER, LETTER_ROW,
    NOTE, PANEL_Y, REF, SPINE_LW, SUPP, TD, TICK, TICK_GRID, TITLE, TITLE_GRID,
    TITLE_PAD, TITLE_ROW, ZERO, add_sig, box_strip, clean, draw_forest,
    group_legend_handles, ols_ci, p_to_stars, panel,
)

OUT = ROOT / "outputs" / "figures" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

ASD_LIGHT = "#E8B4B4"
TD_LIGHT = "#B3B3D9"
SCATTER_S = 20.0
SCATTER_LW = 0.22
REG_LW = 1.70


def save(fig, stem, pad_inches=0.10):
    path = OUT / stem
    for ext in (".png", ".pdf", ".svg"):
        kw = dict(bbox_inches="tight", pad_inches=pad_inches, facecolor="white")
        if ext == ".png":
            kw["dpi"] = 300
        fig.savefig(path.with_suffix(ext), **kw)
    plt.close(fig)
    print(path.with_suffix(".png"))


def scatter_pts(ax, x, y, color, s=SCATTER_S, zorder=3):
    ax.scatter(
        x, y, s=s, facecolor=color, edgecolors="black",
        linewidths=SCATTER_LW, alpha=1.0, zorder=zorder,
    )


def reg_line(ax, x, y, color, n=120):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    coef = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), n)
    ax.plot(
        xx, coef[0] * xx + coef[1], color=color, lw=REG_LW,
        solid_capstyle="round", zorder=2,
    )
    return xx, coef


def p_label(p):
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def pair_box_strip(ax, asd_vals, td_vals, sig, ylabel, title, ylim=None, yticks=None):
    """Fig. 1c grammar: ASD left of box, TD right of box."""
    positions = {"ASD": 0.82, "TD": 1.28}
    fill = {"ASD": ASD, "TD": TD}
    vals_map = {"ASD": np.asarray(asd_vals, float), "TD": np.asarray(td_vals, float)}
    maxima = []
    minima = []
    for group in ("ASD", "TD"):
        vals = vals_map[group]
        vals = vals[np.isfinite(vals)]
        maxima.append(float(np.nanmax(vals)))
        minima.append(float(np.nanmin(vals)))
        box_strip(
            ax, vals, positions[group], fill[group],
            width=0.24, side=-1 if group == "ASD" else +1,
        )
    data_max = max(maxima)
    data_min = min(minima)
    if ylim is None:
        span0 = data_max - data_min
        ymin, ylim_top = data_min - 0.08 * span0, data_max + 0.22 * span0
    else:
        ymin, ylim_top = ylim
    span = ylim_top - ymin
    top = max(ylim_top, data_max + 0.22 * span)
    ax.set_xlim(0.45, 1.55)
    ax.set_xticks([positions["ASD"], positions["TD"]], ["ASD", "TD"])
    ax.set_ylim(ymin, top)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_title(title, pad=TITLE_PAD, fontweight="bold")
    ax.set_xlabel("Group")
    ax.set_ylabel(ylabel)
    span = top - ymin
    kw = {}
    if str(sig).lower().replace(".", "") == "ns":
        kw = dict(fontsize=NOTE, color=INK)
    add_sig(
        ax, positions["ASD"], positions["TD"],
        data_max + 0.055 * span, 0.016 * span, sig, **kw,
    )
    clean(ax)
    return top
