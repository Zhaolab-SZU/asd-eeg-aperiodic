"""Submission figure style constants (Biological Psychiatry v2 palette)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

COL_TD = "#FDB933"
COL_ASD = "#D23538"
COL_EDGE = "#4D4D4D"
COL_GRAY = "#B8B8B8"
COL_SIG = "#333333"
COL_NS = "#9A9A9A"

POSTERIOR_CORE = ["E33", "E36", "E37", "E38"]

RAINCLOUD_POINT_SIZE = 16
RAINCLOUD_POINT_ALPHA = 0.60
RAINCLOUD_VIOLIN_WIDTH = 0.35
RAINCLOUD_JITTER = (0.05, 0.22)
RAINCLOUD_MEDIAN_HW = 0.38
STAT_ANNOT_FONTSIZE = 7
STAT_FOOTNOTE_FONTSIZE = 6.5

TOPO_VLIM = 0.25
FOREST_MARKER_SIZE = 6
CLUSTER_MARKER_SIZE = 12
CLUSTER_RING_SIZE = 13
CLUSTER_RING_WIDTH = 1.5
COL_ROI_SIG = "#333333"
COL_ROI_FADE = "#C5C5C5"
ROBUSTNESS_MARKER_SIZE = 9
ROBUSTNESS_ELINEWIDTH = 2.5
ROBUSTNESS_ANCHOR_ALPHA = 0.5
ZEBRA_BAND_COLOR = "#F4F4F4"

PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "sans-serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
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


def apply_submission_style() -> None:
    plt.rcParams.update(PAPER_RC)


def group_colors() -> dict[str, str]:
    return {"TD": COL_TD, "ASD": COL_ASD}


def save_figure(fig: plt.Figure, out_base: Path, dpi: int = 600) -> None:
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12, 1.08, label, transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="top", ha="left",
    )
