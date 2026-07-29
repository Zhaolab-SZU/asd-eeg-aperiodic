"""BP journal figure style constants and helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_W_IN = 7.48  # 190 mm

COLOR_TD = "#0072B2"
COLOR_ASD = "#D55E00"
COLOR_ROI = "#332288"
COLOR_GRAY = "#666666"
COLOR_NEUTRAL = "#2A2A2A"

POSTERIOR_CORE = ["E33", "E36", "E37", "E38"]
MONTAGE_NAME = "GSN-HydroCel-64_1.0"

BP_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_bp_style() -> None:
    plt.rcParams.update(BP_RC)


def save_figure(fig: plt.Figure, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / name
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_label(ax: plt.Axes, letter: str, x: float = -0.14, y: float = 1.06) -> None:
    ax.text(
        x, y, letter, transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="top", ha="left",
    )


def format_p(p: float) -> str:
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def annotate_box(ax: plt.Axes, text: str, loc: str = "top_right") -> None:
    coords = {
        "top_right": (0.98, 0.98, "right", "top"),
        "top_left": (0.02, 0.98, "left", "top"),
        "bottom_right": (0.98, 0.02, "right", "bottom"),
    }
    x, y, ha, va = coords.get(loc, coords["top_right"])
    ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", alpha=0.95),
    )


def write_caption(fig_num: int, text: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"Figure{fig_num}_caption.txt").write_text(text.strip() + "\n", encoding="utf-8")
