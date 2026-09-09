"""Shared stylesheet for manuscript Figures 1–5 (firebrick/navy).

Outputs go to ``outputs/figures/main/``. Figure CSVs are resolved across
``figure_source_data/``, ``data/figure_source/``, and the nested August copy
``figure_source_data/outputs_figure_source_data/``.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from scipy import stats
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve()
while not (ROOT / "src" / "config.py").is_file():
    if ROOT.parent == ROOT:
        raise RuntimeError("Cannot locate repository root (missing src/config.py)")
    ROOT = ROOT.parent

_DATA_DIRS = (
    ROOT / "figure_source_data",
    ROOT / "data" / "figure_source",
    ROOT / "figure_source_data" / "outputs_figure_source_data",
)
_SUPP_DIRS = (
    ROOT / "figure_source_data" / "supplementary",
    ROOT / "data" / "supplementary_source",
)


class _SearchPath:
    """Path-like locator that searches shipped CSV directories in order."""

    def __init__(self, roots: tuple[Path, ...], *parts: str):
        self._roots = roots
        self._parts = parts

    def __truediv__(self, other: object) -> "_SearchPath":
        return _SearchPath(self._roots, *self._parts, str(other))

    def _resolve(self) -> Path:
        rel = Path(*self._parts) if self._parts else Path()
        for root in self._roots:
            cand = root / rel if self._parts else root
            if cand.exists():
                return cand
        return (self._roots[0] / rel) if self._parts else self._roots[0]

    def __fspath__(self) -> str:
        return str(self._resolve())

    def __str__(self) -> str:
        return str(self._resolve())

    def exists(self) -> bool:
        return self._resolve().exists()

    def is_file(self) -> bool:
        return self._resolve().is_file()

    def is_dir(self) -> bool:
        return self._resolve().is_dir()

    def __getattr__(self, name: str):
        return getattr(self._resolve(), name)


DATA = _SearchPath(_DATA_DIRS)
SUPP = _SearchPath(_SUPP_DIRS)
OUT = ROOT / "outputs" / "figures" / "main"
OUT.mkdir(parents=True, exist_ok=True)

# Current palette
ASD = "#B22222"
TD = "#000080"
INK = "#1A1A1A"
AX = "#222222"
GREY = "#777777"
NS = "#777777"
ZERO = "#D4D4D4"
REF = "#7A7AA8"

# Type scale (~+25–30% vs the previous Nature-column set)
LETTER = 18.0
LETTER_ROW = 14.0
TITLE = 12.0
TITLE_GRID = 9.8
TITLE_ROW = 11.0
LABEL = 12.0
LABEL_GRID = 10.5
LABEL_ROW = 11.0
TICK = 10.5
TICK_GRID = 10.0
NOTE = 9.0
NOTE_SMALL = 8.5
LEGEND = 10.0
SPINE_LW = 2.0
TITLE_PAD = 8.0
PANEL_Y = 1.08

# Marks
VIOLIN_LW = 0.35
BOX_LW = 0.4
FOREST_MS = 5.4
FOREST_ELW = 1.20
FOREST_CAP = 3.0
SCATTER_S = 16.0
REG_LW = 1.80
_SIG_FS = 16.0

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 11.0,
    "font.weight": "normal",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.titlesize": TITLE,
    "axes.titlepad": TITLE_PAD,
    "axes.labelsize": LABEL,
    "xtick.labelsize": TICK,
    "ytick.labelsize": TICK,
    "axes.linewidth": 1.60,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "mathtext.fontset": "dejavusans",
})


def save(fig, stem, pad_inches=0.1):
    path = OUT / stem
    for ext in (".png", ".pdf", ".svg"):
        kw = dict(bbox_inches="tight", pad_inches=pad_inches, facecolor="white")
        if ext == ".png":
            kw["dpi"] = 300
        fig.savefig(path.with_suffix(ext), **kw)
    plt.close(fig)
    print(path.with_suffix(".png"))


def panel(ax, label, x=-0.14, y=PANEL_Y, size=LETTER):
    ax.text(
        x, y, label, transform=ax.transAxes, fontsize=size,
        fontweight="bold", ha="left", va="bottom", color="#1f1f1f",
    )


def clean(ax, ticksize=TICK, lw=SPINE_LW):
    ax.spines["left"].set_linewidth(lw)
    ax.spines["bottom"].set_linewidth(lw)
    ax.spines["left"].set_color(AX)
    ax.spines["bottom"].set_color(AX)
    ax.tick_params(
        axis="both", direction="in", length=4.2, width=lw,
        colors=AX, labelsize=ticksize,
    )
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")


def add_sig(ax, x1, x2, y, h, text, fontsize=_SIG_FS, color=None):
    """Draw significance bracket at the requested y (caller controls height)."""
    ax.plot(
        [x1, x1, x2, x2], [y, y + h, y + h, y],
        color=INK, lw=0.70, clip_on=False, zorder=10,
    )
    is_ns = text.lower().replace(".", "") == "ns"
    if color is None:
        color = NS if is_ns else INK
    if fontsize == _SIG_FS and is_ns:
        fs = max(_SIG_FS - 2, 11)
    else:
        fs = fontsize
    ax.text(
        (x1 + x2) / 2, y + h, text,
        ha="center", va="bottom",
        fontsize=fs,
        fontweight="bold",
        color=color,
        clip_on=False, zorder=11,
    )


def _stable_rng(pos, n, facecolor):
    key = f"{float(pos):.6f}|{int(n)}|{to_hex(facecolor).lower()}".encode("utf-8")
    seed = zlib.crc32(key) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def _strip_side_from_color(facecolor):
    c = to_hex(facecolor).lower()
    if c == to_hex(ASD).lower():
        return -1.0
    if c == to_hex(TD).lower():
        return +1.0
    return -1.0


def box_strip(
    ax,
    vals,
    pos,
    facecolor,
    width=0.18,
    alpha=1.0,
    lw=None,
    fliers=True,
    rng=None,
    side=None,
    gap=None,
    box_width=None,
    show_strip=True,
):
    """Wider box on tick; optional jittered strip beside it (ASD left, TD right)."""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return

    if rng is None:
        rng = _stable_rng(pos, len(vals), facecolor)
    if side is None:
        side = _strip_side_from_color(facecolor)
    side = -1.0 if side < 0 else +1.0

    span = max(width, 0.20)
    box_w = 0.52 * span if box_width is None else float(box_width)
    box_x = pos
    if gap is None:
        gap = 0.10 * span
    strip_half = 0.24 * box_w
    strip_distance = 0.50 * box_w + gap + strip_half
    strip_x = pos + side * strip_distance

    if show_strip:
        offsets = rng.uniform(-strip_half, strip_half, size=len(vals))
        ax.scatter(
            strip_x + offsets, vals,
            s=9.0, facecolor=facecolor, edgecolors=INK, linewidths=0.25,
            alpha=alpha, zorder=3,
        )

    box_lw = SPINE_LW if lw is None else lw
    ax.boxplot(
        vals, positions=[box_x], widths=box_w, patch_artist=True,
        manage_ticks=False, showfliers=False, showcaps=True, whis=1.5, zorder=2,
        boxprops=dict(facecolor=facecolor, edgecolor=INK, linewidth=box_lw, alpha=alpha),
        medianprops=dict(color=INK, linewidth=2.40),
        whiskerprops=dict(color=INK, linewidth=1.85),
        capprops=dict(color=INK, linewidth=1.85),
    )


# Alias used by older figure helpers
violin_box = box_strip


def ols_ci(x, y, x_grid):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = beta[0] + beta[1] * x_grid
    resid = y - X @ beta
    df = max(len(x) - 2, 1)
    mse = np.sum(resid ** 2) / df
    xbar = x.mean()
    sxx = np.sum((x - xbar) ** 2)
    se = np.sqrt(mse * (1 / len(x) + (x_grid - xbar) ** 2 / sxx))
    ci = student_t.ppf(0.975, df) * se
    return yhat, yhat - ci, yhat + ci


def residualize(df, target, covariates=("age_months", "IQ_total")):
    sub = df.dropna(subset=[target, *covariates]).copy()
    X = sub.loc[:, covariates].to_numpy(float)
    X = np.column_stack([np.ones(len(X)), X])
    y = sub[target].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return sub, y - X @ beta


def regression_ci(x, y, x_grid):
    slope, intercept, r, p, _ = stats.linregress(x, y)
    y_hat = intercept + slope * x_grid
    n = len(x)
    x_bar = np.mean(x)
    residual = y - (intercept + slope * x)
    s_err = np.sqrt(np.sum(residual ** 2) / (n - 2))
    s_xx = np.sum((x - x_bar) ** 2)
    tcrit = stats.t.ppf(0.975, n - 2)
    ci = tcrit * s_err * np.sqrt(1 / n + (x_grid - x_bar) ** 2 / s_xx)
    return y_hat, ci


def draw_forest(
    ax, labels, beta, lo, hi, colors, xlabel, xlim, xticks=None,
    ticksize=TICK, zero_color=ZERO, zero_lw=0.8,
    marker_size=FOREST_MS, marker_sizes=None, marker_colors=None,
    marker_fmts=None,
    elinewidth=FOREST_ELW, elinewidths=None, capsize=FOREST_CAP, spine_lw=SPINE_LW,
    y_gap=1.0, zero_ls="-",
):
    y = np.arange(len(labels))[::-1] * y_gap
    sizes = marker_sizes if marker_sizes is not None else [marker_size] * len(labels)
    widths = elinewidths if elinewidths is not None else [elinewidth] * len(labels)
    mcols = marker_colors if marker_colors is not None else colors
    fmts = marker_fmts if marker_fmts is not None else ["o"] * len(labels)
    if zero_lw > 0:
        ax.axvline(0, color=zero_color, lw=zero_lw, ls=zero_ls, zorder=0)
    for i, (b, l, h, c) in enumerate(zip(beta, lo, hi, colors)):
        mc = mcols[i]
        ax.errorbar(
            b, y[i],
            xerr=[[b - l], [h - b]],
            fmt=fmts[i], markersize=sizes[i], mfc=mc, mec=mc, mew=0.45,
            ecolor=c, elinewidth=widths[i], capsize=capsize, capthick=widths[i], zorder=3,
        )
    ax.set_yticks(y, labels)
    ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_xlabel(xlabel, labelpad=6, fontsize=LABEL_ROW)
    ax.tick_params(axis="y", length=0, pad=4)
    clean(ax, ticksize=ticksize, lw=spine_lw)


def group_legend_handles():
    return [
        plt.Line2D(
            [0], [0], marker="s", color="none", label="ASD",
            markerfacecolor=ASD, markeredgecolor="none", markersize=7.5,
        ),
        plt.Line2D(
            [0], [0], marker="s", color="none", label="TD",
            markerfacecolor=TD, markeredgecolor="none", markersize=7.5,
        ),
    ]


def p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."
