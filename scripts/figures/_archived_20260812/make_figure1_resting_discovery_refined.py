from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[3]  # repo root
DATA = ROOT / "data" / "figure_source"
OUT = ROOT / "outputs" / "figures" / "figure1_resting_aperiodic_discovery"
OUT.mkdir(parents=True, exist_ok=True)

ASD = "#C25450"
TD = "#B4C8E0"
TD_DOT = "#4A6FA5"
NEUTRAL = "#333333"
GRAY = "#666666"

mpl.rcParams.update({
    "font.family": "Times New Roman",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8.8,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.labelsize": 9.4,
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.4,
    "axes.linewidth": 1.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def panel(ax, label: str, x=-0.13, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=15,
            fontweight="bold", ha="left", va="bottom", color="#1f1f1f")


def clean(ax):
    ax.spines["left"].set_linewidth(1.15)
    ax.spines["bottom"].set_linewidth(1.15)
    ax.tick_params(axis="both", direction="in", length=3.5, width=1.15)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")


def mean_ci(vals):
    vals = np.asarray(vals, float)
    n = len(vals)
    m = vals.mean()
    se = vals.std(ddof=1) / np.sqrt(n)
    ci = student_t.ppf(0.975, n - 1) * se
    return m, m - ci, m + ci


def add_sig(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            color=NEUTRAL, lw=1.0, clip_on=False)
    is_ns = text.lower().replace(".", "") == "ns"
    ylim = ax.get_ylim()
    label_lift = 0.025 * (ylim[1] - ylim[0]) if is_ns else 0
    ax.text((x1 + x2) / 2, y + h + label_lift, text, ha="center", va="bottom",
            fontsize=13.2 if is_ns else 22.0,
            fontweight="bold",
            color="#111111" if is_ns else ASD, clip_on=False)


def draw_specparam(ax):
    f = np.linspace(1, 40, 240)
    aperiodic = 2.72 - 1.03 * np.log10(f)
    alpha_peak = 0.72 * np.exp(-0.5 * ((f - 9.1) / 1.55) ** 2)
    low_bump = 0.16 * np.exp(-0.5 * ((f - 3.0) / 0.85) ** 2)
    ripple = 0.018 * np.sin(f * 0.85) + 0.012 * np.cos(f * 1.7)
    psd = aperiodic + alpha_peak + low_bump + ripple
    ax.plot(f, psd, color="#42474C", lw=1.0, zorder=3)
    ax.plot(f, aperiodic, color=ASD, lw=0.95, ls=(0, (4, 2)), zorder=2)
    mask = (f >= 5.8) & (f <= 12.5)
    ax.fill_between(f[mask], aperiodic[mask], psd[mask], color="#BFD4E5",
                    alpha=0.20, linewidth=0, zorder=1)
    ax.annotate(
        "periodic\ncomponent", xy=(9.2, psd[np.argmin(abs(f - 9.2))]),
        xytext=(16.0, 2.20), ha="left", va="center",
        fontsize=8.6, color="#666D75", fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="#666D75", lw=0.7)
    )
    ax.annotate(
        "aperiodic\nexponent", xy=(31.0, aperiodic[np.argmin(abs(f - 31.0))]),
        xytext=(25.0, 1.48), ha="left", va="center",
        fontsize=8.6, color="#666D75", fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="#666D75", lw=0.7)
    )
    ax.set_xlim(1, 40)
    ax.set_ylim(0.86, 2.92)
    ax.set_xticks([1, 13, 25, 40])
    ax.set_yticks([])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$\log_{10}$ PSD")
    ax.set_title("Spectral parameterization", fontsize=10.2, pad=5)
    clean(ax)


def violin_panel(ax, df, metric, title, ylabel, sig, ylim=None, yticks=None):
    offsets = {"ASD": 0.84, "TD": 1.16}
    colors = {"ASD": ASD, "TD": TD}
    dot_colors = {"ASD": ASD, "TD": TD_DOT}
    rng = np.random.default_rng(19)
    ymax = -np.inf
    for group in ["ASD", "TD"]:
        vals = df[df["group"].eq(group)][metric].dropna().to_numpy(float)
        pos = offsets[group]
        parts = ax.violinplot(vals, positions=[pos], widths=0.30,
                              showmeans=False, showmedians=False, showextrema=False)
        body = parts["bodies"][0]
        body.set_facecolor(colors[group])
        body.set_edgecolor(colors[group])
        body.set_alpha(0.28)
        body.set_linewidth(0.8)
        jitter = rng.normal(0, 0.025, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=9,
                   color=dot_colors[group], alpha=0.60, edgecolor="white",
                   linewidth=0.35, zorder=3)
        m, lo, hi = mean_ci(vals)
        ymax = max(ymax, hi)
        ax.plot([pos, pos], [lo, hi], color=NEUTRAL, lw=1.0, zorder=5)
        ax.plot([pos - 0.035, pos + 0.035], [lo, lo], color=NEUTRAL, lw=1.0, zorder=5)
        ax.plot([pos - 0.035, pos + 0.035], [hi, hi], color=NEUTRAL, lw=1.0, zorder=5)
        ax.scatter(pos, m, s=28, color=dot_colors[group], edgecolor="white",
                   linewidth=0.55, zorder=6)
    ax.set_xlim(0.50, 1.50)
    ax.set_xticks([offsets["ASD"], offsets["TD"]], ["ASD", "TD"])
    ax.set_title(title, fontsize=10.2, pad=5)
    ax.set_xlabel("Group")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ymin, top = ax.get_ylim()
    y = top - 0.14 * (top - ymin)
    add_sig(ax, offsets["ASD"], offsets["TD"], y, 0.025 * (top - ax.get_ylim()[0]), sig)
    clean(ax)


def main():
    subjects = pd.read_csv(DATA / "fig1_resting_primary_subjects.csv")

    fig = plt.figure(figsize=(6.65, 5.25))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1, 1], height_ratios=[1.0, 1.0],
        left=0.095, right=0.975, bottom=0.105, top=0.93,
        wspace=0.34, hspace=0.42
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel(ax_a, "A", x=-0.08, y=1.01)
    draw_specparam(ax_a)

    panel(ax_b, "B", x=-0.12, y=1.02)
    violin_panel(
        ax_b, subjects, "posterior_exponent",
        "Posterior exponent", "aperiodic exponent", "***",
        ylim=(1.0, 2.65), yticks=[1.0, 1.5, 2.0, 2.5]
    )

    panel(ax_c, "C", x=-0.12, y=1.02)
    violin_panel(
        ax_c, subjects, "global_exponent",
        "Global exponent", "aperiodic exponent", "*",
        ylim=(1.25, 2.45), yticks=[1.5, 2.0, 2.4]
    )

    panel(ax_d, "D", x=-0.12, y=1.02)
    violin_panel(
        ax_d, subjects, "global_offset",
        "Global offset", r"aperiodic offset" + "\n" + r"($\log_{10}\,\mu V^2/Hz$)", "n.s.",
        ylim=(-10.65, -9.45), yticks=[-10.4, -10.0, -9.6]
    )

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", label="ASD",
                   markerfacecolor=ASD, markeredgecolor="white", markersize=5),
        plt.Line2D([0], [0], marker="o", color="none", label="TD",
                   markerfacecolor=TD_DOT, markeredgecolor="white", markersize=5),
    ]
    ax_d.legend(handles=handles, loc="upper right", fontsize=8.2,
                handletextpad=0.35, borderaxespad=0.1)

    stem = OUT / "Figure1_resting_aperiodic_discovery_refined"
    for ext, kwargs in {
        "png": {"dpi": 300},
        "tiff": {"dpi": 600},
        "svg": {},
        "pdf": {},
    }.items():
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight",
                    facecolor="white", **kwargs)
    plt.close(fig)
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
