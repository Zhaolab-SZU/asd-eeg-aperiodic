from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]  # repo root
SRC = ROOT / "data" / "supplementary_source"
OUT = ROOT / "outputs" / "figures" / "supplementary"
BACKUP = OUT.parent / "backup"; BACKUP.mkdir(parents=True, exist_ok=True)
BACKUP.mkdir(parents=True, exist_ok=True)

STEM = "SuppFigS4_aperiodic_isc"

ASD = "#C25450"
TD = "#4A6FA5"
ASD_LIGHT = "#E9A6A2"
TD_LIGHT = "#B8C9E2"
INK = "#333333"
MUTED = "#666666"
REF = "#8A8A8A"

SEGMENTS = ["mentalizing", "pain", "neutral"]
SEG_LABELS = {
    "mentalizing": "Mentalizing",
    "pain": "Pain-related",
    "neutral": "Neutral",
}
TC_LABELS = ["Mentalizing", "Pain-related", "Neutral"]

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


def panel_label(ax, label, x=-0.13, y=1.05):
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


def draw_segment_summary(
    ax,
    subj: pd.DataFrame,
    ci: pd.DataFrame,
    definition: str,
    title: str,
    label: str,
    show_y: bool = True,
):
    panel_label(ax, label)
    rng = np.random.default_rng(20260713 if definition == "td_template" else 20260714)
    centers = np.arange(len(SEGMENTS)) * 1.25
    offsets = {"ASD": -0.18, "TD": 0.18}
    colors = {"ASD": ASD, "TD": TD}
    lights = {"ASD": ASD_LIGHT, "TD": TD_LIGHT}

    for i, segment in enumerate(SEGMENTS):
        for group in ("ASD", "TD"):
            x0 = centers[i] + offsets[group]
            vals = subj[
                (subj["isc_definition"] == definition)
                & (subj["segment"] == segment)
                & (subj["group"] == group)
            ]["isc_value"].dropna().to_numpy()
            parts = ax.violinplot(
                vals,
                positions=[x0],
                widths=0.28,
                showmeans=False,
                showextrema=False,
                showmedians=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(lights[group])
                body.set_edgecolor(colors[group])
                body.set_alpha(0.55)
                body.set_linewidth(0.9)
            jitter = rng.normal(0, 0.022, size=len(vals))
            ax.scatter(
                np.full(len(vals), x0) + jitter,
                vals,
                s=8.5,
                color=colors[group],
                alpha=0.45,
                edgecolor="white",
                linewidth=0.25,
                zorder=3,
            )
            row = ci[
                (ci["isc_definition"] == definition)
                & (ci["segment"] == segment)
                & (ci["group"] == group)
            ].iloc[0]
            yerr = [[row["mean_isc"] - row["ci_low"]], [row["ci_high"] - row["mean_isc"]]]
            ax.errorbar(
                x0,
                row["mean_isc"],
                yerr=yerr,
                fmt="o",
                color=INK,
                ecolor=INK,
                markersize=3.3,
                elinewidth=0.9,
                capsize=3,
                zorder=5,
            )

        p = ci[
            (ci["isc_definition"] == definition)
            & (ci["segment"] == segment)
        ]["fdr_p"].iloc[0]
        y = 0.49
        x1, x2 = centers[i] - 0.34, centers[i] + 0.34
        ax.plot([x1, x1, x2, x2], [y, y + 0.026, y + 0.026, y], color=INK, lw=0.8)
        stars = p_to_stars(float(p))
        ax.text(
            centers[i],
            y + 0.033,
            stars,
            ha="center",
            va="bottom",
            fontsize=11.0 if stars != "n.s." else 8.2,
            color=ASD if stars != "n.s." else MUTED,
            fontweight="bold" if stars != "n.s." else "normal",
        )

    ax.set_xticks(centers)
    ax.set_xticklabels([SEG_LABELS[s] for s in SEGMENTS], fontsize=7)
    ax.set_ylim(-0.36, 0.57)
    ax.set_xlabel("Movie segment", fontsize=7.8, fontweight="bold", labelpad=7)
    if show_y:
        ax.set_ylabel("Aperiodic-ISC (Fisher z)", fontsize=8, fontweight="bold")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.set_title(title, fontsize=8.8, fontweight="bold", color=INK, pad=7)
    style_axes(ax)


def draw_timecourse(ax, tc: pd.DataFrame, segment_label: str, label: str | None = None, show_y: bool = True):
    if label:
        panel_label(ax, label)
    sub = tc[
        (tc["segment_label"] == segment_label)
        & (tc["isc_definition"] == "within_group_loo")
    ].copy()
    common_times = sorted(
        set(sub[sub["group"] == "ASD"]["time_seconds"]).intersection(
            set(sub[sub["group"] == "TD"]["time_seconds"])
        )
    )
    sub = sub[sub["time_seconds"].isin(common_times)]
    window = 21 if segment_label == "Neutral" else 5
    for group, color, light in [("ASD", ASD, ASD_LIGHT), ("TD", TD, TD_LIGHT)]:
        g = sub[sub["group"] == group].sort_values("time_seconds")
        x = g["time_seconds"].to_numpy()
        y = g["isc_mean"].rolling(window, center=True, min_periods=1).mean().to_numpy()
        lo = g["ci_low"].rolling(window, center=True, min_periods=1).mean().to_numpy()
        hi = g["ci_high"].rolling(window, center=True, min_periods=1).mean().to_numpy()
        ax.fill_between(x, lo, hi, color=light, alpha=0.12, linewidth=0)
        ax.plot(x, y, color=color, lw=1.25, label=group)
    ax.set_title(segment_label, fontsize=8.2, fontweight="bold", color=INK, pad=5)
    ax.set_xlabel("Time (s)", fontsize=7.5, fontweight="bold")
    if show_y:
        ax.set_ylabel("ISC z", fontsize=7.5, fontweight="bold")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.set_ylim(-0.34, 0.40)
    ax.set_yticks([-0.3, 0.0, 0.3])
    style_axes(ax)


def draw() -> None:
    backup_existing()
    subj = pd.read_csv(SRC / "s6_isc_subject_level.csv")
    ci = pd.read_csv(SRC / "s6_isc_group_ci.csv")
    tc = pd.read_csv(SRC / "s6_isc_timecourse.csv")

    fig = plt.figure(figsize=(7.45, 5.8))
    gs = fig.add_gridspec(
        2,
        6,
        height_ratios=[1.15, 0.95],
        hspace=0.48,
        wspace=0.36,
    )
    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6])
    draw_segment_summary(ax_a, subj, ci, "td_template", "TD-template ISC", "A", show_y=True)
    draw_segment_summary(ax_b, subj, ci, "within_group_loo", "Within-group LOO ISC", "B", show_y=False)

    legend_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=5.5,
            markerfacecolor=ASD,
            markeredgecolor="white",
            markeredgewidth=0.35,
            label="ASD",
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=5.5,
            markerfacecolor=TD,
            markeredgecolor="white",
            markeredgewidth=0.35,
            label="TD",
        ),
    ]
    fig.legend(
        legend_handles,
        ["ASD", "TD"],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.525),
        ncol=2,
        frameon=False,
        fontsize=7.3,
        handlelength=0.8,
        columnspacing=1.5,
    )

    axes_tc = [fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:6])]
    for ax, seg_label, lab in zip(axes_tc, TC_LABELS, ["C", "D", "E"]):
        draw_timecourse(ax, tc, seg_label, lab, show_y=(lab == "C"))
    fig.subplots_adjust(left=0.065, right=0.990, top=0.935, bottom=0.105)

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
