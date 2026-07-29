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

STEM = "SuppFigS5_synchrony_controls"

ASD = "#C25450"
TD = "#4A6FA5"
ASD_LIGHT = "#E9A6A2"
TD_LIGHT = "#B8C9E2"
INK = "#333333"
MUTED = "#666666"
GREY = "#9A9A9A"
LIGHT_GREY = "#DADADA"

SEGMENTS = ["mentalizing", "pain", "neutral"]
SEG_LABELS = {
    "mentalizing": "Mentalizing",
    "pain": "Pain-related",
    "neutral": "Neutral",
}
METRICS = ["aperiodic_isc", "envelope_isc", "alpha_plv_isc"]
METRIC_LABELS = {
    "aperiodic_isc": "Aperiodic",
    "envelope_isc": "Envelope",
    "alpha_plv_isc": "Alpha PLV",
}
METRIC_COLORS = {
    "aperiodic_isc": INK,
    "envelope_isc": ASD,
    "alpha_plv_isc": "#777777",
}

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


def panel_label(ax, label, x=-0.12, y=1.05):
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


def p_to_stars(p: float | None) -> str:
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def ci_mean(vals: np.ndarray) -> tuple[float, float, float]:
    vals = vals[~np.isnan(vals)]
    mean = float(np.mean(vals))
    sem = stats.sem(vals)
    ci = sem * stats.t.ppf(0.975, len(vals) - 1)
    return mean, mean - ci, mean + ci


def draw_segment_panel(
    ax,
    subj: pd.DataFrame,
    effects: pd.DataFrame,
    segment: str,
    label: str,
    show_ylabel: bool = True,
):
    panel_label(ax, label, x=-0.02, y=1.04)
    rng = np.random.default_rng(20260713 + len(segment))
    metric_centers = np.arange(len(METRICS)) * 1.0
    offsets = {"ASD": -0.14, "TD": 0.14}
    group_colors = {"ASD": ASD, "TD": TD}
    group_lights = {"ASD": ASD_LIGHT, "TD": TD_LIGHT}
    for i, metric in enumerate(METRICS):
        for group in ("ASD", "TD"):
            vals = subj[
                (subj["segment"] == segment)
                & (subj["metric"] == metric)
                & (subj["group"] == group)
            ]["isc_value"].dropna().to_numpy()
            x0 = metric_centers[i] + offsets[group]
            parts = ax.violinplot(
                vals,
                positions=[x0],
                widths=0.24,
                showmeans=False,
                showextrema=False,
                showmedians=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(group_lights[group])
                body.set_edgecolor(group_colors[group])
                body.set_alpha(0.48)
                body.set_linewidth(0.75)
            ax.scatter(
                np.full(len(vals), x0) + rng.normal(0, 0.020, len(vals)),
                vals,
                s=7.4,
                color=group_colors[group],
                alpha=0.48,
                edgecolor="white",
                linewidth=0.20,
                zorder=2,
            )
            mean, lo, hi = ci_mean(vals)
            ax.errorbar(
                x0,
                mean,
                yerr=[[mean - lo], [hi - mean]],
                fmt="o",
                color=group_colors[group],
                ecolor=group_colors[group],
                markersize=3.8,
                elinewidth=1.0,
                capsize=3.0,
                zorder=4,
            )
        eff = effects[(effects["segment"] == segment) & (effects["metric"] == metric)].iloc[0]
        stars = p_to_stars(float(eff["fdr_p"]) if metric != "alpha_plv_isc" else float(eff["p"]))
        y = 0.675
        if stars:
            ax.plot([metric_centers[i] - 0.14, metric_centers[i] - 0.14,
                     metric_centers[i] + 0.14, metric_centers[i] + 0.14],
                    [y, y + 0.025, y + 0.025, y], color=INK, lw=0.75)
            ax.text(
                metric_centers[i],
                y + 0.031,
                stars,
                ha="center",
                va="bottom",
                fontsize=10.6 if stars != "n.s." else 7.4,
                color=ASD if stars != "n.s." else MUTED,
                fontweight="bold" if stars != "n.s." else "normal",
            )
    ax.set_xticks(metric_centers)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=30, ha="right", fontsize=6.6)
    ax.set_ylim(-0.62, 0.78)
    ax.set_yticks([-0.6, -0.2, 0.2, 0.6])
    ax.set_xlabel("Synchrony metric", fontsize=7.5, fontweight="bold", labelpad=6)
    ax.set_title(SEG_LABELS[segment], fontsize=8.7, fontweight="bold", color=INK, pad=10)
    if show_ylabel:
        ax.set_ylabel("ISC (Fisher z)", fontsize=8, fontweight="bold")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    style_axes(ax)


def draw_forest(ax, effects: pd.DataFrame):
    panel_label(ax, "D", x=-0.08, y=1.02)
    rows = []
    for segment in SEGMENTS:
        for metric in METRICS:
            r = effects[(effects["segment"] == segment) & (effects["metric"] == metric)].iloc[0]
            rows.append(
                {
                    "segment": segment,
                    "metric": metric,
                    "label": f"{SEG_LABELS[segment]} | {METRIC_LABELS[metric]}",
                    "est": r["group_effect_td_minus_asd"],
                    "lo": r["ci_low"],
                    "hi": r["ci_high"],
                    "p": r["fdr_p"] if metric != "alpha_plv_isc" else r["p"],
                }
            )
    y = np.arange(len(rows))[::-1]
    last_segment = None
    for yi, row in zip(y, rows):
        color = METRIC_COLORS[row["metric"]]
        alpha = 1.0 if row["metric"] != "alpha_plv_isc" else 0.75
        ax.errorbar(
            row["est"],
            yi,
            xerr=[[row["est"] - row["lo"]], [row["hi"] - row["est"]]],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=4.4,
            elinewidth=1.0,
            capsize=3,
            alpha=alpha,
        )
        stars = p_to_stars(float(row["p"]))
        ax.text(
            0.286,
            yi,
            stars,
            ha="right",
            va="center",
            fontsize=10.2 if stars != "n.s." else 7.2,
            color=ASD if stars != "n.s." else MUTED,
            fontweight="bold" if stars != "n.s." else "normal",
        )
        if last_segment is not None and row["segment"] != last_segment:
            ax.axhline(yi + 0.5, color="#ECECEC", lw=0.8, zorder=0)
        last_segment = row["segment"]
    ax.set_yticks(y)
    ax.set_yticklabels([row["label"] for row in rows], fontsize=6.4)
    ax.set_xlim(-0.04, 0.30)
    ax.set_xlabel("TD − ASD effect", fontsize=8, fontweight="bold")
    ax.set_title("Group effect across synchrony metrics", fontsize=8.7, fontweight="bold", color=INK, pad=7)
    style_axes(ax)


def draw() -> None:
    backup_existing()
    subj = pd.read_csv(SRC / "s7_synchrony_subject_level.csv")
    effects = pd.read_csv(SRC / "s7_synchrony_group_effects.csv")

    fig = plt.figure(figsize=(7.45, 5.55))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.66, wspace=0.24)
    top_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    for i, (ax, segment, label) in enumerate(zip(
        top_axes,
        SEGMENTS,
        ["A", "B", "C"],
    )):
        draw_segment_panel(ax, subj, effects, segment, label, show_ylabel=(i == 0))
    draw_forest(fig.add_subplot(gs[1, :]), effects)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ASD,
                   markeredgecolor=ASD, markersize=4.5, label="ASD"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=TD,
                   markeredgecolor=TD, markersize=4.5, label="TD"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.505),
               frameon=False, ncol=2, fontsize=7.2, handletextpad=0.5, columnspacing=1.5)
    fig.subplots_adjust(left=0.080, right=0.990, top=0.935, bottom=0.095)
    # The bottom forest plot needs a larger left margin for long row labels.
    # Shift only the upper panels leftward after subplots_adjust so A-C are not
    # visually anchored by the lower-panel label column.
    for ax in top_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0 - 0.055, pos.y0, pos.width + 0.018, pos.height])

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
