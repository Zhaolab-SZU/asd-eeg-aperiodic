from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[3]  # repo root
OUT = ROOT / "outputs" / "figures" / "figure5_naturalistic_redesign"
OUT.mkdir(parents=True, exist_ok=True)

ASD = "#C25450"
TD = "#4A6FA5"
NEUTRAL = "#333333"
MID = "#666666"
LIGHT = "#E7EAEE"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8.7,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.labelsize": 9.2,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "axes.linewidth": 1.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def panel(ax, label: str):
    ax.text(-0.13, 1.05, label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", ha="left", va="bottom",
            color="#1f1f1f")


def clean(ax):
    ax.spines["left"].set_linewidth(1.15)
    ax.spines["bottom"].set_linewidth(1.15)
    ax.tick_params(axis="both", direction="out", length=3.5, width=1.15)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")


def mean_ci(vals):
    vals = np.asarray(vals, float)
    n = len(vals)
    m = vals.mean()
    if n < 2:
        return m, m, m
    se = vals.std(ddof=1) / np.sqrt(n)
    ci = student_t.ppf(0.975, n - 1) * se
    return m, m - ci, m + ci


def add_sig(ax, x1, x2, y, h, text, color=ASD):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            color=NEUTRAL, lw=1.0, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom",
            fontsize=11.0, fontweight="bold", color=color, clip_on=False)


def draw_grouped_violins(
    ax,
    df,
    category_col,
    value_col,
    categories,
    category_labels,
    y_label,
    star_map,
    y_lim,
    title=None,
):
    offsets = {"ASD": -0.16, "TD": 0.16}
    colors = {"ASD": ASD, "TD": TD}
    rng = np.random.default_rng(7)
    for i, cat in enumerate(categories):
        base = i + 1
        max_ci = y_lim[0]
        for group in ["ASD", "TD"]:
            vals = df[(df[category_col] == cat) & (df["group"] == group)][value_col].dropna().to_numpy(float)
            pos = base + offsets[group]
            parts = ax.violinplot(
                vals, positions=[pos], widths=0.28,
                showmeans=False, showmedians=False, showextrema=False
            )
            body = parts["bodies"][0]
            body.set_facecolor(colors[group])
            body.set_edgecolor(colors[group])
            body.set_alpha(0.24)
            body.set_linewidth(0.7)
            jitter = rng.normal(0, 0.025, size=len(vals))
            ax.scatter(
                np.full(len(vals), pos) + jitter, vals,
                s=9, color=colors[group], alpha=0.58,
                edgecolor="white", linewidth=0.35, zorder=3
            )
            m, lo, hi = mean_ci(vals)
            max_ci = max(max_ci, hi)
            ax.plot([pos, pos], [lo, hi], color=NEUTRAL, lw=0.9, zorder=5)
            ax.plot([pos - 0.035, pos + 0.035], [lo, lo], color=NEUTRAL, lw=0.9, zorder=5)
            ax.plot([pos - 0.035, pos + 0.035], [hi, hi], color=NEUTRAL, lw=0.9, zorder=5)
            ax.scatter(pos, m, s=24, color=colors[group],
                       edgecolor="white", linewidth=0.55, zorder=6)
        sig = star_map.get(cat, "")
        if sig:
            y = min(y_lim[1] - 0.020, max_ci + 0.022)
            add_sig(ax, base + offsets["ASD"], base + offsets["TD"], y, 0.008, sig)
    ax.set_xticks(np.arange(1, len(categories) + 1), category_labels)
    ax.set_ylabel(y_label)
    ax.set_ylim(*y_lim)
    if title:
        ax.set_title(title, fontsize=9.0, pad=5)
    clean(ax)


def main():
    state = pd.read_csv(ROOT / "data" / "figure_source" / "fig5_rest_movie_exponent_subjects.csv")
    movie = pd.read_csv(ROOT / "data" / "figure_source" / "fig5_movie_isc_subjects.csv")
    hbn = pd.read_csv(ROOT / "data" / "supplementary_source" / "s8_hbn_movie_subjects.csv")

    fig = plt.figure(figsize=(7.35, 2.60))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1, 1, 1],
        left=0.075, right=0.985, bottom=0.22, top=0.84,
        wspace=0.36
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel(ax_a, "A")
    panel(ax_b, "B")
    panel(ax_c, "C")

    # Panel A: interaction plot for state-by-group reconfiguration.
    state_order = ["Rest", "Movie"]
    x_state = np.array([1.0, 2.0])
    for group, color in [("ASD", ASD), ("TD", TD)]:
        z = state[state["group"].eq(group)]
        means = z.groupby("state")["posterior_exponent"].mean().reindex(state_order).to_numpy(float)
        ns = z.groupby("state")["posterior_exponent"].count().reindex(state_order).to_numpy(float)
        sds = z.groupby("state")["posterior_exponent"].std().reindex(state_order).to_numpy(float)
        ci = 1.96 * sds / np.sqrt(ns)
        ax_a.plot(
            x_state, means, color=color, lw=1.9,
            marker="o", ms=5.8, markeredgecolor="white",
            markeredgewidth=0.7, label=group, zorder=4
        )
        ax_a.errorbar(
            x_state, means, yerr=ci, fmt="none", ecolor=color,
            elinewidth=1.15, capsize=3, capthick=1.15, alpha=0.9, zorder=3
        )
        wide = z.pivot_table(
            index="subject_id", columns="state", values="posterior_exponent"
        ).dropna(subset=state_order)
        for _, row in wide.iterrows():
            ax_a.plot(
                x_state, [row["Rest"], row["Movie"]],
                color=color, lw=0.32, alpha=0.08, zorder=1
            )
    ax_a.set_xlim(0.78, 2.22)
    ax_a.set_ylim(1.38, 2.48)
    ax_a.set_xticks(x_state, state_order)
    ax_a.set_ylabel("Posterior exponent")
    ax_a.set_title("Rest-to-movie state modulation", fontsize=9.0, pad=5)
    add_sig(ax_a, 1.0, 2.0, 2.39, 0.030, "***")
    leg = ax_a.get_legend()
    if leg is not None:
        leg.remove()
    clean(ax_a)

    # Panel B: primary movie Aperiodic-ISC subject-level distributions.
    segment_order = ["mentalizing", "pain", "neutral"]
    segment_labels = ["Mental-\nizing", "Pain-\nrelated", "Neutral"]
    draw_grouped_violins(
        ax_b, movie, "segment", "isc_r", segment_order, segment_labels,
        "Aperiodic-ISC (r)", {"mentalizing": "*", "pain": "**", "neutral": "***"},
        (-0.24, 0.45), title="Partly Cloudy"
    )
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", label="ASD",
                   markerfacecolor=ASD, markeredgecolor="white", markersize=5),
        plt.Line2D([0], [0], marker="o", color="none", label="TD",
                   markerfacecolor=TD, markeredgecolor="white", markersize=5),
    ]
    ax_b.legend(handles=handles, loc="upper right", fontsize=7.8,
                handletextpad=0.3, borderaxespad=0.1)

    # Panel C: HBN external convergence distributions.
    hbn = hbn.copy()
    hbn["analysis_label"] = hbn["analysis"].map({
        "sliding_window": "Sliding windows",
        "nonoverlapping_2s_epoch": "2-s epochs",
    })
    draw_grouped_violins(
        ax_c, hbn, "analysis_label", "isc_z",
        ["Sliding windows", "2-s epochs"],
        ["Sliding\nwindows", "2-s epochs"],
        "Posterior Aperiodic-ISC (Fisher z)",
        {"Sliding windows": "*", "2-s epochs": "*"},
        (-0.45, 0.66), title="HBN The Present"
    )
    ax_c.set_xlabel("")
    leg = ax_c.get_legend()
    if leg is not None:
        leg.remove()

    stem = OUT / "Figure5_naturalistic_state_alignment_violin"
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
