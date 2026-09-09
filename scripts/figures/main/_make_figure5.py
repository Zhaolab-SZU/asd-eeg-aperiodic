"""Figure 5 — film-level naturalistic alignment (current design).

Panels
------
a  Rest → Partly Cloudy posterior-exponent state modulation
b  Film-level Aperiodic-ISC: Partly Cloudy (primary) and HBN The Present
   (sliding-window, the parameterization matched to the primary pipeline).
   The HBN 2-s epoch check remains in the supplementary figures.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _grammar_style import (
    ASD, DATA, INK, LABEL, NOTE, PANEL_Y, TD, TITLE_PAD,
    add_sig, box_strip, clean, group_legend_handles, p_to_stars, panel, save,
)

TITLE_F5 = 14.0
CAT_F5 = 12.5
LEGEND_F5 = 12.0


def isc_panel(ax, isc, isc_stats, hbn, hbn_sum):
    """Partly Cloudy and HBN sliding-window ISC, Fig. 1b box+strip grammar."""
    hbn_sw = hbn.loc[hbn["analysis"].eq("sliding_window")]
    hbn_p = float(
        hbn_sum.loc[
            hbn_sum["analysis"].eq("sliding-window posterior Aperiodic-ISC"), "p"
        ].iloc[0]
    )
    specs = [
        (isc, "Partly Cloudy", p_to_stars(float(isc_stats["p_value"]))),
        (hbn_sw, "The Present (HBN)", p_to_stars(hbn_p)),
    ]

    width = 0.24
    box_width = 0.21
    pair_off = 0.18
    gap = 0.050
    bases = np.array([1.18, 2.42])
    fill = {"ASD": ASD, "TD": TD}

    maxima = []
    cond_max = []
    for base, (df, _lab, _sig) in zip(bases, specs):
        cond_vals = []
        for group, sign in (("ASD", -1), ("TD", +1)):
            vals = df.loc[df["group"].eq(group), "isc_z"].dropna().to_numpy(float)
            maxima.append(float(np.nanmax(vals)))
            cond_vals.append(vals)
            box_strip(
                ax, vals, base + sign * pair_off, fill[group],
                width=width, side=-1 if group == "ASD" else +1,
                gap=gap, box_width=box_width,
            )
        cond_max.append(float(np.nanmax(np.concatenate(cond_vals))))

    strip_half = 0.24 * box_width
    strip_distance = 0.50 * box_width + gap + strip_half
    x_left = bases[0] - pair_off - strip_distance - strip_half
    x_right = bases[-1] + pair_off + strip_distance + strip_half
    ax.set_xlim(x_left - 0.32, x_right + 0.32)
    ax.set_xticks(bases, [lab for _df, lab, _sig in specs])
    ax.set_title("Aperiodic-ISC", pad=TITLE_PAD, fontsize=TITLE_F5, fontweight="bold")
    ax.set_xlabel("Movie", fontsize=LABEL)
    ax.set_ylabel("Aperiodic-ISC (Fisher $z$)", fontsize=LABEL)

    ymin, ylim_top = -0.40, 0.52
    data_max = max(maxima)
    span = ylim_top - ymin
    top = max(ylim_top, data_max + 0.14 * span)
    ax.set_ylim(ymin, top)
    ax.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])

    span = top - ymin
    h = 0.016 * span
    for base, (_df, _lab, sig), cmax in zip(bases, specs, cond_max):
        y = cmax + 0.050 * span
        add_sig(ax, base - pair_off, base + pair_off, y, h, sig)
    clean(ax, ticksize=CAT_F5)


def make_figure5():
    state = pd.read_csv(DATA / "fig5_rest_partly_cloudy_exponent_long.csv").copy()
    state["state_plot"] = state["state"].map({"Rest": "Rest", "Partly Cloudy": "Movie"})
    isc = pd.read_csv(DATA / "fig5_partly_cloudy_aperiodic_isc_subjects.csv")
    isc_stats = pd.read_csv(DATA / "fig5_partly_cloudy_aperiodic_isc_stats.csv").iloc[0]
    hbn = pd.read_csv(DATA / "fig5_the_present_aperiodic_isc_subjects.csv").copy()
    hbn_sum = pd.read_csv(DATA / "fig5_the_present_aperiodic_isc.csv")

    fig = plt.figure(figsize=(9.10, 3.72))
    gspec = fig.add_gridspec(
        1, 2,
        width_ratios=[1.00, 1.58],
        left=0.062, right=0.988, bottom=0.275, top=0.84, wspace=0.28,
    )
    ax_a = fig.add_subplot(gspec[0, 0])
    ax_b = fig.add_subplot(gspec[0, 1])
    panel(ax_a, "a", x=-0.12, y=PANEL_Y)
    panel(ax_b, "b", x=-0.10, y=PANEL_Y)

    # ---- a: Rest → Partly Cloudy modulation ----
    order = ["Rest", "Movie"]
    x_state = np.array([1.0, 2.0])
    for group, color in [("ASD", ASD), ("TD", TD)]:
        z = state[state["group"].eq(group)]
        wide = z.pivot_table(
            index="subject_id", columns="state_plot", values="posterior_exponent",
        ).dropna(subset=order)
        for _, row in wide.iterrows():
            ax_a.plot(
                x_state, [row["Rest"], row["Movie"]],
                color=color, lw=0.30, alpha=0.035, zorder=1, solid_capstyle="round",
            )
        means = z.groupby("state_plot")["posterior_exponent"].mean().reindex(order).to_numpy(float)
        ns = z.groupby("state_plot")["posterior_exponent"].count().reindex(order).to_numpy(float)
        sds = z.groupby("state_plot")["posterior_exponent"].std().reindex(order).to_numpy(float)
        ci = 1.96 * sds / np.sqrt(ns)
        ax_a.fill_between(
            x_state, means - ci, means + ci,
            color=color, alpha=0.12, linewidth=0, zorder=2,
        )
        ax_a.plot(
            x_state, means, color=color, lw=2.35, marker="o", ms=5.5,
            markeredgecolor="white", markeredgewidth=0.70, zorder=4, label=group,
        )
    ax_a.set_xlim(0.72, 2.28)
    ax_a.set_ylim(1.35, 2.48)
    ax_a.set_xticks(x_state, order)
    ax_a.set_xlabel("State", fontsize=LABEL)
    ax_a.set_ylabel("Posterior exponent", fontsize=LABEL)
    ax_a.set_title("Rest-to-movie modulation", pad=TITLE_PAD, fontsize=TITLE_F5, fontweight="bold")
    ax_a.text(
        0.97, 0.90,
        "State × Group\n$p$ = 7.2×10$^{-5}$",
        transform=ax_a.transAxes, ha="right", va="top",
        fontsize=NOTE, fontweight="bold", color=INK, linespacing=1.15,
    )
    clean(ax_a, ticksize=CAT_F5)

    isc_panel(ax_b, isc, isc_stats, hbn, hbn_sum)

    fig.legend(
        handles=group_legend_handles(),
        loc="lower center",
        ncol=2,
        fontsize=LEGEND_F5,
        frameon=False,
        handletextpad=0.40,
        columnspacing=1.35,
        bbox_to_anchor=(0.5, 0.015),
    )
    save(fig, "Figure5")
    print(
        "Partly Cloudy film-level:",
        f"n_ASD={int(isc_stats['n_asd'])}, n_TD={int(isc_stats['n_td'])},",
        f"p={float(isc_stats['p_value']):.3e}",
    )


if __name__ == "__main__":
    make_figure5()
