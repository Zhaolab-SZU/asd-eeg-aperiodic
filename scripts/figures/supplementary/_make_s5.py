"""Supplementary Figure S5 — HBN The Present ISC and resting convergence."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _si_style import (
    ASD, INK, LABEL_ROW, LEGEND, LETTER_ROW, PANEL_Y, SPINE_LW, SUPP, TD, TICK_GRID,
    TITLE_GRID, TITLE_PAD, ZERO,
    add_sig, box_strip, clean, draw_forest, group_legend_handles, panel,
    p_to_stars, save,
)


def isc_panel(ax, hbn, summary):
    specs = [
        ("sliding_window", "Sliding\nwindows"),
        ("nonoverlapping_2s_epoch", "2-s epochs"),
    ]
    width = 0.24
    box_width = 0.21
    pair_off = 0.18
    gap = 0.050
    bases = np.array([1.18, 2.42])
    fill = {"ASD": ASD, "TD": TD}

    maxima = []
    cond_max = []
    stars = []
    for base, (key, _lab) in zip(bases, specs):
        cond_vals = []
        for group, sign in (("ASD", -1), ("TD", +1)):
            vals = hbn.loc[
                hbn["analysis"].eq(key) & hbn["group"].eq(group), "isc_z"
            ].dropna().to_numpy(float)
            maxima.append(float(np.nanmax(vals)))
            cond_vals.append(vals)
            box_strip(
                ax, vals, base + sign * pair_off, fill[group],
                width=width, side=-1 if group == "ASD" else +1,
                gap=gap, box_width=box_width,
            )
        cond_max.append(float(np.nanmax(np.concatenate(cond_vals))))
        p = float(summary.loc[summary["analysis"].eq(key), "p"].iloc[0])
        stars.append(p_to_stars(p))

    strip_half = 0.24 * box_width
    strip_distance = 0.50 * box_width + gap + strip_half
    x_left = bases[0] - pair_off - strip_distance - strip_half
    x_right = bases[-1] + pair_off + strip_distance + strip_half
    ax.set_xlim(x_left - 0.32, x_right + 0.32)
    ax.set_xticks(bases, [lab for _k, lab in specs])
    ax.set_title("The Present Aperiodic-ISC", pad=TITLE_PAD, fontweight="bold")
    ax.set_ylabel("Aperiodic-ISC (Fisher $z$)", fontsize=LABEL_ROW)

    ymin, ylim_top = -0.50, 0.78
    data_max = max(maxima)
    span = ylim_top - ymin
    top = max(ylim_top, data_max + 0.14 * span)
    ax.set_ylim(ymin, top)
    ax.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6])

    span = top - ymin
    h = 0.016 * span
    for base, cmax, sig in zip(bases, cond_max, stars):
        add_sig(ax, base - pair_off, base + pair_off, cmax + 0.050 * span, h, sig)
    clean(ax)


def make_s5():
    hbn = pd.read_csv(SUPP / "s8_hbn_movie_subjects.csv")
    summary = pd.read_csv(SUPP / "s8_hbn_movie_summary.csv")
    models = pd.read_csv(SUPP / "s8_hbn_resting_models.csv")

    fig = plt.figure(figsize=(8.70, 3.62))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.28, 1.00],
        left=0.062, right=0.988, bottom=0.235, top=0.84, wspace=0.30,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    panel(ax_a, "a", x=-0.10, y=PANEL_Y, size=LETTER_ROW)
    panel(ax_b, "b", x=-0.16, y=PANEL_Y, size=LETTER_ROW)

    isc_panel(ax_a, hbn, summary)

    order = [
        ("nuclear_knee", "primary_covariates", "Knee, OLS"),
        ("nuclear_knee", "welch_group_comparison", "Knee, Welch"),
        ("nuclear_knee", "paired_t", "Knee, paired"),
        ("legacy_fixed", "primary_covariates", "Fixed, OLS"),
        ("legacy_fixed", "welch_group_comparison", "Fixed, Welch"),
        ("legacy_fixed", "paired_t", "Fixed, paired"),
    ]
    labels, beta, lo, hi, colors, fmts = [], [], [], [], [], []
    for pipeline, model_type, label in order:
        row = models[(models["pipeline"] == pipeline) & (models["model_type"] == model_type)].iloc[0]
        est = float(row["estimate_td_minus_asd"])
        if pd.notna(row["ci_low"]) and pd.notna(row["ci_high"]):
            clo, chi = float(row["ci_low"]), float(row["ci_high"])
        elif pd.notna(row["se"]):
            clo, chi = est - 1.96 * float(row["se"]), est + 1.96 * float(row["se"])
        else:
            clo, chi = est, est
        labels.append(label)
        beta.append(est)
        lo.append(clo)
        hi.append(chi)
        colors.append(INK if pipeline == "nuclear_knee" else "#777777")
        fmts.append("^" if pipeline == "nuclear_knee" else "o")

    draw_forest(
        ax_b, labels, beta, lo, hi, colors,
        "TD − ASD posterior exponent",
        (-0.09, 0.37),
        xticks=[0.00, 0.15, 0.30],
        ticksize=TICK_GRID,
        zero_color=ZERO,
        zero_lw=0.90,
        zero_ls=(0, (3.5, 2.2)),
        marker_size=5.2,
        marker_fmts=fmts,
        marker_colors=colors,
        elinewidth=0.70,
        capsize=2.6,
        spine_lw=SPINE_LW,
        y_gap=0.72,
    )
    ax_b.set_title("HBN eyes-open resting", pad=TITLE_PAD, fontsize=TITLE_GRID)
    ax_b.set_ylim(-0.55, 4.05)

    fig.legend(
        handles=group_legend_handles(),
        loc="lower center", ncol=2, fontsize=LEGEND, frameon=False,
        handletextpad=0.40, columnspacing=1.35, bbox_to_anchor=(0.5, 0.015),
    )
    save(fig, "SupplementaryFigureS5")


if __name__ == "__main__":
    make_s5()
