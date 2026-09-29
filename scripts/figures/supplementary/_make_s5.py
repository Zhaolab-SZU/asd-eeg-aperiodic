"""Supplementary Figure S5 — HBN The Present parameterization check."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _si_style import (
    ASD, LABEL_ROW, LEGEND, SUPP, TD,
    add_sig, box_strip, clean, group_legend_handles, p_to_stars, save,
)


def make_s5():
    hbn = pd.read_csv(SUPP / "s8_hbn_movie_subjects.csv")
    summary = pd.read_csv(SUPP / "s8_hbn_movie_summary.csv")
    specs = [
        ("sliding_window", "Sliding\nwindows"),
        ("nonoverlapping_2s_epoch", "2-s epochs"),
    ]
    bases = np.array([1.18, 2.42])
    pair_off = 0.18
    maxima, stars = [], []

    fig, ax = plt.subplots(figsize=(5.25, 3.62))
    for base, (key, _label) in zip(bases, specs):
        condition_values = []
        for group, sign, color in (("ASD", -1, ASD), ("TD", +1, TD)):
            vals = hbn.loc[
                hbn["analysis"].eq(key) & hbn["group"].eq(group), "isc_z"
            ].dropna().to_numpy(float)
            condition_values.append(vals)
            box_strip(
                ax, vals, base + sign * pair_off, color,
                width=0.24, side=sign, gap=0.050, box_width=0.21,
            )
        maxima.append(float(np.nanmax(np.concatenate(condition_values))))
        p = float(summary.loc[summary["analysis"].eq(key), "p"].iloc[0])
        stars.append(p_to_stars(p))

    ax.set_xlim(0.62, 2.98)
    ax.set_xticks(bases, [label for _, label in specs])
    ax.set_title("The Present Aperiodic-ISC", fontweight="bold")
    ax.set_ylabel("Aperiodic-ISC (Fisher $z$)", fontsize=LABEL_ROW)
    ymin, top = -0.50, 0.78
    ax.set_ylim(ymin, top)
    ax.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    span = top - ymin
    for base, maximum, star in zip(bases, maxima, stars):
        add_sig(
            ax, base - pair_off, base + pair_off,
            maximum + 0.050 * span, 0.016 * span, star,
        )
    clean(ax)
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.25, top=0.82)
    fig.legend(
        handles=group_legend_handles(), loc="lower center", ncol=2,
        fontsize=LEGEND, frameon=False, handletextpad=0.40,
        columnspacing=1.35, bbox_to_anchor=(0.5, 0.015),
    )
    save(fig, "SupplementaryFigureS5")


if __name__ == "__main__":
    make_s5()
