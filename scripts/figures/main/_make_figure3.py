"""Figure 3 — TD-referenced temporal alignment during naturalistic viewing.

Panels
------
a  Partly Cloudy Aperiodic-ISC in the primary movie spectral-QC cohort.
b  HBN The Present sliding-window Aperiodic-ISC for cross-stimulus support.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _grammar_style import (
    ASD, DATA, INK, LABEL, NOTE, PANEL_Y, TD, TITLE_PAD,
    add_sig, box_strip, clean, p_to_stars, panel, save,
)

TITLE_F3 = 14.0
CAT_F3 = 12.0


def single_isc_panel(ax, df, title, p_value, panel_label):
    """Draw one direct-labelled ASD-versus-TD Aperiodic-ISC comparison."""
    groups = [("ASD", ASD, 0.82, -1), ("TD", TD, 1.18, +1)]
    maxima = []
    labels = []
    for group, color, pos, side in groups:
        vals = df.loc[df["group"].eq(group), "isc_z"].dropna().to_numpy(float)
        maxima.append(float(np.nanmax(vals)))
        labels.append(f"{group}\n$n$ = {len(vals)}")
        box_strip(
            ax, vals, pos, color, width=0.20, box_width=0.14,
            gap=0.032, side=side,
        )

    ax.set_xlim(0.48, 1.52)
    ax.set_ylim(-0.40, 0.52)
    ax.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    ax.set_xticks([0.82, 1.18], labels)
    ax.set_title(title, pad=TITLE_PAD, fontsize=TITLE_F3, fontweight="bold")
    span = 0.92
    add_sig(
        ax, 0.82, 1.18, max(maxima) + 0.055 * span,
        0.016 * span, p_to_stars(float(p_value)),
    )
    panel(ax, panel_label, x=-0.12, y=PANEL_Y)
    clean(ax, ticksize=CAT_F3)


def make_figure3():
    isc = pd.read_csv(DATA / "fig5_partly_cloudy_aperiodic_isc_subjects.csv")
    isc_stats = pd.read_csv(DATA / "fig5_partly_cloudy_aperiodic_isc_stats.csv").iloc[0]
    hbn = pd.read_csv(DATA / "fig5_the_present_aperiodic_isc_subjects.csv")
    hbn_sum = pd.read_csv(DATA / "fig5_the_present_aperiodic_isc.csv")
    hbn_sw = hbn.loc[hbn["analysis"].eq("sliding_window")].copy()
    hbn_p = float(
        hbn_sum.loc[
            hbn_sum["analysis"].eq("sliding-window posterior Aperiodic-ISC"), "p"
        ].iloc[0]
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(9.10, 3.73), sharey=True,
        gridspec_kw={
            "left": 0.080, "right": 0.985, "bottom": 0.245,
            "top": 0.82, "wspace": 0.24,
        },
    )
    single_isc_panel(axes[0], isc, "Partly Cloudy", isc_stats["p_value"], "a")
    single_isc_panel(axes[1], hbn_sw, "The Present (HBN)", hbn_p, "b")
    axes[0].set_ylabel("Aperiodic-ISC (Fisher $z$)", fontsize=LABEL)
    axes[1].set_ylabel("")
    fig.text(
        0.5, 0.035, "TD-referenced temporal alignment during naturalistic viewing",
        ha="center", va="bottom", fontsize=NOTE, color=INK,
    )
    save(fig, "Figure3")
    print(
        "Partly Cloudy:",
        f"n_ASD={int(isc_stats['n_asd'])}, n_TD={int(isc_stats['n_td'])},",
        f"p={float(isc_stats['p_value']):.3e}; HBN p={hbn_p:.5f}",
    )


if __name__ == "__main__":
    make_figure3()
