"""Supplementary Figure S4 — Partly Cloudy synchrony controls (current design)."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _si_style import (
    DATA, LEGEND, LETTER_ROW, PANEL_Y, group_legend_handles, pair_box_strip, panel,
    p_to_stars, save,
)

SYNC = DATA / "film_level_synchrony_controls"


def make_s4():
    isc = pd.read_csv(DATA / "fig5_partly_cloudy_aperiodic_isc_subjects.csv")
    env = pd.read_csv(SYNC / "partly_cloudy_envelope_isc.csv")
    plv = pd.read_csv(SYNC / "partly_cloudy_alpha_plv_isc.csv")
    summary = pd.read_csv(SYNC / "partly_cloudy_synchrony_controls.csv").set_index("isc_type")

    p_a = float(summary.loc["Partly Cloudy Aperiodic-ISC", "p_value"])
    p_b = float(summary.loc["Partly Cloudy broadband envelope ISC", "p_value"])
    p_c = float(summary.loc["Partly Cloudy alpha PLV ISC", "p_value"])

    fig = plt.figure(figsize=(8.70, 3.62))
    gs = fig.add_gridspec(
        1, 3, wspace=0.32, left=0.055, right=0.988, top=0.84, bottom=0.235,
    )
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[0, i]) for i in range(3)]

    panel(ax_a, "a", x=-0.12, y=PANEL_Y, size=LETTER_ROW)
    pair_box_strip(
        ax_a,
        isc.loc[isc["group"].eq("ASD"), "isc_z"],
        isc.loc[isc["group"].eq("TD"), "isc_z"],
        p_to_stars(p_a),
        "Aperiodic-ISC (Fisher $z$)", "Aperiodic-ISC",
        ylim=(-0.28, 0.42),
    )
    panel(ax_b, "b", x=-0.12, y=PANEL_Y, size=LETTER_ROW)
    pair_box_strip(
        ax_b,
        env.loc[env["group"].eq("ASD"), "isc_z"],
        env.loc[env["group"].eq("TD"), "isc_z"],
        p_to_stars(p_b),
        "Envelope ISC (Fisher $z$)", "Broadband envelope ISC",
        ylim=(-0.25, 1.20),
    )
    panel(ax_c, "c", x=-0.14, y=PANEL_Y, size=LETTER_ROW)
    pair_box_strip(
        ax_c,
        plv.loc[plv["group"].eq("ASD"), "isc_z"],
        plv.loc[plv["group"].eq("TD"), "isc_z"],
        p_to_stars(p_c),
        "Alpha PLV ISC (Fisher $z$)", "Alpha PLV ISC",
        ylim=(0.00, 0.145),
    )

    fig.legend(
        handles=group_legend_handles(),
        loc="lower center", ncol=2, fontsize=LEGEND, frameon=False,
        handletextpad=0.40, columnspacing=1.35, bbox_to_anchor=(0.5, 0.015),
    )
    save(fig, "SupplementaryFigureS4")
    print(
        "S4:",
        f"A p={p_a:.3e}; B p={p_b:.3e}; C p={p_c:.3e}",
    )


if __name__ == "__main__":
    make_s4()
