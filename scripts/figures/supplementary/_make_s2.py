"""Supplementary Figure S2 — developmental sensitivity (current design)."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _si_style import (
    ASD, ASD_LIGHT, DATA, INK, LABEL_GRID, LEGEND, LETTER_ROW, NOTE, PANEL_Y, REF,
    SPINE_LW, SUPP, TD, TD_LIGHT, TICK_GRID, TITLE_GRID, TITLE_PAD, ZERO, add_sig,
    box_strip, clean, draw_forest, group_legend_handles, panel, p_to_stars,
    reg_line, save, scatter_pts,
)


def make_s2():
    pred = pd.read_csv(SUPP / "s4_development_predictions_with_ci.csv")
    inter = pd.read_csv(SUPP / "s4_development_interactions.csv")
    subj = pd.read_csv(DATA / "fig3_development_subjects.csv")

    fig = plt.figure(figsize=(7.20, 6.20))
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1.05, 1.00], width_ratios=[1.10, 0.95, 1.20],
        hspace=0.52, wspace=0.42, left=0.075, right=0.985, top=0.94, bottom=0.08,
    )
    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])

    panel(ax_a, "a", size=LETTER_ROW, y=PANEL_Y)
    colors = {"ASD": ASD, "TD": TD}
    light = {"ASD": ASD_LIGHT, "TD": TD_LIGHT}
    for group in ["ASD", "TD"]:
        lin = pred[(pred["model"] == "linear_interaction") & (pred["group"] == group)].sort_values("age_months")
        spl = pred[(pred["model"] == "spline_interaction") & (pred["group"] == group)].sort_values("age_months")
        x = lin["age_months"].to_numpy() / 12
        ax_a.fill_between(
            x, lin["ci_low"].to_numpy(), lin["ci_high"].to_numpy(),
            color=light[group], alpha=0.35, linewidth=0, zorder=1,
        )
        ax_a.plot(
            x, lin["predicted_exponent"].to_numpy(), color=colors[group],
            lw=1.70, solid_capstyle="round", label=group, zorder=3,
        )
        ax_a.plot(
            spl["age_months"].to_numpy() / 12, spl["predicted_exponent"].to_numpy(),
            color=colors[group], lw=1.05, ls=(0, (3.2, 2.4)), zorder=2,
        )
    ax_a.set_title("Linear and spline age trajectories", pad=TITLE_PAD, fontsize=TITLE_GRID)
    ax_a.set_xlabel("Age (years)", fontsize=LABEL_GRID)
    ax_a.set_ylabel("Posterior exponent", fontsize=LABEL_GRID)
    ax_a.set_xlim(3.2, 11.3)
    ax_a.set_ylim(1.58, 2.20)
    ax_a.legend(
        handles=group_legend_handles(), loc="upper right", fontsize=LEGEND,
        handlelength=1.15, borderaxespad=0.12,
    )
    clean(ax_a, ticksize=TICK_GRID, lw=SPINE_LW)

    keep = ["Primary", "IQ-balanced matched", "Strict-QC OLS", "Strict-QC robust (robust)"]
    data = inter[(inter["region"] == "posterior") & (inter["model"].isin(keep))].copy()
    data["model"] = pd.Categorical(data["model"], keep, ordered=True)
    data = data.sort_values("model")
    panel(ax_b, "b", size=LETTER_ROW, y=PANEL_Y)
    draw_forest(
        ax_b,
        ["Primary", "IQ-balanced", "Strict QC OLS", "Strict QC robust"],
        data["estimate_group_by_age"].to_numpy(float),
        data["ci_low"].to_numpy(float),
        data["ci_high"].to_numpy(float),
        [INK] * 4,
        "Age × group β (per month)",
        (-0.0025, 0.0110),
        xticks=[0.000, 0.004, 0.008],
        ticksize=TICK_GRID,
        zero_color=ZERO,
        zero_lw=0.90,
        zero_ls=(0, (3.5, 2.2)),
        marker_size=5.6,
        marker_fmts=["^", "^", "o", "o"],
        elinewidth=0.70,
        capsize=2.8,
        spine_lw=SPINE_LW,
        y_gap=0.68,
    )
    ax_b.set_ylim(-0.68, 2.72)
    ax_b.set_title("Age × group sensitivity", pad=TITLE_PAD, fontsize=TITLE_GRID)

    panel(ax_c, "c", size=LETTER_ROW, y=PANEL_Y)
    asd_z = subj[subj["group"] == "ASD"]["posterior_td_reference_z"].dropna().to_numpy()
    bins = np.linspace(-5.5, 3.2, 16)
    ax_c.hist(asd_z, bins=bins, color=ASD, alpha=1.0, edgecolor=INK, linewidth=0.35)
    ax_c.axvline(0, color=REF, lw=0.85, ls=(0, (3.5, 2.2)))
    ax_c.text(
        0.62, 0.94, "TD reference", transform=ax_c.transAxes,
        ha="left", va="top", fontsize=NOTE, color=REF,
    )
    ax_c.set_title("ASD TD-reference deviation", pad=TITLE_PAD, fontsize=TITLE_GRID)
    ax_c.set_xlabel("Deviation z-score", fontsize=LABEL_GRID)
    ax_c.set_ylabel("Count", fontsize=LABEL_GRID)
    clean(ax_c, ticksize=TICK_GRID, lw=SPINE_LW)

    panel(ax_d, "d", size=LETTER_ROW, y=PANEL_Y)
    asd_df = subj[(subj["group"] == "ASD") & subj["posterior_td_reference_z"].notna()].copy()
    x = asd_df["age_months"].to_numpy() / 12
    yv = asd_df["posterior_td_reference_z"].to_numpy()
    scatter_pts(ax_d, x, yv, ASD)
    reg_line(ax_d, x, yv, ASD)
    ax_d.text(
        0.04, 0.96, "r = −0.33\np = 0.008",
        transform=ax_d.transAxes, ha="left", va="top",
        fontsize=NOTE, color=INK, fontweight="bold", linespacing=1.15,
    )
    ax_d.set_title("Deviation vs age in ASD", pad=TITLE_PAD, fontsize=TITLE_GRID)
    ax_d.set_xlabel("Age (years)", fontsize=LABEL_GRID)
    ax_d.set_ylabel("Deviation z-score", fontsize=LABEL_GRID)
    ax_d.set_xlim(3.4, 11.2)
    ax_d.set_ylim(-5.70, 4.05)
    ax_d.set_yticks([-4, -2, 0, 2])
    clean(ax_d, ticksize=TICK_GRID, lw=SPINE_LW)

    panel(ax_e, "e", size=LETTER_ROW, y=PANEL_Y)
    younger = asd_df[asd_df["age_months"] <= 72]["posterior_td_reference_z"].to_numpy(float)
    older = asd_df[asd_df["age_months"] > 72]["posterior_td_reference_z"].to_numpy(float)
    specs = [(younger, "≤72 mo", -1), (older, ">72 mo", +1)]
    pos = [0.82, 1.28]
    maxima = []
    for (vals, _lab, side), x0 in zip(specs, pos):
        vals = vals[np.isfinite(vals)]
        maxima.append(float(vals.max()))
        box_strip(ax_e, vals, x0, ASD, width=0.24, side=side)
    data_max = max(maxima)
    ymin, ylim_top = -5.70, 4.05
    ax_e.set_xlim(0.45, 1.55)
    ax_e.set_ylim(ymin, ylim_top)
    ax_e.set_yticks([-4, -2, 0, 2])
    ax_e.set_xticks(
        pos,
        [f"{lab}\nn = {int(np.isfinite(v).sum())}" for v, lab, _s in specs],
    )
    _, p = stats.ttest_ind(younger, older, equal_var=False, nan_policy="omit")
    span = ylim_top - ymin
    add_sig(ax_e, pos[0], pos[1], data_max + 0.028 * span, 0.014 * span, p_to_stars(p))
    ax_e.set_title("ASD age subgroup", pad=TITLE_PAD, fontsize=TITLE_GRID)
    ax_e.set_ylabel("Deviation z-score", fontsize=LABEL_GRID)
    clean(ax_e, ticksize=TICK_GRID, lw=SPINE_LW)

    save(fig, "SupplementaryFigureS2", pad_inches=0.08)


if __name__ == "__main__":
    make_s2()
