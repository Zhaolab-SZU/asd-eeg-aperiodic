"""Supplementary Figure S3 — IAF controls (current design)."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator
from scipy import stats

from _si_style import (
    ASD, INK, LABEL_GRID, LETTER_ROW, NOTE, PANEL_Y, SPINE_LW, SUPP, TD, TICK_GRID,
    TITLE_GRID, TITLE_PAD, ZERO,
    clean, draw_forest, p_label, pair_box_strip, panel, p_to_stars,
    reg_line, save, scatter_pts,
)


def make_s3():
    df = pd.read_csv(SUPP / "s5_iaf_subjects.csv")
    models = pd.read_csv(SUPP / "s5_iaf_models.csv")

    fig = plt.figure(figsize=(6.80, 5.70))
    gs = fig.add_gridspec(
        2, 2, hspace=0.48, wspace=0.40,
        left=0.105, right=0.955, top=0.94, bottom=0.09,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    asd_iaf = df.loc[df["group"] == "ASD", "posterior_iaf"].dropna().to_numpy()
    td_iaf = df.loc[df["group"] == "TD", "posterior_iaf"].dropna().to_numpy()
    _, p_iaf = stats.ttest_ind(asd_iaf, td_iaf, equal_var=False, nan_policy="omit")
    panel(ax_a, "a", size=LETTER_ROW, y=PANEL_Y)
    pair_box_strip(
        ax_a, asd_iaf, td_iaf, p_to_stars(p_iaf),
        "Posterior IAF (Hz)", "Posterior IAF",
        ylim=(6.5, 12.2), yticks=[7, 9, 11],
    )

    panel(ax_b, "b", size=LETTER_ROW, y=PANEL_Y)
    for group, color in [("ASD", ASD), ("TD", TD)]:
        sub = df[(df["group"] == group) & df["posterior_iaf"].notna()].copy()
        x = sub["age_months"].to_numpy() / 12
        yv = sub["posterior_iaf"].to_numpy()
        scatter_pts(ax_b, x, yv, color, s=18)
        xx, coef = reg_line(ax_b, x, yv, color)
        ax_b.text(
            xx[-1] + 0.12, coef[0] * xx[-1] + coef[1], group,
            ha="left", va="center", fontsize=NOTE, color=color, fontweight="bold",
            clip_on=False,
        )
    ax_b.set_title("Posterior IAF vs age", pad=TITLE_PAD, fontsize=TITLE_GRID)
    ax_b.set_xlabel("Age (years)", fontsize=LABEL_GRID)
    ax_b.set_ylabel("Posterior IAF (Hz)", fontsize=LABEL_GRID)
    ax_b.set_xlim(3.2, 12.2)
    ax_b.yaxis.set_major_locator(FixedLocator([8, 9, 10, 11]))
    clean(ax_b, ticksize=TICK_GRID, lw=SPINE_LW)

    panel(ax_c, "c", size=LETTER_ROW, y=PANEL_Y)
    asd = df[df["group"] == "ASD"].dropna(subset=["posterior_exponent_deviation_z", "iaf_deviation_z"])
    x = asd["iaf_deviation_z"].to_numpy()
    yv = asd["posterior_exponent_deviation_z"].to_numpy()
    rho = models.loc[models["term"] == "Spearman_rho", "estimate"].iloc[0]
    p = models.loc[models["term"] == "Spearman_rho", "p"].iloc[0]
    scatter_pts(ax_c, x, yv, ASD)
    reg_line(ax_c, x, yv, ASD)
    rho_txt = f"{rho:.2f}".replace("-", "−")
    ax_c.text(
        0.04, 0.96, f"ρ = {rho_txt}\n{p_label(p)}",
        transform=ax_c.transAxes, ha="left", va="top",
        fontsize=NOTE, color=INK, fontweight="bold", linespacing=1.15,
    )
    ax_c.set_title("Exponent vs IAF deviation", pad=TITLE_PAD, fontsize=TITLE_GRID)
    ax_c.set_xlabel("IAF deviation z-score", fontsize=LABEL_GRID)
    ax_c.set_ylabel("Exponent deviation z-score", fontsize=LABEL_GRID)
    ax_c.yaxis.set_major_locator(FixedLocator([-4, -2, 0, 2]))
    clean(ax_c, ticksize=TICK_GRID, lw=SPINE_LW)

    panel(ax_d, "d", size=LETTER_ROW, y=PANEL_Y)
    age_adj = models.loc[models["model"] == "posterior exponent group x age adjusted for posterior IAF"].iloc[0]
    dev = pd.read_csv(SUPP / "s4_development_interactions.csv")
    unadj = dev[(dev["model"] == "Primary") & (dev["region"] == "posterior")].iloc[0]
    draw_forest(
        ax_d,
        ["Unadjusted", "IAF-adjusted"],
        [float(unadj.estimate_group_by_age), float(age_adj.estimate)],
        [float(unadj.ci_low), float(age_adj.ci_low)],
        [float(unadj.ci_high), float(age_adj.ci_high)],
        [INK, INK],
        "Age × group β",
        (-0.001, 0.0105),
        xticks=[0.000, 0.004, 0.008],
        ticksize=TICK_GRID,
        zero_color=ZERO,
        zero_lw=0.90,
        zero_ls=(0, (3.5, 2.2)),
        marker_size=5.6,
        marker_fmts=["^", "o"],
        elinewidth=0.70,
        capsize=2.8,
        spine_lw=SPINE_LW,
        y_gap=0.68,
    )
    ax_d.set_ylim(-0.55, 1.55)
    ax_d.set_title("Posterior age × group", pad=TITLE_PAD, fontsize=TITLE_GRID)

    save(fig, "SupplementaryFigureS3", pad_inches=0.12)


if __name__ == "__main__":
    make_s3()
