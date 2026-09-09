"""Figure 3 — developmental divergence (current design).

Box+strip age-split panel with raised significance bracket.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _grammar_style import (
    ASD, DATA, INK, LABEL_GRID, LEGEND, LETTER_ROW, NOTE, PANEL_Y, REF, SPINE_LW,
    SUPP, TICK_GRID, TITLE_GRID, TITLE_PAD, TD,
    add_sig, box_strip, clean, draw_forest, ols_ci, panel, save,
)


def make_figure3():
    df = pd.read_csv(DATA / "fig3_development_subjects.csv").copy()
    df["age_years"] = df["age_months"] / 12
    asd = df[df.group.eq("ASD")].copy()
    sens = pd.read_csv(SUPP / "s4_development_interactions.csv")
    sens = sens[sens["region"].eq("posterior") & sens["model"].isin([
        "Primary", "IQ-balanced matched", "Strict-QC OLS", "Strict-QC robust (robust)",
    ])].drop_duplicates("model")
    order = ["Primary", "IQ-balanced matched", "Strict-QC OLS", "Strict-QC robust (robust)"]
    sens["model"] = pd.Categorical(sens["model"], categories=order, ordered=True)
    sens = sens.sort_values("model")

    fig_w, fig_h = 7.20, 6.20
    fig = plt.figure(figsize=(fig_w, fig_h))
    h_box = 0.352
    w_ac = 0.328
    left = 0.112
    gap = 0.148
    x_bd = left + w_ac + gap
    w_bd = 0.352
    y_c = 0.108
    y_a = y_c + h_box + 0.158
    ax_a = fig.add_axes([left, y_a, w_ac, h_box])
    ax_b = fig.add_axes([x_bd, y_a, w_bd, h_box])
    ax_c = fig.add_axes([left, y_c, w_ac, h_box])
    ax_d = fig.add_axes([x_bd, y_c, w_bd, h_box])
    for ax, lab in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")]:
        panel(ax, lab, size=LETTER_ROW, y=PANEL_Y)

    rng = np.random.default_rng(17)
    for group, color, z_sc in [("ASD", ASD, 4), ("TD", TD, 3)]:
        sub = df[df.group.eq(group)].dropna(subset=["age_years", "posterior_exponent"])
        xg = np.linspace(3.3, 11.0, 120)
        yhat, lo, hi = ols_ci(sub.age_years, sub.posterior_exponent, xg)
        ax_a.plot(xg, yhat, color=color, lw=1.70, solid_capstyle="round",
                  zorder=2, label=group)
        ax_a.scatter(
            sub.age_years + rng.normal(0, 0.015, len(sub)),
            sub.posterior_exponent,
            s=20.0, facecolor=color, edgecolors="black", linewidths=0.22,
            alpha=1.0, zorder=z_sc,
        )
    ax_a.set_xlim(3.2, 11.2)
    ax_a.set_ylim(1.05, 2.58)
    ax_a.set_yticks([1.2, 1.6, 2.0, 2.4])
    ax_a.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax_a.set_title("Age-related variation", pad=TITLE_PAD, fontsize=TITLE_GRID)
    ax_a.set_xlabel("Age (years)", fontsize=LABEL_GRID)
    ax_a.set_ylabel("Posterior aperiodic exponent", fontsize=LABEL_GRID)
    ax_a.legend(loc="upper right", fontsize=LEGEND, handlelength=1.15, borderaxespad=0.12)
    clean(ax_a, ticksize=TICK_GRID, lw=SPINE_LW)

    draw_forest(
        ax_b,
        ["Primary", "IQ-balanced", "Strict QC OLS", "Strict QC robust"],
        sens["estimate_group_by_age"].to_numpy(float),
        sens["ci_low"].to_numpy(float),
        sens["ci_high"].to_numpy(float),
        [INK] * 4,
        "Age × group β (per month)",
        (-0.0025, 0.0110),
        xticks=[0.000, 0.004, 0.008],
        ticksize=TICK_GRID,
        zero_color="#D0D0D0",
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

    ax_c.scatter(
        asd.age_years, asd.posterior_td_reference_z,
        s=20.0, facecolor=ASD, edgecolors="black", linewidths=0.22,
        alpha=1.0, zorder=3,
    )
    xg = np.linspace(3.3, 11.0, 120)
    yhat, lo, hi = ols_ci(asd.age_years, asd.posterior_td_reference_z, xg)
    ax_c.plot(xg, yhat, color=ASD, lw=1.70, solid_capstyle="round", zorder=2)
    ax_c.axhline(0, color=REF, lw=0.85, ls=(0, (3.5, 2.2)), zorder=0)
    ax_c.text(
        10.9, 0.12, "TD reference (z = 0)", ha="right", va="bottom",
        fontsize=NOTE, color=REF, fontweight="normal",
    )
    ax_c.text(
        0.04, 0.96, "r = −0.33\np = 0.008",
        transform=ax_c.transAxes, ha="left", va="top",
        fontsize=NOTE, color=INK, fontweight="bold",
    )
    ax_c.set_xlim(3.2, 11.2)
    ax_c.set_ylim(-5.70, 4.05)
    ax_c.set_yticks([-4, -2, 0, 2])
    ax_c.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax_c.set_title("TD-reference deviation", pad=TITLE_PAD, fontsize=TITLE_GRID)
    ax_c.set_xlabel("Age (years)", fontsize=LABEL_GRID)
    ax_c.set_ylabel("Posterior exponent z-score", fontsize=LABEL_GRID)
    clean(ax_c, ticksize=TICK_GRID, lw=SPINE_LW)

    specs = [
        ("younger_or_equal_72mo", "Younger ASD\nn = 10", -1),
        ("older", "Older ASD\nn = 51", +1),
    ]
    pos = [0.82, 1.28]
    maxima = []
    for (key, _label, side), x in zip(specs, pos):
        vals = asd[asd.age_group_72mo.eq(key)].posterior_td_reference_z.dropna().to_numpy(float)
        maxima.append(float(vals.max()))
        box_strip(ax_d, vals, x, ASD, width=0.24, side=side)
    ax_d.set_xlim(0.45, 1.55)
    ymin, ylim_top = -5.70, 4.05
    data_max = max(maxima)
    ax_d.set_ylim(ymin, ylim_top)
    ax_d.set_yticks([-4, -2, 0, 2])
    ax_d.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax_d.set_xticks(pos, [lab for _, lab, _side in specs])
    span = ylim_top - ymin
    add_sig(
        ax_d, pos[0], pos[1],
        data_max + 0.028 * span, 0.014 * span, "*",
    )
    ax_d.set_title("Exploratory age split", pad=TITLE_PAD, fontsize=TITLE_GRID)
    ax_d.set_ylabel("Posterior exponent z-score", fontsize=LABEL_GRID)
    clean(ax_d, ticksize=TICK_GRID, lw=SPINE_LW)
    save(fig, "Figure3", pad_inches=0.08)


if __name__ == "__main__":
    make_figure3()
