"""Figure 5 — symptom-linked resting posterior exponent associations."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _grammar_style import (
    ASD, DATA, INK, LABEL_ROW, LETTER_ROW, NOTE, PANEL_Y, SPINE_LW,
    TITLE_PAD, TITLE_ROW, clean, panel, regression_ci, residualize, save,
)


def make_figure5():
    subjects = pd.read_csv(DATA / "fig4_clinical_subjects.csv")
    raw = pd.read_csv(DATA / "fig4_clinical_models.csv")
    raw = raw[raw["note"].eq("domain_framework")].copy()
    raw["label"] = raw["outcome"].map({
        "ADOS total": "ADOS total",
        "ADOS Social Affect": "Social Affect",
        "ADOS Restricted and Repetitive Behavior": "RRB",
    })
    models = raw.dropna(subset=["label"]).set_index("label")
    base_df, x_resid = residualize(subjects, "posterior_exponent")
    base = base_df[["subject_id"]].copy()
    base["x"] = x_resid
    outcomes = [
        ("ADOS total", "ADOS_total", "ADOS total residual", "a"),
        ("Social Affect", "ADOS_SA", "Social Affect residual", "b"),
        ("RRB", "ADOS_RRB", "RRB residual", "c"),
    ]
    ylims = {
        "ADOS total": (-9.4, 9.4),
        "Social Affect": (-6.2, 7.4),
        "RRB": (-3.4, 3.75),
    }
    yticks = {
        "ADOS total": [-8, -4, 0, 4, 8],
        "Social Affect": [-5, 0, 5],
        "RRB": [-2, 0, 2],
    }

    fig_w, fig_h = 7.65, 3.05
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    h_ax = 0.600
    w_ax = h_ax * fig_h / fig_w
    left, gap, y0 = 0.090, 0.058, 0.235
    axes = [fig.add_axes([left + i * (w_ax + gap), y0, w_ax, h_ax]) for i in range(3)]
    for ax, (label, col, ylabel, letter) in zip(axes, outcomes):
        y_df, y_resid = residualize(subjects, col)
        merged = base.merge(
            pd.DataFrame({"subject_id": y_df["subject_id"], "y": y_resid}),
            on="subject_id", how="inner",
        )
        stat = models.loc[label]
        x = merged["x"].to_numpy(float)
        y = merged["y"].to_numpy(float)
        x_grid = np.linspace(-0.48, 0.48, 240)
        y_hat, _ = regression_ci(x, y, x_grid)
        ax.plot(x_grid, y_hat, color=ASD, lw=1.70, solid_capstyle="round", zorder=2)
        ax.scatter(
            x, y, s=20.0, facecolor=ASD, edgecolors="black",
            linewidths=0.22, alpha=1.0, zorder=3,
        )
        ax.set_title(label, pad=TITLE_PAD, fontsize=TITLE_ROW)
        ax.set_xlabel("Posterior exponent residual", fontsize=LABEL_ROW)
        ax.set_ylabel(ylabel, fontsize=LABEL_ROW)
        ax.set_xlim(-0.50, 0.50)
        ax.set_xticks([-0.4, 0.0, 0.4])
        ax.set_ylim(*ylims[label])
        ax.set_yticks(yticks[label])
        panel(ax, letter, x=-0.16, y=PANEL_Y, size=LETTER_ROW)
        q = float(stat["FDR_q"])
        q_text = f"q = {q:.3f}" if q < 0.1 else f"q = {q:.2f}"
        ax.text(
            0.04, 0.96, f"partial r = {stat['partial_r']:.2f}\n{q_text}",
            transform=ax.transAxes, ha="left", va="top", fontsize=NOTE,
            fontweight="bold", color=INK,
        )
        clean(ax, lw=SPINE_LW)
    save(fig, "Figure5")


if __name__ == "__main__":
    make_figure5()
