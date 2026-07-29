from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.ticker import FixedLocator


ROOT = Path(__file__).resolve().parents[3]  # repo root
SRC = ROOT / "data" / "supplementary_source"
OUT = ROOT / "outputs" / "figures" / "supplementary"
BACKUP = OUT.parent / "backup"; BACKUP.mkdir(parents=True, exist_ok=True)
BACKUP.mkdir(parents=True, exist_ok=True)

STEM = "SuppFigS3_iaf_controls"
TABLE_STEM = "SuppTable_IAF_control_models"

ASD = "#C25450"
TD = "#4A6FA5"
ASD_LIGHT = "#E9A6A2"
TD_LIGHT = "#B8C9E2"
INK = "#333333"
MUTED = "#666666"
NONSIG = "#9A9A9A"
REF = "#8A8A8A"
SHADE = "#F3F3F3"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def backup_existing() -> None:
    for ext in ("png", "pdf", "svg", "tiff"):
        src = OUT / f"{STEM}.{ext}"
        if src.exists():
            dst = BACKUP / f"{STEM}.{ext}"
            if not dst.exists():
                shutil.copy2(src, dst)


def panel_label(ax, label, x=-0.18, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=13, fontweight="bold", color=INK)


def style_axes(ax):
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(labelsize=7, colors=INK, width=0.9, length=3.2)
    ax.grid(False)


def p_label(p):
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def panel_a(ax, df):
    panel_label(ax, "A")
    rng = np.random.default_rng(20260713)
    for i, (group, color, light) in enumerate([("ASD", ASD, ASD_LIGHT), ("TD", TD, TD_LIGHT)]):
        vals = df.loc[df["group"] == group, "posterior_iaf"].dropna().to_numpy()
        parts = ax.violinplot(vals, positions=[i], widths=0.62, showmeans=False, showextrema=False, showmedians=False)
        for body in parts["bodies"]:
            body.set_facecolor(light)
            body.set_edgecolor(color)
            body.set_alpha(0.75)
        ax.scatter(np.full(len(vals), i) + rng.normal(0, 0.045, len(vals)), vals,
                   s=17, color=color, alpha=0.68, edgecolor="white", linewidth=0.4, zorder=3)
        mean = np.mean(vals)
        ci = stats.sem(vals) * stats.t.ppf(0.975, len(vals) - 1)
        ax.errorbar(i, mean, yerr=ci, fmt="o", color=INK, ecolor=INK,
                    capsize=4, markersize=4.2, lw=1.0, zorder=4)
    asd = df.loc[df["group"] == "ASD", "posterior_iaf"].dropna()
    td = df.loc[df["group"] == "TD", "posterior_iaf"].dropna()
    _, p = stats.ttest_ind(asd, td, equal_var=False)
    y = max(df["posterior_iaf"].dropna()) + 0.42
    ax.plot([0, 0, 1, 1], [y, y + 0.12, y + 0.12, y], color=INK, lw=0.8)
    ax.text(0.5, y + 0.14, "n.s.", ha="center", va="bottom", fontsize=7.8, color=MUTED)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ASD", "TD"], fontsize=7)
    ax.set_title("Posterior IAF", fontsize=8.7, fontweight="bold", color=INK, pad=6)
    ax.set_ylabel("Posterior IAF (Hz)", fontsize=8, fontweight="bold")
    ax.set_ylim(6.5, y + 0.45)
    ax.yaxis.set_major_locator(FixedLocator([7, 9, 11]))
    style_axes(ax)


def panel_b(ax, df):
    panel_label(ax, "B")
    for group, color, light in [("ASD", ASD, ASD_LIGHT), ("TD", TD, TD_LIGHT)]:
        sub = df[(df["group"] == group) & df["posterior_iaf"].notna()].copy()
        x = sub["age_months"].to_numpy() / 12
        y = sub["posterior_iaf"].to_numpy()
        ax.scatter(x, y, s=18, color=color, alpha=0.65, edgecolor="white", linewidth=0.4)
        coef = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        yy = coef[0] * xx + coef[1]
        ax.plot(xx, yy, color=color, lw=1.15)
        ax.text(xx[-1] + 0.10, yy[-1], group, ha="left", va="center",
                fontsize=6.5, color=color, fontweight="bold")
    ax.set_title("Posterior IAF vs age", fontsize=8.7, fontweight="bold", color=INK, pad=6)
    ax.set_xlabel("Age (years)", fontsize=8, fontweight="bold")
    ax.set_ylabel("Posterior IAF (Hz)", fontsize=8, fontweight="bold")
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 0.45)
    ax.yaxis.set_major_locator(FixedLocator([8, 9, 10, 11]))
    style_axes(ax)


def forest(ax, rows, title, xlabel, xlim, annotate=True):
    y = np.arange(len(rows))[::-1]
    for yi, row in zip(y, rows):
        est, lo, hi, p, color = row["estimate"], row["ci_low"], row["ci_high"], row["p"], row["color"]
        ax.errorbar(est, yi, xerr=[[est - lo], [hi - est]], fmt="o",
                    color=color, ecolor=color, capsize=3, elinewidth=1.0,
                    markersize=4.6)
        if annotate:
            label = f"{est:.4f} [{lo:.4f}, {hi:.4f}]"
            ax.text(xlim[1] + (xlim[1] - xlim[0]) * 0.04, yi, label,
                    ha="left", va="center", fontsize=5.6, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([row["label"] for row in rows], fontsize=6.2)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.set_xlabel(xlabel, fontsize=8, fontweight="bold")
    ax.set_title(title, fontsize=8.7, fontweight="bold", color=INK, pad=6)
    style_axes(ax)


def panel_c(ax, df, models):
    panel_label(ax, "C")
    asd = df[df["group"] == "ASD"].dropna(subset=["posterior_exponent_deviation_z", "iaf_deviation_z"])
    x = asd["iaf_deviation_z"].to_numpy()
    y = asd["posterior_exponent_deviation_z"].to_numpy()
    rho = models.loc[models["term"] == "Spearman_rho", "estimate"].iloc[0]
    p = models.loc[models["term"] == "Spearman_rho", "p"].iloc[0]
    ax.scatter(x, y, s=20, color=ASD, alpha=0.68, edgecolor="white", linewidth=0.45)
    coef = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, coef[0] * xx + coef[1], color=INK, lw=1.1)
    ax.text(0.96, 0.94, f"Spearman ρ = {rho:.2f}; {p_label(p)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.2, color=INK)
    ax.set_title("Exponent vs IAF deviation", fontsize=8.7, fontweight="bold", color=INK, pad=6)
    ax.set_xlabel("IAF deviation z-score", fontsize=8, fontweight="bold")
    ax.set_ylabel("Exponent deviation z-score", fontsize=8, fontweight="bold")
    ax.yaxis.set_major_locator(FixedLocator([-4, -2, 0, 2]))
    style_axes(ax)


def panel_d(ax, models):
    panel_label(ax, "D")
    age_adj = models.loc[models["model"] == "posterior exponent group x age adjusted for posterior IAF"].iloc[0]
    # Unadjusted primary age x group coefficient from the development source table.
    dev = pd.read_csv(SRC / "s4_development_interactions.csv")
    unadj = dev[(dev["model"] == "Primary") & (dev["region"] == "posterior")].iloc[0]
    rows = [
        dict(label="Unadjusted", estimate=unadj.estimate_group_by_age, ci_low=unadj.ci_low,
             ci_high=unadj.ci_high, p=unadj.p, color=ASD),
        dict(label="IAF-adjusted", estimate=age_adj.estimate, ci_low=age_adj.ci_low,
             ci_high=age_adj.ci_high, p=age_adj.p, color=ASD),
    ]
    forest(ax, rows, "Posterior age×group", "Beta", (-0.001, 0.0105), annotate=False)


def export_iaf_table(models: pd.DataFrame) -> None:
    order = [
        ("Global IAF group effect", "global IAF group effect"),
        ("Global IAF age × group", "global IAF group x age"),
        ("Posterior IAF age × group", "posterior IAF group x age"),
        ("Posterior exponent age × group, IAF-adjusted",
         "posterior exponent group x age adjusted for posterior IAF"),
        ("ASD exponent-deviation vs IAF-deviation Spearman rho",
         "exponent-deviation vs IAF-deviation correlation"),
    ]
    rows = []
    for label, model in order:
        r = models.loc[models["model"] == model].iloc[0]
        rows.append(
            {
                "analysis": label,
                "outcome": r.outcome,
                "term": r.term,
                "estimate": r.estimate,
                "SE": r.se,
                "CI_low": r.ci_low,
                "CI_high": r.ci_high,
                "p": r.p,
                "n": int(r.n),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / f"{TABLE_STEM}.csv", index=False)


def draw() -> None:
    backup_existing()
    df = pd.read_csv(SRC / "s5_iaf_subjects.csv")
    models = pd.read_csv(SRC / "s5_iaf_models.csv")
    export_iaf_table(models)

    fig = plt.figure(figsize=(6.5, 5.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                          width_ratios=[1.0, 1.0],
                          hspace=0.54, wspace=0.46)
    panel_a(fig.add_subplot(gs[0, 0]), df)
    panel_b(fig.add_subplot(gs[0, 1]), df)
    panel_c(fig.add_subplot(gs[1, 0]), df, models)
    panel_d(fig.add_subplot(gs[1, 1]), models)
    fig.subplots_adjust(left=0.105, right=0.965, top=0.94, bottom=0.085)

    for ext, kw in {
        "png": {"dpi": 600},
        "tiff": {"dpi": 600},
        "pdf": {},
        "svg": {},
    }.items():
        fig.savefig(OUT / f"{STEM}.{ext}", bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)
    print(OUT / f"{STEM}.png")


if __name__ == "__main__":
    draw()
