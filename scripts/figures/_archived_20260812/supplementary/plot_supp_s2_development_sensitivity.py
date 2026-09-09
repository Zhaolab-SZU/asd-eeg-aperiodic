from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]  # repo root
SRC = ROOT / "data" / "supplementary_source"
MAIN_SRC = ROOT / "data" / "figure_source"
OUT = ROOT / "outputs" / "figures" / "supplementary"
BACKUP = OUT.parent / "backup"; BACKUP.mkdir(parents=True, exist_ok=True)
BACKUP.mkdir(parents=True, exist_ok=True)

STEM = "SuppFigS2_development_sensitivity"

ASD = "#C25450"
TD = "#4A6FA5"
ASD_LIGHT = "#E9A6A2"
TD_LIGHT = "#B8C9E2"
NEUTRAL = "#4D4D4D"
NONSIG = "#9A9A9A"
REF = "#8A8A8A"
INK = "#333333"
MUTED = "#666666"
LIGHT = "#D7D7D7"
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
    ax.tick_params(labelsize=7, colors=INK)
    ax.grid(False)


def p_label(p):
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def panel_a(ax, pred):
    panel_label(ax, "A")
    colors = {"ASD": ASD, "TD": TD}
    light = {"ASD": ASD_LIGHT, "TD": TD_LIGHT}
    for group in ["ASD", "TD"]:
        lin = pred[(pred["model"] == "linear_interaction") & (pred["group"] == group)].sort_values("age_months")
        spl = pred[(pred["model"] == "spline_interaction") & (pred["group"] == group)].sort_values("age_months")
        x = lin["age_months"].to_numpy() / 12
        ax.fill_between(x, lin["ci_low"].to_numpy(), lin["ci_high"].to_numpy(),
                        color=light[group], alpha=0.30, linewidth=0)
        ax.plot(x, lin["predicted_exponent"].to_numpy(), color=colors[group],
                lw=1.5, label=f"{group} linear")
        ax.plot(spl["age_months"].to_numpy() / 12, spl["predicted_exponent"].to_numpy(),
                color=colors[group], lw=1.05, ls=(0, (3, 2)), label=f"{group} spline")
    ax.set_title("Linear and spline age trajectories", fontsize=8.8, fontweight="bold", color=INK, pad=6)
    ax.set_xlabel("Age (years)", fontsize=8, fontweight="bold")
    ax.set_ylabel("Posterior exponent", fontsize=8, fontweight="bold")
    ax.set_xlim(3.2, 11.3)
    ax.set_ylim(1.58, 2.20)
    ax.legend(loc="upper right", fontsize=5.2, handlelength=1.7, borderaxespad=0.2, labelspacing=0.25)
    style_axes(ax)


def panel_b(ax, inter):
    panel_label(ax, "B")
    keep = ["Primary", "IQ-balanced matched", "Strict-QC OLS", "Strict-QC robust (robust)"]
    data = inter[(inter["region"] == "posterior") & (inter["model"].isin(keep))].copy()
    data["model"] = pd.Categorical(data["model"], keep, ordered=True)
    data = data.sort_values("model")
    labels = ["Primary", "IQ-balanced", "Strict-QC OLS", "Strict-QC robust"]
    y = np.arange(len(data))[::-1]
    for yi, (_, row), lab in zip(y, data.iterrows(), labels):
        sig = row["p"] < 0.05
        color = ASD if sig else NONSIG
        ax.errorbar(row["estimate_group_by_age"], yi,
                    xerr=[[row["estimate_group_by_age"] - row["ci_low"]],
                          [row["ci_high"] - row["estimate_group_by_age"]]],
                    fmt="o", color=color, ecolor=color, elinewidth=1.0,
                    capsize=3, markersize=4.6, zorder=3)
        txt = f"{row['estimate_group_by_age']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        ax.text(0.0102, yi, txt, va="center", ha="left", fontsize=5.7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.3)
    ax.set_xlabel("Age × group beta", fontsize=8, fontweight="bold")
    ax.set_xlim(-0.0018, 0.0155)
    ax.set_ylim(-0.65, len(data)-0.35)
    ax.set_title("Age × group sensitivity", fontsize=8.8, fontweight="bold", color=INK, pad=6)
    style_axes(ax)


def panel_c(ax, subj):
    panel_label(ax, "C")
    asd = subj[subj["group"] == "ASD"]["posterior_td_reference_z"].dropna().to_numpy()
    bins = np.linspace(-5.5, 3.2, 16)
    ax.hist(asd, bins=bins, color=ASD, alpha=0.75, edgecolor="white", linewidth=0.7)
    ax.axvline(0, color=REF, lw=1.1)
    ax.text(0.10, ax.get_ylim()[1] * 0.92, "TD reference", ha="left", va="top",
            fontsize=5.8, color=REF)
    ax.set_title("ASD TD-reference deviation", fontsize=8.4, fontweight="bold", color=INK, pad=6)
    ax.set_xlabel("Deviation z-score", fontsize=8, fontweight="bold")
    ax.set_ylabel("Count", fontsize=8, fontweight="bold")
    style_axes(ax)


def panel_d(ax, subj):
    panel_label(ax, "D")
    asd = subj[(subj["group"] == "ASD") & subj["posterior_td_reference_z"].notna()].copy()
    x = asd["age_months"].to_numpy() / 12
    y = asd["posterior_td_reference_z"].to_numpy()
    rho, p = stats.spearmanr(x, y)
    ax.scatter(x, y, s=20, color=ASD, alpha=0.70, edgecolor="white", linewidth=0.45)
    coef = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, coef[0] * xx + coef[1], color=INK, lw=1.1)
    ax.text(0.04, 0.93, f"ρ = {rho:.2f}\n{p_label(p)}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.2, color=INK)
    ax.set_title("Deviation vs age in ASD", fontsize=8.4, fontweight="bold", color=INK, pad=6)
    ax.set_xlabel("Age (years)", fontsize=8, fontweight="bold")
    ax.set_ylabel("Deviation z-score", fontsize=8, fontweight="bold")
    ax.set_xlim(3.4, 11.2)
    ax.set_ylim(-5.6, 3.2)
    style_axes(ax)


def panel_e(ax, subj):
    panel_label(ax, "E")
    asd = subj[(subj["group"] == "ASD") & subj["posterior_td_reference_z"].notna()].copy()
    groups = [("≤72 mo", asd[asd["age_months"] <= 72]["posterior_td_reference_z"].to_numpy()),
              (">72 mo", asd[asd["age_months"] > 72]["posterior_td_reference_z"].to_numpy())]
    rng = np.random.default_rng(20260713)
    for i, (_, vals) in enumerate(groups):
        parts = ax.violinplot(vals, positions=[i], widths=0.64, showmeans=False, showmedians=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(ASD if i == 1 else ASD_LIGHT)
            body.set_edgecolor(ASD if i == 1 else ASD_LIGHT)
            body.set_alpha(0.75)
        jitter = rng.normal(0, 0.045, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, color=ASD, alpha=0.68,
                   edgecolor="white", linewidth=0.4, zorder=3)
        mean = np.mean(vals)
        sem = stats.sem(vals)
        ci = sem * stats.t.ppf(0.975, len(vals) - 1)
        ax.errorbar(i, mean, yerr=ci, fmt="o", color=INK, ecolor=INK,
                    capsize=4, markersize=4.5, lw=1.0, zorder=4)
    _, p = stats.ttest_ind(groups[0][1], groups[1][1], equal_var=False, nan_policy="omit")
    y = max(np.nanmax(groups[0][1]), np.nanmax(groups[1][1])) + 0.28
    ax.plot([0, 0, 1, 1], [y, y + 0.16, y + 0.16, y], color=INK, lw=0.8)
    ax.text(0.5, y + 0.14, "*" if p < 0.05 else "n.s.", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=ASD if p < 0.05 else MUTED)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([groups[0][0], groups[1][0]], fontsize=6.5)
    ax.set_title("ASD age subgroup", fontsize=8.4, fontweight="bold", color=INK, pad=13)
    ax.set_ylabel("Deviation z-score", fontsize=8, fontweight="bold")
    ax.set_ylim(-5.6, 3.2)
    style_axes(ax)


def draw() -> None:
    backup_existing()
    pred = pd.read_csv(SRC / "s4_development_predictions_with_ci.csv")
    inter = pd.read_csv(SRC / "s4_development_interactions.csv")
    subj = pd.read_csv(MAIN_SRC / "fig3_development_subjects.csv")

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], width_ratios=[1.2, 1.0, 1.0],
                          hspace=0.58, wspace=0.45)
    panel_a(fig.add_subplot(gs[0, :2]), pred)
    panel_b(fig.add_subplot(gs[0, 2]), inter)
    panel_c(fig.add_subplot(gs[1, 0]), subj)
    panel_d(fig.add_subplot(gs[1, 1]), subj)
    panel_e(fig.add_subplot(gs[1, 2]), subj)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.08)

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
