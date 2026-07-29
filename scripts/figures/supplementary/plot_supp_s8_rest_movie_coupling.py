from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]  # repo root
SRC = ROOT / "data" / "supplementary_source"
OUT = ROOT / "outputs" / "figures" / "supplementary"
BACKUP = OUT.parent / "backup"; BACKUP.mkdir(parents=True, exist_ok=True)
BACKUP.mkdir(parents=True, exist_ok=True)

STEM = "SuppFigS8_rest_movie_coupling"

ASD = "#C25450"
TD = "#4A6FA5"
INK = "#333333"
MUTED = "#666666"
GREY = "#9A9A9A"
LIGHT_GREY = "#DADADA"

SEGMENTS = ["mental", "pain", "neutral"]
SEG_LABELS = {
    "mental": "Mentalizing",
    "pain": "Pain-related",
    "neutral": "Neutral",
}
COHORT_LABELS = {
    "overlapping": "Overlap cohort",
    "dual_paradigm_matched": "Dual-paradigm matched",
}

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


def style_axes(ax):
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(labelsize=7, colors=INK, width=0.9, length=3.2)
    ax.grid(False)


def panel_label(ax, label, x=-0.13, y=1.05):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )


def p_to_label(p: float | None) -> str:
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def draw_scatter(ax, subjects: pd.DataFrame, models: pd.DataFrame):
    panel_label(ax, "A")
    for group, color in [("ASD", ASD), ("TD", TD)]:
        sub = subjects[subjects["group"] == group].dropna(
            subset=["resting_posterior_exponent", "neutral_aperiodic_isc"]
        )
        x = sub["resting_posterior_exponent"].to_numpy()
        y = sub["neutral_aperiodic_isc"].to_numpy()
        ax.scatter(x, y, s=16, color=color, alpha=0.55, edgecolor="white", linewidth=0.35)
        coef = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        yy = coef[0] * xx + coef[1]
        ax.plot(xx, yy, color=color, lw=1.15)
        ax.plot([], [], color=color, lw=1.15, marker="o", markersize=3.5,
                label=group)
    row = models[models["cohort"] == "overlapping"].iloc[0]
    ax.text(
        0.07,
        0.88,
        f"β = {row['interaction_beta']:.2f}\nq = {row['fdr_p']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
        color=INK,
        linespacing=1.18,
    )
    ax.set_title("Neutral rest–movie coupling", fontsize=8.8, fontweight="bold", color=INK, pad=6)
    ax.set_xlabel("Resting posterior exponent", fontsize=8, fontweight="bold")
    ax.set_ylabel("Neutral Aperiodic-ISC", fontsize=8, fontweight="bold")
    ax.set_xlim(1.20, 2.48)
    ax.set_ylim(-0.32, 0.52)
    ax.legend(loc="upper right", frameon=False, fontsize=6.5, handlelength=1.4)
    style_axes(ax)


def draw_bootstrap_forest(ax, boot: pd.DataFrame):
    panel_label(ax, "C", x=-0.10, y=1.04)
    # Order by video segment, then cohort, so the same movie condition is adjacent.
    rows = []
    for segment in SEGMENTS:
        for cohort in ["overlapping", "dual_paradigm_matched"]:
            r = boot[(boot["cohort"] == cohort) & (boot["segment"] == segment)].iloc[0]
            rows.append(
                {
                    "label": f"{'Overlap' if cohort == 'overlapping' else 'Matched'}: {SEG_LABELS[segment]}",
                    "cohort": cohort,
                    "est": r["median_beta"],
                    "lo": r["ci_low"],
                    "hi": r["ci_high"],
                    "p": r["bootstrap_p"],
                }
            )
    y = np.arange(len(rows))[::-1]
    for yi, row in zip(y, rows):
        # Use black/grey only: red would incorrectly imply significance.
        color = INK if row["cohort"] == "overlapping" else GREY
        ax.errorbar(
            row["est"],
            yi,
            xerr=[[row["est"] - row["lo"]], [row["hi"] - row["est"]]],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=4.2,
            elinewidth=1.0,
            capsize=3,
        )
        lab = p_to_label(float(row["p"]))
        ax.text(
            1.18,
            yi,
            lab,
            ha="right",
            va="center",
            fontsize=7.2 if lab == "n.s." else 10.2,
            color=MUTED if lab == "n.s." else ASD,
            fontweight="normal" if lab == "n.s." else "bold",
        )
    # Separators between segment pairs (Mentalizing / Pain / Neutral).
    ax.axhline(3.5, color="#ECECEC", lw=0.8)
    ax.axhline(1.5, color="#ECECEC", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=6.3)
    ax.set_xlim(-0.95, 1.22)
    ax.set_xlabel("Bootstrap interaction β", fontsize=8, fontweight="bold")
    ax.set_title("Bootstrap sensitivity across segments", fontsize=8.8, fontweight="bold", color=INK, pad=6)
    style_axes(ax)


def draw_model_forest(ax, models: pd.DataFrame):
    panel_label(ax, "B", x=-0.12, y=1.04)
    rows = []
    for cohort in ["overlapping", "dual_paradigm_matched"]:
        r = models[models["cohort"] == cohort].iloc[0]
        se = float(r["se"])
        est = float(r["interaction_beta"])
        rows.append(
            {
                "label": COHORT_LABELS[cohort],
                "est": est,
                "lo": est - 1.96 * se,
                "hi": est + 1.96 * se,
                "p": r["fdr_p"] if pd.notna(r["fdr_p"]) else r["raw_p"],
                "cohort": cohort,
            }
        )
    y = np.arange(len(rows))[::-1]
    for yi, row in zip(y, rows):
        color = ASD if row["cohort"] == "overlapping" else GREY
        ax.errorbar(
            row["est"],
            yi,
            xerr=[[row["est"] - row["lo"]], [row["hi"] - row["est"]]],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=4.5,
            elinewidth=1.0,
            capsize=3,
        )
        lab = p_to_label(float(row["p"]))
        ax.text(
            0.68,
            yi,
            lab,
            ha="right",
            va="center",
            fontsize=7.2 if lab == "n.s." else 10.2,
            color=MUTED if lab == "n.s." else ASD,
            fontweight="normal" if lab == "n.s." else "bold",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=6.7)
    ax.set_xlim(-0.45, 0.72)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xlabel("Neutral interaction β", fontsize=8, fontweight="bold")
    ax.set_title("Primary neutral-model estimates", fontsize=8.8, fontweight="bold", color=INK, pad=6)
    style_axes(ax)


def draw() -> None:
    backup_existing()
    subjects = pd.read_csv(SRC / "s9_coupling_subjects.csv")
    models = pd.read_csv(SRC / "s9_coupling_models.csv")
    boot = pd.read_csv(SRC / "s9_coupling_bootstrap_summary.csv")

    fig = plt.figure(figsize=(7.45, 5.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.0, 1.0],
                          hspace=0.58, wspace=0.40)
    top_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    draw_scatter(top_axes[0], subjects, models)
    draw_model_forest(top_axes[1], models)
    draw_bootstrap_forest(fig.add_subplot(gs[1, :]), boot)
    fig.subplots_adjust(left=0.105, right=0.990, top=0.91, bottom=0.115)
    # The lower forest panel needs a larger left margin for row labels; shift the
    # upper panels back left so the top row does not visually drift right.
    for ax in top_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0 - 0.045, pos.y0, pos.width + 0.015, pos.height])

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
