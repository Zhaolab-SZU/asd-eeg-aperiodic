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
OUT = ROOT / "outputs" / "figures" / "supplementary"
BACKUP = OUT.parent / "backup"; BACKUP.mkdir(parents=True, exist_ok=True)
BACKUP.mkdir(parents=True, exist_ok=True)

STEM = "SuppFigS6_envelope_adjusted_isc"

ASD = "#C25450"
TD = "#4A6FA5"
INK = "#333333"
MUTED = "#666666"
GREY = "#9A9A9A"
LIGHT_GREY = "#DADADA"
RAW = "#9A9A9A"
ADJ = ASD

SEGMENTS = ["mentalizing", "pain", "neutral"]
SEG_LABELS = {
    "mentalizing": "Mentalizing",
    "mental": "Mentalizing",
    "pain": "Pain-related",
    "neutral": "Neutral",
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


def panel_label(ax, label, x=-0.12, y=1.06):
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


def pivot_subjects(subj: pd.DataFrame) -> pd.DataFrame:
    keep = subj[subj["metric"].isin(["aperiodic_isc", "envelope_isc"])].copy()
    wide = keep.pivot_table(
        index=["subject_id", "group", "segment"],
        columns="metric",
        values="isc_value",
    ).reset_index()
    return wide.dropna(subset=["aperiodic_isc", "envelope_isc"])


def draw_scatter(
    ax,
    wide: pd.DataFrame,
    adj: pd.DataFrame,
    segment: str,
    label: str,
    *,
    show_ylabel: bool = True,
):
    panel_label(ax, label)
    sub = wide[wide["segment"] == segment]
    for group, color in [("ASD", ASD), ("TD", TD)]:
        g = sub[sub["group"] == group]
        ax.scatter(
            g["envelope_isc"],
            g["aperiodic_isc"],
            s=12,
            color=color,
            alpha=0.45,
            edgecolor="white",
            linewidth=0.25,
        )
    x = sub["envelope_isc"].to_numpy()
    y = sub["aperiodic_isc"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    ax.plot(xx, slope * xx + intercept, color=INK, lw=1.1)
    row = adj[adj["event_type"].isin([segment, "mental" if segment == "mentalizing" else segment])].iloc[0]
    ax.text(
        0.05,
        0.94,
        f"r = {row['pearson_r']:.2f}\nshared = {row['shared_variance_pct']:.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color=INK,
        linespacing=1.15,
    )
    # Direct, per-panel group key avoids a global legend colliding with panel titles.
    legend_y = 0.94
    for i, (group, color) in enumerate([("ASD", ASD), ("TD", TD)]):
        y_pos = legend_y - i * 0.085
        ax.scatter(
            0.74,
            y_pos,
            s=22,
            color=color,
            alpha=0.92,
            edgecolor="white",
            linewidth=0.35,
            transform=ax.transAxes,
            clip_on=False,
            zorder=5,
        )
        ax.text(
            0.79,
            y_pos,
            group,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.5,
            color=INK,
        )
    ax.set_title(SEG_LABELS[segment], fontsize=8.5, fontweight="bold", color=INK, pad=5)
    ax.set_xlim(-0.62, 1.18)
    ax.set_ylim(-0.36, 0.46)
    ax.set_xlabel("Envelope ISC", fontsize=7.5, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Aperiodic-ISC", fontsize=7.5, fontweight="bold")
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
    style_axes(ax)


def draw_adjusted_forest(ax, adj: pd.DataFrame):
    panel_label(ax, "D", x=-0.08, y=1.04)
    rows = []
    for seg in ["mental", "pain", "neutral"]:
        row = adj[adj["event_type"] == seg].iloc[0]
        raw = row["raw_mean_z_td"] - row["raw_mean_z_asd"]
        raw_se = abs(raw / row["raw_t"]) if row["raw_t"] != 0 else np.nan
        raw_ci = (raw - 1.96 * raw_se, raw + 1.96 * raw_se)
        adjusted = row["adj_mean_z_td_at_mean_env"] - row["adj_mean_z_asd_at_mean_env"]
        adj_se = row["envelope_adjusted_group_se"]
        adj_ci = (adjusted - 1.96 * adj_se, adjusted + 1.96 * adj_se)
        rows.append((SEG_LABELS[seg], raw, raw_ci[0], raw_ci[1], adjusted, adj_ci[0], adj_ci[1], row))
    y = np.arange(len(rows))[::-1]
    for yi, (label, raw, raw_lo, raw_hi, adjusted, adj_lo, adj_hi, row) in zip(y, rows):
        ax.plot([raw, adjusted], [yi + 0.09, yi - 0.09], color="#CFCFCF", lw=0.9, zorder=1)
        ax.errorbar(
            raw,
            yi + 0.09,
            xerr=[[raw - raw_lo], [raw_hi - raw]],
            fmt="o",
            color=RAW,
            ecolor=RAW,
            markersize=4.0,
            elinewidth=1.0,
            capsize=3,
            label="Raw" if yi == y[0] else None,
        )
        ax.errorbar(
            adjusted,
            yi - 0.09,
            xerr=[[adjusted - adj_lo], [adj_hi - adjusted]],
            fmt="o",
            color=ADJ,
            ecolor=ADJ,
            markersize=4.4,
            elinewidth=1.0,
            capsize=3,
            label="Envelope-adjusted" if yi == y[0] else None,
        )
        q = row["envelope_adjusted_group_fdr_p"]
        stars = "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "n.s."
        ax.text(
            0.178,
            yi - 0.09,
            stars,
            ha="right",
            va="center",
            fontsize=10.6 if stars != "n.s." else 7.2,
            color=ASD if stars != "n.s." else MUTED,
            fontweight="bold" if stars != "n.s." else "normal",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.8)
    ax.set_xlim(-0.02, 0.185)
    ax.set_xlabel("TD − ASD Aperiodic-ISC", fontsize=8, fontweight="bold")
    ax.set_title("Raw vs envelope-adjusted group effect", fontsize=8.6, fontweight="bold", color=INK, pad=6)
    raw_handle = plt.Line2D([0], [0], marker="o", color=RAW, markerfacecolor=RAW,
                            markersize=4.0, lw=1.0, label="Raw")
    adj_handle = plt.Line2D([0], [0], marker="o", color=ADJ, markerfacecolor=ADJ,
                            markersize=4.2, lw=1.0, label="Envelope-adjusted")
    ax.legend(
        handles=[raw_handle, adj_handle],
        loc="upper center",
        bbox_to_anchor=(0.69, 0.93),
        frameon=False,
        fontsize=6.5,
        handlelength=1.4,
        borderaxespad=0.0,
    )
    style_axes(ax)


def draw_retained(ax, adj: pd.DataFrame):
    panel_label(ax, "E", x=-0.13, y=1.04)
    data = adj.set_index("event_type").loc[["mental", "pain", "neutral"]].reset_index()
    labels = [SEG_LABELS[s] for s in data["event_type"]]
    vals = data["effect_retained_pct"].to_numpy()
    colors = [GREY if v < 50 else ASD for v in vals]
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, width=0.56, edgecolor="none", alpha=0.92)
    ax.axhline(100, color="#CFCFCF", lw=0.8, ls=(0, (3, 2)))
    for xi, v in zip(x, vals):
        ax.text(xi, v + 5, f"{v:.0f}%", ha="center", va="bottom",
                fontsize=6.8, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=6.8)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Effect retained (%)", fontsize=8, fontweight="bold")
    ax.set_title("Aperiodic-specific effect retained", fontsize=8.6, fontweight="bold", color=INK, pad=6)
    style_axes(ax)


def draw() -> None:
    backup_existing()
    subj = pd.read_csv(SRC / "s7_synchrony_subject_level.csv")
    adj = pd.read_csv(SRC / "s7_envelope_adjusted.csv")
    wide = pivot_subjects(subj)

    fig = plt.figure(figsize=(7.45, 5.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.02], hspace=0.55, wspace=0.40)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4], sharey=ax_a)
    ax_c = fig.add_subplot(gs[0, 4:6], sharey=ax_a)
    scatter_axes = [ax_a, ax_b, ax_c]
    for i, (ax, segment, lab) in enumerate(
        zip(scatter_axes, SEGMENTS, ["A", "B", "C"])
    ):
        draw_scatter(ax, wide, adj, segment, lab, show_ylabel=(i == 0))
        # Keep left spine/ticks on B/C for panel framing, but no shared-y numeric labels.
        if i > 0:
            ax.tick_params(axis="y", labelleft=False)
            ax.set_ylabel("")
    draw_adjusted_forest(fig.add_subplot(gs[1, 0:4]), adj)
    draw_retained(fig.add_subplot(gs[1, 4:6]), adj)
    fig.subplots_adjust(left=0.070, right=0.990, top=0.935, bottom=0.135)

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
