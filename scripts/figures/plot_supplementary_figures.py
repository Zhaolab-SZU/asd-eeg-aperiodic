# Ideal release name: plot_supplementary_figures.py
# Original path: scripts/plot_supplementary_figures.py
# Note: Plot Supp Figs S1/S4–S7
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""Plot Supplementary Figures S1, S4, S5, S6, S7 (20260713 design)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPP_DIR = PROJECT_ROOT / "figure_source_data" / "supplementary"
OUT_DIR = PROJECT_ROOT / "working" / "final_figures" / "supplementary"

COL_TD = "#FDB933"
COL_ASD = "#D23538"
COL_EDGE = "#4D4D4D"
COL_GRAY = "#666666"
COL_POST = "#333333"
COL_ENV = "#888888"
COL_ALPHA = "#BBBBBB"

SEGMENT_ORDER = ["mentalizing", "pain", "neutral"]
SEGMENT_LABELS = {"mentalizing": "Mentalizing", "pain": "Pain-related", "neutral": "Neutral"}
EVENT_MAP = {"mental": "mentalizing", "pain": "pain", "neutral": "neutral"}
POSTERIOR_CORE = ["E33", "E36", "E37", "E38"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, letter: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")


def save_fig(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _raincloud(ax: plt.Axes, data: pd.DataFrame, value_col: str, group_col: str = "group") -> None:
    colors = {"TD": COL_TD, "ASD": COL_ASD}
    rng = np.random.default_rng(42)
    for i, g in enumerate(["TD", "ASD"]):
        vals = data.loc[data[group_col] == g, value_col].dropna().values
        if len(vals) == 0:
            continue
        color = colors[g]
        vp = ax.violinplot([vals], positions=[i], widths=0.55, showextrema=False)
        for body in vp["bodies"]:
            verts = body.get_paths()[0].vertices
            m = np.mean(verts[:, 0])
            verts[:, 0] = np.clip(verts[:, 0], -np.inf, m)
            body.set_facecolor(color)
            body.set_alpha(0.35)
            body.set_edgecolor("none")
        med = np.median(vals)
        q1, q3 = np.percentile(vals, [25, 75])
        ax.plot([i - 0.08, i + 0.08], [med, med], color="black", lw=1.2, zorder=4)
        ax.plot([i, i], [q1, q3], color="0.35", lw=1.0, zorder=3)
        jitter = rng.uniform(0.12, 0.32, size=len(vals))
        ax.scatter(i + jitter, vals, s=12, alpha=0.6, color=color, edgecolors="white", linewidths=0.2, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["TD", "ASD"])


def _format_p(p: float) -> str:
    if pd.isna(p):
        return "n.s."
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def plot_s1() -> None:
    surv = pd.read_csv(SUPP_DIR / "s2_loocv_electrode_survival.csv")
    crit = pd.read_csv(SUPP_DIR / "s2_loocv_criteria_summary.csv")
    elec = surv.set_index("electrode")

    fig = plt.figure(figsize=(7.2, 3.0))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.1, 1.1, 1.0], wspace=0.45)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "A")
    y = np.arange(4)
    ax_a.barh(y, [elec.loc[ch, "survival_percent"] for ch in POSTERIOR_CORE], color=COL_POST, height=0.55)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(POSTERIOR_CORE)
    ax_a.set_xlabel("FDR survival (%)")
    ax_a.set_xlim(85, 101)
    ax_a.set_title("FDR-significant survival", fontsize=8.5)
    for i, ch in enumerate(POSTERIOR_CORE):
        ax_a.text(elec.loc[ch, "survival_percent"] + 0.3, i, f"{elec.loc[ch, 'survival_percent']:.1f}%", va="center", fontsize=7)

    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "B")
    ax_b.barh(y, [elec.loc[ch, "uncorrected_survival_percent"] for ch in POSTERIOR_CORE], color="#AAAAAA", height=0.55)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(POSTERIOR_CORE)
    ax_b.set_xlabel("Uncorrected p < .05 survival (%)")
    ax_b.set_xlim(85, 101)
    ax_b.set_title("Uncorrected survival", fontsize=8.5)

    ax_c = fig.add_subplot(gs[0, 2])
    panel_label(ax_c, "C")
    ax_c.axis("off")
    lines = [
        "LOOCV summary (138 folds)",
        "",
    ]
    for _, row in crit.iterrows():
        if row["criterion"] in {"all four electrodes", "at least three", "at least one"}:
            lines.append(f"{row['criterion']}: {int(row['n_folds_satisfied'])}/{int(row['total_folds'])} ({row['survival_percent']:.1f}%)")
    ax_c.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=8, family="monospace", transform=ax_c.transAxes)

    save_fig(fig, "SuppFigS1_loocv_fdr_survival")


def plot_s4() -> None:
    subj = pd.read_csv(SUPP_DIR / "s6_isc_subject_level.csv")
    tc = pd.read_csv(SUPP_DIR / "s6_isc_timecourse.csv")

    fig = plt.figure(figsize=(7.5, 6.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1.1], hspace=0.48, wspace=0.38)

    for idx, (isc_def, title) in enumerate([("td_template", "TD-template"), ("within_group_loo", "Within-group LOO")]):
        ax = fig.add_subplot(gs[0, idx])
        panel_label(ax, "A" if idx == 0 else "B")
        sub = subj[subj["isc_definition"] == isc_def].copy()
        sub["segment"] = pd.Categorical(sub["segment"], SEGMENT_ORDER, ordered=True)
        positions = []
        xticks = []
        xlabels = []
        pos = 0
        rng = np.random.default_rng(42)
        for seg in SEGMENT_ORDER:
            for gi, g in enumerate(["TD", "ASD"]):
                vals = sub.loc[(sub["segment"] == seg) & (sub["group"] == g), "isc_value"].values
                color = COL_TD if g == "TD" else COL_ASD
                if len(vals):
                    vp = ax.violinplot([vals], positions=[pos], widths=0.5, showextrema=False)
                    for body in vp["bodies"]:
                        body.set_facecolor(color)
                        body.set_alpha(0.35)
                        body.set_edgecolor("none")
                    jitter = rng.uniform(-0.12, 0.12, len(vals))
                    ax.scatter(pos + jitter, vals, s=8, alpha=0.55, color=color, edgecolors="none", zorder=3)
                positions.append(pos)
                pos += 1
            xticks.append(pos - 1)
            xlabels.append(SEGMENT_LABELS[seg])
            pos += 0.6
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=7.5)
        ax.set_ylabel("Aperiodic-ISC (Fisher z)")
        ax.set_title(title, fontsize=8.5)
        ax.axhline(0, color=COL_GRAY, lw=0.6, zorder=0)

    ax_leg = fig.add_subplot(gs[0, 2])
    ax_leg.axis("off")
    panel_label(ax_leg, " ", x=-0.05)
    from matplotlib.lines import Line2D

    ax_leg.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_TD, markersize=8, label="TD"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_ASD, markersize=8, label="ASD"),
        ],
        loc="center",
        frameon=False,
        title="Group",
    )

    seg_tc = ["Mentalizing", "Pain-related", "Neutral"]
    letters = ["C", "D", "E"]
    for i, seg in enumerate(seg_tc):
        ax = fig.add_subplot(gs[1, i])
        panel_label(ax, letters[i])
        seg_data = tc[tc["segment_label"] == seg]
        for g, color in [("TD", COL_TD), ("ASD", COL_ASD)]:
            gsub = seg_data[seg_data["group"] == g].sort_values("time_seconds")
            ax.plot(gsub["time_seconds"], gsub["isc_mean"], color=color, lw=1.8, label=g)
            ax.fill_between(gsub["time_seconds"], gsub["ci_low"], gsub["ci_high"], color=color, alpha=0.15, linewidth=0)
        ax.axhline(0, color=COL_GRAY, lw=0.6)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Within-group ISC (z)")
        ax.set_title(seg, fontsize=8.5)
        if i == 2:
            ax.legend(frameon=False, loc="upper right", fontsize=7)

    save_fig(fig, "SuppFigS4_naturalistic_isc")


def plot_s5() -> None:
    ge = pd.read_csv(PROJECT_ROOT / "figure_source_data" / "s7_synchrony_group_effects.csv")
    metric_map = {
        "aperiodic_isc": "Aperiodic",
        "envelope_isc": "Envelope",
        "alpha_plv_isc": "Alpha PLV",
    }

    fig = plt.figure(figsize=(7.5, 5.0))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1.0], hspace=0.55, wspace=0.32)

    for col, (seg, letter) in enumerate(zip(SEGMENT_ORDER, ["A", "B", "C"])):
        ax = fig.add_subplot(gs[0, col])
        panel_label(ax, letter, x=-0.2)
        x = np.arange(3)
        width = 0.35
        for mi, metric in enumerate(metric_map):
            row = ge[(ge["segment"] == seg) & (ge["metric"] == metric)].iloc[0]
            ax.bar(
                mi - width / 2,
                row["mean_td"],
                width,
                color=COL_TD,
                edgecolor=COL_EDGE,
                linewidth=0.4,
                label="TD" if col == 0 and mi == 0 else None,
            )
            ax.bar(
                mi + width / 2,
                row["mean_asd"],
                width,
                color=COL_ASD,
                edgecolor=COL_EDGE,
                linewidth=0.4,
                label="ASD" if col == 0 and mi == 0 else None,
            )
            ptxt = _format_p(row["p"])
            ymax = max(row["mean_td"], row["mean_asd"], 0)
            ax.text(mi, ymax + 0.03, ptxt, ha="center", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels([metric_map[m] for m in metric_map], fontsize=7, rotation=20, ha="right")
        ax.set_title(SEGMENT_LABELS[seg], fontsize=8.5)
        if col == 0:
            ax.set_ylabel("Within-group ISC (Fisher z)")
        ax.axhline(0, color=COL_GRAY, lw=0.6)

    ax_d = fig.add_subplot(gs[1, :])
    panel_label(ax_d, "D", x=-0.06, y=1.02)
    rows = []
    for seg in SEGMENT_ORDER:
        for metric in metric_map:
            row = ge[(ge["segment"] == seg) & (ge["metric"] == metric)].iloc[0]
            rows.append(
                {
                    "label": f"{SEGMENT_LABELS[seg][:4]} · {metric_map[metric]}",
                    "beta": row["group_effect_td_minus_asd"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "p": row["p"],
                }
            )
    df = pd.DataFrame(rows)
    y = np.arange(len(df))
    colors = [COL_ASD if (p < 0.05 and not pd.isna(p)) else COL_GRAY for p in df["p"]]
    ax_d.axvline(0, color=COL_GRAY, lw=0.8)
    for i, row in df.iterrows():
        ax_d.plot([row["ci_low"], row["ci_high"]], [i, i], color=colors[i], lw=1.2)
        ax_d.plot(row["beta"], i, "o", color=colors[i], ms=5)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(df["label"], fontsize=7)
    ax_d.set_xlabel("Group effect β (TD − ASD)")
    ax_d.invert_yaxis()

    fig.axes[0].legend(frameon=False, loc="upper left", fontsize=7)
    save_fig(fig, "SuppFigS5_classic_synchrony_controls")


def plot_s6() -> None:
    env = pd.read_csv(SUPP_DIR / "s7_envelope_adjusted.csv")
    env["segment"] = env["event_type"].map(EVENT_MAP)

    fig = plt.figure(figsize=(7.2, 3.2))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "A")
    event_order = ["mental", "pain", "neutral"]
    markers = ["o", "s", "^"]
    for ev, marker in zip(event_order, markers):
        row = env[env["event_type"] == ev].iloc[0]
        ax_a.scatter(
            row["pearson_r"],
            row["shared_variance_pct"],
            s=60,
            marker=marker,
            color=COL_POST,
            label=SEGMENT_LABELS[EVENT_MAP[ev]],
        )
    ax_a.set_xlabel("Pearson r (Aperiodic vs Envelope ISC)")
    ax_a.set_ylabel("Shared variance (%)")
    ax_a.legend(frameon=False, fontsize=7, loc="upper left")

    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "B")
    y = np.arange(3)
    raw = []
    adj = []
    labels = []
    for ev in ["mental", "pain", "neutral"]:
        row = env[env["event_type"] == ev].iloc[0]
        raw.append(-row["raw_cohen_d_asd_minus_td"])
        adj.append(row["envelope_adjusted_group_beta_z"])
        labels.append(SEGMENT_LABELS[EVENT_MAP[ev]])
    ax_b.axvline(0, color=COL_GRAY, lw=0.8)
    for i in range(3):
        ax_b.plot(raw[i], i, "o", color="#AAAAAA", ms=7, label="Raw" if i == 0 else None)
        ax_b.plot(adj[i], i, "o", color=COL_POST, ms=7, label="Envelope-adjusted" if i == 0 else None)
        ax_b.plot([raw[i], adj[i]], [i, i], "-", color=COL_GRAY, lw=0.8)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labels, fontsize=7.5)
    ax_b.set_xlabel("Group effect (TD − ASD)")
    ax_b.legend(frameon=False, fontsize=7, loc="lower right")

    ax_c = fig.add_subplot(gs[0, 2])
    panel_label(ax_c, "C")
    retain = [env[env["event_type"] == ev]["effect_retained_pct"].iloc[0] for ev in ["mental", "pain", "neutral"]]
    ax_c.bar(np.arange(3), retain, color=COL_POST, width=0.6)
    ax_c.set_xticks(np.arange(3))
    ax_c.set_xticklabels(["Mental.", "Pain", "Neutral"], fontsize=7.5)
    ax_c.set_ylabel("Effect retained (%)")
    ax_c.set_ylim(0, 105)
    for i, v in enumerate(retain):
        ax_c.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=7)

    save_fig(fig, "SuppFigS6_envelope_adjusted")


def plot_s7() -> None:
    subj = pd.read_csv(SUPP_DIR / "s8_hbn_movie_subjects.csv")
    summ = pd.read_csv(SUPP_DIR / "s8_hbn_movie_summary.csv")

    fig = plt.figure(figsize=(7.2, 3.6))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.95], wspace=0.38)

    analyses = [
        ("sliding_window", "Sliding window", "A"),
        ("nonoverlapping_2s_epoch", "2-s epochs", "B"),
    ]
    for i, (analysis, title, letter) in enumerate(analyses):
        ax = fig.add_subplot(gs[0, i])
        panel_label(ax, letter)
        sub = subj[subj["analysis"] == analysis]
        _raincloud(ax, sub, "isc_z")
        ax.set_title(title, fontsize=8.5)
        ax.set_ylabel("Posterior Aperiodic-ISC (Fisher z)" if i == 0 else "")
        td_mean = summ[(summ["analysis"] == analysis) & (summ["group"] == "TD")]["mean"].iloc[0]
        asd_mean = summ[(summ["analysis"] == analysis) & (summ["group"] == "ASD")]["mean"].iloc[0]
        p_val = summ[(summ["analysis"] == analysis) & (summ["group"] == "TD")]["p"].iloc[0]
        ax.text(
            0.98,
            0.98,
            f"Δz = {asd_mean - td_mean:.3f}\n{_format_p(p_val)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CCCCCC", alpha=0.95),
        )

    ax_c = fig.add_subplot(gs[0, 2])
    panel_label(ax_c, "C")
    x = np.arange(2)
    width = 0.35
    for gi, (g, color) in enumerate([("TD", COL_TD), ("ASD", COL_ASD)]):
        means = []
        ses = []
        for analysis, _, _ in analyses:
            gsub = subj[subj["analysis"] == analysis]
            vals = gsub.loc[gsub["group"] == g, "isc_z"].dropna().values
            means.append(np.mean(vals))
            ses.append(stats.sem(vals) if len(vals) > 1 else 0.0)
        ax_c.bar(x + (gi - 0.5) * width, means, width, yerr=ses, color=color, label=g, capsize=3, edgecolor=COL_EDGE, linewidth=0.4)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(["Sliding\nwindow", "2-s\nepochs"], fontsize=7.5)
    ax_c.set_ylabel("Group mean ISC (z)")
    ax_c.legend(frameon=False, fontsize=7, loc="upper right")

    save_fig(fig, "SuppFigS7_hbn_external_convergence")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot supplementary figures S1/S4/S5/S6/S7.")
    parser.add_argument("--fig", nargs="+", default=["S1", "S4", "S5", "S6", "S7"], help="Figures to render")
    args = parser.parse_args()

    apply_style()
    mapping = {"S1": plot_s1, "S4": plot_s4, "S5": plot_s5, "S6": plot_s6, "S7": plot_s7}
    for key in args.fig:
        key = key.upper()
        if key not in mapping:
            raise SystemExit(f"Unknown figure: {key}")
        print(f"Plotting {key}...")
        mapping[key]()
        print(f"  -> {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
