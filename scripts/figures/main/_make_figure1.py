"""Figure 1 — spectral parameterization + group distributions.

Box+strip panels with firebrick/navy palette. Output → ``outputs/figures/main/``.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd

from _grammar_style import (
    ASD, DATA, LABEL, LEGEND, NOTE, PANEL_Y, TD, TITLE_PAD, add_sig,
    box_strip, clean, group_legend_handles, panel, save,
)

NEUTRAL = "#222222"
PSD = "#3A3F44"
APER = "#1F1F1F"
PERIODIC = "#B8B8B8"
LEADER = "#888888"


def y_at(x, xs, ys):
    return float(np.interp(x, xs, ys))


def tick_label(ax, x, y_line, text, side="below", tick=0.07):
    if side == "below":
        ax.plot([x, x], [y_line - 0.03, y_line - 0.03 - tick], color=LEADER, lw=0.65, zorder=6)
        ax.text(
            x, y_line - 0.03 - tick - 0.04, text, ha="center", va="top",
            fontsize=LABEL, color="#1A1A1A", fontweight="bold", zorder=6,
        )
    else:
        ax.plot([x, x], [y_line + 0.03, y_line + 0.03 + tick], color=LEADER, lw=0.65, zorder=6)
        ax.text(
            x, y_line + 0.03 + tick + 0.04, text, ha="center", va="bottom",
            fontsize=LABEL, color="#1A1A1A", fontweight="bold", zorder=6,
        )


def draw_specparam(ax):
    f = np.logspace(np.log10(1.0), np.log10(40.0), 420)
    b = 2.52
    chi = 1.05
    aperiodic = b - chi * np.log10(f)
    alpha = 0.58 * np.exp(-0.5 * ((f - 9.2) / 1.55) ** 2)
    beta = 0.10 * np.exp(-0.5 * ((f - 20.0) / 2.4) ** 2)
    ripple = 0.016 * np.sin(np.log10(f) * 14.0)
    psd = aperiodic + alpha + beta + ripple

    mask = (f >= 5.8) & (f <= 12.6)
    hatch_lw = mpl.rcParams["hatch.linewidth"]
    mpl.rcParams["hatch.linewidth"] = 0.60
    periodic = ax.fill_between(
        f[mask], aperiodic[mask], psd[mask],
        facecolor="white",
        edgecolor="#1A1A1A",
        hatch="///",
        linewidth=0.35,
        zorder=1,
    )
    periodic.set_facecolor("white")
    periodic.set_edgecolor("#1A1A1A")
    mpl.rcParams["hatch.linewidth"] = hatch_lw
    ax.plot(f, psd, color=PSD, lw=1.05, zorder=3)
    ax.plot(f, aperiodic, color=APER, lw=1.15, ls=(0, (3.2, 1.6)), zorder=4)
    ax.scatter([1.0], [b], s=14, color=APER, zorder=5)

    ax.set_xscale("log")
    ax.set_xlim(0.9, 42)
    ax.set_ylim(0.52, 3.22)
    ax.set_xticks([1, 10, 40])
    ax.set_xticklabels(["1", "10", "40"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([1.0, 1.5, 2.0, 2.5])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"Log$_{10}$ PSD")
    ax.set_title("Spectral parameterization", pad=TITLE_PAD)

    ax.text(
        1.05, b + 0.14, "offset", ha="left", va="bottom",
        fontsize=LABEL, color="#1A1A1A", fontweight="bold", zorder=6,
    )
    tick_label(ax, 9.2, y_at(9.2, f, psd), "periodic", side="above", tick=0.16)
    tick_label(ax, 22.0, y_at(22.0, f, aperiodic), "exponent", side="below", tick=0.12)
    clean(ax)


def box_strip_panel(ax, df, metric, title, ylabel, sig, ylim, yticks, show_strip=True):
    positions = {"ASD": 0.82, "TD": 1.28}
    fill = {"ASD": ASD, "TD": TD}

    for group in ["ASD", "TD"]:
        vals = df.loc[df["group"].eq(group), metric].dropna().to_numpy(float)
        box_strip(
            ax, vals, positions[group], fill[group],
            width=0.24, side=-1 if group == "ASD" else +1,
            show_strip=show_strip,
        )

    ax.set_xlim(0.45, 1.55)
    ax.set_xticks([positions["ASD"], positions["TD"]], ["ASD", "TD"])
    ax.set_title(title, pad=TITLE_PAD, fontweight="bold")
    ax.set_xlabel("Group")
    ax.set_ylabel(ylabel)

    all_vals = np.concatenate([
        df.loc[df["group"].eq(g), metric].dropna().to_numpy(float)
        for g in ["ASD", "TD"]
    ])
    data_max = float(np.nanmax(all_vals))
    ymin = ylim[0]
    head = 0.18 * (data_max - ymin)
    top = max(ylim[1], data_max + head)
    ax.set_ylim(ymin, top)
    ax.set_yticks(yticks)

    span = top - ymin
    y = data_max + 0.055 * span
    h = 0.016 * span
    add_sig(
        ax, positions["ASD"], positions["TD"], y, h, sig,
        fontsize=NOTE, color="#1A1A1A",
    )
    clean(ax)


def exponent_panel(ax, df, box_width=None, show_strip=True):
    """Global and posterior exponent; boxes sized to read like panel c."""
    specs = [
        ("global_exponent", "Global", "*"),
        ("posterior_exponent", "Posterior", "***"),
    ]
    width = 0.24
    if box_width is None:
        box_width = 0.21
    pair_off = 0.18
    gap = 0.050
    bases = np.array([1.18, 2.42])
    fill = {"ASD": ASD, "TD": TD}

    maxima = []
    for base, (metric, _lab, _sig) in zip(bases, specs):
        for group, sign in (("ASD", -1), ("TD", +1)):
            vals = df.loc[df["group"].eq(group), metric].dropna().to_numpy(float)
            maxima.append(float(np.nanmax(vals)))
            box_strip(
                ax, vals, base + sign * pair_off, fill[group],
                width=width, side=-1 if group == "ASD" else +1,
                gap=gap, box_width=box_width, show_strip=show_strip,
            )

    if show_strip:
        strip_half = 0.24 * box_width
        strip_distance = 0.50 * box_width + gap + strip_half
        x_left = bases[0] - pair_off - strip_distance - strip_half
        x_right = bases[-1] + pair_off + strip_distance + strip_half
        pad = 0.32
    else:
        x_left = bases[0] - pair_off - 0.50 * box_width
        x_right = bases[-1] + pair_off + 0.50 * box_width
        pad = 0.28
    ax.set_xlim(x_left - pad, x_right + pad)
    ax.set_xticks(bases, [lab for _m, lab, _s in specs])
    ax.set_title("Aperiodic exponent", pad=TITLE_PAD, fontweight="bold")
    ax.set_xlabel("Region")
    ax.set_ylabel("Aperiodic exponent")

    ymin, ylim_top = 1.00, 2.62
    data_max = max(maxima)
    span = ylim_top - ymin
    top = max(ylim_top, data_max + 0.14 * span)
    ax.set_ylim(ymin, top)
    ax.set_yticks([1.0, 1.5, 2.0, 2.5])

    span = top - ymin
    h = 0.016 * span
    for base, (metric, _lab, sig) in zip(bases, specs):
        cond_max = float(np.nanmax(df[metric].to_numpy(float)))
        y = cond_max + 0.050 * span
        add_sig(ax, base - pair_off, base + pair_off, y, h, sig)
    clean(ax)


def _render_figure1(show_strip=True, stem="Figure1"):
    subjects = pd.read_csv(DATA / "fig1_resting_primary_subjects.csv")
    fig = plt.figure(figsize=(8.70, 3.62))
    gspec = fig.add_gridspec(
        1, 3, left=0.052, right=0.988, bottom=0.235, top=0.84,
        width_ratios=[1.02, 1.52, 1.00], wspace=0.30,
    )
    ax_a = fig.add_subplot(gspec[0, 0])
    ax_b = fig.add_subplot(gspec[0, 1])
    ax_c = fig.add_subplot(gspec[0, 2])

    panel(ax_a, "a", x=-0.10, y=PANEL_Y)
    draw_specparam(ax_a)

    panel(ax_b, "b", x=-0.10, y=PANEL_Y)
    exponent_panel(ax_b, subjects, show_strip=show_strip)

    panel(ax_c, "c", x=-0.14, y=PANEL_Y)
    box_strip_panel(
        ax_c, subjects, "global_offset", "Global offset",
        r"Aperiodic offset (log$_{10}$ $\mu V^2/Hz$)",
        "n.s.", ylim=(-10.65, -9.45), yticks=[-10.4, -10.0, -9.6],
        show_strip=show_strip,
    )

    fig.legend(
        handles=group_legend_handles(),
        loc="lower center",
        ncol=2,
        fontsize=LEGEND,
        frameon=False,
        handletextpad=0.40,
        columnspacing=1.35,
        bbox_to_anchor=(0.5, 0.015),
    )
    save(fig, stem)


def make_figure1():
    _render_figure1(show_strip=True, stem="Figure1")
    _render_figure1(show_strip=False, stem="Figure1_box")


if __name__ == "__main__":
    make_figure1()
