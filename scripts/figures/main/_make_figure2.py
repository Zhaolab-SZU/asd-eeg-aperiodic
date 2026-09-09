"""Figure 2 — spatial distribution + robustness (current design)."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle

from _grammar_style import (
    ASD, AX, DATA, INK, LETTER, NOTE, NOTE_SMALL, TD, TICK_GRID, TITLE_GRID,
    draw_forest, save,
)


# ============================================================

def firebrick_navy_cmap():
    """Diverging map for beta(TD - ASD).

    Negative values:
        ASD > TD → firebrick

    Positive values:
        TD > ASD → navy
    """
    colors = [
        ASD,
        "#F2E4E4",
        "#F7F7F7",
        "#E8EAF0",
        TD,
    ]

    return LinearSegmentedColormap.from_list(
        "firebrick_navy",
        colors,
        N=256,
    )


def make_figure2():
    import pandas as pd
    import mne
    from mne.channels.layout import _find_topomap_coords

    robust = pd.read_csv(
        DATA / "fig2_posterior_robustness.csv"
    )

    fig = plt.figure(
        figsize=(8.15, 3.55),
        dpi=300,
    )

    axes_top = 0.855

    ax_a = fig.add_axes([
        0.042,
        0.155,
        0.370,
        axes_top - 0.155,
    ])

    cax_w = 0.370 * 0.34

    cax = fig.add_axes([
        0.042 + (0.370 - cax_w) / 2,
        0.118,
        cax_w,
        0.014,
    ])

    x_b = 0.500
    w_b = 0.470

    ax_b = fig.add_axes([
        x_b,
        0.155,
        w_b,
        axes_top - 0.155,
    ])

    sphere = 0.08

    posterior = [
        "E33",
        "E36",
        "E37",
        "E38",
    ]

    callouts = {
        "E33": (-0.056, -0.112),
        "E36": (-0.020, -0.124),
        "E37": (0.020, -0.124),
        "E38": (0.056, -0.112),
    }

    effects = pd.read_csv(
        DATA
        / "fig2_topomap_rebuild"
        / "data"
        / "channel_group_effects.csv"
    )

    montage = mne.channels.make_standard_montage(
        "GSN-HydroCel-64_1.0"
    )

    ch_names = [
        c
        for c in montage.ch_names
        if c.startswith("E")
    ]

    coef = (
        effects
        .set_index("channel")["beta_TD_minus_ASD"]
        .to_dict()
    )

    values = np.array([
        coef.get(c, np.nan)
        for c in ch_names
    ])

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=250.0,
        ch_types="eeg",
    )

    info.set_montage(montage)

    pos = _find_topomap_coords(
        info,
        picks="eeg",
        sphere=sphere,
    )

    name_xy = {
        name: pos[i]
        for i, name in enumerate(ch_names)
    }

    levels = np.linspace(
        -0.15,
        0.15,
        7,
    )

    cmap = firebrick_navy_cmap()

    im, contour_set = mne.viz.plot_topomap(
        values,
        info,
        axes=ax_a,
        show=False,
        cmap=cmap,
        vlim=(-0.15, 0.15),
        contours=levels,
        sensors=False,
        outlines="head",
        sphere=sphere,
        image_interp="cubic",
        res=256,
        extrapolate="head",
    )

    for image in ax_a.images:
        image.set_interpolation("bilinear")

    clip = Circle(
        (0.0, 0.0),
        sphere,
        transform=ax_a.transData,
    )

    clip.set_visible(False)
    ax_a.add_patch(clip)

    for image in ax_a.images:
        image.set_clip_path(clip)

    if contour_set is not None:
        collections = getattr(
            contour_set,
            "collections",
            None,
        )

        if collections:
            for coll in collections:
                coll.set_clip_path(clip)
                coll.set_edgecolor("#111111")
                coll.set_linewidth(0.45)

        else:
            contour_set.set_clip_path(clip)
            contour_set.set_edgecolor("#111111")
            contour_set.set_linewidths(0.45)

    for artist in (
        list(ax_a.patches)
        + list(ax_a.lines)
    ):
        if artist is clip:
            continue

        if hasattr(artist, "set_edgecolor"):
            artist.set_edgecolor("#1A1A1A")

        if hasattr(artist, "set_color"):
            artist.set_color("#1A1A1A")

        if hasattr(artist, "set_linewidth"):
            artist.set_linewidth(0.75)

    posterior_set = set(posterior)

    for name, xy in name_xy.items():
        if (
            name in posterior_set
            or float(np.hypot(*xy)) > sphere + 1e-9
        ):
            continue

        ax_a.plot(
            xy[0],
            xy[1],
            "o",
            color="#1A1A1A",
            markersize=1.8,
            alpha=0.85,
            markeredgewidth=0,
            zorder=4,
            clip_on=False,
        )

    for name in posterior:
        xy = name_xy[name]
        lx, ly = callouts[name]

        ax_a.plot(
            [xy[0], lx],
            [xy[1], ly + 0.003],
            "-",
            color="#1A1A1A",
            lw=0.45,
            zorder=7,
        )

        ax_a.plot(
            xy[0],
            xy[1],
            "o",
            color="#1A1A1A",
            markersize=4.6,
            markeredgewidth=0,
            zorder=8,
        )

        ax_a.text(
            lx,
            ly,
            name,
            ha="center",
            va="top",
            fontsize=NOTE,
            fontweight="bold",
            color="#1A1A1A",
        )

    ax_a.set_ylim(
        -0.138,
        0.110,
    )

    ax_a.set_aspect(
        "equal",
        adjustable="box",
    )

    ax_a.axis("off")

    title_y = axes_top + 0.012

    for x_center, title in (
        (
            0.042 + 0.370 / 2,
            "Spatial distribution",
        ),
        (
            x_b + w_b / 2,
            "Posterior effect robustness",
        ),
    ):
        fig.text(
            x_center,
            title_y,
            title,
            ha="center",
            va="bottom",
            fontsize=TITLE_GRID,
            fontweight="bold",
            color=AX,
        )

    sm = mpl.cm.ScalarMappable(
        norm=Normalize(
            vmin=-0.15,
            vmax=0.15,
        ),
        cmap=cmap,
    )

    cbar = fig.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
    )

    cbar.set_ticks([
        -0.15,
        0,
        0.15,
    ])

    cbar.set_ticklabels([
        "−0.15",
        "0",
        "0.15",
    ])

    cbar.ax.tick_params(
        labelsize=NOTE,
        width=1.40,
        length=2.6,
        colors=AX,
    )

    cbar.outline.set_linewidth(1.40)
    cbar.outline.set_edgecolor(AX)

    cbar.set_label(
        "Channel-wise β (TD − ASD)",
        fontsize=NOTE_SMALL,
        fontweight="bold",
        color=AX,
        labelpad=3,
    )

    order = [
        "primary",
        "IQ-balanced matched",
        "strict-QC",
        "low-gamma adjusted",
        "ICLabel",
    ]

    names = {
        "primary": "Primary",
        "IQ-balanced matched": "IQ-balanced",
        "strict-QC": "Strict QC",
        "low-gamma adjusted": "Low-gamma adj.",
        "ICLabel": "ICLabel",
    }

    df = robust[
        robust["analysis"].isin(order)
    ].copy()

    df["analysis"] = pd.Categorical(
        df["analysis"],
        categories=order,
        ordered=True,
    )

    df = (
        df.sort_values("analysis")
        .reset_index(drop=True)
    )

    draw_forest(
        ax_b,
        [
            names[str(a)]
            for a in df["analysis"]
        ],
        df["beta_TD_minus_ASD"].to_numpy(float),
        df["ci_low"].to_numpy(float),
        df["ci_high"].to_numpy(float),
        [INK] * 5,
        "Posterior exponent β (TD − ASD)",
        (0.00, 0.22),
        xticks=[
            0.00,
            0.05,
            0.10,
            0.15,
            0.20,
        ],
        ticksize=TICK_GRID,
        zero_color="#D0D0D0",
        zero_lw=0,
        marker_size=5.6,
        marker_fmts=["^"] * 5,
        elinewidth=0.70,
        capsize=2.8,
    )

    ax_b.set_ylim(
        -0.70,
        len(df) - 0.35,
    )

    fig.text(
        0.020,
        0.955,
        "a",
        fontsize=LETTER,
        fontweight="bold",
        ha="left",
        va="top",
    )

    fig.text(
        0.455,
        0.955,
        "b",
        fontsize=LETTER,
        fontweight="bold",
        ha="left",
        va="top",
    )

    save(fig, "Figure2")


if __name__ == "__main__":
    make_figure2()
