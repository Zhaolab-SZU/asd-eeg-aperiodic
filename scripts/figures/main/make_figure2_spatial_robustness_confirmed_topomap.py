from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFilter
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe


ROOT = Path(__file__).resolve().parents[3]  # repo root
OUTDIR = ROOT / "outputs/figures/figure2_spatial_robustness"
OUTDIR.mkdir(parents=True, exist_ok=True)

TOPOMAP = ROOT / "data/figure_source/fig2_topomap_rebuild/output/panel_A_channelwise_effect_topomap.png"
CMAP_STOPS = ROOT / "data/figure_source/fig2_topomap_rebuild/data/colormap_reference_tmap_cbar_stops.csv"
ROBUST = ROOT / "data/figure_source/fig2_posterior_robustness.csv"

ASD = "#C25450"
ASD_DARK = "#9E3F3D"
LIGHT_GREY = "#D7DCE3"
AX = "#222222"
DEEP_GREY = "#4D4D4D"


def setup_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.15,
            "axes.labelsize": 11.8,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10.8,
            "ytick.labelsize": 11.0,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 3.4,
            "ytick.major.size": 3.4,
        }
    )


def pstars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def draw_forest(ax, df):
    order = ["primary", "IQ-balanced matched", "strict-QC", "low-gamma adjusted", "ICLabel"]
    labels = {
        "primary": "Primary",
        "IQ-balanced matched": "IQ-balanced",
        "strict-QC": "Strict QC",
        "low-gamma adjusted": "Low-gamma adj.",
        "ICLabel": "ICLabel",
    }
    df = df[df["analysis"].isin(order)].copy()
    df["analysis"] = pd.Categorical(df["analysis"], categories=order, ordered=True)
    df = df.sort_values("analysis", ascending=False).reset_index(drop=True)
    y = np.arange(len(df))

    primary = df.loc[df["analysis"].astype(str) == "primary"].iloc[0]
    ax.axvspan(primary["ci_low"], primary["ci_high"], color=ASD, alpha=0.060, zorder=0)
    ax.axvline(0, color=DEEP_GREY, lw=1.0, zorder=1)

    shades = {
        "primary": ASD_DARK,
        "IQ-balanced matched": "#B5524F",
        "strict-QC": "#BD615D",
        "low-gamma adjusted": "#C7726E",
        "ICLabel": "#A85B57",
    }
    for i, row in df.iterrows():
        analysis = str(row["analysis"])
        color = shades[analysis]
        marker = "o"
        ms = 7.4
        x = row["beta_TD_minus_ASD"]
        ax.errorbar(
            x,
            y[i],
            xerr=[[x - row["ci_low"]], [row["ci_high"] - x]],
            fmt=marker,
            markersize=ms,
            mfc=color,
            mec=color,
            mew=0.8,
            ecolor=color,
            elinewidth=1.15,
            capsize=4,
            capthick=1.15,
            zorder=3,
        )
        star = pstars(float(row["p"]))
        star_color = "#D73027" if star != "n.s." else "#9A9A9A"
        star_weight = "bold" if star != "n.s." else "normal"
        ax.text(
            0.232,
            y[i] + 0.03,
            star,
            ha="left",
            va="center",
            fontsize=22.0 if star != "n.s." else 13.0,
            fontweight=star_weight,
            color=star_color,
            clip_on=False,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([labels[str(a)] for a in df["analysis"]])
    ax.set_ylim(-0.42, len(df) - 0.72)
    ax.set_xlim(0.0, 0.238)
    ax.set_xticks([0.00, 0.05, 0.10, 0.15, 0.20])
    ax.set_xlabel("Posterior exponent β (TD - ASD)", labelpad=7)
    ax.tick_params(axis="x", direction="in", width=1.2, length=3.8, color=AX, labelcolor=AX)
    ax.tick_params(axis="y", length=0, pad=4, labelcolor=AX)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(10.8)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("heavy")
        tick.set_fontsize(11.2)
    for spine in ["right", "top"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_linewidth(1.15)
    ax.spines["left"].set_color(AX)
    ax.spines["bottom"].set_linewidth(1.15)
    ax.spines["bottom"].set_color(AX)


def load_confirmed_topomap():
    im = Image.open(TOPOMAP).convert("RGBA")
    # Remove the original oversized panel label while preserving Cursor/MNE's
    # approved topographic interpolation, colorbar, and electrode callouts.
    patch = Image.new("RGBA", (320, 240), (255, 255, 255, 255))
    im.alpha_composite(patch, (0, 0))
    # Use the approved raster for the brain/topomap only. The colorbar is
    # redrawn as vector text below so its bar can be shorter without shrinking
    # the tick labels.
    brain = im.crop((420, 350, 2135, 2360))
    brain = recolor_topomap_raster(brain)

    canvas = Image.new("RGBA", (1680, 2010), (255, 255, 255, 255))
    brain_h = 1860
    brain_w = int(brain.width * brain_h / brain.height)
    brain = brain.resize((brain_w, brain_h), Image.Resampling.LANCZOS)
    canvas.alpha_composite(brain, (0, 65))
    crop = canvas
    # Lightly strengthen existing dark strokes (head outline, ears, labels)
    # without drawing any new geometry.
    rgb = crop.convert("RGB")
    arr = np.asarray(rgb)
    dark = np.all(arr < 70, axis=2).astype("uint8") * 255
    expanded = Image.fromarray(dark, mode="L").filter(ImageFilter.MaxFilter(3))
    added = Image.new("RGBA", crop.size, (0, 0, 0, 96))
    strengthened = crop.copy()
    strengthened.alpha_composite(Image.composite(added, Image.new("RGBA", crop.size, (0, 0, 0, 0)), expanded))
    crop = strengthened
    return crop


def reference_cmap():
    stops = pd.read_csv(CMAP_STOPS)
    cdict = {"red": [], "green": [], "blue": []}
    for row in stops.itertuples(index=False):
        r, g, b = to_rgb(row.hex_color)
        pos = float(row.position)
        cdict["red"].append((pos, r, r))
        cdict["green"].append((pos, g, g))
        cdict["blue"].append((pos, b, b))
    return LinearSegmentedColormap("reference_tmap_cbar", cdict, N=256)


def target_cmap():
    return LinearSegmentedColormap.from_list(
        "publication_rdbu",
        ["#2166AC", "#67A9CF", "#F7F7F7", "#F4A582", "#B2182B"],
        N=256,
    )


def recolor_topomap_raster(img):
    """Map the approved Cursor/MNE raster from its original colormap to the
    final deep-blue/white/brick-red scale while preserving black lines/text and
    white background.
    """
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba).copy()
    rgb = arr[..., :3].astype(float) / 255.0
    alpha = arr[..., 3]

    # Keep background and dark ink unchanged.
    is_white = np.all(arr[..., :3] > 242, axis=2)
    is_dark = np.all(arr[..., :3] < 90, axis=2)
    recolor_mask = (alpha > 0) & (~is_white) & (~is_dark)

    old_lut = reference_cmap()(np.linspace(0, 1, 256))[:, :3]
    new_lut = target_cmap()(np.linspace(0, 1, 256))[:, :3]
    pix = rgb[recolor_mask]
    # Chunked nearest-colormap lookup; preserves the spatial pattern, only
    # swaps palette family.
    idx = np.empty(len(pix), dtype=int)
    chunk = 120000
    for start in range(0, len(pix), chunk):
        part = pix[start : start + chunk]
        dist = ((part[:, None, :] - old_lut[None, :, :]) ** 2).sum(axis=2)
        idx[start : start + chunk] = dist.argmin(axis=1)
    arr[..., :3][recolor_mask] = np.clip(new_lut[idx] * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def main():
    setup_style()
    robust = pd.read_csv(ROBUST)
    topomap = load_confirmed_topomap()

    fig = plt.figure(figsize=(8.35, 3.42), dpi=300)
    ax_a = fig.add_axes([0.000, 0.035, 0.405, 0.910])
    cax = fig.add_axes([0.370, 0.315, 0.012, 0.390])
    ax_b = fig.add_axes([0.610, 0.205, 0.350, 0.655])

    ax_a.imshow(topomap)
    label_cfg = {
        "E33": {"label": (420, 1730), "dot": (635, 1180)},
        "E36": {"label": (650, 1810), "dot": (770, 1230)},
        "E37": {"label": (972, 1885), "dot": (900, 1380)},
        "E38": {"label": (1252, 1730), "dot": (1080, 1180)},
    }
    for cfg in label_cfg.values():
        x, y = cfg["label"]
        ax_a.add_patch(
            Rectangle(
                (x - 132, y - 82),
                264,
                164,
                facecolor="white",
                edgecolor="none",
                alpha=1.0,
                zorder=9,
            )
        )
    for label, cfg in label_cfg.items():
        x, y = cfg["label"]
        ax_a.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=14.2,
            fontweight="black",
            color=AX,
            path_effects=[pe.withStroke(linewidth=0.35, foreground=AX)],
            zorder=11,
        )
    ax_a.axis("off")
    sm = mpl.cm.ScalarMappable(norm=Normalize(vmin=-0.15, vmax=0.15), cmap=target_cmap())
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([-0.15, 0, 0.15])
    cbar.set_ticklabels(["-0.15", "0", "0.15"])
    cbar.ax.tick_params(labelsize=9.6, width=1.2, length=3.6, colors=AX)
    cbar.outline.set_linewidth(1.0)
    cbar.outline.set_edgecolor(AX)
    cbar.set_label("β (TD - ASD)", fontsize=9.8, fontweight="bold", color=AX, labelpad=5)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    draw_forest(ax_b, robust)

    fig.text(0.030, 0.925, "A", fontsize=19, fontweight="bold", ha="left", va="top")
    fig.text(0.535, 0.925, "B", fontsize=19, fontweight="bold", ha="left", va="top")

    stem = OUTDIR / "Figure2_spatial_localization_robustness_confirmed_topomap"
    for ext in ["png", "tiff"]:
        fig.savefig(f"{stem}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.04)
    for ext in ["pdf", "svg"]:
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.04)
    print(f"{stem}.png")


if __name__ == "__main__":
    main()
