#!/usr/bin/env python
"""Fig. 2A topomap rebuild — plot-only package (frozen channel statistics).

Reads pre-exported CSVs from ./data/; does NOT refit channel OLS or recompute FDR.
Layout constants match scripts/plot_fig2a_channelwise_topomap.py as of 2026-07-11.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Circle
from mne.channels.layout import _find_topomap_coords

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"
DEFAULT_OUT_DIR = PACKAGE_ROOT / "output"

EFFECTS_CSV = DATA_DIR / "channel_group_effects.csv"
CMAP_STOPS_CSV = DATA_DIR / "colormap_reference_tmap_cbar_stops.csv"
POSTERIOR_CSV = DATA_DIR / "posterior_cluster_callouts.csv"
PLOT_CONFIG_JSON = PACKAGE_ROOT / "plot_config.json"

MONTAGE_NAME = "GSN-HydroCel-64_1.0"
POSTERIOR_CORE = ["E33", "E36", "E37", "E38"]
TOPO_SPHERE = 0.08
TOPO_CONTOUR_LEVELS = 7
TOPO_RES = 256
TOPO_VLIM_DEFAULT = 0.15
FDR_ALPHA = 0.05

PANEL_LABEL = "A"
COLOR_HEAD = "#1A1A1A"
COLOR_CONTOUR = "#444444"
COLOR_SENSOR = COLOR_HEAD
TOPO_CMAP_NAME = "reference_tmap_cbar"

POSTERIOR_CALLOUT = {
    "E33": {"label_xy": (-0.052, -0.104), "ha": "center", "va": "top"},
    "E36": {"label_xy": (-0.018, -0.110), "ha": "center", "va": "top"},
    "E37": {"label_xy": (0.018, -0.116), "ha": "center", "va": "top"},
    "E38": {"label_xy": (0.052, -0.104), "ha": "center", "va": "top"},
}
CALLOUT_MARKER_SIZE = 8.5
CALLOUT_FONT_SIZE = 8.0
CALLOUT_LINE_WIDTH = 0.55


def _load_plot_config() -> dict:
    if PLOT_CONFIG_JSON.exists():
        return json.loads(PLOT_CONFIG_JSON.read_text(encoding="utf-8"))
    return {}


def _load_cmap_stops() -> list[tuple[float, str]]:
    if CMAP_STOPS_CSV.exists():
        df = pd.read_csv(CMAP_STOPS_CSV)
        return [(float(r.position), str(r.hex_color)) for r in df.itertuples(index=False)]
    return [
        (0.00, "#392862"),
        (0.10, "#1175ae"),
        (0.20, "#37a5c9"),
        (0.30, "#79c9da"),
        (0.40, "#bbe2ef"),
        (0.50, "#f2edee"),
        (0.60, "#ffe2cf"),
        (0.70, "#fcbd98"),
        (0.80, "#fb8d66"),
        (0.90, "#fb4d3e"),
        (1.00, "#e30227"),
    ]


def _make_topo_cmap() -> LinearSegmentedColormap:
    cdict: dict[str, list[tuple[float, float]]] = {"red": [], "green": [], "blue": []}
    for pos, color in _load_cmap_stops():
        r, g, b = to_rgb(color)
        cdict["red"].append((pos, r, r))
        cdict["green"].append((pos, g, g))
        cdict["blue"].append((pos, b, b))
    return LinearSegmentedColormap(TOPO_CMAP_NAME, segmentdata=cdict, N=256)


def _colorbar_ticks(vlim: float) -> np.ndarray:
    half = round(vlim / 2.0, 3)
    return np.array([-vlim, -half, 0.0, half, vlim])


def _format_cbar_tick(val: float) -> str:
    text = f"{val:.3f}".rstrip("0").rstrip(".")
    return text if text else "0"


def apply_theme_classic() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "text.color": COLOR_HEAD,
            "axes.labelcolor": COLOR_HEAD,
            "xtick.color": COLOR_HEAD,
            "ytick.color": COLOR_HEAD,
            "axes.grid": False,
        }
    )


def load_channel_effects() -> pd.DataFrame:
    if not EFFECTS_CSV.exists():
        raise FileNotFoundError(f"Missing {EFFECTS_CSV}")
    df = pd.read_csv(EFFECTS_CSV)
    required = {"channel", "beta_TD_minus_ASD", "p", "q"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{EFFECTS_CSV.name} missing columns: {sorted(missing)}")
    df["channel"] = df["channel"].astype(str)
    if "fdr_significant" not in df.columns:
        df["fdr_significant"] = df["q"].astype(float) < FDR_ALPHA
    if "is_posterior_roi" not in df.columns:
        df["is_posterior_roi"] = df["channel"].isin(POSTERIOR_CORE)
    return df.sort_values("channel").reset_index(drop=True)


def _contour_levels(vlim: float) -> np.ndarray:
    return np.linspace(-vlim, vlim, TOPO_CONTOUR_LEVELS)


def _is_inside_head(xy: np.ndarray, *, radius: float = TOPO_SPHERE) -> bool:
    return float(np.hypot(xy[0], xy[1])) <= radius + 1e-9


def _clip_topomap_to_head(ax: plt.Axes, *, radius: float = TOPO_SPHERE) -> Circle:
    clip = Circle((0.0, 0.0), radius, transform=ax.transData)
    clip.set_visible(False)
    ax.add_patch(clip)
    for artist in ax.images:
        artist.set_clip_path(clip)
    for coll in ax.collections:
        coll.set_clip_path(clip)
    return clip


def _style_topomap_contours(contour_set) -> None:
    if contour_set is None:
        return
    for coll in getattr(contour_set, "collections", []):
        coll.set_edgecolor(COLOR_CONTOUR)
        coll.set_linewidth(0.45)
        coll.set_alpha(0.85)


def _style_topomap_outline(ax: plt.Axes) -> None:
    for artist in ax.patches + ax.lines:
        if hasattr(artist, "set_edgecolor"):
            artist.set_edgecolor(COLOR_HEAD)
        if hasattr(artist, "set_color"):
            artist.set_color(COLOR_HEAD)
        if hasattr(artist, "set_linewidth"):
            artist.set_linewidth(0.75)


def _annotate_channel_callout(
    ax: plt.Axes,
    xy: np.ndarray,
    label: str,
    *,
    label_xy: tuple[float, float],
    ha: str = "center",
    va: str = "top",
) -> None:
    x, y = float(xy[0]), float(xy[1])
    lx, ly = label_xy
    ax.plot([lx, x], [ly, y], "-", color=COLOR_HEAD, lw=CALLOUT_LINE_WIDTH, solid_capstyle="round", zorder=7)
    ax.plot(x, y, "o", color=COLOR_HEAD, markersize=CALLOUT_MARKER_SIZE, markeredgewidth=0, zorder=8)
    ax.text(lx, ly, label, ha=ha, va=va, fontsize=CALLOUT_FONT_SIZE, fontweight="bold", color=COLOR_HEAD, zorder=9)


def plot_fig2a_topomap(
    channel_df: pd.DataFrame,
    *,
    out_dir: Path,
    out_stem: str = "panel_A_channelwise_effect_topomap",
    dpi: int = 600,
    vlim: float = TOPO_VLIM_DEFAULT,
) -> dict:
    apply_theme_classic()
    out_dir.mkdir(parents=True, exist_ok=True)

    montage = mne.channels.make_standard_montage(MONTAGE_NAME)
    ch_names = [c for c in montage.ch_names if c.startswith("E")]
    coef_map = channel_df.set_index("channel")["beta_TD_minus_ASD"].to_dict()
    fdr_map = channel_df.set_index("channel")["fdr_significant"].to_dict()
    values = np.array([coef_map.get(c, np.nan) for c in ch_names])
    vlim_val = float(vlim)
    contour_levels = _contour_levels(vlim_val)

    info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types="eeg")
    info.set_montage(montage)
    pos_2d = _find_topomap_coords(info, picks="eeg", sphere=TOPO_SPHERE)
    name_to_xy = {name: pos_2d[i] for i, name in enumerate(ch_names)}

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    topo_cmap = _make_topo_cmap()
    im, contour_set = mne.viz.plot_topomap(
        values,
        info,
        axes=ax,
        show=False,
        cmap=topo_cmap,
        vlim=(-vlim_val, vlim_val),
        contours=contour_levels,
        sensors=False,
        outlines="head",
        sphere=TOPO_SPHERE,
        image_interp="cubic",
        res=TOPO_RES,
        extrapolate="head",
    )
    for image in ax.images:
        image.set_interpolation("bilinear")
    _clip_topomap_to_head(ax, radius=TOPO_SPHERE)
    _style_topomap_contours(contour_set)
    _style_topomap_outline(ax)

    posterior_set = set(POSTERIOR_CORE)
    for name, xy in name_to_xy.items():
        if name in posterior_set or not _is_inside_head(xy):
            continue
        ax.plot(xy[0], xy[1], "o", color=COLOR_SENSOR, markersize=1.8, alpha=0.85, markeredgewidth=0, zorder=4)

    for ch in POSTERIOR_CORE:
        if ch not in name_to_xy or not _is_inside_head(name_to_xy[ch]):
            continue
        if not fdr_map.get(ch, False):
            continue
        cfg = POSTERIOR_CALLOUT.get(ch, {})
        _annotate_channel_callout(
            ax,
            name_to_xy[ch],
            ch,
            label_xy=cfg.get("label_xy", (0.0, -0.11)),
            ha=cfg.get("ha", "center"),
            va=cfg.get("va", "top"),
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.025, shrink=0.70, aspect=22)
    cbar.set_label("β (TD − ASD)", fontsize=8, labelpad=6)
    cbar.ax.tick_params(labelsize=7, width=0.5, length=3, color=COLOR_HEAD)
    cbar.outline.set_edgecolor(COLOR_HEAD)
    cbar.outline.set_linewidth(0.5)
    cbar_ticks = _colorbar_ticks(vlim_val)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.set_yticklabels([_format_cbar_tick(t) for t in cbar_ticks])

    ax.text(-0.08, 1.06, PANEL_LABEL, transform=ax.transAxes, fontsize=15, fontweight="bold", ha="left", va="top", color=COLOR_HEAD)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.06, right=0.88, top=0.96, bottom=0.10)

    png_path = out_dir / f"{out_stem}.png"
    tiff_path = out_dir / f"{out_stem}.tiff"
    pdf_path = out_dir / f"{out_stem}.pdf"
    svg_path = out_dir / f"{out_stem}.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(tiff_path, dpi=dpi, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "vlim": vlim_val,
        "colormap": TOPO_CMAP_NAME,
        "n_channels": len(channel_df),
        "fdr_channels": channel_df.loc[channel_df["fdr_significant"], "channel"].tolist(),
        "outputs": {
            "png": str(png_path),
            "tiff": str(tiff_path),
            "pdf": str(pdf_path),
            "svg": str(svg_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Fig. 2A topomap from frozen CSV inputs.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-stem", type=str, default="panel_A_channelwise_effect_topomap")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--vlim", type=float, default=TOPO_VLIM_DEFAULT, help="Symmetric colour limit (default 0.15).")
    args = parser.parse_args()

    channel_df = load_channel_effects()
    meta = plot_fig2a_topomap(channel_df, out_dir=args.out_dir, out_stem=args.out_stem, dpi=args.dpi, vlim=args.vlim)
    report_path = args.out_dir / f"{args.out_stem}_report.json"
    report = {
        "package_root": str(PACKAGE_ROOT),
        "effects_csv": str(EFFECTS_CSV),
        "plot_config": _load_plot_config(),
        "load_mode": "frozen_csv_only_no_recompute",
        **meta,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
