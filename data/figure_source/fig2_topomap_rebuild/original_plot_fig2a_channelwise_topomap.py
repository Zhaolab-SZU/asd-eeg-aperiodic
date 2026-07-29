#!/usr/bin/env python
"""Fig. 2A — channel-wise resting aperiodic exponent group effect (TD − ASD) topomap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Circle
from mne.channels.layout import _find_topomap_coords

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MONTAGE_NAME = "GSN-HydroCel-64_1.0"
POSTERIOR_CORE = ["E33", "E36", "E37", "E38"]
TOPO_SPHERE = 0.08
TOPO_CONTOUR_LEVELS = 7
TOPO_RES = 256
TOPO_VLIM_CLIP = (0.12, 0.16)
TOPO_VLIM_DEFAULT = 0.15
FDR_ALPHA = 0.05

MODEL_FORMULA = (
    "aperiodic_exponent ~ C(group) + age_months + C(sex) + IQ_total + usable_epochs"
)

PANEL_LABEL = "A"
SUGGESTED_CAPTION = (
    "Only E33, E36, E37, and E38 survived whole-scalp FDR correction."
)

COLOR_HEAD = "#1A1A1A"
COLOR_CONTOUR = "#444444"
COLOR_SENSOR = COLOR_HEAD

# External labels + straight leader lines (reference: column outside head, bold text).
POSTERIOR_CALLOUT = {
    "E33": {"label_xy": (-0.052, -0.104), "ha": "center", "va": "top"},
    "E36": {"label_xy": (-0.018, -0.110), "ha": "center", "va": "top"},
    "E37": {"label_xy": (0.018, -0.116), "ha": "center", "va": "top"},
    "E38": {"label_xy": (0.052, -0.104), "ha": "center", "va": "top"},
}
CALLOUT_MARKER_SIZE = 8.5
CALLOUT_FONT_SIZE = 8.0
CALLOUT_LINE_WIDTH = 0.55

# Colormap sampled from reference Δt-value colorbar (pixel-extracted, trimmed edges).
TOPO_CMAP_NAME = "reference_tmap_cbar"
REFERENCE_CMAP_STOPS = [
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

DEFAULT_INPUT_CANDIDATES = [
    PROJECT_ROOT / "outputs/figure_source_data/fig2_channel_level.csv",
    PROJECT_ROOT / "figure_source_data/topomap_channel_effects_mne_layout.csv",
    PROJECT_ROOT / "figures_submission/v3/data/channel_effects.csv",
    PROJECT_ROOT / "derivatives/stats/channel_level_analysis.csv",
]

SOURCE_OUT = PROJECT_ROOT / "figure_source_data/fig2a_channelwise_topomap_source.csv"
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "working/final_figures/figure2_spatial_robustness"
)
DEFAULT_OUT_STEM = "panel_A_channelwise_effect_topomap"


def _make_fig2a_topo_cmap() -> LinearSegmentedColormap:
    """Reference Δt colorbar palette (pixel-extracted from submission style guide)."""
    cdict: dict[str, list[tuple[float, float]]] = {"red": [], "green": [], "blue": []}
    for pos, color in REFERENCE_CMAP_STOPS:
        r, g, b = to_rgb(color)
        cdict["red"].append((pos, r, r))
        cdict["green"].append((pos, g, g))
        cdict["blue"].append((pos, b, b))
    return LinearSegmentedColormap(TOPO_CMAP_NAME, segmentdata=cdict, N=256)


def _colorbar_ticks(vlim: float) -> np.ndarray:
    """Five symmetric ticks like reference Δt colorbar (-2 … 2)."""
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


def _find_input_csv(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return explicit
    for path in DEFAULT_INPUT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No channel-level group model CSV found. Run scripts/10_channel_level_analysis.py "
        "or provide --input-csv."
    )


def _normalize_channel_table(raw: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    df = raw.copy()
    rename_map = {
        "ch_name": "channel",
        "electrode": "channel",
        "coef": "beta_TD_minus_ASD",
        "beta": "beta_TD_minus_ASD",
        "estimate": "beta_TD_minus_ASD",
        "pvalue": "p",
        "p_value": "p",
        "pvalue_fdr": "q",
        "fdr_q": "q",
        "FDR_q": "q",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "beta_TD_minus_ASD" not in df.columns:
        raise ValueError(f"{source_path} missing group effect column (beta/coef).")

    if "channel" not in df.columns:
        raise ValueError(f"{source_path} missing channel column.")

    df["channel"] = df["channel"].astype(str)
    if "SE" not in df.columns and "se" in df.columns:
        df["SE"] = df["se"]

    if "p" not in df.columns and "pvalue" in df.columns:
        df["p"] = df["pvalue"]

    if "q" not in df.columns:
        if "significant_fdr" in df.columns and "p" in df.columns:
            from src.stats_utils import fdr_correction

            reject, q = fdr_correction(df["p"].astype(float).values, alpha=FDR_ALPHA)
            df["q"] = q
            df["fdr_significant"] = reject
        elif "p" in df.columns:
            from src.stats_utils import fdr_correction

            reject, q = fdr_correction(df["p"].astype(float).values, alpha=FDR_ALPHA)
            df["q"] = q
            df["fdr_significant"] = reject
        else:
            df["q"] = np.nan
            df["fdr_significant"] = False
    else:
        df["fdr_significant"] = df["q"].astype(float) < FDR_ALPHA

    if "SE" in df.columns:
        df["t_TD_minus_ASD"] = df["beta_TD_minus_ASD"] / df["SE"]
    else:
        df["t_TD_minus_ASD"] = np.nan

    df["is_posterior_roi"] = df["channel"].isin(POSTERIOR_CORE)
    return df.sort_values("channel").reset_index(drop=True)


def recompute_from_specparam(cfg_path: Path | None = None) -> pd.DataFrame:
    """Fallback: per-channel OLS + BH-FDR from specparam channel QC table."""
    from src.config import load_config
    from src.io_utils import attach_usable_epochs, exclude_specparam_low_quality, load_analysis_participants
    from src.stats_utils import fdr_correction, run_ols

    cfg = load_config(cfg_path)
    deriv = Path(cfg["paths"]["derivatives_root"])
    ch_path = deriv / "specparam" / "specparam_channel_results_qc.csv"
    channel_df = pd.read_csv(ch_path)
    if "fit_valid" in channel_df.columns:
        channel_df = channel_df[channel_df["fit_valid"]]

    participants = load_analysis_participants(cfg)
    participants = attach_usable_epochs(participants, deriv)
    participants = exclude_specparam_low_quality(participants, deriv)

    rows: list[dict] = []
    for ch, sub_ch in channel_df.groupby("channel"):
        df = participants.merge(sub_ch, on=["subject_id", "group"], how="inner")
        df = df.dropna(
            subset=["aperiodic_exponent", "group", "age_months", "sex", "IQ_total", "usable_epochs"]
        )
        if len(df) < 10:
            continue
        res = run_ols(MODEL_FORMULA, df)
        group_terms = [t for t in res.params.index if t.startswith("C(group)")]
        if not group_terms:
            continue
        term = group_terms[0]
        beta = float(res.params[term])
        se = float(res.bse[term])
        p = float(res.pvalues[term])
        rows.append(
            {
                "channel": str(ch),
                "beta_TD_minus_ASD": beta,
                "SE": se,
                "t_TD_minus_ASD": beta / se if se else np.nan,
                "p": p,
                "n_obs": int(res.nobs),
            }
        )

    out = pd.DataFrame(rows)
    reject, q = fdr_correction(out["p"].values, alpha=FDR_ALPHA)
    out["q"] = q
    out["fdr_significant"] = reject
    out["is_posterior_roi"] = out["channel"].isin(POSTERIOR_CORE)
    return out.sort_values("channel").reset_index(drop=True)


def load_channel_data(input_csv: Path | None, recompute: bool) -> tuple[pd.DataFrame, Path, str]:
    if recompute:
        df = recompute_from_specparam()
        return df, Path("recomputed_from_specparam"), "recomputed"

    path = _find_input_csv(input_csv)
    df = _normalize_channel_table(pd.read_csv(path), path)
    return df, path, "existing_csv"


def _auto_topo_vlim(values: np.ndarray, percentile: float = 88.0) -> float:
    """Symmetric colour limits — ignore single-channel outliers (e.g. E63 β≈0.28)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return TOPO_VLIM_CLIP[1]
    scale = float(np.percentile(np.abs(finite), percentile))
    lo, hi = TOPO_VLIM_CLIP
    return round(float(np.clip(scale, lo, hi)), 2)


def _contour_levels(vlim: float) -> np.ndarray:
    """Evenly spaced contour lines across the display range."""
    return np.linspace(-vlim, vlim, TOPO_CONTOUR_LEVELS)


def _is_inside_head(xy: np.ndarray, *, radius: float = TOPO_SPHERE) -> bool:
    """True when electrode 2D position lies within the head circle."""
    return float(np.hypot(xy[0], xy[1])) <= radius + 1e-9


def _clip_topomap_to_head(ax: plt.Axes, *, radius: float = TOPO_SPHERE) -> Circle:
    """Clip interpolated topomap image/contours to the head circle (no bleed past outline)."""
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
    """Straight leader line from external bold label to scalp electrode."""
    x, y = float(xy[0]), float(xy[1])
    lx, ly = label_xy
    ax.plot(
        [lx, x],
        [ly, y],
        "-",
        color=COLOR_HEAD,
        lw=CALLOUT_LINE_WIDTH,
        solid_capstyle="round",
        zorder=7,
    )
    ax.plot(
        x,
        y,
        "o",
        color=COLOR_HEAD,
        markersize=CALLOUT_MARKER_SIZE,
        markeredgewidth=0,
        zorder=8,
    )
    ax.text(
        lx,
        ly,
        label,
        ha=ha,
        va=va,
        fontsize=CALLOUT_FONT_SIZE,
        fontweight="bold",
        color=COLOR_HEAD,
        zorder=9,
    )


def plot_fig2a_topomap(
    channel_df: pd.DataFrame,
    *,
    out_dir: Path,
    out_stem: str,
    dpi: int = 600,
    vlim: float | None = TOPO_VLIM_DEFAULT,
) -> dict:
    apply_theme_classic()
    out_dir.mkdir(parents=True, exist_ok=True)

    montage = mne.channels.make_standard_montage(MONTAGE_NAME)
    ch_names = [c for c in montage.ch_names if c.startswith("E")]
    coef_map = channel_df.set_index("channel")["beta_TD_minus_ASD"].to_dict()
    fdr_map = channel_df.set_index("channel")["fdr_significant"].to_dict()
    values = np.array([coef_map.get(c, np.nan) for c in ch_names])
    vlim_val = float(vlim) if vlim is not None else _auto_topo_vlim(values)
    contour_levels = _contour_levels(vlim_val)

    info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types="eeg")
    info.set_montage(montage)
    pos_2d = _find_topomap_coords(info, picks="eeg", sphere=TOPO_SPHERE)
    name_to_xy = {name: pos_2d[i] for i, name in enumerate(ch_names)}

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    topo_cmap = _make_fig2a_topo_cmap()
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
        ax.plot(
            xy[0],
            xy[1],
            "o",
            color=COLOR_SENSOR,
            markersize=1.8,
            alpha=0.85,
            markeredgewidth=0,
            zorder=4,
        )

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

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.038,
        pad=0.025,
        shrink=0.70,
        aspect=22,
    )
    cbar.set_label("β (TD − ASD)", fontsize=8, labelpad=6)
    cbar.ax.tick_params(labelsize=7, width=0.5, length=3, color=COLOR_HEAD)
    cbar.outline.set_edgecolor(COLOR_HEAD)
    cbar.outline.set_linewidth(0.5)
    cbar_ticks = _colorbar_ticks(vlim_val)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.set_yticklabels([_format_cbar_tick(t) for t in cbar_ticks])

    ax.text(
        -0.08,
        1.06,
        PANEL_LABEL,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top",
        color=COLOR_HEAD,
    )

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

    source_cols = [
        "channel",
        "beta_TD_minus_ASD",
        "SE",
        "t_TD_minus_ASD",
        "p",
        "q",
        "fdr_significant",
        "is_posterior_roi",
    ]
    source_df = channel_df[[c for c in source_cols if c in channel_df.columns]].copy()
    SOURCE_OUT.parent.mkdir(parents=True, exist_ok=True)
    source_df.to_csv(SOURCE_OUT, index=False)

    fdr_channels = channel_df.loc[channel_df["fdr_significant"], "channel"].tolist()
    posterior_stats = channel_df[channel_df["is_posterior_roi"]][
        ["channel", "beta_TD_minus_ASD", "p", "q"]
    ]

    return {
        "panel_label": PANEL_LABEL,
        "panel_style": "fig2_main_panel_A",
        "annotation_style": "external_straight_callout_labels",
        "suggested_caption": SUGGESTED_CAPTION,
        "n_channels": len(channel_df),
        "n_fdr_significant": int(channel_df["fdr_significant"].sum()),
        "fdr_channels": fdr_channels,
        "vlim": vlim_val,
        "colormap": TOPO_CMAP_NAME,
        "colormap_source": "pixel_extract_reference_delta_t_colorbar",
        "posterior_stats": posterior_stats.to_dict(orient="records"),
        "outputs": {
            "png": str(png_path),
            "tiff": str(tiff_path),
            "pdf": str(pdf_path),
            "svg": str(svg_path),
            "source_csv": str(SOURCE_OUT),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fig. 2A channel-wise group-effect topomap.")
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--recompute", action="store_true", help="Re-fit channel OLS from specparam QC.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-stem", type=str, default=DEFAULT_OUT_STEM)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--vlim",
        type=float,
        default=TOPO_VLIM_DEFAULT,
        help="Symmetric colour limit (default: 0.15).",
    )
    args = parser.parse_args()

    channel_df, input_path, load_mode = load_channel_data(args.input_csv, args.recompute)
    meta = plot_fig2a_topomap(
        channel_df,
        out_dir=args.out_dir,
        out_stem=args.out_stem,
        dpi=args.dpi,
        vlim=args.vlim,
    )

    report = {
        "input_data": str(input_path),
        "load_mode": load_mode,
        "model_formula": MODEL_FORMULA,
        "group_coding": "TD − ASD (positive β: TD > ASD, ASD exponent lower / flatter)",
        **meta,
    }
    report_path = args.out_dir / f"{args.out_stem}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Input: {input_path} ({load_mode})")
    print(f"Model: {MODEL_FORMULA}")
    print(f"Channels: {meta['n_channels']}")
    print(f"FDR-significant channels (q < {FDR_ALPHA}): {meta['n_fdr_significant']}")
    print(f"FDR list: {', '.join(meta['fdr_channels'])}")
    print("Posterior ROI:")
    for row in meta["posterior_stats"]:
        print(
            f"  {row['channel']}: beta={row['beta_TD_minus_ASD']:.4f}, "
            f"p={row['p']:.4g}, q={row['q']:.4g}"
        )
    print("Outputs:")
    for k, v in meta["outputs"].items():
        print(f"  {k}: {v}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
