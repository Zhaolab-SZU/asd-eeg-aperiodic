# Ideal release name: plot_posterior_robustness_forest.py
# Original path: scripts/plot_posterior_robustness_forest.py
# Note: Fig.2B robustness forest
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""Publication-ready posterior robustness figures: forest (default), dumbbell, or slope."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLOR_EARTH_RED = "#B5655D"
COLOR_TEXT = "#333333"
COLOR_REF_ZERO = "#C8C8C8"
COLOR_REF_PRIMARY = "#D8D8D8"
COLOR_LINE_MUTED = "#C9A8A3"
COLOR_DUMBELL_BAR = "#C8C8C8"
COLOR_CONNECTOR = "#E6DCDA"

X_AXIS_END = 0.20
ERROR_LINEWIDTH = 1.0
TEXT_GAP = 0.012
PLOT_X_MAX = 0.215

X_LEFT = 0.0
X_RIGHT = 1.0
SLOPE_Y_PAD = 0.018

DEFAULT_ROBUSTNESS_CSV = (
    PROJECT_ROOT / "outputs" / "figure_source_data" / "fig2_posterior_robustness.csv"
)

MODEL_ORDER = [
    "Primary model",
    "IQ-balanced cohort",
    "Strict specparam-QC",
    "Low-gamma adjusted",
    "ICLabel artifact branch",
]

# Fallback aligned with fig2_posterior_robustness.csv (strict recompute on n=90).
ROBUSTNESS_FALLBACK = [
    {
        "model": "Primary model",
        "beta": 0.133,
        "se": 0.034,
        "ci_low": 0.066,
        "ci_high": 0.200,
        "p_label": "< 0.001",
        "is_primary": True,
    },
    {
        "model": "IQ-balanced cohort",
        "beta": 0.123,
        "se": 0.033,
        "ci_low": 0.064,
        "ci_high": 0.193,
        "p_label": "< 0.001",
        "is_primary": False,
    },
    {
        "model": "Strict specparam-QC",
        "beta": 0.139,
        "se": 0.034,
        "ci_low": 0.071,
        "ci_high": 0.207,
        "p_label": "< 0.001",
        "is_primary": False,
    },
    {
        "model": "Low-gamma adjusted",
        "beta": 0.102,
        "se": 0.035,
        "ci_low": 0.033,
        "ci_high": 0.171,
        "p_label": "0.004",
        "is_primary": False,
    },
    {
        "model": "ICLabel artifact branch",
        "beta": 0.121,
        "se": 0.034,
        "ci_low": 0.054,
        "ci_high": 0.189,
        "p_label": "< 0.001",
        "is_primary": False,
    },
]

ANALYSIS_LABEL_MAP = {
    "primary": "Primary model",
    "IQ-balanced matched": "IQ-balanced cohort",
    "strict-QC": "Strict specparam-QC",
    "low-gamma adjusted": "Low-gamma adjusted",
    "ICLabel": "ICLabel artifact branch",
}

SLOPE_SHORT_LABELS = {
    "Primary model": "Primary model",
    "IQ-balanced cohort": "IQ-balanced",
    "Strict specparam-QC": "Strict-QC",
    "Low-gamma adjusted": "Low-gamma adj.",
    "ICLabel artifact branch": "ICLabel",
}

DUMBELL_ROW_LABELS = {
    "Primary model": "Primary model",
    "IQ-balanced cohort": "IQ-balanced cohort",
    "Strict specparam-QC": "Strict specparam-QC",
    "Low-gamma adjusted": "Low-gamma adjusted",
    "ICLabel artifact branch": "ICLabel artifact branch",
}


def apply_classic_style() -> None:
    apply_theme_classic()


def apply_theme_classic() -> None:
    """ggplot2 theme_classic analogue for matplotlib."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "text.color": COLOR_TEXT,
            "axes.labelcolor": COLOR_TEXT,
            "xtick.color": COLOR_TEXT,
            "ytick.color": COLOR_TEXT,
            "axes.grid": False,
        }
    )


def _p_label(p_val: float) -> str:
    return "< 0.001" if p_val < 0.001 else f"{p_val:.3f}".rstrip("0").rstrip(".")


def load_robustness_data(path: Path | None = None) -> pd.DataFrame:
    """Load five main robustness rows from figure source-data CSV."""
    csv_path = path or DEFAULT_ROBUSTNESS_CSV
    if not csv_path.exists():
        return pd.DataFrame(ROBUSTNESS_FALLBACK)

    raw = pd.read_csv(csv_path)
    rows: list[dict] = []
    for key, model in ANALYSIS_LABEL_MAP.items():
        sub = raw[raw["analysis"].astype(str).str.lower() == key.lower()]
        if sub.empty:
            sub = raw[raw["analysis"].astype(str).str.contains(key.split()[0], case=False, na=False)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        beta = float(r["beta_TD_minus_ASD"])
        ci_low = float(r["ci_low"]) if pd.notna(r.get("ci_low")) else np.nan
        ci_high = float(r["ci_high"]) if pd.notna(r.get("ci_high")) else np.nan
        se = float(r["SE"]) if pd.notna(r.get("SE")) else np.nan
        if np.isnan(se) and not np.isnan(ci_low) and not np.isnan(ci_high):
            se = (ci_high - ci_low) / (2 * 1.96)
        if np.isnan(ci_low) or np.isnan(ci_high):
            ci_low = beta - 1.96 * se
            ci_high = beta + 1.96 * se
        rows.append(
            {
                "model": model,
                "beta": beta,
                "se": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_label": _p_label(float(r["p"])),
                "is_primary": key == "primary",
            }
        )
    if len(rows) != 5:
        return pd.DataFrame(ROBUSTNESS_FALLBACK)
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
    return out.sort_values("model")


def prepare_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ci_low" not in out.columns or out["ci_low"].isna().any():
        out["ci_low"] = out["beta"] - 1.96 * out["se"]
        out["ci_high"] = out["beta"] + 1.96 * out["se"]
    out["beta_ci_label"] = out.apply(
        lambda r: f"{r['beta']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]",
        axis=1,
    )
    out["p_text"] = out["p_label"].apply(lambda p: f"p {p}")
    return out


def _ax_y_to_fig_y(ax: plt.Axes, y: float, fig: plt.Figure) -> float:
    """Map a data-space y coordinate to figure-fraction y for aligned side annotations."""
    _, y_disp = ax.transData.transform((0.0, y))
    return float(fig.transFigure.inverted().transform((0.0, y_disp))[1])


def _add_forest_table(
    fig: plt.Figure,
    ax: plt.Axes,
    rows: list,
    y_positions: list[float],
) -> None:
    """Right-aligned stats table separated from the plot area."""
    pos = ax.get_position()
    col_beta_x = pos.x1 + 0.03
    col_p_x = pos.x1 + 0.27
    header_y = _ax_y_to_fig_y(ax, y_positions[0] - 0.62, fig)

    fig.text(col_beta_x, header_y, "β [95% CI]", fontsize=9, fontweight="semibold", color=COLOR_TEXT, va="center")
    fig.text(col_p_x, header_y, "p", fontsize=9, fontweight="semibold", color=COLOR_TEXT, va="center")
    fig.add_artist(
        plt.Line2D(
            [col_beta_x, col_p_x + 0.07],
            [header_y - 0.028, header_y - 0.028],
            transform=fig.transFigure,
            color="#DDDDDD",
            linewidth=0.6,
        )
    )

    for i, (y, row) in enumerate(zip(y_positions, rows)):
        y_fig = _ax_y_to_fig_y(ax, y, fig)
        is_pri = i == 0
        fw = "semibold" if is_pri else "normal"
        fc = COLOR_TEXT if is_pri else "#555555"
        fs = 9.5 if is_pri else 9.0
        fig.text(col_beta_x, y_fig, row.beta_ci_label, fontsize=fs, fontweight=fw, color=fc, va="center")
        fig.text(col_p_x, y_fig, row.p_text, fontsize=fs, fontweight=fw, color=fc, va="center")


def plot_robustness_forest(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Standard journal forest plot with primary CI reference band and side table."""
    apply_theme_classic()
    df = prepare_table(df)
    primary_row = df.loc[df["is_primary"]].iloc[0]
    primary_beta = float(primary_row.beta)
    primary_ci_low = float(primary_row.ci_low)
    primary_ci_high = float(primary_row.ci_high)

    models = df["model"].tolist()
    y_pos = np.arange(len(models), dtype=float)
    rows = list(df.itertuples(index=False))

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
    else:
        fig = ax.figure

    ax.axvspan(
        primary_ci_low,
        primary_ci_high,
        color="#E6E6E6",
        alpha=0.9,
        zorder=0,
    )
    ax.axvline(primary_beta, color="#9A9A9A", linestyle="--", linewidth=0.85, zorder=1)
    ax.axvline(0, color=COLOR_REF_ZERO, linestyle="--", linewidth=0.6, zorder=1)

    for i, row in enumerate(rows):
        ax.hlines(
            y_pos[i],
            row.ci_low,
            row.ci_high,
            colors=COLOR_EARTH_RED,
            linewidth=0.75,
            zorder=2,
        )
        ax.scatter(
            row.beta,
            y_pos[i],
            s=40,
            c=COLOR_EARTH_RED,
            edgecolors="none",
            zorder=3,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlim(0, PLOT_X_MAX)
    ax.set_xticks(np.arange(0, X_AXIS_END + 0.01, 0.05))
    ax.set_xlabel("Posterior exponent β (TD − ASD)")
    ax.set_ylabel("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["left"].set_color(COLOR_TEXT)
    ax.spines["bottom"].set_color(COLOR_TEXT)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", width=0.5, length=4)
    ax.grid(False)

    fig.subplots_adjust(left=0.30, right=0.46, top=0.94, bottom=0.14)
    _add_forest_table(fig, ax, rows, y_pos.tolist())
    return fig


def _stagger_label_ys(
    betas: list[float],
    *,
    min_sep: float = 0.028,
) -> list[float]:
    """Nudge annotation y positions to avoid overlap when betas cluster."""
    order = np.argsort(betas)
    ys = list(betas)
    for i in range(1, len(order)):
        idx = order[i]
        prev_idx = order[i - 1]
        if ys[idx] - ys[prev_idx] < min_sep:
            ys[idx] = ys[prev_idx] + min_sep
    return ys


def plot_robustness_slope(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Summary slope graph: primary anchor (left) → sensitivity models (right)."""
    apply_classic_style()
    df = prepare_table(df)
    primary = df.loc[df["is_primary"]].iloc[0]
    others = df.loc[~df["is_primary"]].copy()
    primary_beta = float(primary.beta)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
    else:
        fig = ax.figure

    y_min = float(min(df["ci_low"].min(), primary_beta) - SLOPE_Y_PAD)
    y_max = float(max(df["ci_high"].max(), primary_beta) + SLOPE_Y_PAD)

    ax.axhspan(
        primary_beta - 0.015,
        primary_beta + 0.015,
        color=COLOR_EARTH_RED,
        alpha=0.07,
        zorder=0,
    )
    ax.axhline(primary_beta, color=COLOR_REF_PRIMARY, linestyle=":", linewidth=0.8, zorder=1)
    ax.axhline(0, color=COLOR_REF_ZERO, linestyle="--", linewidth=0.6, zorder=1, xmin=0, xmax=0.42)

    others_list = list(others.itertuples(index=False))
    label_ys = _stagger_label_ys([float(r.beta) for r in others_list])

    for row, label_y in zip(others_list, label_ys):
        beta = float(row.beta)
        ax.plot(
            [X_LEFT, X_RIGHT],
            [primary_beta, beta],
            color=COLOR_LINE_MUTED,
            linewidth=1.4,
            alpha=0.75,
            solid_capstyle="round",
            zorder=2,
        )
        ax.vlines(
            X_RIGHT,
            row.ci_low,
            row.ci_high,
            colors=COLOR_EARTH_RED,
            linewidth=ERROR_LINEWIDTH,
            zorder=3,
        )
        ax.scatter(
            X_RIGHT,
            beta,
            s=46,
            c=COLOR_EARTH_RED,
            edgecolors="none",
            zorder=4,
        )
        short = SLOPE_SHORT_LABELS.get(row.model, row.model)
        label_x = X_RIGHT + 0.06
        if abs(label_y - beta) > 0.004:
            ax.plot(
                [X_RIGHT + 0.02, label_x - 0.01],
                [beta, label_y],
                color="#BBBBBB",
                linewidth=0.5,
                zorder=3,
            )
        ax.text(
            label_x,
            label_y,
            f"{short}  β = {row.beta:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]",
            ha="left",
            va="center",
            fontsize=8.5,
            color=COLOR_TEXT,
        )
        ax.text(
            label_x,
            label_y - 0.013,
            row.p_text,
            ha="left",
            va="top",
            fontsize=7.5,
            color="#666666",
        )

    ax.scatter(
        X_LEFT,
        primary_beta,
        marker="s",
        s=72,
        c=COLOR_EARTH_RED,
        edgecolors="white",
        linewidths=0.6,
        zorder=5,
    )
    ax.text(
        X_LEFT - 0.07,
        primary_beta,
        f"β = {primary_beta:.3f}\n[{primary.ci_low:.3f}, {primary.ci_high:.3f}]",
        ha="right",
        va="center",
        fontsize=9,
        color=COLOR_TEXT,
        linespacing=1.25,
    )
    ax.text(
        X_LEFT - 0.07,
        primary_beta - 0.028,
        primary.p_text,
        ha="right",
        va="top",
        fontsize=8,
        color="#666666",
    )

    ax.set_xlim(-0.22, 1.78)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([X_LEFT, X_RIGHT])
    ax.set_xticklabels(["Primary model", "Sensitivity analyses"], fontsize=10)
    ax.set_ylabel("Posterior exponent β (TD − ASD)")
    ax.set_xlabel("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["left"].set_color(COLOR_TEXT)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", width=0.5, length=4)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.6, zorder=0)

    fig.subplots_adjust(left=0.14, right=0.98, top=0.94, bottom=0.12)
    return fig


def _draw_dumbbell(
    ax: plt.Axes,
    y: float,
    ci_low: float,
    beta: float,
    ci_high: float,
    *,
    primary: bool = False,
) -> None:
    """Horizontal CI dumbbell: caps at CI ends, filled dot at beta."""
    bar_lw = 1.35 if primary else 0.9
    bar_color = COLOR_EARTH_RED if primary else COLOR_DUMBELL_BAR
    cap_s = 28 if primary else 20
    dot_s = 58 if primary else 38

    ax.plot(
        [ci_low, ci_high],
        [y, y],
        color=bar_color,
        linewidth=bar_lw,
        solid_capstyle="round",
        zorder=2,
    )
    if primary:
        ax.scatter(
            [ci_low, ci_high],
            [y, y],
            s=cap_s,
            c=COLOR_EARTH_RED,
            edgecolors="none",
            zorder=3,
        )
        ax.scatter(
            beta,
            y,
            marker="s",
            s=dot_s,
            c=COLOR_EARTH_RED,
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )
    else:
        ax.scatter(
            [ci_low, ci_high],
            [y, y],
            s=cap_s,
            facecolors="white",
            edgecolors=COLOR_EARTH_RED,
            linewidths=0.75,
            zorder=3,
        )
        ax.scatter(beta, y, s=dot_s, c=COLOR_EARTH_RED, edgecolors="none", alpha=0.9, zorder=4)


def plot_robustness_dumbbell(df: pd.DataFrame) -> plt.Figure:
    """Dumbbell plot: primary anchor on top, sensitivity rows below, stats table on right."""
    apply_classic_style()
    df = prepare_table(df)
    primary = df.loc[df["is_primary"]].iloc[0]
    sensitivity = df.loc[~df["is_primary"]].copy()
    primary_beta = float(primary.beta)

    fig, ax = plt.subplots(figsize=(8.8, 4.6))

    y_primary = 0.0
    y_sens = np.arange(1, len(sensitivity) + 1, dtype=float) + 0.35
    rows_ordered = [primary, *list(sensitivity.itertuples(index=False))]
    y_positions = [y_primary, *y_sens.tolist()]

    ax.axvspan(
        primary_beta - 0.01,
        primary_beta + 0.01,
        color=COLOR_EARTH_RED,
        alpha=0.05,
        zorder=0,
    )
    ax.axvline(primary_beta, color=COLOR_REF_PRIMARY, linestyle=":", linewidth=0.75, zorder=1)
    ax.axvline(0, color=COLOR_REF_ZERO, linestyle="--", linewidth=0.6, zorder=1)
    ax.axhline(y_primary + 0.55, color="#EEEEEE", linewidth=0.8, zorder=0)

    _draw_dumbbell(
        ax,
        y_primary,
        float(primary.ci_low),
        primary_beta,
        float(primary.ci_high),
        primary=True,
    )

    for row, y in zip(sensitivity.itertuples(index=False), y_sens):
        ax.plot(
            [primary_beta, row.beta],
            [y_primary, y],
            color=COLOR_CONNECTOR,
            linewidth=0.7,
            zorder=1,
        )
        _draw_dumbbell(ax, y, float(row.ci_low), float(row.beta), float(row.ci_high), primary=False)
        label = DUMBELL_ROW_LABELS.get(row.model, row.model)
        ax.text(-0.008, y, label, ha="right", va="center", fontsize=9, color="#666666")

    ax.text(
        -0.008,
        y_primary,
        DUMBELL_ROW_LABELS.get(primary.model, primary.model),
        ha="right",
        va="center",
        fontsize=9.5,
        fontweight="semibold",
        color=COLOR_TEXT,
    )

    ax.set_xlim(-0.002, PLOT_X_MAX)
    ax.set_ylim(float(max(y_sens) + 0.55), -0.35)
    ax.set_xticks(np.arange(0, X_AXIS_END + 0.01, 0.05))
    ax.set_xlabel("Posterior exponent β (TD − ASD)")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["bottom"].set_color(COLOR_TEXT)
    ax.tick_params(axis="x", width=0.5, length=4)

    fig.subplots_adjust(left=0.26, right=0.50, top=0.90, bottom=0.14)
    _add_forest_table(fig, ax, rows_ordered, y_positions)
    return fig


def build_figure(
    out_dir: Path,
    *,
    dpi: int = 300,
    data_csv: Path | None = None,
    style: str = "forest",
) -> tuple[Path, Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_robustness_data(data_csv)

    if style == "forest":
        fig = plot_robustness_forest(df)
        stem = "posterior_robustness_forest"
    elif style == "slope":
        fig = plot_robustness_slope(df)
        stem = "posterior_robustness_slope"
    elif style == "dumbbell":
        fig = plot_robustness_dumbbell(df)
        stem = "posterior_robustness_dumbbell"
    else:
        raise ValueError(f"Unknown style: {style!r} (use 'dumbbell', 'slope', or 'forest')")

    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    svg_path = out_dir / f"{stem}.svg"
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return pdf_path, png_path, svg_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Posterior robustness figure (dumbbell, forest, or slope)."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "submission",
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=DEFAULT_ROBUSTNESS_CSV,
        help="fig2_posterior_robustness.csv (default: outputs/figure_source_data/).",
    )
    parser.add_argument(
        "--style",
        choices=["forest", "dumbbell", "slope"],
        default="forest",
        help="Plot type: forest (default), dumbbell, or slope graph.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    pdf_path, png_path, svg_path = build_figure(
        args.out_dir, dpi=args.dpi, data_csv=args.data_csv, style=args.style
    )
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")
    strict = load_robustness_data(args.data_csv)
    row = strict[strict["model"] == "Strict specparam-QC"].iloc[0]
    print(
        f"Strict specparam-QC: beta={row['beta']:.3f} "
        f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] p {row['p_label']}"
    )


if __name__ == "__main__":
    main()
