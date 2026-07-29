"""Reusable plot primitives for BP figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib import patches as mpatches

from src.bp_figures.style import (
    COLOR_ASD,
    COLOR_GRAY,
    COLOR_ROI,
    COLOR_TD,
    annotate_box,
    format_p,
)

try:
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False


def raincloud(
    ax: plt.Axes,
    data: pd.DataFrame,
    value_col: str,
    group_col: str = "group",
    ylabel: str = "",
    stats_text: str = "",
    groups: tuple[str, str] = ("TD", "ASD"),
) -> None:
    colors = {groups[0]: COLOR_TD, groups[1]: COLOR_ASD}
    positions = [0, 1]
    rng = np.random.default_rng(42)
    for i, g in enumerate(groups):
        vals = data.loc[data[group_col] == g, value_col].dropna().values
        if len(vals) == 0:
            continue
        color = colors[g]
        vp = ax.violinplot([vals], positions=[i], widths=0.55, showextrema=False, showmeans=False, showmedians=False)
        for body in vp["bodies"]:
            verts = body.get_paths()[0].vertices
            m = np.mean(verts[:, 0])
            verts[:, 0] = np.clip(verts[:, 0], -np.inf, m)
            body.set_facecolor(color)
            body.set_alpha(0.35)
            body.set_edgecolor("none")
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([i - 0.08, i + 0.08], [med, med], color="black", lw=1.2, zorder=4)
        ax.plot([i, i], [q1, q3], color="0.35", lw=1.0, zorder=3)
        ax.plot([i - 0.05, i + 0.05], [q1, q1], color="0.35", lw=1.0, zorder=3)
        ax.plot([i - 0.05, i + 0.05], [q3, q3], color="0.35", lw=1.0, zorder=3)
        jitter = rng.uniform(0.12, 0.32, size=len(vals))
        ax.scatter(i + jitter, vals, s=14, alpha=0.65, color=color, edgecolors="white", linewidths=0.25, zorder=5)
    ax.set_xticks(positions)
    ax.set_xticklabels(list(groups))
    ax.set_ylabel(ylabel)
    if stats_text:
        annotate_box(ax, stats_text)


def forest_h(
    ax: plt.Axes,
    labels: list[str],
    betas: list[float],
    ci_low: list[float] | None = None,
    ci_high: list[float] | None = None,
    colors: list[str] | None = None,
    xlabel: str = "β (TD − ASD)",
    ref_line: float | None = None,
    annotations: list[str] | None = None,
) -> None:
    y = np.arange(len(labels))
    ax.axvline(0, color=COLOR_GRAY, lw=0.8, zorder=0)
    if ref_line is not None:
        ax.axvline(ref_line, color=COLOR_GRAY, lw=0.8, ls="--", zorder=0)
    for i, b in enumerate(betas):
        c = colors[i] if colors else "#2A2A2A"
        if ci_low is not None and ci_high is not None:
            ax.errorbar(
                b, i, xerr=[[b - ci_low[i]], [ci_high[i] - b]],
                fmt="o", color=c, ecolor=c, elinewidth=1.1, capsize=3, ms=5.5, zorder=3,
            )
        else:
            ax.plot(b, i, "o", color=c, ms=5.5, zorder=3)
        if annotations and i < len(annotations) and annotations[i]:
            ax.text(
                (ci_high[i] if ci_high else b) + 0.008, i, annotations[i],
                va="center", ha="left", fontsize=6.5, color="#444444",
            )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()


def fit_line_ci(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        nan = np.full_like(x_grid, np.nan)
        return nan, nan, nan
    slope, intercept, _, _, _ = stats.linregress(x, y)
    pred = intercept + slope * x_grid
    n = len(x)
    y_hat = intercept + slope * x
    se = np.sqrt(np.sum((y - y_hat) ** 2) / max(n - 2, 1))
    x_mean = x.mean()
    denom = np.sum((x - x_mean) ** 2)
    se_line = se * np.sqrt(1 / n + (x_grid - x_mean) ** 2 / denom) if denom > 0 else np.zeros_like(x_grid)
    return pred, pred - 1.96 * se_line, pred + 1.96 * se_line


def residualize(y: pd.Series, covariates: pd.DataFrame) -> pd.Series:
    sub = pd.concat([y.rename("y"), covariates], axis=1).dropna()
    if len(sub) < 5 or not HAS_SM:
        out = y.copy()
        return out - out.mean()
    x = sm.add_constant(sub[covariates.columns])
    model = sm.OLS(sub["y"], x).fit()
    resid = sub["y"] - model.fittedvalues
    return resid.reindex(y.index)


def cohort_flow(ax: plt.Axes, stages: list[tuple]) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    y_positions = [7.8, 5.0, 2.2]
    for idx, (title, n, na, nt, exclude) in enumerate(stages):
        y = y_positions[idx]
        rect = mpatches.FancyBboxPatch(
            (1.2, y - 0.55), 7.6, 1.1, boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor="white", edgecolor="#888888", linewidth=0.9,
        )
        ax.add_patch(rect)
        sub = f"N = {n}; ASD = {na}, TD = {nt}"
        ax.text(5, y + 0.12, title, ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(5, y - 0.22, sub, ha="center", va="center", fontsize=7.5)
        if exclude:
            ax.add_patch(mpatches.FancyBboxPatch(
                (8.3, y - 0.42), 1.5, 0.84, boxstyle="round,pad=0.01",
                facecolor="#F5F5F5", edgecolor="#BBBBBB", linewidth=0.7,
            ))
            ax.text(9.05, y, exclude, ha="center", va="center", fontsize=6, color="#555555")
        if idx < len(stages) - 1:
            ax.annotate("", xy=(5, y_positions[idx + 1] + 0.55), xytext=(5, y - 0.55),
                        arrowprops=dict(arrowstyle="->", color="#666666", lw=1.0))


def specparam_exemplar(ax: plt.Axes) -> None:
    freqs = np.logspace(np.log10(1), np.log10(40), 250)
    aperiodic = 2.2e-3 * freqs ** (-2.0)
    alpha_peak = 0.35 * np.exp(-0.5 * ((np.log(freqs) - np.log(10)) / 0.22) ** 2)
    periodic = alpha_peak * aperiodic.mean() * 12
    raw = aperiodic + periodic
    ax.loglog(freqs, raw, color="#AAAAAA", lw=1.4, label="Raw spectrum")
    ax.loglog(freqs, aperiodic, color="#333333", lw=2.0, ls="--", label="Aperiodic fit")
    idx4 = np.argmin(np.abs(freqs - 4))
    idx10 = np.argmin(np.abs(freqs - 10))
    ax.annotate("Aperiodic\nexponent", xy=(4, aperiodic[idx4] * 1.2), xytext=(14, 3e-4),
                fontsize=7.5, arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))
    ax.annotate("Offset", xy=(28, aperiodic[-40] * 2.5), xytext=(20, 1.2e-3),
                fontsize=7.5, arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))
    ax.scatter([10], [raw[idx10]], s=28, c=COLOR_ROI, zorder=5)
    ax.text(10, raw[idx10] * 2.2, "α peak (~10 Hz)", ha="center", fontsize=7.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (log scale)")
    ax.legend(frameon=False, loc="upper right", fontsize=7)


def segment_bars(
    ax: plt.Axes,
    segments: list[str],
    td_vals: list[float],
    asd_vals: list[float],
    pvals: list[float | None],
    ylabel: str = "Aperiodic-ISC (r)",
    title: str = "",
) -> None:
    x = np.arange(len(segments))
    w = 0.34
    ax.bar(x - w / 2, td_vals, w, color=COLOR_TD, alpha=0.88, label="TD", edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, asd_vals, w, color=COLOR_ASD, alpha=0.88, label="ASD", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(segments, fontsize=7.5)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    if title:
        ax.set_title(title, fontsize=8.5, pad=4)
    ymax = np.nanmax(td_vals + asd_vals) if np.any(np.isfinite(td_vals + asd_vals)) else 0.15
    for i, p in enumerate(pvals):
        if p is not None and np.isfinite(p):
            ax.text(i, ymax * 1.06, format_p(p), ha="center", fontsize=6.5)
    ax.set_ylim(bottom=0, top=ymax * 1.22 if np.isfinite(ymax) else None)
