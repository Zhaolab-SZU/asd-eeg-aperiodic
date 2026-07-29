from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[3]  # repo root
DATA = ROOT / "data" / "figure_source"
OUT = ROOT / "outputs" / "figures" / "figure3_development"
OUT.mkdir(parents=True, exist_ok=True)

ASD = "#C25450"
TD_FILL = "#B4C8E0"
TD_DOT = "#4A6FA5"
NEUTRAL = "#333333"
GRAY = "#666666"
LIGHT = "#E7EAEE"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8.7,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.labelsize": 9.2,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "axes.linewidth": 1.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def panel(ax, label: str):
    ax.text(-0.14, 1.05, label, transform=ax.transAxes,
            fontsize=15, fontweight="bold", ha="left", va="bottom",
            color="#1f1f1f")


def clean(ax):
    ax.spines["left"].set_linewidth(1.15)
    ax.spines["bottom"].set_linewidth(1.15)
    ax.tick_params(axis="both", direction="out", length=3.5, width=1.15)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")


def ols_ci(x, y, x_grid):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = beta[0] + beta[1] * x_grid
    resid = y - X @ beta
    df = len(x) - 2
    mse = np.sum(resid ** 2) / df
    xbar = x.mean()
    sxx = np.sum((x - xbar) ** 2)
    se = np.sqrt(mse * (1 / len(x) + (x_grid - xbar) ** 2 / sxx))
    ci = student_t.ppf(0.975, df) * se
    return yhat, yhat - ci, yhat + ci


def mean_ci(vals):
    vals = np.asarray(vals, float)
    n = len(vals)
    m = vals.mean()
    se = vals.std(ddof=1) / np.sqrt(n)
    ci = student_t.ppf(0.975, n - 1) * se
    return m, m - ci, m + ci


def add_sig(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            color=NEUTRAL, lw=1.0, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom",
            fontsize=11.5, fontweight="bold", color=ASD, clip_on=False)


def main():
    df = pd.read_csv(DATA / "fig3_development_subjects.csv").copy()
    df["age_years"] = df["age_months"] / 12
    asd = df[df.group.eq("ASD")].copy()

    fig = plt.figure(figsize=(7.35, 2.65))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1, 1, 1],
        left=0.075, right=0.985, bottom=0.22, top=0.86,
        wspace=0.34
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    for ax, lab in [(ax_a, "A"), (ax_b, "B"), (ax_c, "C")]:
        panel(ax, lab)

    # A. Age trajectories by group.
    rng = np.random.default_rng(17)
    for group, color, fill, zorder in [
        ("ASD", ASD, ASD, 3),
        ("TD", TD_DOT, TD_FILL, 2),
    ]:
        sub = df[df.group.eq(group)].dropna(subset=["age_years", "posterior_exponent"])
        ax_a.scatter(
            sub.age_years + rng.normal(0, 0.015, len(sub)),
            sub.posterior_exponent,
            s=18, color=color, alpha=0.62, edgecolor="white",
            linewidth=0.35, zorder=zorder
        )
        xg = np.linspace(3.3, 11.0, 120)
        yhat, lo, hi = ols_ci(sub.age_years, sub.posterior_exponent, xg)
        ax_a.fill_between(xg, lo, hi, color=fill, alpha=0.14, linewidth=0, zorder=1)
        ax_a.plot(xg, yhat, color=color, lw=1.75, zorder=4, label=group)
    ax_a.set_xlim(3.2, 11.2)
    ax_a.set_ylim(1.18, 2.52)
    ax_a.set_xlabel("Age (years)")
    ax_a.set_ylabel("Posterior aperiodic\nexponent")
    ax_a.legend(loc="upper right", fontsize=7.8, handlelength=1.4, borderaxespad=0.2)
    clean(ax_a)

    # B. TD-reference z-score deviation.
    ax_b.scatter(
        asd.age_years, asd.posterior_td_reference_z,
        s=18, color=ASD, alpha=0.66, edgecolor="white",
        linewidth=0.35, zorder=3
    )
    xg = np.linspace(3.3, 11.0, 120)
    yhat, lo, hi = ols_ci(asd.age_years, asd.posterior_td_reference_z, xg)
    ax_b.fill_between(xg, lo, hi, color=ASD, alpha=0.15, linewidth=0, zorder=1)
    ax_b.plot(xg, yhat, color=ASD, lw=1.75, zorder=4)
    ax_b.axhline(0, color="#8D93A6", lw=1.0, ls=(0, (4, 3)), zorder=0)
    r_b, p_b = pearsonr(
        asd.age_years.to_numpy(float),
        asd.posterior_td_reference_z.to_numpy(float)
    )
    p_txt = "p < .001" if p_b < 0.001 else f"p = {p_b:.3f}"
    ax_b.text(
        0.96, 0.88, rf"$r$ = {r_b:.2f}" + "\n" + p_txt,
        transform=ax_b.transAxes, ha="right", va="top",
        fontsize=8.0, color=NEUTRAL, fontweight="bold"
    )
    ax_b.set_xlim(3.2, 11.2)
    ax_b.set_ylim(-5.45, 3.2)
    ax_b.set_xlabel("Age (years)")
    ax_b.set_ylabel("Posterior exponent z-score\n(vs age-matched TD)")
    clean(ax_b)

    # C. ASD deviation by age subgroup.
    subgroups = [("younger_or_equal_72mo", "Younger ASD"), ("older", "Older ASD")]
    vals_by = [asd[asd.age_group_72mo.eq(k)].posterior_td_reference_z.dropna().to_numpy(float) for k, _ in subgroups]
    colors = ["#E69A98", ASD]
    rng = np.random.default_rng(23)
    for i, (vals, color) in enumerate(zip(vals_by, colors), start=1):
        parts = ax_c.violinplot(vals, positions=[i], widths=0.56,
                                showmeans=False, showmedians=False, showextrema=False)
        body = parts["bodies"][0]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.24)
        body.set_linewidth(0.8)
        ax_c.scatter(
            np.full(len(vals), i) + rng.normal(0, 0.035, len(vals)),
            vals, s=18, color=color, alpha=0.70,
            edgecolor="white", linewidth=0.35, zorder=3
        )
        m, lo, hi = mean_ci(vals)
        ax_c.plot([i, i], [lo, hi], color=NEUTRAL, lw=1.0, zorder=5)
        ax_c.plot([i - 0.055, i + 0.055], [lo, lo], color=NEUTRAL, lw=1.0, zorder=5)
        ax_c.plot([i - 0.055, i + 0.055], [hi, hi], color=NEUTRAL, lw=1.0, zorder=5)
        ax_c.scatter(i, m, s=28, color=color, edgecolor="white", linewidth=0.55, zorder=6)
    add_sig(ax_c, 1, 2, 2.68, 0.18, "*")
    ax_c.set_xlim(0.55, 2.45)
    ax_c.set_ylim(-5.45, 3.0)
    ax_c.set_xticks([1, 2], [label for _, label in subgroups])
    ax_c.set_ylabel("Posterior exponent z-score\n(vs age-matched TD)")
    clean(ax_c)

    stem = OUT / "Figure3_developmental_divergence_refined"
    for ext, kwargs in {
        "png": {"dpi": 300},
        "tiff": {"dpi": 600},
        "svg": {},
        "pdf": {},
    }.items():
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight",
                    facecolor="white", **kwargs)
    plt.close(fig)
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
