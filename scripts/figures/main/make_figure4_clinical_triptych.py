from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]  # repo root
OUTDIR = ROOT / "outputs/figures/figure4_clinical"
OUTDIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = ROOT / "data/figure_source/fig4_clinical_subjects.csv"
MODELS = OUTDIR / "panel_B_symptom_domain_partial_correlations_source.csv"

ASD = "#C25450"
ASD_DARK = "#A94442"
GREY = "#666666"
GREY_LIGHT = "#D9DDE3"
AX = "#222222"


def setup_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.65,
            "axes.labelsize": 10.6,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "xtick.major.width": 1.45,
            "ytick.major.width": 1.45,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
        }
    )


def residualize(df, target, covariates=("age_months", "IQ_total")):
    sub = df.dropna(subset=[target, *covariates]).copy()
    X = sub.loc[:, covariates].to_numpy(float)
    X = np.column_stack([np.ones(len(X)), X])
    y = sub[target].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return sub, y - X @ beta


def regression_ci(x, y, x_grid):
    slope, intercept, r, p, _ = stats.linregress(x, y)
    y_hat = intercept + slope * x_grid
    n = len(x)
    x_bar = np.mean(x)
    residual = y - (intercept + slope * x)
    s_err = np.sqrt(np.sum(residual**2) / (n - 2))
    s_xx = np.sum((x - x_bar) ** 2)
    tcrit = stats.t.ppf(0.975, n - 2)
    ci = tcrit * s_err * np.sqrt(1 / n + (x_grid - x_bar) ** 2 / s_xx)
    return y_hat, ci


def style_axis(ax):
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AX)
        ax.spines[spine].set_linewidth(1.65)
    ax.tick_params(direction="out", width=1.45, length=4.0, color=AX, labelcolor=AX)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def panel(ax, x, y, title, ylabel, stat, letter):
    significant = stat["FDR_q"] < 0.05
    line_color = ASD_DARK if significant else GREY
    fill_color = ASD if significant else GREY

    x_grid = np.linspace(-0.48, 0.48, 240)
    y_hat, ci = regression_ci(x, y, x_grid)

    ax.scatter(
        x,
        y,
        s=29,
        facecolor=ASD,
        edgecolor="white",
        linewidth=0.52,
        alpha=0.66,
        zorder=3,
    )
    ax.fill_between(
        x_grid,
        y_hat - ci,
        y_hat + ci,
        color=fill_color,
        alpha=0.13 if significant else 0.11,
        linewidth=0,
        zorder=1,
    )
    ax.plot(x_grid, y_hat, color=line_color, linewidth=1.55, zorder=4)

    ax.set_title(title, fontsize=11.0, fontweight="bold", pad=8)
    ax.set_xlabel("Posterior exponent residual", labelpad=6)
    ax.set_ylabel(ylabel, labelpad=7)
    ax.set_xlim(-0.50, 0.50)
    ax.set_xticks([-0.4, 0.0, 0.4])
    style_axis(ax)

    ax.text(-0.20, 1.08, letter, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    q_text = "n.s." if stat["FDR_q"] >= 0.05 else f"q = {stat['FDR_q']:.3f}"
    q_color = GREY if stat["FDR_q"] >= 0.05 else AX
    ax.text(
        0.97,
        0.96,
        f"partial r = {stat['partial_r']:.2f}\n{q_text}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=q_color,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.74, pad=1.4),
    )


def main():
    setup_style()
    subjects = pd.read_csv(SUBJECTS)
    models = pd.read_csv(MODELS).set_index("label")

    base_df, x_resid = residualize(subjects, "posterior_exponent")
    base = base_df[["subject_id"]].copy()
    base["posterior_exponent_residual_age_iq"] = x_resid

    outcomes = [
        ("Total", "ADOS_total", "ADOS total residual", "A"),
        ("Social Affect", "ADOS_SA", "ADOS Social Affect residual", "B"),
        ("RRB", "ADOS_RRB", "ADOS RRB residual", "C"),
    ]

    residual_tables = []
    fig, axes = plt.subplots(1, 3, figsize=(8.3, 3.05), dpi=300, sharex=True)

    ylims = {
        "Social Affect": (-6.2, 7.4),
        "Total": (-9.4, 9.4),
        "RRB": (-3.4, 3.75),
    }
    yticks = {
        "Social Affect": [-5, 0, 5],
        "Total": [-8, -4, 0, 4, 8],
        "RRB": [-2, 0, 2],
    }

    for ax, (label, col, ylabel, letter) in zip(axes, outcomes):
        y_df, y_resid = residualize(subjects, col)
        merged = base.merge(
            pd.DataFrame({"subject_id": y_df["subject_id"], f"{col}_residual_age_iq": y_resid}),
            on="subject_id",
            how="inner",
        )
        stat = models.loc[label]
        panel(
            ax,
            merged["posterior_exponent_residual_age_iq"].to_numpy(float),
            merged[f"{col}_residual_age_iq"].to_numpy(float),
            label,
            ylabel,
            stat,
            letter,
        )
        ax.set_ylim(*ylims[label])
        ax.set_yticks(yticks[label])
        residual_tables.append(merged)

    fig.subplots_adjust(left=0.085, right=0.992, bottom=0.245, top=0.865, wspace=0.43)

    stem = OUTDIR / "Figure4_clinical_relevance_triptych"
    for ext in ["png", "tiff"]:
        fig.savefig(f"{stem}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.04)
    for ext in ["pdf", "svg"]:
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.04)

    source = residual_tables[0]
    for tbl in residual_tables[1:]:
        source = source.merge(tbl, on=["subject_id", "posterior_exponent_residual_age_iq"], how="outer")
    source.to_csv(OUTDIR / "panel_ABC_clinical_residual_scatter_source.csv", index=False)

    print(f"{stem}.png")
    print(models.reset_index()[["label", "n", "partial_r", "ci_low", "ci_high", "FDR_q"]].to_string(index=False))


if __name__ == "__main__":
    main()
