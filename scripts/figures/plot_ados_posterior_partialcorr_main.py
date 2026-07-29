# Ideal release name: plot_ados_posterior_partialcorr_main.py
# Original path: scripts/64_plot_ados_posterior_partialcorr_main.py
# Note: Fig.4 ADOS partial-correlation panels
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
64_plot_ados_posterior_partialcorr_main.py
------------------------------------------
可视化主流程（ASD n=62）中：
ADOS_SA / ADOS_total 与 posterior_exponent 的 age+IQ 校正后关系。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

from src.stats_utils import fdr_correction


def residualize(y: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """返回 y 对协变量回归后的残差。"""
    x = sm.add_constant(cov, has_constant="add")
    model = sm.OLS(y, x).fit()
    return pd.Series(model.resid, index=y.index)


def main() -> None:
    root = PROJECT_ROOT
    feat_csv = root / "outputs" / "tables" / "resting_features_locked.csv"
    part_csv = root / "data" / "participants" / "participants.csv"
    out_dir = root / "outputs" / "figures" / "clinical"
    out_dir.mkdir(parents=True, exist_ok=True)

    feat = pd.read_csv(feat_csv)
    part = pd.read_csv(part_csv)

    feat["subject_id"] = feat["subject_id"].astype(str)
    part["subject_id"] = part["subject_id"].astype(str)

    df = feat.merge(
        part[["subject_id", "group", "age_months", "IQ_total", "ADOS_SA", "ADOS_total"]],
        on=["subject_id", "group"],
        how="left",
    )
    df = df[df["group"].astype(str).str.upper() == "ASD"].copy()

    targets = ["ADOS_SA", "ADOS_total"]
    plot_rows = []
    ann: dict[str, dict[str, float | int]] = {}
    raw_p: list[float] = []
    for y in targets:
        sub = df.dropna(subset=[y, "posterior_exponent", "age_months", "IQ_total"]).copy()
        sub["x_resid"] = residualize(sub["posterior_exponent"], sub[["age_months", "IQ_total"]])
        sub["y_resid"] = residualize(sub[y], sub[["age_months", "IQ_total"]])
        sub["outcome"] = y
        plot_rows.append(sub[["subject_id", "outcome", "x_resid", "y_resid"]])

        r, p = stats.pearsonr(sub["x_resid"], sub["y_resid"])
        ann[y] = {"r": float(r), "p": float(p), "n": int(len(sub))}
        raw_p.append(float(p))

    _, fdr_q = fdr_correction(np.array(raw_p, dtype=float))
    for i, y in enumerate(targets):
        ann[y]["pfdr"] = float(fdr_q[i])

    plot_df = pd.concat(plot_rows, axis=0, ignore_index=True)

    sns.set_style("white")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=False, sharey=False)
    title_map = {
        "ADOS_SA": "ADOS Social Affect (Age+IQ adjusted residuals)",
        "ADOS_total": "ADOS Total (Age+IQ adjusted residuals)",
    }

    for ax, y in zip(axes, targets):
        sub = plot_df[plot_df["outcome"] == y]
        sns.regplot(
            data=sub,
            x="x_resid",
            y="y_resid",
            ci=95,
            scatter_kws={"s": 40, "alpha": 0.72, "color": "#2a6fbb", "edgecolor": "none"},
            line_kws={"color": "#1b1b1b", "linewidth": 2.0},
            ax=ax,
        )
        sns.despine(ax=ax, top=True, right=True)
        ax.set_title(title_map[y], fontsize=11.5)
        ax.set_xlabel("Posterior exponent residual", fontsize=11)
        ax.set_ylabel(f"{y} residual", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)

        if y in ann:
            t = ann[y]
            txt = (
                f"Main ASD, n={t['n']}\n"
                f"partial r = {t['r']:.3f}\n"
                f"p = {t['p']:.4f}, FDR = {t['pfdr']:.4f}"
            )
            ax.text(
                0.98,
                0.98,
                txt,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
            )

    fig.suptitle("ASD Main Cohort: Posterior Exponent vs ADOS (Partial-Correlation View)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_png = out_dir / "ados_posterior_partialcorr_main_n62.png"
    out_pdf = out_dir / "ados_posterior_partialcorr_main_n62.pdf"
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
