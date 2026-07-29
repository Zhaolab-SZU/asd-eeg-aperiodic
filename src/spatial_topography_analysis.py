"""Resting aperiodic exponent spatial-topography / mesoscale heterogeneity (exploratory)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols

from src.config import load_config
from src.extension_analysis import COL_ASD, COL_TD, load_main_qc_cohort
from src.io_utils import save_csv
from src.spectral_maturation_analysis import POSTERIOR_CORE, _posterior_channel_mean
from src.stats_utils import fdr_correction, spearman_correlation

logger = logging.getLogger(__name__)

POSTERIOR_CHANNELS = POSTERIOR_CORE
N_EGI_CHANNELS = 64
MIN_CHANNEL_RATIO = 0.80
GROUP_REF = "ASD"
GROUP_TERM = f"C(group, Treatment(reference='{GROUP_REF}'))[T.TD]"
MONTAGE_NAME = "GSN-HydroCel-64_1.0"

DISTANCE_BINS: list[tuple[str, float, float]] = [
    ("0-3", 0.0, 3.0),
    ("3-6", 3.0, 6.0),
    ("6-9", 6.0, 9.0),
    ("9-12", 9.0, 12.0),
    ("12-15", 12.0, 15.0),
    (">15", 15.0, np.inf),
]

BIN_TO_WIDE = {
    "0-3": "absdiff_0_3",
    "3-6": "absdiff_3_6",
    "6-9": "absdiff_6_9",
    "9-12": "absdiff_9_12",
    "12-15": "absdiff_12_15",
    ">15": "absdiff_gt15",
}

CLINICAL_VARS = [
    "ADOS_total",
    "ADOS_SA",
    "ADOS_RRB",
    "CARS_total",
    "language_score",
]

PLOT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _apply_plot_style() -> None:
    plt.rcParams.update(PLOT_RC)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_channel_distance_matrix(montage_name: str = MONTAGE_NAME) -> tuple[pd.DataFrame, pd.DataFrame]:
    """EGI-64 inter-electrode distances (cm). Returns square + long tables."""
    montage = mne.channels.make_standard_montage(montage_name)
    ch_names = sorted(
        [c for c in montage.ch_names if c.startswith("E")],
        key=lambda x: int(x[1:]),
    )
    pos = montage.get_positions()["ch_pos"]
    n = len(ch_names)
    dist_m = np.zeros((n, n))
    for i, ci in enumerate(ch_names):
        pi = np.array(pos[ci])
        for j, cj in enumerate(ch_names):
            pj = np.array(pos[cj])
            dist_m[i, j] = np.linalg.norm(pi - pj)

    dist_cm = dist_m * 100.0
    square = pd.DataFrame(dist_cm, index=ch_names, columns=ch_names)
    square.index.name = "channel_i"
    rows = []
    for i, ci in enumerate(ch_names):
        for j, cj in enumerate(ch_names):
            if j <= i:
                continue
            rows.append({"channel_i": ci, "channel_j": cj, "distance_cm": dist_cm[i, j]})
    long_df = pd.DataFrame(rows)
    return square, long_df


def _assign_distance_bin(distance_cm: float) -> str | None:
    for label, lo, hi in DISTANCE_BINS:
        if lo <= distance_cm < hi:
            return label
    return None


def load_channel_exponent_cohort(cfg: dict[str, Any], deriv: Path) -> pd.DataFrame:
    """Primary spectral cohort: channel-level QC-valid exponents + demographics."""
    ch_path = deriv / "specparam" / "specparam_channel_results_qc.csv"
    if not ch_path.exists():
        raise FileNotFoundError(f"缺少通道 specparam 文件: {ch_path}")

    ch_df = pd.read_csv(ch_path)
    ch_df["subject_id"] = ch_df["subject_id"].astype(str)
    if "fit_valid" in ch_df.columns:
        ch_df = ch_df[ch_df["fit_valid"]].copy()

    participants = load_main_qc_cohort(cfg, deriv)
    participants["subject_id"] = participants["subject_id"].astype(str)

    sp_qc = deriv / "specparam" / "specparam_qc_summary_subject.csv"
    if sp_qc.exists():
        sp = pd.read_csv(sp_qc)
        sp["subject_id"] = sp["subject_id"].astype(str)
        if "mean_r_squared" in sp.columns:
            participants = participants.merge(
                sp[["subject_id", "mean_r_squared"]],
                on="subject_id",
                how="left",
            )

    demo_cols = [
        "subject_id",
        "group",
        "age_months",
        "sex",
        "IQ_total",
        "usable_epochs",
        "ADOS_total",
        "ADOS_SA",
        "ADOS_RRB",
        "CARS_total",
        "language_score",
    ]
    demo = participants[[c for c in demo_cols if c in participants.columns]].copy()
    demo = demo.rename(columns={"age_months": "age", "IQ_total": "IQ"})

    ch_df = ch_df.merge(demo, on=["subject_id", "group"], how="inner")
    ch_df = ch_df.rename(
        columns={
            "aperiodic_exponent": "exponent",
            "mean_r_squared": "mean_R2",
        }
    )
    if "mean_R2" not in ch_df.columns and "r_squared" in ch_df.columns:
        sub_r2 = (
            ch_df.groupby("subject_id", as_index=False)["r_squared"]
            .mean()
            .rename(columns={"r_squared": "mean_R2"})
        )
        ch_df = ch_df.drop(columns=["mean_R2"], errors="ignore").merge(sub_r2, on="subject_id", how="left")

    return ch_df


def build_subject_topography_qc(
    ch_df: pd.DataFrame,
    posterior_exp: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-subject topography QC (>=80% valid channels)."""
    rows: list[dict[str, Any]] = []
    for sid, sub in ch_df.groupby("subject_id"):
        exp_by_ch = sub.dropna(subset=["exponent"]).drop_duplicates("channel")
        n_valid = len(exp_by_ch)
        pct = 100.0 * n_valid / N_EGI_CHANNELS
        if pct < MIN_CHANNEL_RATIO * 100:
            continue
        meta = sub.iloc[0]
        row: dict[str, Any] = {
            "subject_id": str(sid),
            "group": meta["group"],
            "age": meta.get("age", np.nan),
            "sex": meta.get("sex", np.nan),
            "IQ": meta.get("IQ", np.nan),
            "usable_epochs": meta.get("usable_epochs", np.nan),
            "mean_R2": meta.get("mean_R2", np.nan),
            "n_valid_channels": n_valid,
            "percent_valid_channels": pct,
            "global_exponent_mean": float(exp_by_ch["exponent"].mean()),
            "global_exponent_sd": float(exp_by_ch["exponent"].std(ddof=1)) if n_valid > 1 else np.nan,
        }
        if posterior_exp is not None and str(sid) in posterior_exp["subject_id"].astype(str).values:
            pe = posterior_exp.loc[posterior_exp["subject_id"].astype(str) == str(sid), "posterior_exponent"]
            row["posterior_exponent"] = float(pe.iloc[0]) if len(pe) else np.nan
        else:
            row["posterior_exponent"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute_subject_dissimilarity(
    ch_df: pd.DataFrame,
    dist_long: pd.DataFrame,
    subject_qc: pd.DataFrame,
    *,
    channel_filter: set[str] | None = None,
    exclude_channels: set[str] | None = None,
    posterior_pairs_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Subject-level mean |exp_i - exp_j| by distance bin.

    channel_filter: if set, only these channels used.
    exclude_channels: drop these channels before pairing.
    posterior_pairs_only: keep pairs with >=1 channel in POSTERIOR_CHANNELS.
    """
    dist_map = {
        (r.channel_i, r.channel_j): r.distance_cm
        for r in dist_long.itertuples(index=False)
    }
    # symmetrize
    for (a, b), d in list(dist_map.items()):
        dist_map[(b, a)] = d

    included = set(subject_qc["subject_id"].astype(str))
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []

    for sid, sub in ch_df.groupby("subject_id"):
        sid = str(sid)
        if sid not in included:
            continue
        exp_map = (
            sub.dropna(subset=["exponent"])
            .drop_duplicates("channel")
            .set_index("channel")["exponent"]
            .to_dict()
        )
        channels = list(exp_map.keys())
        if channel_filter is not None:
            channels = [c for c in channels if c in channel_filter]
        if exclude_channels:
            channels = [c for c in channels if c not in exclude_channels]

        bin_sums: dict[str, list[float]] = {label: [] for label, _, _ in DISTANCE_BINS}
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                ci, cj = channels[i], channels[j]
                if posterior_pairs_only and not (
                    ci in POSTERIOR_CHANNELS or cj in POSTERIOR_CHANNELS
                ):
                    continue
                d_cm = dist_map.get((ci, cj))
                if d_cm is None:
                    continue
                bin_label = _assign_distance_bin(d_cm)
                if bin_label is None:
                    continue
                absdiff = abs(exp_map[ci] - exp_map[cj])
                bin_sums[bin_label].append(absdiff)

        if not any(bin_sums.values()):
            continue
        meta = subject_qc.loc[subject_qc["subject_id"].astype(str) == sid].iloc[0]
        wide: dict[str, Any] = {
            "subject_id": sid,
            "group": meta["group"],
            "age": meta["age"],
            "sex": meta["sex"],
            "IQ": meta["IQ"],
            "usable_epochs": meta["usable_epochs"],
            "mean_R2": meta["mean_R2"],
            "global_exponent_mean": meta["global_exponent_mean"],
            "global_exponent_sd": meta["global_exponent_sd"],
        }
        for label, _, _ in DISTANCE_BINS:
            vals = bin_sums[label]
            mean_v = float(np.mean(vals)) if vals else np.nan
            n_pairs = len(vals)
            long_rows.append(
                {
                    **{k: wide[k] for k in ("subject_id", "group", "age", "sex", "IQ", "usable_epochs", "mean_R2")},
                    "distance_bin": label,
                    "mean_absdiff": mean_v,
                    "n_pairs": n_pairs,
                }
            )
            col = BIN_TO_WIDE[label]
            wide[col] = mean_v
        wide["mesoscale_absdiff_6_9"] = wide.get("absdiff_6_9", np.nan)
        wide_rows.append(wide)

    return pd.DataFrame(long_rows), pd.DataFrame(wide_rows)


def _cohen_f2_group(full_r2: float, reduced_r2: float) -> float:
    if np.isnan(full_r2) or np.isnan(reduced_r2) or full_r2 >= 1.0:
        return np.nan
    delta = full_r2 - reduced_r2
    denom = 1.0 - full_r2
    return float(delta / denom) if denom > 0 else np.nan


def fit_group_model_hc3(
    df: pd.DataFrame,
    outcome: str,
    extra_covariates: list[str] | None = None,
) -> dict[str, Any]:
    """OLS + HC3; ASD reference; report TD − ASD on group term."""
    cov = ["age", "C(sex)", "IQ", "usable_epochs", "mean_R2"]
    if extra_covariates:
        cov = extra_covariates + cov
    cov_str = " + ".join(cov)
    formula = f"{outcome} ~ C(group, Treatment(reference='{GROUP_REF}')) + {cov_str}"
    sub = df.dropna(subset=[outcome, "group", "age", "sex", "IQ", "usable_epochs", "mean_R2"])
    if len(sub) < 15:
        return {"formula": formula, "n": len(sub), "error": "insufficient_n"}

    try:
        res = ols(formula, data=sub).fit(cov_type="HC3")
    except Exception as exc:
        return {"formula": formula, "n": len(sub), "error": str(exc)}

    if GROUP_TERM not in res.params.index:
        return {"formula": formula, "n": len(sub), "error": f"missing {GROUP_TERM}"}

    beta = float(res.params[GROUP_TERM])
    se = float(res.bse[GROUP_TERM])
    ci = res.conf_int().loc[GROUP_TERM]
    p = float(res.pvalues[GROUP_TERM])

    reduced_formula = f"{outcome} ~ {cov_str}"
    try:
        res_red = ols(reduced_formula, data=sub).fit()
        cohen_f2 = _cohen_f2_group(float(res.rsquared), float(res_red.rsquared))
        partial_r2 = cohen_f2  # semipartial R² equivalent for single added predictor
    except Exception:
        cohen_f2 = np.nan
        partial_r2 = np.nan

    return {
        "formula": formula,
        "outcome": outcome,
        "beta_TD_minus_ASD": beta,
        "SE_HC3": se,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p": p,
        "n": int(res.nobs),
        "r_squared": float(res.rsquared),
        "partial_R2": partial_r2,
        "Cohen_f2": cohen_f2,
    }


def fit_age_interaction_model(df: pd.DataFrame, outcome: str = "mesoscale_absdiff_6_9") -> dict[str, Any]:
    formula = (
        f"{outcome} ~ C(group, Treatment(reference='{GROUP_REF}')) * age "
        f"+ C(sex) + IQ + usable_epochs + mean_R2"
    )
    sub = df.dropna(subset=[outcome, "group", "age", "sex", "IQ", "usable_epochs", "mean_R2"])
    if len(sub) < 20:
        return {"formula": formula, "n": len(sub), "error": "insufficient_n"}

    res = ols(formula, data=sub).fit(cov_type="HC3")
    int_term = f"C(group, Treatment(reference='{GROUP_REF}'))[T.TD]:age"
    rows = {
        "formula": formula,
        "outcome": outcome,
        "n": int(res.nobs),
        "ASD_age_slope": float(res.params.get("age", np.nan)),
        "ASD_age_slope_se": float(res.bse.get("age", np.nan)),
    }
    if int_term in res.params.index:
        ci = res.conf_int().loc[int_term]
        rows.update(
            {
                "group_x_age_beta": float(res.params[int_term]),
                "group_x_age_SE": float(res.bse[int_term]),
                "group_x_age_ci_low": float(ci[0]),
                "group_x_age_ci_high": float(ci[1]),
                "group_x_age_p": float(res.pvalues[int_term]),
                "TD_age_slope": float(res.params.get("age", 0) + res.params[int_term]),
            }
        )
    else:
        rows["error"] = f"missing {int_term}"
    return rows


def fit_distancebin_group_models(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bin_label in [b[0] for b in DISTANCE_BINS]:
        sub = long_df[long_df["distance_bin"] == bin_label].copy()
        if sub.empty:
            continue
        res = fit_group_model_hc3(sub, "mean_absdiff")
        rows.append({"distance_bin": bin_label, **res})
    out = pd.DataFrame(rows)
    if "p" in out.columns:
        reject, q = fdr_correction(out["p"].values)
        out["FDR_q"] = q
        out["FDR_significant"] = reject
    return out


def partial_correlation_residual(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    cov_cols: list[str],
) -> dict[str, Any]:
    sub = df.dropna(subset=[y_col, x_col] + cov_cols)
    if len(sub) < 8:
        return {"n": len(sub), "partial_r": np.nan, "raw_p": np.nan}

    cov_str = " + ".join(cov_cols)
    ry = ols(f"{y_col} ~ {cov_str}", data=sub).fit().resid
    rx = ols(f"{x_col} ~ {cov_str}", data=sub).fit().resid
    r, p = stats.pearsonr(ry, rx)
    return {"n": int(len(sub)), "partial_r": float(r), "raw_p": float(p)}


def fit_posterior_mesoscale_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "posterior_exponent" not in df.columns:
        return pd.DataFrame(rows)

    models = [
        (
            "mesoscale_on_posterior",
            "mesoscale_absdiff_6_9 ~ posterior_exponent + age + C(sex) + IQ + "
            f"C(group, Treatment(reference='{GROUP_REF}'))",
            ["posterior_exponent", "age", "sex", "IQ", "group"],
        ),
        (
            "posterior_on_mesoscale",
            "posterior_exponent ~ mesoscale_absdiff_6_9 + "
            f"C(group, Treatment(reference='{GROUP_REF}')) + age + C(sex) + IQ "
            "+ usable_epochs + mean_R2",
            [
                "mesoscale_absdiff_6_9",
                "group",
                "age",
                "sex",
                "IQ",
                "usable_epochs",
                "mean_R2",
            ],
        ),
    ]
    for model_name, formula, cols in models:
        sub = df.dropna(subset=cols + ["mesoscale_absdiff_6_9", "posterior_exponent"])
        if len(sub) < 15:
            continue
        try:
            res = ols(formula, data=sub).fit(cov_type="HC3")
        except Exception as exc:
            rows.append({"model": model_name, "error": str(exc), "n": len(sub)})
            continue
        focus = (
            "posterior_exponent"
            if model_name == "mesoscale_on_posterior"
            else "mesoscale_absdiff_6_9"
        )
        if focus not in res.params.index:
            gterm = GROUP_TERM
            terms = [focus, gterm] if gterm in res.params.index else [focus]
        else:
            terms = [focus]
        for term in terms:
            if term not in res.params.index:
                continue
            ci = res.conf_int().loc[term]
            rows.append(
                {
                    "model": model_name,
                    "term": term,
                    "beta": float(res.params[term]),
                    "SE_HC3": float(res.bse[term]),
                    "ci_low": float(ci[0]),
                    "ci_high": float(ci[1]),
                    "p": float(res.pvalues[term]),
                    "n": int(res.nobs),
                    "r_squared": float(res.rsquared),
                }
            )
    return pd.DataFrame(rows)


def run_clinical_associations(df: pd.DataFrame) -> pd.DataFrame:
    asd = df[df["group"] == "ASD"].copy()
    rows = []
    for clinical in CLINICAL_VARS:
        if clinical not in asd.columns:
            continue
        if pd.to_numeric(asd[clinical], errors="coerce").notna().sum() < 8:
            continue
        res = partial_correlation_residual(
            asd,
            clinical,
            "mesoscale_absdiff_6_9",
            ["age", "IQ"],
        )
        rows.append(
            {
                "clinical_score": clinical,
                "EEG_metric": "mesoscale_absdiff_6_9",
                **res,
            }
        )
    out = pd.DataFrame(rows)
    if len(out) and "raw_p" in out.columns:
        _, q = fdr_correction(out["raw_p"].values)
        out["FDR_q"] = q
    return out


# ---- Figures ----


def plot_distancebin_group_effect(models_df: pd.DataFrame, out_path: Path) -> None:
    _apply_plot_style()
    sub = models_df.dropna(subset=["beta_TD_minus_ASD", "ci_low", "ci_high"]).copy()
    if sub.empty:
        return
    order = [b[0] for b in DISTANCE_BINS]
    sub["distance_bin"] = pd.Categorical(sub["distance_bin"], categories=order, ordered=True)
    sub = sub.sort_values("distance_bin")
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        x,
        sub["beta_TD_minus_ASD"],
        yerr=[
            sub["beta_TD_minus_ASD"] - sub["ci_low"],
            sub["ci_high"] - sub["beta_TD_minus_ASD"],
        ],
        fmt="o-",
        color=COL_TD,
        capsize=4,
        lw=1.5,
    )
    ax.axhline(0, color=COL_ASD, ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["distance_bin"].astype(str))
    if "6-9" in order:
        idx = order.index("6-9")
        ax.axvspan(idx - 0.4, idx + 0.4, alpha=0.12, color="gray", label="6–9 cm mesoscale")
    ax.set_xlabel("Electrode distance bin (cm)")
    ax.set_ylabel("TD − ASD β (mean |Δexponent|)")
    ax.set_title("Group effect on topographic heterogeneity by distance")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_group_mean_dissimilarity_curve(long_df: pd.DataFrame, out_path: Path) -> None:
    _apply_plot_style()
    order = [b[0] for b in DISTANCE_BINS]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for grp, color in [("ASD", COL_ASD), ("TD", COL_TD)]:
        sub = long_df[long_df["group"] == grp]
        means = sub.groupby("distance_bin")["mean_absdiff"].mean().reindex(order)
        sem = sub.groupby("distance_bin")["mean_absdiff"].sem().reindex(order)
        x = np.arange(len(order))
        ax.errorbar(x, means, yerr=sem, fmt="o-", label=grp, color=color, capsize=3, lw=1.5)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    meso_i = order.index("6-9")
    ax.axvspan(meso_i - 0.4, meso_i + 0.4, alpha=0.12, color="gray")
    ax.set_xlabel("Electrode distance bin (cm)")
    ax.set_ylabel("Mean |Δexponent| (subject-level)")
    ax.set_title("Group mean topographic dissimilarity curve")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_mesoscale_age_interaction(wide_df: pd.DataFrame, interaction: dict[str, Any], out_path: Path) -> None:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for grp, color in [("ASD", COL_ASD), ("TD", COL_TD)]:
        sub = wide_df[wide_df["group"] == grp].dropna(subset=["age", "mesoscale_absdiff_6_9"])
        ax.scatter(sub["age"], sub["mesoscale_absdiff_6_9"], alpha=0.55, s=28, color=color, label=f"{grp} raw")
        if len(sub) >= 3:
            z = np.polyfit(sub["age"], sub["mesoscale_absdiff_6_9"], 1)
            xline = np.linspace(sub["age"].min(), sub["age"].max(), 50)
            ax.plot(xline, np.poly1d(z)(xline), color=color, lw=2, ls="--")
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Mesoscale heterogeneity (6–9 cm mean |Δexponent|)")
    if "group_x_age_p" in interaction:
        ax.set_title(
            f"Mesoscale heterogeneity × age (interaction p={interaction.get('group_x_age_p', np.nan):.3f})"
        )
    else:
        ax.set_title("Mesoscale heterogeneity × age")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_posterior_vs_mesoscale(wide_df: pd.DataFrame, out_path: Path) -> None:
    _apply_plot_style()
    sub = wide_df.dropna(subset=["posterior_exponent", "mesoscale_absdiff_6_9"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for grp, color in [("ASD", COL_ASD), ("TD", COL_TD)]:
        g = sub[sub["group"] == grp]
        ax.scatter(
            g["mesoscale_absdiff_6_9"],
            g["posterior_exponent"],
            alpha=0.6,
            s=30,
            color=color,
            label=grp,
        )
    ax.set_xlabel("Mesoscale heterogeneity (6–9 cm)")
    ax.set_ylabel("Posterior exponent (E33/E36/E37/E38)")
    ax.set_title("Posterior mean vs mesoscale topography heterogeneity")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, out_path)


def _auto_conclusion(mesoscale_model: dict[str, Any]) -> str:
    beta = mesoscale_model.get("beta_TD_minus_ASD", np.nan)
    p = mesoscale_model.get("p", np.nan)
    if np.isnan(beta) or np.isnan(p):
        return (
            "本探索分析未能完成 mesoscale 组间模型，请检查样本量与协变量完整性。"
        )
    if beta < 0 and p < 0.05:
        return (
            "结果提示 ASD 儿童在 6–9 cm mesoscale 上表现出更高的 aperiodic exponent topographic "
            "heterogeneity。该发现与主研究 posterior exponent 异常方向一致，可作为补充探索证据，"
            "但不替代 posterior exponent 主结果。"
        )
    if abs(beta) < 1e-6 or p >= 0.05:
        return (
            "本探索分析未发现 ASD 儿童在 6–9 cm mesoscale 上存在稳定的 aperiodic exponent "
            "topographic heterogeneity。该结果不否定主研究 posterior exponent 发现，但提示当前样本中 "
            "posterior mean effect 未明显扩展为 whole-head mesoscale topography effect。"
        )
    return (
        "本探索分析未支持预期的 ASD mesoscale topographic heterogeneity 增强，结果方向与假设不一致。"
        "该分析应仅作为探索性补充，不纳入主结论。"
    )


def write_report_zh(
    out_path: Path,
    *,
    n_ch_records: int,
    n_spatial: int,
    n_asd: int,
    n_td: int,
    mesoscale_model: dict[str, Any],
    distbin_models: pd.DataFrame,
    age_model: dict[str, Any],
    posterior_rel: pd.DataFrame,
    clinical: pd.DataFrame,
    sensitivity: pd.DataFrame,
    montage_note: str,
) -> None:
    conclusion = _auto_conclusion(mesoscale_model)
    beta = mesoscale_model.get("beta_TD_minus_ASD", np.nan)
    p = mesoscale_model.get("p", np.nan)
    n = mesoscale_model.get("n", "—")

    dist_lines = []
    if len(distbin_models):
        for _, r in distbin_models.iterrows():
            dist_lines.append(
                f"- **{r['distance_bin']} cm**：β(TD−ASD)={r.get('beta_TD_minus_ASD', np.nan):.4f}, "
                f"p={r.get('p', np.nan):.4f}, FDR q={r.get('FDR_q', np.nan):.4f}"
            )

    clin_lines = []
    if len(clinical):
        for _, r in clinical.iterrows():
            clin_lines.append(
                f"- **{r['clinical_score']}**：partial r={r.get('partial_r', np.nan):.3f}, "
                f"p={r.get('raw_p', np.nan):.4f}, n={r.get('n', '—')}"
            )
    else:
        clin_lines.append("- 临床变量有效样本不足或未合并，未报告稳定关联。")

    sens_lines = []
    for _, r in sensitivity.iterrows():
        sens_lines.append(
            f"- **{r['analysis']}**：β={r.get('beta_TD_minus_ASD', np.nan):.4f}, p={r.get('p', np.nan):.4f}, n={r.get('n', '—')}"
        )

    post_lines = []
    if len(posterior_rel):
        for _, r in posterior_rel.iterrows():
            post_lines.append(
                f"- {r['model']} / {r.get('term', '')}: β={r.get('beta', np.nan):.4f}, p={r.get('p', np.nan):.4f}"
            )
    else:
        post_lines.append("- 未拟合 posterior–mesoscale 联合模型。")

    text = f"""# Resting EEG aperiodic exponent spatial-topography exploratory analysis

## 分析目的

本补充分析受文献 *Cortical neural state topography reveals mesoscale heterogeneity in autism* 启发，用于探索 ASD 儿童静息态 aperiodic exponent 是否除后部均值降低外，还存在全头 exponent 空间拓扑的 mesoscale 组织异常。定位为 **exploratory / supplementary spatial-topography analysis**，不替代主分析 posterior exponent、age×group、TD 规范模型或 ADOS 主线。

## 输入数据与 QC

- **通道级来源**：`derivatives/specparam/specparam_channel_results_qc.csv`（`fit_valid==True`）。
- **分析队列**：与主研究一致（`participants_analysis` + specparam 被试 QC + ROI 可用）。
- **空间拓扑纳入规则**：每被试至少 **80%**（≥52/64）QC 有效通道；不对缺失通道插值。
- **通道记录数（QC 后）**：{n_ch_records}
- **纳入空间拓扑分析被试**：{n_spatial}（ASD {n_asd}，TD {n_td}）

## 方法

- **电极坐标**：MNE 标准 montage `{MONTAGE_NAME}`。{montage_note}
- **距离矩阵**：电极对欧氏距离（头皮坐标，单位 **cm**），保存于 `channel_distance_matrix.csv`。
- **异质性指标**：对被试所有有效通道对计算 `|exponent_i − exponent_j|`，按距离 bin（0–3、3–6、**6–9**、9–12、12–15、>15 cm）取平均。
- **预设 mesoscale bin**：**6–9 cm**（`mesoscale_absdiff_6_9`）。
- **组间模型**：`outcome ~ group + age + sex + IQ + usable_epochs + mean_R2`，**OLS + HC3**，**ASD 为参照**，报告 **TD − ASD** β。对 absdiff 结局，**β < 0 表示 ASD 异质性更高**。
- **FDR**：对各 distance bin 的 group 效应进行 Benjamini–Hochberg 校正。

## 主要结果（6–9 cm mesoscale）

- **模型**：`mesoscale_absdiff_6_9 ~ group + age + sex + IQ + usable_epochs + mean_R2`
- **TD − ASD β**：{beta:.4f}
- **HC3 SE**：{mesoscale_model.get('SE_HC3', np.nan):.4f}
- **95% CI**：[{mesoscale_model.get('ci_low', np.nan):.4f}, {mesoscale_model.get('ci_high', np.nan):.4f}]
- **p**：{p:.4f}
- **n**：{n}
- **R²**：{mesoscale_model.get('r_squared', np.nan):.4f}；**Cohen f²（group）**：{mesoscale_model.get('Cohen_f2', np.nan):.4f}

## 距离尺度曲线

{chr(10).join(dist_lines) if dist_lines else '- 无有效 bin 模型结果。'}

图：`distancebin_group_effect_curve.png`、`group_mean_dissimilarity_curve.png`。

## 发育交互

- **模型**：`mesoscale_absdiff_6_9 ~ group × age + sex + IQ + usable_epochs + mean_R2`
- **group × age β**：{age_model.get('group_x_age_beta', np.nan):.4f}（p={age_model.get('group_x_age_p', np.nan):.4f}）
- **ASD 年龄斜率**：{age_model.get('ASD_age_slope', np.nan):.4f}
- **TD 年龄斜率**：{age_model.get('TD_age_slope', np.nan):.4f}

图：`mesoscale_age_interaction_plot.png`。

## 与 posterior exponent 的关系

{chr(10).join(post_lines)}

图：`posterior_vs_mesoscale_absdiff.png`。

## 临床关联（ASD-only）

对 ADOS/CARS/language 等变量，在控制 age、IQ 后计算与 `mesoscale_absdiff_6_9` 的偏相关。

{chr(10).join(clin_lines)}

## Posterior cluster sensitivity

对三种通道集分别重算 mesoscale 指标并拟合同一 group 模型：

{chr(10).join(sens_lines)}

## 解释与限制

- 本分析为 **exploratory supplementary**，非主结果、非外部验证。
- `mean_absdiff` 仅为 topographic heterogeneity 的 **proxy**，不是直接 E/I 或皮层神经状态测量。
- 头皮电极距离 ≠ 皮层测地距离；64 导空间分辨率有限。
- 阴性结果 **不否定** 主分析 posterior exponent 发现；阳性结果亦 **不宜** 过度推断为皮层 mesoscale 机制。

## 最终结论

{conclusion}
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def run_spatial_topography_analysis(cfg: dict[str, Any] | None = None) -> dict[str, Path]:
    cfg = cfg or load_config()
    deriv = Path(cfg["paths"]["derivatives_root"])
    out_dir = Path(cfg["paths"]["outputs_root"]) / "spatial_topography"
    out_dir.mkdir(parents=True, exist_ok=True)

    square_dist, long_dist = build_channel_distance_matrix(
        cfg.get("eeg", {}).get("montage", MONTAGE_NAME)
    )
    save_csv(square_dist.reset_index(), out_dir / "channel_distance_matrix.csv")
    save_csv(long_dist, out_dir / "channel_distance_matrix_long.csv")

    ch_df = load_channel_exponent_cohort(cfg, deriv)
    raw_ch = pd.read_csv(deriv / "specparam" / "specparam_channel_results_qc.csv")
    if "fit_valid" in raw_ch.columns:
        raw_valid = raw_ch[raw_ch["fit_valid"]]
    else:
        raw_valid = raw_ch
    post_exp = _posterior_channel_mean(raw_valid, "aperiodic_exponent")
    post_exp = post_exp.rename(columns={"aperiodic_exponent": "posterior_exponent"})

    subject_qc = build_subject_topography_qc(ch_df, post_exp)
    save_csv(subject_qc, out_dir / "subject_topography_qc.csv")

    long_diss, wide_diss = compute_subject_dissimilarity(
        ch_df, long_dist, subject_qc
    )
    wide_diss = wide_diss.merge(
        subject_qc[["subject_id", "posterior_exponent"]],
        on="subject_id",
        how="left",
    )
    participants = load_main_qc_cohort(cfg, deriv)
    participants["subject_id"] = participants["subject_id"].astype(str)
    clin_cols = ["subject_id"] + [c for c in CLINICAL_VARS if c in participants.columns]
    wide_diss = wide_diss.merge(participants[clin_cols], on="subject_id", how="left")
    save_csv(long_diss, out_dir / "subject_distancebin_dissimilarity.csv")
    save_csv(wide_diss, out_dir / "subject_mesoscale_metrics.csv")

    mesoscale_model = fit_group_model_hc3(wide_diss, "mesoscale_absdiff_6_9")
    save_csv(pd.DataFrame([mesoscale_model]), out_dir / "mesoscale_group_model.csv")

    distbin_models = fit_distancebin_group_models(long_diss)
    save_csv(distbin_models, out_dir / "distancebin_group_models.csv")

    age_model = fit_age_interaction_model(wide_diss)
    save_csv(pd.DataFrame([age_model]), out_dir / "mesoscale_age_interaction_model.csv")

    posterior_rel = fit_posterior_mesoscale_models(wide_diss)
    save_csv(posterior_rel, out_dir / "posterior_mesoscale_relationship.csv")

    clinical = run_clinical_associations(wide_diss)
    save_csv(clinical, out_dir / "mesoscale_clinical_associations.csv")

    # Sensitivity
    sens_rows = []
    variants = [
        ("A_all_channels", None, None, False),
        ("B_exclude_posterior_cluster", None, set(POSTERIOR_CHANNELS), False),
        ("C_posterior_related_pairs_only", None, None, True),
    ]
    for name, ch_filter, excl, post_only in variants:
        _, wide_v = compute_subject_dissimilarity(
            ch_df,
            long_dist,
            subject_qc,
            channel_filter=ch_filter,
            exclude_channels=excl,
            posterior_pairs_only=post_only,
        )
        if wide_v.empty:
            sens_rows.append({"analysis": name, "error": "no_subjects"})
            continue
        res = fit_group_model_hc3(wide_v, "mesoscale_absdiff_6_9")
        res["analysis"] = name
        sens_rows.append(res)
    sensitivity = pd.DataFrame(sens_rows)
    save_csv(sensitivity, out_dir / "posterior_cluster_sensitivity.csv")

    # Figures
    plot_distancebin_group_effect(distbin_models, out_dir / "distancebin_group_effect_curve.png")
    plot_group_mean_dissimilarity_curve(long_diss, out_dir / "group_mean_dissimilarity_curve.png")
    plot_mesoscale_age_interaction(wide_diss, age_model, out_dir / "mesoscale_age_interaction_plot.png")
    plot_posterior_vs_mesoscale(wide_diss, out_dir / "posterior_vs_mesoscale_absdiff.png")

    n_asd = int((wide_diss["group"] == "ASD").sum())
    n_td = int((wide_diss["group"] == "TD").sum())
    montage_note = (
        "坐标来自 MNE 内置 GSN-HydroCel-64_1.0 标准布局（单位由 m 转为 cm），"
        "为头皮电极位置的近似而非个体解剖配准。"
    )
    write_report_zh(
        out_dir / "spatial_topography_report_zh.md",
        n_ch_records=len(ch_df),
        n_spatial=len(wide_diss),
        n_asd=n_asd,
        n_td=n_td,
        mesoscale_model=mesoscale_model,
        distbin_models=distbin_models,
        age_model=age_model,
        posterior_rel=posterior_rel,
        clinical=clinical,
        sensitivity=sensitivity,
        montage_note=montage_note,
    )

    logger.info("Spatial topography 分析完成 → %s", out_dir)
    return {
        "output_dir": out_dir,
        "report": out_dir / "spatial_topography_report_zh.md",
    }
