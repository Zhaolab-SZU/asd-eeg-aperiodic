"""Partial overlap analysis: Aperiodic-ISC vs broadband envelope ISC."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

EVENT_TYPES_DEFAULT = ("mental", "pain", "neutral")


def _cohen_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    na, nb = len(group_a), len(group_b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = group_a.var(ddof=1), group_b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled <= 0:
        return float("nan")
    return float((group_a.mean() - group_b.mean()) / pooled)


def load_merged_within_group_isc(
    aperiodic_path: Path,
    envelope_path: Path,
    event_types: Iterable[str] = EVENT_TYPES_DEFAULT,
) -> pd.DataFrame:
    """Merge subject-level within-group Aperiodic-ISC and envelope ISC."""
    ap = pd.read_csv(aperiodic_path)
    env = pd.read_csv(envelope_path)
    for df in (ap, env):
        df["group"] = df["group"].astype(str).str.upper()

    merged = ap.merge(
        env[["subject_id", "event_type", "isc_r", "isc_z"]],
        on=["subject_id", "event_type"],
        suffixes=("_aperiodic", "_envelope"),
        how="inner",
    )
    return merged[merged["event_type"].isin(list(event_types))].copy()


def run_aperiodic_envelope_partial_analysis(
    merged: pd.DataFrame,
    event_types: Iterable[str] = EVENT_TYPES_DEFAULT,
) -> pd.DataFrame:
    """
    Test whether Aperiodic-ISC group differences remain after controlling envelope ISC.

    Primary model (per event): Fisher-z Aperiodic-ISC ~ group + Fisher-z envelope ISC.
    """
    rows: list[dict[str, float | str | int]] = []
    for ev in event_types:
        sub = merged[merged["event_type"] == ev].dropna(
            subset=["isc_z_aperiodic", "isc_z_envelope"]
        ).copy()
        asd = sub.loc[sub["group"] == "ASD", "isc_z_aperiodic"].to_numpy()
        td = sub.loc[sub["group"] == "TD", "isc_z_aperiodic"].to_numpy()

        raw_t, raw_p = stats.ttest_ind(asd, td, equal_var=False)
        pearson_r, pearson_p = stats.pearsonr(sub["isc_z_aperiodic"], sub["isc_z_envelope"])
        spearman_rho, _ = stats.spearmanr(sub["isc_z_aperiodic"], sub["isc_z_envelope"])

        model = smf.ols(
            'isc_z_aperiodic ~ C(group, Treatment(reference="TD")) + isc_z_envelope',
            data=sub,
        ).fit()
        group_key = 'C(group, Treatment(reference="TD"))[T.ASD]'
        group_beta = float(model.params[group_key])
        group_se = float(model.bse[group_key])
        group_p = float(model.pvalues[group_key])
        env_beta = float(model.params["isc_z_envelope"])
        env_p = float(model.pvalues["isc_z_envelope"])
        resid_sd = float(model.resid.std(ddof=1))
        partial_d = group_beta / resid_sd if resid_sd > 0 else float("nan")

        env_mean = float(sub["isc_z_envelope"].mean())
        pred_td = float(
            model.predict(pd.DataFrame({"group": ["TD"], "isc_z_envelope": [env_mean]})).iloc[0]
        )
        pred_asd = float(
            model.predict(pd.DataFrame({"group": ["ASD"], "isc_z_envelope": [env_mean]})).iloc[0]
        )

        raw_d = _cohen_d(asd, td)
        rows.append(
            {
                "event_type": ev,
                "n_total": int(len(sub)),
                "n_asd": int(len(asd)),
                "n_td": int(len(td)),
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_rho": float(spearman_rho),
                "shared_variance_pct": float(pearson_r**2 * 100),
                "raw_mean_z_asd": float(np.nanmean(asd)),
                "raw_mean_z_td": float(np.nanmean(td)),
                "raw_mean_r_asd": float(np.tanh(np.nanmean(asd))),
                "raw_mean_r_td": float(np.tanh(np.nanmean(td))),
                "raw_t": float(raw_t),
                "raw_p": float(raw_p),
                "raw_cohen_d_asd_minus_td": raw_d,
                "ancova_group_beta_z": group_beta,
                "ancova_group_se": group_se,
                "ancova_group_p": group_p,
                "ancova_envelope_beta": env_beta,
                "ancova_envelope_p": env_p,
                "ancova_r2": float(model.rsquared),
                "partial_cohen_d_asd_minus_td": float(partial_d),
                "partial_effect_retained_pct": float(abs(partial_d / raw_d) * 100)
                if np.isfinite(raw_d) and raw_d != 0
                else float("nan"),
                "adj_mean_z_td_at_mean_env": pred_td,
                "adj_mean_z_asd_at_mean_env": pred_asd,
                "adj_mean_r_td_at_mean_env": float(np.tanh(pred_td)),
                "adj_mean_r_asd_at_mean_env": float(np.tanh(pred_asd)),
            }
        )

    out = pd.DataFrame(rows)
    if len(out):
        _, fdr_p, _, _ = multipletests(out["ancova_group_p"], method="fdr_bh")
        out["ancova_group_fdr_p"] = fdr_p
    return out


def build_manuscript_summary_table(partial_df: pd.DataFrame) -> pd.DataFrame:
    """Compact table for supplementary reporting."""
    rows = []
    for _, r in partial_df.iterrows():
        rows.append(
            {
                "event_type": r["event_type"],
                "isc_type": "Aperiodic-ISC (envelope-adjusted)",
                "n_asd": r["n_asd"],
                "n_td": r["n_td"],
                "td_effect": r["adj_mean_r_td_at_mean_env"],
                "asd_effect": r["adj_mean_r_asd_at_mean_env"],
                "p_value": r["ancova_group_p"],
                "fdr_p_value": r.get("ancova_group_fdr_p", np.nan),
                "shared_variance_with_envelope_pct": r["shared_variance_pct"],
                "partial_cohen_d": r["partial_cohen_d_asd_minus_td"],
            }
        )
    return pd.DataFrame(rows)
