#!/usr/bin/env python
"""
compute_partly_cloudy_isc_models.py
-----------------------------------
Group models for Partly Cloudy TD-template Aperiodic-ISC.

A) Unadjusted ASD vs TD (Welch on Fisher-z)
B) Covariate-adjusted OLS: group + age + sex + FSIQ
C) Data-quality sensitivity: + n_overlap_points + mean_r_squared
D) Matched-cohort sensitivity (movie post-QC matched participants), if available

All statistics are written from executed models; nothing is fabricated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "src" / "config.py").is_file():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError("Cannot locate repository root (missing src/config.py)")
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import save_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Partly Cloudy Aperiodic-ISC group models")
    p.add_argument("--config", type=str, default="config/config_task_movie.yaml")
    p.add_argument(
        "--subject_isc_csv",
        type=str,
        default="derivatives/derivatives_task_movie/stats/partly_cloudy_isc/partly_cloudy_aperiodic_isc_subject_level.csv",
    )
    p.add_argument(
        "--participants_csv",
        type=str,
        default="derivatives/derivatives_task_movie/participants_analysis.csv",
    )
    p.add_argument(
        "--movie_qc_csv",
        type=str,
        default="derivatives/derivatives_task_movie/specparam/specparam_qc_summary_subject.csv",
    )
    p.add_argument(
        "--matched_csv",
        type=str,
        default="data/participants/participants_task_movie_matched_postqc.csv",
    )
    return p.parse_args()


def _welch_row(df: pd.DataFrame, label: str) -> dict:
    asd = df.loc[df["group"] == "ASD", "isc_z"].dropna().to_numpy(dtype=float)
    td = df.loc[df["group"] == "TD", "isc_z"].dropna().to_numpy(dtype=float)
    if len(asd) < 2 or len(td) < 2:
        return {
            "model": label,
            "n_asd": int(len(asd)),
            "n_td": int(len(td)),
            "asd_mean_z": np.nan,
            "td_mean_z": np.nan,
            "td_minus_asd_z": np.nan,
            "asd_mean_r": np.nan,
            "td_mean_r": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "cohen_d_td_minus_asd": np.nan,
            "ci95_low_td_minus_asd": np.nan,
            "ci95_high_td_minus_asd": np.nan,
        }
    res = stats.ttest_ind(td, asd, equal_var=False, nan_policy="omit")
    diff = float(np.mean(td) - np.mean(asd))
    se = float(np.sqrt(np.var(td, ddof=1) / len(td) + np.var(asd, ddof=1) / len(asd)))
    df_w = float(res.df) if hasattr(res, "df") else float(len(td) + len(asd) - 2)
    tcrit = stats.t.ppf(0.975, df_w) if np.isfinite(df_w) and df_w > 0 else np.nan
    pooled = np.sqrt(
        ((len(asd) - 1) * np.var(asd, ddof=1) + (len(td) - 1) * np.var(td, ddof=1))
        / max(len(asd) + len(td) - 2, 1)
    )
    d = float(diff / pooled) if pooled > 0 else np.nan
    return {
        "model": label,
        "n_asd": int(len(asd)),
        "n_td": int(len(td)),
        "asd_mean_z": float(np.mean(asd)),
        "td_mean_z": float(np.mean(td)),
        "td_minus_asd_z": diff,
        "asd_mean_r": float(np.tanh(np.mean(asd))),
        "td_mean_r": float(np.tanh(np.mean(td))),
        "t_stat": float(res.statistic),
        "p_value": float(res.pvalue),
        "cohen_d_td_minus_asd": d,
        "ci95_low_td_minus_asd": float(diff - tcrit * se) if np.isfinite(tcrit) else np.nan,
        "ci95_high_td_minus_asd": float(diff + tcrit * se) if np.isfinite(tcrit) else np.nan,
    }


def _ols_group_effect(df: pd.DataFrame, formula: str, label: str) -> dict:
    model = smf.ols(formula, data=df).fit(cov_type="HC3")
    # TD coded as 1 via C(group, Treatment(reference='ASD'))
    key = None
    for k in model.params.index:
        if "group" in k and "TD" in k:
            key = k
            break
    if key is None:
        raise RuntimeError(f"No group TD coefficient in {label}: {list(model.params.index)}")
    beta = float(model.params[key])
    se = float(model.bse[key])
    p = float(model.pvalues[key])
    ci = model.conf_int().loc[key]
    return {
        "model": label,
        "formula": formula,
        "n_total": int(model.nobs),
        "n_asd": int((df["group"] == "ASD").sum()),
        "n_td": int((df["group"] == "TD").sum()),
        "beta_td_minus_asd": beta,
        "se": se,
        "t_stat": float(model.tvalues[key]),
        "p_value": p,
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "r_squared": float(model.rsquared),
    }


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = load_config(cfg_path)
    log = setup_logging(cfg, name="partly_cloudy_isc_models")

    isc_path = (PROJECT_ROOT / args.subject_isc_csv).resolve()
    if not isc_path.exists():
        raise FileNotFoundError(
            f"[ANALYSIS BLOCKED: missing Partly Cloudy ISC subject file: {isc_path}. "
            "Run scripts/68_compute_aperiodic_isc.py first.]"
        )

    isc = pd.read_csv(isc_path)
    isc["subject_id"] = isc["subject_id"].astype(str)
    isc["group"] = isc["group"].astype(str).str.upper()
    isc = isc[np.isfinite(isc["isc_z"])].copy()

    demo = pd.read_csv(PROJECT_ROOT / args.participants_csv)
    demo["subject_id"] = demo["subject_id"].astype(str)
    demo["group"] = demo["group"].astype(str).str.upper()
    demo["sex"] = demo["sex"].astype(str).str.upper().str.strip()
    demo = demo.rename(columns={"IQ_total": "FSIQ"})

    qc = pd.read_csv(PROJECT_ROOT / args.movie_qc_csv)
    qc["subject_id"] = qc["subject_id"].astype(str)

    df = isc.merge(
        demo[["subject_id", "group", "age_months", "sex", "FSIQ"]],
        on=["subject_id", "group"],
        how="left",
    )
    df = df.merge(qc[["subject_id", "mean_r_squared"]], on="subject_id", how="left")
    df["n_overlap_points"] = pd.to_numeric(df["n_overlap_points"], errors="coerce")
    df["valid_window_prop"] = df["n_overlap_points"] / df["n_overlap_points"].max()

    out_dir = PROJECT_ROOT / "outputs" / "tables" / "partly_cloudy_isc"
    out_dir.mkdir(parents=True, exist_ok=True)

    welch_rows = [_welch_row(df, "A_unadjusted_welch")]
    ols_rows = []

    df_b = df.dropna(subset=["isc_z", "age_months", "sex", "FSIQ"]).copy()
    df_b = df_b[df_b["sex"].isin(["M", "F"])]
    ols_rows.append(
        _ols_group_effect(
            df_b,
            "isc_z ~ C(group, Treatment(reference='ASD')) + age_months + C(sex) + FSIQ",
            "B_demographics_adjusted",
        )
    )

    df_c = df_b.dropna(subset=["n_overlap_points", "mean_r_squared"]).copy()
    ols_rows.append(
        _ols_group_effect(
            df_c,
            "isc_z ~ C(group, Treatment(reference='ASD')) + age_months + C(sex) + FSIQ "
            "+ n_overlap_points + mean_r_squared",
            "C_demographics_plus_data_quality",
        )
    )

    matched_path = PROJECT_ROOT / args.matched_csv
    matched_note = ""
    if matched_path.exists():
        matched = pd.read_csv(matched_path)
        matched["subject_id"] = matched["subject_id"].astype(str)
        matched_ids = set(matched["subject_id"])
        df_m = df[df["subject_id"].isin(matched_ids)].copy()
        welch_rows.append(_welch_row(df_m, "D_matched_postqc_welch"))
        df_m2 = df_m.dropna(subset=["isc_z", "age_months", "sex", "FSIQ"])
        df_m2 = df_m2[df_m2["sex"].isin(["M", "F"])]
        if len(df_m2) >= 20 and df_m2["group"].nunique() == 2:
            ols_rows.append(
                _ols_group_effect(
                    df_m2,
                    "isc_z ~ C(group, Treatment(reference='ASD')) + age_months + C(sex) + FSIQ",
                    "D_matched_postqc_demographics_adjusted",
                )
            )
        matched_note = f"matched_n={len(df_m)}"
    else:
        matched_note = f"[ANALYSIS BLOCKED: matched cohort file missing: {matched_path}]"
        log.warning(matched_note)

    welch_df = pd.DataFrame(welch_rows)
    ols_df = pd.DataFrame(ols_rows)
    save_csv(welch_df, out_dir / "partly_cloudy_aperiodic_isc_group_model_welch.csv")
    save_csv(ols_df, out_dir / "partly_cloudy_aperiodic_isc_group_model.csv")
    save_csv(df, out_dir / "partly_cloudy_aperiodic_isc_analysis_frame.csv")

    meta = {
        "n_subjects": int(len(df)),
        "n_asd": int((df["group"] == "ASD").sum()),
        "n_td": int((df["group"] == "TD").sum()),
        "matched_note": matched_note,
        "subject_isc_csv": str(isc_path),
    }
    (out_dir / "partly_cloudy_isc_models_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("=== Welch ===")
    print(welch_df.to_string(index=False))
    print("\n=== OLS ===")
    print(ols_df.to_string(index=False))
    log.info("Wrote models to %s", out_dir)


if __name__ == "__main__":
    main()
