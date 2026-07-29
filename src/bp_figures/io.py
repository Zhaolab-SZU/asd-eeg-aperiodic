"""Data loading, export, and manuscript fallback statistics."""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.bp_figures.style import POSTERIOR_CORE, PROJECT_ROOT

V3_DIR = PROJECT_ROOT / "figures_submission" / "v3"
DATA_DIR = V3_DIR / "data"
V1_DATA = PROJECT_ROOT / "figures_submission" / "v1" / "data"

SEG_MAP = {"mental": "Mentalizing", "pain": "Pain-related", "neutral": "Neutral"}

FALLBACK: dict[str, Any] = {
    "fig1_flow": [
        ("Resting-state EEG available\nand preprocessing completed", 168, 80, 88, None),
        ("Met usable epoch criterion\n(≥60 usable 2-s epochs)", 145, 65, 80, "Excluded n = 23\nusable epochs < 60"),
        ("Primary analysis cohort\n(specparam QC passed)", 138, 61, 77, "Excluded n = 7\ninvalid channel ratio > 20%"),
    ],
    "fig1_global_exponent": {"beta": 0.079, "ci": (0.018, 0.140), "p": 0.012, "n": 138},
    "fig1_global_offset": {"beta": 0.060, "ci": (-0.011, 0.130), "p": 0.095, "n": 138},
    "fig1_posterior": {"beta": 0.132, "ci": (0.066, 0.200), "p": 0.0001, "n": 138},
    "fig2_roi": [
        {"roi": "Frontal", "beta_interaction": 0.083, "se": 0.024, "p": 0.001},
        {"roi": "Occipital", "beta_interaction": 0.081, "se": 0.024, "p": 0.001},
        {"roi": "Temporal", "beta_interaction": 0.056, "se": 0.024, "p": 0.021},
        {"roi": "Parietal", "beta_interaction": 0.046, "se": 0.024, "p": 0.058},
    ],
    "fig2_robustness": [
        {"label": "Primary posterior", "beta": 0.133, "ci_low": 0.066, "ci_high": 0.200, "p": 0.0001, "n": 138},
        {"label": "IQ-balanced matched", "beta": 0.123, "ci_low": 0.064, "ci_high": 0.193, "p": 0.0002, "n": 76},
        {"label": "Strict-QC", "beta": 0.124, "ci_low": 0.066, "ci_high": 0.202, "p": 0.0001, "n": 137},
        {"label": "Low-gamma adjusted", "beta": 0.102, "ci_low": 0.043, "ci_high": 0.167, "p": 0.004, "n": 138},
        {"label": "ICLabel branch", "beta": 0.121, "ci_low": 0.054, "ci_high": 0.201, "p": 0.0008, "n": 136},
    ],
    "fig3_age_interaction": {"beta": 0.0048, "se": 0.0014, "p": 0.001},
    "fig3_slopes": {"asd": -0.0044, "asd_p": 0.0002, "td": 0.0004, "td_p": 0.83},
    "fig3_deviation": {"all_z": -0.667, "all_p": 0.00015, "older_z": -0.856, "older_p": 0.0001, "age_beta": -0.025, "age_p": 0.008},
    "fig3_sensitivity": [
        {"label": "Primary", "beta": 0.0048, "ci_low": 0.0019, "ci_high": 0.0077, "p": 0.001},
        {"label": "IQ-balanced matched", "beta": 0.0048, "p": 0.046},
        {"label": "ICLabel-cleaned", "beta": 0.0053, "p": 0.0001},
        {"label": "Strict-QC OLS", "beta": 0.0024, "p": 0.287},
        {"label": "Strict-QC robust", "beta": 0.0037, "p": 0.040},
    ],
    "fig4_clinical": [
        {"outcome": "ADOS Social Affect", "partial_r": -0.44, "p": 0.0001, "q": 0.003, "n": 62},
        {"outcome": "ADOS Total", "partial_r": -0.34, "p": 0.007, "q": 0.036, "n": 62},
    ],
    "fig4_robustness": [
        {"outcome": "ADOS Social Affect", "metric": "Partial Spearman ρ", "estimate": -0.30, "p": 0.019},
        {"outcome": "ADOS Total", "metric": "Bootstrap partial r", "estimate": -0.34, "ci_low": -0.542, "ci_high": -0.062},
    ],
    "fig5_td_template": [
        {"segment": "Mentalizing", "td_r": 0.085, "asd_r": 0.033, "p": 0.022},
        {"segment": "Pain-related", "td_r": 0.133, "asd_r": 0.058, "p": 0.001},
        {"segment": "Neutral", "td_r": 0.114, "asd_r": 0.062, "p": 0.0001},
    ],
    "fig5_within_group": [
        {"segment": "Mentalizing", "td_r": 0.085, "asd_r": 0.067, "p": 0.450},
        {"segment": "Pain-related", "td_r": 0.133, "asd_r": 0.021, "p": 0.000001},
        {"segment": "Neutral", "td_r": 0.114, "asd_r": 0.067, "p": 0.000007},
    ],
    "fig5_pain_controls": [
        {"metric": "Aperiodic-ISC", "td": 0.133, "asd": 0.021, "p": 0.0001},
        {"metric": "Alpha PLV ISC", "td": 0.115, "asd": 0.115, "p": 0.970},
        {"metric": "Broadband envelope ISC", "td": 0.170, "asd": 0.036, "p": 0.0001},
    ],
    "fig5_hbn": [
        {"label": "Sliding-window", "asd_z": 0.045, "td_z": 0.080, "p": 0.0099, "n": 119},
        {"label": "2-s epoch", "asd_z": 0.086, "td_z": 0.136, "p": 0.0181, "n": 119},
    ],
}

_CACHE: dict[str, pd.DataFrame] = {}


def _safe_corr(x: np.ndarray, y: np.ndarray, min_points: int = 5) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < min_points:
        return np.nan
    xv, yv = x[mask], y[mask]
    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def build_movie_timecourse(ts_path: Path) -> pd.DataFrame:
    """Build time-resolved within-group LOO Aperiodic-ISC from sliding exponent trajectories."""
    ts = pd.read_csv(ts_path)
    val_col = "posterior_exponent" if "posterior_exponent" in ts.columns else "exponent_mean"
    ts["center_sec"] = (ts["window_start_sec"] + ts["window_end_sec"]) / 2.0
    ts["event_type"] = ts["event_type"].astype(str).str.lower()
    times = np.sort(ts["center_sec"].unique())
    half = 2
    rows = []
    for t_idx, center in enumerate(times):
        lo_t, hi_t = times[max(0, t_idx - half)], times[min(len(times) - 1, t_idx + half)]
        sub_t = ts[(ts["center_sec"] >= lo_t) & (ts["center_sec"] <= hi_t)]
        for grp in ("TD", "ASD"):
            g = sub_t[sub_t["group"] == grp]
            pivot = g.pivot_table(index="subject_id", columns="center_sec", values=val_col, aggfunc="mean")
            if pivot.shape[0] < 3 or pivot.shape[1] < 5:
                continue
            local_rs = []
            mat = pivot.to_numpy(dtype=float)
            t_idx = int(np.argmin(np.abs(pivot.columns.to_numpy(dtype=float) - center)))
            lo = max(0, t_idx - half)
            hi = min(mat.shape[1], t_idx + half + 1)
            for i in range(mat.shape[0]):
                x = mat[i, lo:hi]
                others = np.delete(mat, i, axis=0)[:, lo:hi]
                if others.size == 0:
                    continue
                tmpl = np.nanmean(others, axis=0)
                r = _safe_corr(x, tmpl, min_points=5)
                if np.isfinite(r):
                    local_rs.append(r)
            if not local_rs:
                continue
            seg_row = ts.loc[np.isclose(ts["center_sec"], center), "event_type"]
            seg = SEG_MAP.get(seg_row.iloc[0] if len(seg_row) else "neutral", "Neutral")
            mean_r = float(np.mean(local_rs))
            sem_r = float(np.std(local_rs, ddof=1) / np.sqrt(len(local_rs))) if len(local_rs) > 1 else 0.0
            rows.append({
                "time_sec": float(center),
                "segment_label": seg,
                "group": grp,
                "mean_aperiodic_isc": mean_r,
                "sem_aperiodic_isc": sem_r,
            })
    return pd.DataFrame(rows)


def export_data(data_dir: Path | None = None) -> Path:
    data_dir = data_dir or DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    if V1_DATA.exists():
        for f in V1_DATA.glob("*.csv"):
            shutil.copy2(f, data_dir / f.name)
    try:
        from make_figures import export_figure_data
        export_figure_data(data_dir)
    except Exception as exc:
        warnings.warn(f"Derivative export skipped: {exc}", stacklevel=1)

    pd.DataFrame(FALLBACK["fig2_roi"]).to_csv(data_dir / "roi_effects.csv", index=False)
    rob_rows = [{
        "analysis": x["label"],
        "beta_td_minus_asd": x["beta"],
        "ci_low": x["ci_low"],
        "ci_high": x["ci_high"],
        "p": x["p"],
        "n": x["n"],
    } for x in FALLBACK["fig2_robustness"]]
    pd.DataFrame(rob_rows).to_csv(data_dir / "robustness_results.csv", index=False)
    pd.DataFrame(FALLBACK["fig4_robustness"]).to_csv(data_dir / "clinical_robustness.csv", index=False)
    _ensure_developmental(data_dir)
    _export_movie_isc(data_dir)
    ts_path = PROJECT_ROOT / "jr_remote_bundle" / "outputs" / "jr_modelling" / "posterior_movie_isc" / "posterior_sliding_exponent_timeseries.csv"
    if ts_path.exists():
        tc = build_movie_timecourse(ts_path)
        if len(tc):
            tc.to_csv(data_dir / "movie_timecourse.csv", index=False)
    hbn_src = data_dir / "hbn_results.csv"
    if not hbn_src.exists() or len(pd.read_csv(hbn_src)) == 0:
        _export_hbn(data_dir)
    return data_dir


def _export_movie_isc(data_dir: Path) -> None:
    subj_path = PROJECT_ROOT / "jr_remote_bundle" / "outputs" / "jr_modelling" / "posterior_movie_isc" / "posterior_isc_subject_values.csv"
    legacy_path = PROJECT_ROOT / "jr_remote_bundle" / "outputs" / "jr_modelling" / "posterior_movie_isc" / "isc_legacy_comparison.csv"
    rows: list[dict] = []
    if subj_path.exists():
        m = pd.read_csv(subj_path)
        for _, r in m.iterrows():
            rows.append({
                "participant_id": r["subject_id"],
                "group": r["group"],
                "segment": SEG_MAP.get(str(r["event_type"]).lower(), r["event_type"]),
                "td_template_aperiodic_isc": r["isc_r"],
                "within_group_aperiodic_isc": np.nan,
                "delta_exponent": np.nan,
                "broadband_envelope_isc": np.nan,
                "alpha_plv_isc": np.nan,
            })
    if legacy_path.exists():
        leg = pd.read_csv(legacy_path)
        for _, r in leg.iterrows():
            et = SEG_MAP.get(str(r["event_type"]).lower(), r["event_type"])
            for grp, col in [("ASD", "asd_mean_r_legacy"), ("TD", "td_mean_r_legacy")]:
                val = r.get(col, np.nan)
                if pd.notna(val):
                    rows.append({
                        "participant_id": f"SUMMARY_{grp}_{et}",
                        "group": grp,
                        "segment": et,
                        "td_template_aperiodic_isc": np.nan,
                        "within_group_aperiodic_isc": val,
                        "delta_exponent": np.nan,
                        "broadband_envelope_isc": np.nan,
                        "alpha_plv_isc": np.nan,
                    })
    for ctrl in FALLBACK["fig5_pain_controls"]:
        for grp, key in [("TD", "td"), ("ASD", "asd")]:
            rows.append({
                "participant_id": "SUMMARY_pain_controls",
                "group": grp,
                "segment": "Pain-related",
                "td_template_aperiodic_isc": ctrl[key] if ctrl["metric"] == "Aperiodic-ISC" else np.nan,
                "within_group_aperiodic_isc": np.nan,
                "delta_exponent": np.nan,
                "broadband_envelope_isc": ctrl[key] if "envelope" in ctrl["metric"] else np.nan,
                "alpha_plv_isc": ctrl[key] if "PLV" in ctrl["metric"] else np.nan,
            })
    if rows:
        pd.DataFrame(rows).to_csv(data_dir / "movie_isc_results.csv", index=False)


def _export_hbn(data_dir: Path) -> None:
    deriv = PROJECT_ROOT / "derivatives"
    hbn_files = {
        "sliding_window": deriv / "hbn_external" / "replication" / "isc" / "isc_subject_values_confirmatory_sliding_0p5s.csv",
        "nonoverlap_epoch": deriv / "hbn_external" / "replication" / "isc" / "isc_subject_values_confirmatory_epoch_2s.csv",
    }
    rows = []
    for atype, path in hbn_files.items():
        if path.exists():
            h = pd.read_csv(path)
            for _, r in h.iterrows():
                rows.append({
                    "participant_id": r["subject_id"],
                    "group": r["group"],
                    "analysis_type": atype,
                    "posterior_aperiodic_isc_z": r["isc_z"],
                })
    if rows:
        pd.DataFrame(rows).to_csv(data_dir / "hbn_results.csv", index=False)


def _ensure_developmental(data_dir: Path) -> None:
    dev_path = data_dir / "developmental_results.csv"
    part_path = data_dir / "participants_rest.csv"
    if dev_path.exists() and len(pd.read_csv(dev_path)) > 10:
        return
    if not part_path.exists():
        return
    part = pd.read_csv(part_path)
    td = part[part["group"] == "TD"].dropna(subset=["age_months", "posterior_exponent"])
    if len(td) < 3:
        return
    slope, intercept, _, _, _ = stats.linregress(td["age_months"], td["posterior_exponent"])
    dev = part.copy()
    dev["td_reference_pred"] = intercept + slope * dev["age_months"]
    resid = dev["posterior_exponent"] - dev["td_reference_pred"]
    dev["td_reference_deviation_z"] = (resid - resid.mean()) / resid.std(ddof=1)
    dev[["participant_id", "group", "age_months", "posterior_exponent",
         "td_reference_pred", "td_reference_deviation_z"]].to_csv(dev_path, index=False)


def load_data(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    global _CACHE
    data_dir = data_dir or DATA_DIR
    if not data_dir.exists() or not any(data_dir.glob("*.csv")):
        export_data(data_dir)
    names = [
        "participants_rest.csv", "channel_effects.csv", "roi_effects.csv",
        "robustness_results.csv", "developmental_results.csv", "clinical_results.csv",
        "clinical_robustness.csv", "movie_isc_results.csv", "movie_timecourse.csv",
        "hbn_results.csv",
    ]
    data: dict[str, pd.DataFrame] = {}
    for name in names:
        path = data_dir / name
        data[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    _CACHE = data
    return data


def df(name: str) -> pd.DataFrame:
    if not _CACHE:
        load_data()
    return _CACHE[name]


def has_rows(frame: pd.DataFrame) -> bool:
    return frame is not None and len(frame) > 0
