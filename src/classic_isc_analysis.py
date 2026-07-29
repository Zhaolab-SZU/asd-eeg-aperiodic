"""Classic envelope and alpha-phase ISC controls for movie data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import ttest_ind

from src.aperiodic_isc_analysis import (
    EVENT_TYPES_ALL,
    build_concat_keys,
    build_subject_concat_series,
    compute_within_group_isc,
    fisher_z,
    safe_corr,
)

POSTERIOR_CHANNELS = ("E33", "E36", "E37", "E38")
BROADBAND_BAND = (0.5, 45.0)
ALPHA_BAND = (8.0, 13.0)


def sliding_window_means(
    signal: np.ndarray,
    sfreq: float,
    window_sec: float,
    step_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return window center times (s) and mean signal in each window."""
    x = np.asarray(signal, dtype=float)
    win = max(int(round(window_sec * sfreq)), 1)
    step = max(int(round(step_sec * sfreq)), 1)
    if x.size < win:
        return np.array([]), np.array([])

    starts = np.arange(0, x.size - win + 1, step)
    centers = (starts + win / 2.0) / sfreq
    values = np.array([np.nanmean(x[s : s + win]) for s in starts], dtype=float)
    return centers, values


def load_posterior_mean_trace(raw_path: Path, channels: tuple[str, ...] = POSTERIOR_CHANNELS) -> tuple[np.ndarray, float]:
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    picks = [c for c in channels if c in raw.ch_names]
    if len(picks) < len(channels):
        missing = set(channels) - set(picks)
        raise ValueError(f"{raw_path.name}: missing channels {sorted(missing)}")
    data = raw.get_data(picks=picks).mean(axis=0)
    return data, float(raw.info["sfreq"])


def compute_envelope_timeseries(
    raw_path: Path,
    window_sec: float,
    step_sec: float,
    l_freq: float = BROADBAND_BAND[0],
    h_freq: float = BROADBAND_BAND[1],
) -> pd.DataFrame:
    data, sfreq = load_posterior_mean_trace(raw_path)
    if not np.isclose(l_freq, 0.5) or not np.isclose(h_freq, 45.0):
        data = mne.filter.filter_data(data, sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False)
    envelope = np.abs(hilbert(data))
    centers, values = sliding_window_means(envelope, sfreq, window_sec, step_sec)
    return pd.DataFrame({"center_sec": centers, "signal_mean": values})


def compute_alpha_phase_timeseries(
    raw_path: Path,
    window_sec: float,
    step_sec: float,
    l_freq: float = ALPHA_BAND[0],
    h_freq: float = ALPHA_BAND[1],
) -> pd.DataFrame:
    """Circular mean alpha phase in each sliding window."""
    data, sfreq = load_posterior_mean_trace(raw_path)
    alpha = mne.filter.filter_data(data, sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False)
    phase = np.angle(hilbert(alpha))
    win = max(int(round(window_sec * sfreq)), 1)
    step = max(int(round(step_sec * sfreq)), 1)
    if phase.size < win:
        return pd.DataFrame({"center_sec": [], "signal_mean": []})

    starts = np.arange(0, phase.size - win + 1, step)
    centers = (starts + win / 2.0) / sfreq
    values = []
    for s in starts:
        seg = phase[s : s + win]
        values.append(float(np.angle(np.mean(np.exp(1j * seg)))))
    return pd.DataFrame({"center_sec": centers, "signal_mean": values})


def build_subject_timeseries_table(
    participants: pd.DataFrame,
    preproc_dir: Path,
    extractor,
    window_sec: float,
    step_sec: float,
) -> pd.DataFrame:
    rows = []
    for _, row in participants.iterrows():
        sid = str(row["subject_id"])
        group = str(row["group"]).upper()
        raw_path = preproc_dir / f"{sid}-raw.fif"
        if not raw_path.exists():
            continue
        ts = extractor(raw_path, window_sec, step_sec)
        for i, rec in ts.iterrows():
            rows.append(
                {
                    "subject_id": sid,
                    "group": group,
                    "window_index": int(i),
                    "window_start_sec": float(rec["center_sec"] - window_sec / 2.0),
                    "window_end_sec": float(rec["center_sec"] + window_sec / 2.0),
                    "center_sec": float(rec["center_sec"]),
                    "signal_mean": float(rec["signal_mean"]),
                }
            )
    return pd.DataFrame(rows)


def _circular_mean_phase(phases: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.angle(np.mean(np.exp(1j * phases), axis=axis))


def compute_within_group_alpha_plv(
    subject_concat: pd.DataFrame,
    min_overlap_points: int,
    event_types: Iterable[str],
    phase_col: str = "signal_mean",
) -> pd.DataFrame:
    """Leave-one-out PLV within TD and ASD groups on concatenated event phase series."""
    rows = []
    for ev in event_types:
        sub_ev = subject_concat[subject_concat["event_type"] == ev].copy()
        mat = sub_ev.pivot_table(
            index=["subject_id", "group"],
            columns="concat_index",
            values=phase_col,
            aggfunc="mean",
        ).sort_index(axis=1)
        idx_df = mat.index.to_frame(index=False)
        idx_df.columns = ["subject_id", "group"]

        for group in ("TD", "ASD"):
            g_mask = idx_df["group"] == group
            g_vals = mat[g_mask.to_numpy()]
            g_subjects = idx_df.loc[g_mask, "subject_id"].tolist()
            if g_vals.shape[0] < 2:
                continue
            phase_mat = g_vals.to_numpy(dtype=float)
            for i, sid in enumerate(g_subjects):
                xi = phase_mat[i]
                others = np.delete(phase_mat, i, axis=0)
                mask = np.isfinite(xi)
                for j in range(others.shape[0]):
                    mask &= np.isfinite(others[j])
                n_overlap = int(mask.sum())
                if n_overlap < min_overlap_points:
                    plv = np.nan
                else:
                    tmpl = _circular_mean_phase(others[:, mask], axis=0)
                    plv = float(np.abs(np.mean(np.exp(1j * (xi[mask] - tmpl)))))
                rows.append(
                    {
                        "subject_id": sid,
                        "group": group,
                        "event_type": ev,
                        "isc_r": plv,
                        "isc_z": fisher_z(plv) if np.isfinite(plv) else np.nan,
                        "n_overlap_points": n_overlap,
                        "template_type": f"{group}_LOO",
                        "isc_method": "within_group_alpha_plv",
                    }
                )
    return pd.DataFrame(rows)


def summarize_group_plv_tests(plv_df: pd.DataFrame, event_types: Iterable[str]) -> pd.DataFrame:
    stats_rows = []
    for ev in event_types:
        sub = plv_df[plv_df["event_type"] == ev].copy()
        asd = sub.loc[sub["group"] == "ASD", "isc_r"].dropna().to_numpy()
        td = sub.loc[sub["group"] == "TD", "isc_r"].dropna().to_numpy()
        if len(asd) < 2 or len(td) < 2:
            t_stat, p_val = np.nan, np.nan
        else:
            res = ttest_ind(asd, td, equal_var=False, nan_policy="omit")
            t_stat, p_val = float(res.statistic), float(res.pvalue)
        stats_rows.append(
            {
                "event_type": ev,
                "n_asd": int(len(asd)),
                "n_td": int(len(td)),
                "asd_mean_plv": float(np.nanmean(asd)) if len(asd) else np.nan,
                "td_mean_plv": float(np.nanmean(td)) if len(td) else np.nan,
                "mean_diff_asd_minus_td": float(np.nanmean(asd) - np.nanmean(td)) if len(asd) and len(td) else np.nan,
                "t_stat": t_stat,
                "p_value": p_val,
            }
        )
    return pd.DataFrame(stats_rows)


def summarize_group_envelope_tests(envelope_isc: pd.DataFrame, event_types: Iterable[str]) -> pd.DataFrame:
    stats_rows = []
    for ev in event_types:
        sub = envelope_isc[envelope_isc["event_type"] == ev].copy()
        asd = sub.loc[sub["group"] == "ASD", "isc_z"].dropna().to_numpy()
        td = sub.loc[sub["group"] == "TD", "isc_z"].dropna().to_numpy()
        if len(asd) < 2 or len(td) < 2:
            t_stat, p_val = np.nan, np.nan
        else:
            res = ttest_ind(asd, td, equal_var=False, nan_policy="omit")
            t_stat, p_val = float(res.statistic), float(res.pvalue)
        stats_rows.append(
            {
                "event_type": ev,
                "n_asd": int(len(asd)),
                "n_td": int(len(td)),
                "asd_mean_r": float(np.tanh(np.nanmean(asd))) if len(asd) else np.nan,
                "td_mean_r": float(np.tanh(np.nanmean(td))) if len(td) else np.nan,
                "asd_mean_z": float(np.nanmean(asd)) if len(asd) else np.nan,
                "td_mean_z": float(np.nanmean(td)) if len(td) else np.nan,
                "t_stat": t_stat,
                "p_value": p_val,
            }
        )
    return pd.DataFrame(stats_rows)


def build_control_comparison_table(
    aperiodic_stats: pd.DataFrame,
    envelope_stats: pd.DataFrame,
    alpha_stats: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for ev in EVENT_TYPES_ALL:
        ap = aperiodic_stats[aperiodic_stats["event_type"] == ev]
        env = envelope_stats[envelope_stats["event_type"] == ev]
        al = alpha_stats[alpha_stats["event_type"] == ev]
        if len(ap):
            r = ap.iloc[0]
            rows.append(
                {
                    "event_type": ev,
                    "isc_type": "Aperiodic-ISC (exponent)",
                    "n_asd": r["n_asd"],
                    "n_td": r["n_td"],
                    "asd_effect": r["asd_mean_r"],
                    "td_effect": r["td_mean_r"],
                    "p_value": r["p_value"],
                }
            )
        if len(env):
            r = env.iloc[0]
            rows.append(
                {
                    "event_type": ev,
                    "isc_type": "Broadband envelope ISC",
                    "n_asd": r["n_asd"],
                    "n_td": r["n_td"],
                    "asd_effect": r["asd_mean_r"],
                    "td_effect": r["td_mean_r"],
                    "p_value": r["p_value"],
                }
            )
        if len(al):
            r = al.iloc[0]
            rows.append(
                {
                    "event_type": ev,
                    "isc_type": "Alpha phase PLV (LOO)",
                    "n_asd": r["n_asd"],
                    "n_td": r["n_td"],
                    "asd_effect": r["asd_mean_plv"],
                    "td_effect": r["td_mean_plv"],
                    "p_value": r["p_value"],
                }
            )
    return pd.DataFrame(rows)
