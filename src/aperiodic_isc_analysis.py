"""Time-resolved aperiodic exponent inter-subject correlation (Aperiodic-ISC)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

EVENT_TYPES_BASE = ("mental", "pain")
EVENT_TYPES_ALL = ("mental", "pain", "neutral")


def required_cols(df: pd.DataFrame, cols: set[str], name: str) -> None:
    miss = cols - set(df.columns)
    if miss:
        raise ValueError(f"{name} 缺少列: {sorted(miss)}")


def safe_corr(x: np.ndarray, y: np.ndarray, min_points: int) -> tuple[float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < min_points:
        return np.nan, n
    xv = x[mask]
    yv = y[mask]
    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return np.nan, n
    return float(np.corrcoef(xv, yv)[0, 1]), n


def fisher_z(r: float) -> float:
    if pd.isna(r):
        return np.nan
    r = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(r))


def build_concat_keys(ts_bins: np.ndarray, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    events = events.copy()
    events["event_type"] = events["event_type"].astype(str).str.strip().str.lower()
    events = events[events["event_type"].isin(EVENT_TYPES_BASE)].copy()
    events["start"] = pd.to_numeric(events["onset_sec"], errors="coerce")
    events["end"] = pd.to_numeric(events["onset_sec"], errors="coerce") + pd.to_numeric(
        events["duration_sec"], errors="coerce"
    )
    events = events.dropna(subset=["start", "end"]).sort_values("start").reset_index(drop=True)

    for b in np.sort(np.unique(np.round(ts_bins, 6))):
        labels = []
        for _, ev in events.iterrows():
            if float(ev["start"]) <= float(b) < float(ev["end"]):
                labels.append(str(ev["event_type"]))
        if "mental" in labels:
            event_type = "mental"
        elif "pain" in labels:
            event_type = "pain"
        else:
            event_type = "neutral"
        rows.append({"center_sec": float(b), "event_type": event_type})

    out = pd.DataFrame(rows)
    for ev in EVENT_TYPES_ALL:
        idx = out.index[out["event_type"] == ev].to_numpy()
        out.loc[out["event_type"] == ev, "concat_index"] = np.arange(len(idx), dtype=int)
    out["concat_index"] = out["concat_index"].astype(int)
    return out


def build_concat_keys_overall(ts_bins: np.ndarray) -> pd.DataFrame:
    bins = np.sort(np.unique(np.round(ts_bins, 6)))
    return pd.DataFrame(
        {
            "center_sec": bins.astype(float),
            "event_type": "overall",
            "concat_index": np.arange(len(bins), dtype=int),
        }
    )


def load_exponent_timeseries(
    ts_path: Path,
    movie_analysis_path: Path | None = None,
    movie_qc_path: Path | None = None,
) -> pd.DataFrame:
    ts = pd.read_csv(Path(ts_path))
    required_cols(ts, {"subject_id", "group", "window_start_sec", "window_end_sec", "exponent_mean"}, "timeseries")
    ts = ts.copy()
    ts["subject_id"] = ts["subject_id"].astype(str)
    ts["group"] = ts["group"].astype(str).str.upper()
    ts["center_sec"] = np.round((ts["window_start_sec"] + ts["window_end_sec"]) / 2.0, 6)

    if movie_analysis_path is not None and Path(movie_analysis_path).exists():
        movie_analysis = pd.read_csv(Path(movie_analysis_path))
        required_cols(movie_analysis, {"subject_id", "group"}, "movie_analysis_csv")
        movie_analysis["subject_id"] = movie_analysis["subject_id"].astype(str)
        movie_analysis["group"] = movie_analysis["group"].astype(str).str.upper()
        allowed_pairs = movie_analysis[["subject_id", "group"]].drop_duplicates()
        ts = ts.merge(allowed_pairs, on=["subject_id", "group"], how="inner")

    if movie_qc_path is not None and Path(movie_qc_path).exists():
        movie_qc = pd.read_csv(Path(movie_qc_path))
        required_cols(movie_qc, {"subject_id", "low_quality_subject"}, "movie_specparam_qc_csv")
        movie_qc["subject_id"] = movie_qc["subject_id"].astype(str)
        bad_ids = set(
            movie_qc.loc[pd.to_numeric(movie_qc["low_quality_subject"], errors="coerce") == 1, "subject_id"].tolist()
        )
        if bad_ids:
            ts = ts[~ts["subject_id"].isin(bad_ids)].copy()
    return ts


def build_subject_concat_series(ts: pd.DataFrame, keys_df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for (sid, grp), sub in ts.groupby(["subject_id", "group"], sort=False):
        sub2 = (
            sub[["center_sec", "exponent_mean"]]
            .groupby("center_sec", as_index=False)
            .mean()
            .copy()
        )
        merged = keys_df.merge(sub2, on="center_sec", how="left")
        merged["subject_id"] = sid
        merged["group"] = grp
        out_rows.append(
            merged[
                [
                    "subject_id",
                    "group",
                    "event_type",
                    "concat_index",
                    "center_sec",
                    "exponent_mean",
                ]
            ]
        )
    return pd.concat(out_rows, ignore_index=True)


def _pivot_event_matrix(
    subject_concat: pd.DataFrame,
    event_type: str,
    value_col: str = "exponent_mean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub_ev = subject_concat[subject_concat["event_type"] == event_type].copy()
    mat = sub_ev.pivot_table(
        index=["subject_id", "group"],
        columns="concat_index",
        values=value_col,
        aggfunc="mean",
    )
    mat = mat.sort_index(axis=1)
    idx_df = mat.index.to_frame(index=False)
    idx_df.columns = ["subject_id", "group"]
    return idx_df, mat


def compute_td_template_isc(
    subject_concat: pd.DataFrame,
    min_overlap_points: int,
    event_types: Iterable[str],
    value_col: str = "exponent_mean",
) -> pd.DataFrame:
    rows = []
    for ev in event_types:
        idx_df, mat = _pivot_event_matrix(subject_concat, ev, value_col=value_col)
        td_mask = idx_df["group"] == "TD"
        asd_mask = idx_df["group"] == "ASD"
        td_vals = mat[td_mask.to_numpy()]
        asd_vals = mat[asd_mask.to_numpy()]

        if td_vals.shape[0] < 2:
            raise RuntimeError(f"{ev}: TD 样本不足（至少需要2名用于留一模板）")

        td_subjects = idx_df.loc[td_mask, "subject_id"].tolist()
        for i, sid in enumerate(td_subjects):
            x = td_vals.iloc[i].to_numpy(dtype=float)
            others = td_vals.drop(td_vals.index[i])
            tmpl = others.mean(axis=0, skipna=True).to_numpy(dtype=float)
            r, n_overlap = safe_corr(x, tmpl, min_points=min_overlap_points)
            rows.append(
                {
                    "subject_id": sid,
                    "group": "TD",
                    "event_type": ev,
                    "isc_r": r,
                    "isc_z": fisher_z(r),
                    "n_overlap_points": n_overlap,
                    "template_type": "TD_LOO",
                    "isc_method": "td_template",
                }
            )

        td_template = td_vals.mean(axis=0, skipna=True).to_numpy(dtype=float)
        asd_subjects = idx_df.loc[asd_mask, "subject_id"].tolist()
        for i, sid in enumerate(asd_subjects):
            x = asd_vals.iloc[i].to_numpy(dtype=float)
            r, n_overlap = safe_corr(x, td_template, min_points=min_overlap_points)
            rows.append(
                {
                    "subject_id": sid,
                    "group": "ASD",
                    "event_type": ev,
                    "isc_r": r,
                    "isc_z": fisher_z(r),
                    "n_overlap_points": n_overlap,
                    "template_type": "TD_FULL",
                    "isc_method": "td_template",
                }
            )
    return pd.DataFrame(rows)


def compute_within_group_isc(
    subject_concat: pd.DataFrame,
    min_overlap_points: int,
    event_types: Iterable[str],
    value_col: str = "exponent_mean",
) -> pd.DataFrame:
    """组内留一模板 Aperiodic-ISC：TD 对 TD 模板，ASD 对 ASD 模板。"""
    rows = []
    for ev in event_types:
        idx_df, mat = _pivot_event_matrix(subject_concat, ev, value_col=value_col)
        for group in ("TD", "ASD"):
            g_mask = idx_df["group"] == group
            g_vals = mat[g_mask.to_numpy()]
            g_subjects = idx_df.loc[g_mask, "subject_id"].tolist()
            if g_vals.shape[0] < 2:
                continue
            for i, sid in enumerate(g_subjects):
                x = g_vals.iloc[i].to_numpy(dtype=float)
                others = g_vals.drop(g_vals.index[i])
                tmpl = others.mean(axis=0, skipna=True).to_numpy(dtype=float)
                r, n_overlap = safe_corr(x, tmpl, min_points=min_overlap_points)
                rows.append(
                    {
                        "subject_id": sid,
                        "group": group,
                        "event_type": ev,
                        "isc_r": r,
                        "isc_z": fisher_z(r),
                        "n_overlap_points": n_overlap,
                        "template_type": f"{group}_LOO",
                        "isc_method": "within_group",
                    }
                )
    return pd.DataFrame(rows)


def compute_pairwise_mean_isc(
    subject_concat: pd.DataFrame,
    min_overlap_points: int,
    event_types: Iterable[str],
    value_col: str = "exponent_mean",
) -> pd.DataFrame:
    """组内所有被试对的 exponent 序列相关均值（被试级无 LOO 标量，仅组级汇总）。"""
    rows = []
    for ev in event_types:
        idx_df, mat = _pivot_event_matrix(subject_concat, ev, value_col=value_col)
        for group in ("TD", "ASD"):
            g_mask = idx_df["group"] == group
            g_vals = mat[g_mask.to_numpy()]
            n_sub = g_vals.shape[0]
            if n_sub < 2:
                continue
            pair_rs = []
            for i in range(n_sub):
                xi = g_vals.iloc[i].to_numpy(dtype=float)
                for j in range(i + 1, n_sub):
                    xj = g_vals.iloc[j].to_numpy(dtype=float)
                    r, _ = safe_corr(xi, xj, min_points=min_overlap_points)
                    if np.isfinite(r):
                        pair_rs.append(r)
            rows.append(
                {
                    "group": group,
                    "event_type": ev,
                    "n_subjects": n_sub,
                    "n_pairs": len(pair_rs),
                    "mean_pairwise_r": float(np.mean(pair_rs)) if pair_rs else np.nan,
                    "mean_pairwise_z": fisher_z(float(np.mean(pair_rs))) if pair_rs else np.nan,
                    "isc_method": "pairwise_mean",
                }
            )
    return pd.DataFrame(rows)


def compute_time_resolved_group_isc(
    subject_concat: pd.DataFrame,
    local_window_bins: int,
    min_overlap_points: int,
    event_types: Iterable[str],
    value_col: str = "exponent_mean",
) -> pd.DataFrame:
    """在每个时间点用局部窗口计算组内平均 LOO 相关，得到 ISC(t)。"""
    rows = []
    half = max(int(local_window_bins) // 2, 1)
    for ev in event_types:
        idx_df, mat = _pivot_event_matrix(subject_concat, ev, value_col=value_col)
        n_time = mat.shape[1]
        center_secs = (
            subject_concat.loc[subject_concat["event_type"] == ev, ["concat_index", "center_sec"]]
            .drop_duplicates()
            .sort_values("concat_index")
        )
        for group in ("TD", "ASD"):
            g_mask = idx_df["group"] == group
            g_vals = mat[g_mask.to_numpy()]
            n_sub = g_vals.shape[0]
            if n_sub < 2:
                continue
            for t_idx in range(n_time):
                lo = max(0, t_idx - half)
                hi = min(n_time, t_idx + half + 1)
                local_rs = []
                for i in range(n_sub):
                    x = g_vals.iloc[i, lo:hi].to_numpy(dtype=float)
                    others = g_vals.drop(g_vals.index[i], errors="ignore")
                    tmpl = others.iloc[:, lo:hi].mean(axis=0, skipna=True).to_numpy(dtype=float)
                    r, _ = safe_corr(x, tmpl, min_points=min_overlap_points)
                    if np.isfinite(r):
                        local_rs.append(r)
                center_sec = float(center_secs.loc[center_secs["concat_index"] == t_idx, "center_sec"].iloc[0])
                rows.append(
                    {
                        "group": group,
                        "event_type": ev,
                        "concat_index": int(t_idx),
                        "center_sec": center_sec,
                        "mean_loo_r": float(np.mean(local_rs)) if local_rs else np.nan,
                        "mean_loo_z": fisher_z(float(np.mean(local_rs))) if local_rs else np.nan,
                        "n_subjects": n_sub,
                        "local_window_bins": int(hi - lo),
                    }
                )
    return pd.DataFrame(rows)


def summarize_group_isc_tests(isc_df: pd.DataFrame, event_types: Iterable[str]) -> pd.DataFrame:
    stats_rows = []
    for ev in event_types:
        sub = isc_df[isc_df["event_type"] == ev].copy()
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
                "asd_mean_z": float(np.nanmean(asd)) if len(asd) else np.nan,
                "td_mean_z": float(np.nanmean(td)) if len(td) else np.nan,
                "asd_mean_r": float(np.tanh(np.nanmean(asd))) if len(asd) else np.nan,
                "td_mean_r": float(np.tanh(np.nanmean(td))) if len(td) else np.nan,
                "mean_diff_asd_minus_td_z": float(np.nanmean(asd) - np.nanmean(td)) if len(asd) and len(td) else np.nan,
                "t_stat": t_stat,
                "p_value": p_val,
            }
        )
    return pd.DataFrame(stats_rows)
