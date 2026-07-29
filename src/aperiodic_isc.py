"""Aperiodic-ISC：非周期 exponent 时序与 TD 模板相关。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def zscore_series(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return x
    sd = np.nanstd(x, ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return x * 0.0
    return (x - np.nanmean(x)) / sd


def resample_series(series: np.ndarray, n_target: int) -> np.ndarray:
    """将不等长时序重采样至统一长度（片段 ISC 对齐）。"""
    x = np.asarray(series, dtype=float)
    if len(x) == 0 or n_target < 3:
        return np.full(n_target, np.nan)
    if len(x) == n_target:
        return x
    src = np.linspace(0.0, 1.0, len(x))
    dst = np.linspace(0.0, 1.0, n_target)
    out = np.interp(dst, src, x)
    return out


def compute_isc_r(
    subject_series: np.ndarray,
    template_series: np.ndarray,
    min_overlap: int = 10,
    resample_to: int | None = 80,
) -> tuple[float, int]:
    """Pearson r（原始值）；返回 (r, n_overlap)。"""
    s = np.asarray(subject_series, dtype=float)
    t = np.asarray(template_series, dtype=float)
    if resample_to is not None and len(s) >= min_overlap and len(t) >= min_overlap:
        n_use = resample_to
        s = resample_series(s, n_use)
        t = resample_series(t, n_use)
    else:
        n_use = min(len(s), len(t))
        s, t = s[:n_use], t[:n_use]
    mask = np.isfinite(s) & np.isfinite(t)
    if mask.sum() < min_overlap:
        return np.nan, int(mask.sum())
    r, _ = stats.pearsonr(s[mask], t[mask])
    return float(r), int(mask.sum())


def fisher_z(r: float) -> float:
    if not np.isfinite(r):
        return np.nan
    r = np.clip(r, -0.999999, 0.999999)
    return float(np.arctanh(r))


def build_td_loo_template(
    series_by_subject: dict[str, np.ndarray],
    groups: dict[str, str],
    exclude_subject: str | None = None,
    resample_to: int = 80,
    min_windows: int = 15,
) -> np.ndarray:
    """TD 被试 leave-one-out 模板：重采样至统一长度后逐点平均。"""
    td_ids = [sid for sid, g in groups.items() if g == "TD" and sid != exclude_subject]
    if not td_ids:
        return np.array([], dtype=float)
    arrs = []
    for sid in td_ids:
        if sid not in series_by_subject:
            continue
        s = series_by_subject[sid]
        if len(s) < min_windows:
            continue
        arrs.append(resample_series(s, resample_to))
    if not arrs:
        return np.array([], dtype=float)
    stack = np.vstack(arrs)
    return np.nanmean(stack, axis=0)


def compute_cohort_isc(
    series_df: pd.DataFrame,
    subject_col: str = "subject_id",
    group_col: str = "group",
    value_col: str = "exponent",
    event_col: str = "event_type",
    template: str = "TD_LOO",
    min_overlap: int = 10,
    resample_to: int = 80,
    min_windows: int = 15,
) -> pd.DataFrame:
    """
    计算被试级 ISC。

    series_df 须含逐窗 exponent 及 event_type（mental/pain/neutral）。
    每被试每 event：将该 event 全部窗口按时间顺序拼接后计算 ISC。
    """
    rows: list[dict[str, Any]] = []
    for event_type, ev_df in series_df.groupby(event_col):
        series_by_subject: dict[str, np.ndarray] = {}
        groups: dict[str, str] = {}
        for sid, g in ev_df.groupby(subject_col):
            sid_s = str(sid)
            sub = g.sort_values(["window_start_sec", "window_index"], na_position="last")
            series_by_subject[sid_s] = sub[value_col].to_numpy(dtype=float)
            groups[sid_s] = str(g[group_col].iloc[0])

        for sid, y in series_by_subject.items():
            grp = groups[sid]
            if len(y) < min_windows:
                rows.append(
                    {
                        "subject_id": sid,
                        "group": grp,
                        "event_type": event_type,
                        "isc_r": np.nan,
                        "isc_z": np.nan,
                        "n_overlap_points": len(y),
                        "template_type": template,
                    }
                )
                continue
            if template == "TD_LOO":
                tmpl = build_td_loo_template(
                    series_by_subject,
                    groups,
                    exclude_subject=sid if grp == "TD" else None,
                    resample_to=resample_to,
                    min_windows=min_windows,
                )
            else:
                tmpl = build_td_loo_template(
                    series_by_subject,
                    groups,
                    exclude_subject=None,
                    resample_to=resample_to,
                    min_windows=min_windows,
                )

            r, n_ov = compute_isc_r(y, tmpl, min_overlap=min_overlap, resample_to=resample_to)
            rows.append(
                {
                    "subject_id": sid,
                    "group": grp,
                    "event_type": event_type,
                    "isc_r": r,
                    "isc_z": fisher_z(r),
                    "n_overlap_points": n_ov,
                    "template_type": template,
                }
            )
    return pd.DataFrame(rows)


def compute_trajectory_metrics(timeseries: np.ndarray) -> dict[str, float]:
    """Trajectory-level summary metrics for ISC mechanism analysis."""
    from src.jr_dynamic_disambiguation import compute_trajectory_features

    return compute_trajectory_features(timeseries)


def compute_lagged_isc(
    subject_series: np.ndarray,
    template_series: np.ndarray,
    max_lag: int = 10,
    resample_to: int | None = 80,
    min_overlap: int = 10,
) -> dict[str, float]:
    """
    Compute zero-lag and lagged ISC profile.

    Returns zero_lag_isc, max_lag_isc, optimal_lag, and lag_profile as string.
    """
    s = resample_series(np.asarray(subject_series, dtype=float), resample_to or 80)
    t = resample_series(np.asarray(template_series, dtype=float), resample_to or 80)
    n = len(s)
    profile: list[float] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            ss, tt = s[-lag:], t[: n + lag]
        elif lag > 0:
            ss, tt = s[: n - lag], t[lag:]
        else:
            ss, tt = s, t
        mask = np.isfinite(ss) & np.isfinite(tt)
        if mask.sum() < min_overlap:
            profile.append(float("nan"))
        else:
            r, _ = stats.pearsonr(ss[mask], tt[mask])
            profile.append(float(r))

    arr = np.asarray(profile, dtype=float)
    valid = np.isfinite(arr)
    zero_idx = max_lag
    zero_lag = float(arr[zero_idx]) if valid[zero_idx] else float("nan")
    if valid.any():
        best_i = int(np.nanargmax(arr))
        return {
            "zero_lag_isc": zero_lag,
            "max_lag_isc": float(arr[best_i]),
            "optimal_lag": int(best_i - max_lag),
            "lag_profile": "|".join(f"{v:.4f}" if np.isfinite(v) else "nan" for v in arr),
        }
    return {
        "zero_lag_isc": float("nan"),
        "max_lag_isc": float("nan"),
        "optimal_lag": 0,
        "lag_profile": "",
    }


def isc_group_tests(isc_df: pd.DataFrame) -> pd.DataFrame:
    """各 event_type 的 ASD vs TD Welch t 检验。"""
    rows = []
    for ev, sub in isc_df.groupby("event_type"):
        asd = sub.loc[sub["group"] == "ASD", "isc_z"].dropna()
        td = sub.loc[sub["group"] == "TD", "isc_z"].dropna()
        if len(asd) < 2 or len(td) < 2:
            continue
        t, p = stats.ttest_ind(asd, td, equal_var=False)
        rows.append(
            {
                "event_type": ev,
                "n_asd": len(asd),
                "n_td": len(td),
                "asd_mean_z": float(asd.mean()),
                "td_mean_z": float(td.mean()),
                "asd_mean_r": float(sub.loc[sub["group"] == "ASD", "isc_r"].mean()),
                "td_mean_r": float(sub.loc[sub["group"] == "TD", "isc_r"].mean()),
                "t_stat": float(t),
                "p_value": float(p),
            }
        )
    return pd.DataFrame(rows)
