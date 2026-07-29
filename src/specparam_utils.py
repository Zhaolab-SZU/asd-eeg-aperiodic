"""specparam / FOOOF 拟合工具。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.io_utils import save_csv

logger = logging.getLogger(__name__)

try:
    from specparam import SpectralModel
except ImportError:
    try:
        from fooof import SpectralModel  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "请安装 specparam: pip install specparam"
        ) from exc


# 频段标签与频率范围（用于从峰列表提取）
BAND_DEFS = {
    "theta": (4.0, 7.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 40.0),
}

PEAK_COLUMNS = [
    "theta_cf", "theta_pw", "theta_bw",
    "alpha_cf", "alpha_pw", "alpha_bw",
    "beta_cf", "beta_pw", "beta_bw",
    "low_gamma_cf", "low_gamma_pw", "low_gamma_bw",
]


def _nan_fit_result() -> dict[str, Any]:
    cols = [
        "aperiodic_exponent",
        "aperiodic_offset",
        "r_squared",
        "fit_error",
        "n_peaks",
        *[f"{b}_{m}" for b in BAND_DEFS for m in ("cf", "pw", "bw")],
    ]
    fit = {k: np.nan for k in cols}
    fit["n_peaks"] = 0
    return fit


def _extract_freq_columns(psd_df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    freq_cols = [c for c in psd_df.columns if c.startswith("power_")]
    if not freq_cols:
        return [], np.array([], dtype=float)
    freqs = np.array([float(c.replace("power_", "")) for c in freq_cols], dtype=float)
    order = np.argsort(freqs)
    sorted_cols = [freq_cols[i] for i in order]
    sorted_freqs = freqs[order]
    return sorted_cols, sorted_freqs


def _extract_fit_from_model(
    model: SpectralModel,
    sp_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    从拟合后的 SpectralModel 提取参数。

    specparam 2.x 使用 get_params / get_metrics；fooof 1.x / specparam 1.x 使用 *_ 后缀属性。
    """
    aperiodic_mode = sp_cfg.get("aperiodic_mode", "fixed")

    if hasattr(model, "get_params"):
        results = getattr(model, "results", None)
        if results is not None and hasattr(results, "has_model") and not results.has_model:
            raise RuntimeError("specparam 拟合未成功 (has_model=False)")

        ap_params = np.asarray(model.get_params("aperiodic"))
        peak_raw = model.get_params("peak")
        peak_params = None if peak_raw is None else np.asarray(peak_raw)
        if peak_params is not None and peak_params.size == 0:
            peak_params = None

        r_squared = float(model.get_metrics("gof"))
        fit_error = float(model.get_metrics("error"))
    else:
        if not hasattr(model, "aperiodic_params_"):
            raise RuntimeError("specparam 拟合未成功 (无 aperiodic_params_)")

        ap_params = np.asarray(model.aperiodic_params_)
        peak_params = getattr(model, "peak_params_", None)
        if peak_params is not None and len(peak_params) == 0:
            peak_params = None

        r_squared = float(model.r_squared_)
        fit_error = float(model.error_)

    # fixed:  [offset, exponent]
    # knee:   [offset, knee_frequency, exponent] — exponent 为高频段斜率
    knee = np.nan
    if aperiodic_mode == "fixed":
        offset, exponent = float(ap_params[0]), float(ap_params[1])
    else:
        offset = float(ap_params[0])
        knee = float(ap_params[1])
        exponent = float(ap_params[2])

    n_peaks = len(peak_params) if peak_params is not None else 0
    band_peaks = extract_band_peaks(peak_params)

    out: dict[str, Any] = {
        "aperiodic_exponent": exponent,
        "aperiodic_offset": offset,
        "r_squared": r_squared,
        "fit_error": fit_error,
        "n_peaks": int(n_peaks),
        **band_peaks,
    }
    if aperiodic_mode != "fixed":
        out["aperiodic_knee"] = knee
    return out


def _assign_peak_to_band(
    peak_cf: float,
    peak_pw: float,
    peak_bw: float,
    band_defs: dict[str, tuple[float, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """将单个峰分配到频段；若该频段已有峰，保留功率更高者。"""
    band_defs = band_defs or BAND_DEFS
    assignment: dict[str, dict[str, float]] = {}
    for band, (lo, hi) in band_defs.items():
        if lo <= peak_cf <= hi:
            if band not in assignment or peak_pw > assignment[band]["pw"]:
                assignment[band] = {"cf": peak_cf, "pw": peak_pw, "bw": peak_bw}
    return assignment


def extract_band_peaks(
    peak_params: np.ndarray | None,
    band_defs: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """
    从 specparam 峰参数矩阵提取各频段 cf/pw/bw。

    无峰时返回 NaN。
    """
    band_defs = band_defs or BAND_DEFS
    result: dict[str, float] = {}
    for band in band_defs:
        result[f"{band}_cf"] = np.nan
        result[f"{band}_pw"] = np.nan
        result[f"{band}_bw"] = np.nan

    if peak_params is None or len(peak_params) == 0:
        return result

    band_peaks: dict[str, dict[str, float]] = {}
    for peak in peak_params:
        cf, pw, bw = peak[0], peak[1], peak[2]
        assigned = _assign_peak_to_band(cf, pw, bw, band_defs)
        for band, vals in assigned.items():
            if band not in band_peaks or vals["pw"] > band_peaks[band]["pw"]:
                band_peaks[band] = vals

    for band, vals in band_peaks.items():
        result[f"{band}_cf"] = vals["cf"]
        result[f"{band}_pw"] = vals["pw"]
        result[f"{band}_bw"] = vals["bw"]
    return result


def fit_specparam_channel(
    freqs: np.ndarray,
    power: np.ndarray,
    sp_cfg: dict[str, Any],
) -> dict[str, Any]:
    """对单通道 PSD 拟合 specparam。"""
    freq_range = sp_cfg.get("freq_range", [1.0, 40.0])
    model = SpectralModel(
        peak_width_limits=tuple(sp_cfg.get("peak_width_limits", [1, 8])),
        max_n_peaks=sp_cfg.get("max_n_peaks", 6),
        min_peak_height=sp_cfg.get("min_peak_height", 0.1),
        peak_threshold=sp_cfg.get("peak_threshold", 2.0),
        aperiodic_mode=sp_cfg.get("aperiodic_mode", "fixed"),
        verbose=sp_cfg.get("verbose", False),
    )
    # specparam 使用线性功率；若 PSD 为 dB 需先转换
    model.fit(freqs, power, freq_range)
    return _extract_fit_from_model(model, sp_cfg)


def fit_subject_specparam(
    psd_df: pd.DataFrame,
    sp_cfg: dict[str, Any],
) -> pd.DataFrame:
    """对被试所有通道拟合 specparam。"""
    rows = []
    subject_id = psd_df["subject_id"].iloc[0]
    group = psd_df["group"].iloc[0]

    for ch, sub in psd_df.groupby("channel"):
        sub = sub.sort_values("frequency")
        freqs = sub["frequency"].values
        power = sub["power"].values
        try:
            fit = fit_specparam_channel(freqs, power, sp_cfg)
        except Exception as exc:
            logger.warning("%s %s 拟合失败: %s", subject_id, ch, exc)
            fit = _nan_fit_result()

        rows.append({
            "subject_id": subject_id,
            "group": group,
            "channel": ch,
            **fit,
        })
    return pd.DataFrame(rows)


def fit_subject_specparam_dynamic(
    psd_df: pd.DataFrame,
    sp_cfg: dict[str, Any],
) -> pd.DataFrame:
    """对滑窗 PSD 宽表做时变 specparam 拟合（窗口 x 通道）。"""
    freq_cols, freqs = _extract_freq_columns(psd_df)
    if len(freq_cols) == 0:
        raise ValueError("滑窗 PSD 缺少 power_* 频率列")

    required = {"subject_id", "group", "window_index", "window_start_sec", "window_end_sec", "channel"}
    missing = [c for c in required if c not in psd_df.columns]
    if missing:
        raise ValueError(f"滑窗 PSD 缺少必需列: {missing}")

    dynamic_cfg = sp_cfg.get("dynamic", {})
    fit_mode = str(dynamic_cfg.get("fit_mode", "global_mean")).lower()
    if fit_mode not in {"global_mean", "channel", "roi_mean"}:
        raise ValueError("specparam.dynamic.fit_mode 仅支持: global_mean | roi_mean | channel")

    rows = []
    sid = str(psd_df["subject_id"].iloc[0])
    if fit_mode in {"global_mean", "roi_mean"}:
        group_cols = ["subject_id", "group", "window_index", "window_start_sec", "window_end_sec"]
        if fit_mode == "roi_mean":
            target_channels_raw = dynamic_cfg.get("target_channels", [])
            target_channels = {str(ch).strip().upper() for ch in target_channels_raw}
            if not target_channels:
                raise ValueError("roi_mean 模式要求配置 specparam.dynamic.target_channels")
            channel_series = psd_df["channel"].astype(str).str.upper()
            roi_df = psd_df[channel_series.isin(target_channels)].copy()
            if roi_df.empty:
                raise ValueError(
                    f"{sid}: roi_mean 无可用目标通道，target_channels={sorted(target_channels)}"
                )
            # 标记每窗口用于平均的通道数，便于后续 QC。
            channel_count_df = (
                roi_df.groupby(group_cols, as_index=False)["channel"]
                .count()
                .rename(columns={"channel": "aggregated_channel_n"})
            )
            win_df = roi_df.groupby(group_cols, as_index=False)[freq_cols].mean()
            win_df = win_df.merge(channel_count_df, on=group_cols, how="left")
            channel_label = "ROI_TARGET"
        else:
            win_df = psd_df.groupby(group_cols, as_index=False)[freq_cols].mean()
            channel_count_df = (
                psd_df.groupby(group_cols, as_index=False)["channel"]
                .nunique()
                .rename(columns={"channel": "aggregated_channel_n"})
            )
            win_df = win_df.merge(channel_count_df, on=group_cols, how="left")
            channel_label = "GLOBAL"
        for _, rec in win_df.iterrows():
            power = rec[freq_cols].to_numpy(dtype=float)
            try:
                fit = fit_specparam_channel(freqs, power, sp_cfg)
            except Exception as exc:
                logger.warning("%s win=%s %s 拟合失败: %s", sid, rec["window_index"], fit_mode.upper(), exc)
                fit = _nan_fit_result()
            rows.append(
                {
                    "subject_id": rec["subject_id"],
                    "group": rec["group"],
                    "window_index": int(rec["window_index"]),
                    "window_start_sec": float(rec["window_start_sec"]),
                    "window_end_sec": float(rec["window_end_sec"]),
                    "channel": channel_label,
                    "aggregated_channel_n": int(rec.get("aggregated_channel_n", 0)),
                    **fit,
                }
            )
    else:
        for _, rec in psd_df.iterrows():
            power = rec[freq_cols].to_numpy(dtype=float)
            try:
                fit = fit_specparam_channel(freqs, power, sp_cfg)
            except Exception as exc:
                logger.warning(
                    "%s win=%s ch=%s 拟合失败: %s",
                    sid,
                    rec["window_index"],
                    rec["channel"],
                    exc,
                )
                fit = _nan_fit_result()
            rows.append(
                {
                    "subject_id": rec["subject_id"],
                    "group": rec["group"],
                    "window_index": int(rec["window_index"]),
                    "window_start_sec": float(rec["window_start_sec"]),
                    "window_end_sec": float(rec["window_end_sec"]),
                    "channel": rec["channel"],
                    **fit,
                }
            )
    return pd.DataFrame(rows)


def summarize_dynamic_exponent_timeseries(dynamic_fit_df: pd.DataFrame) -> pd.DataFrame:
    """按窗口汇总通道级动态拟合，得到可对齐电影标签的时间序列。"""
    group_cols = ["subject_id", "group", "window_index", "window_start_sec", "window_end_sec"]
    out = (
        dynamic_fit_df.groupby(group_cols, as_index=False)
        .agg(
            exponent_mean=("aperiodic_exponent", "mean"),
            exponent_median=("aperiodic_exponent", "median"),
            offset_mean=("aperiodic_offset", "mean"),
            r_squared_mean=("r_squared", "mean"),
            valid_channel_n=("aperiodic_exponent", lambda x: int(x.notna().sum())),
        )
        .sort_values(["subject_id", "window_index"])
    )
    return out


def run_specparam_batch(
    participants: pd.DataFrame,
    psd_dir: Path,
    out_csv: Path,
    cfg: dict[str, Any],
    sp_cfg_override: dict[str, Any] | None = None,
    psd_suffix: str = "_psd.csv",
    dynamic_summary_out_csv: Path | None = None,
) -> pd.DataFrame:
    """批量 specparam 拟合。"""
    sp_cfg = dict(cfg.get("specparam", {}))
    if sp_cfg_override:
        sp_cfg.update(sp_cfg_override)

    all_results = []
    is_dynamic_mode = None
    for _, row in participants.iterrows():
        sid = row["subject_id"]
        psd_path = psd_dir / f"{sid}{psd_suffix}"
        if not psd_path.exists():
            logger.warning("跳过 %s: 无 PSD 文件", sid)
            continue
        psd_df = pd.read_csv(psd_path)
        has_dynamic_cols = {"window_index", "window_start_sec", "window_end_sec"}.issubset(psd_df.columns)
        if is_dynamic_mode is None:
            is_dynamic_mode = has_dynamic_cols
        if has_dynamic_cols:
            result = fit_subject_specparam_dynamic(psd_df, sp_cfg)
        else:
            result = fit_subject_specparam(psd_df, sp_cfg)
        all_results.append(result)

    if not all_results:
        raise FileNotFoundError(
            f"未找到任何 PSD 文件于 {psd_dir}。请先运行 03_compute_psd.py"
        )

    combined = pd.concat(all_results, ignore_index=True)
    save_csv(combined, out_csv)
    if is_dynamic_mode and dynamic_summary_out_csv is not None:
        dyn_summary = summarize_dynamic_exponent_timeseries(combined)
        save_csv(dyn_summary, dynamic_summary_out_csv)
    return combined
