"""specparam 与预处理质量控制。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.io_utils import save_csv

logger = logging.getLogger(__name__)


def _knee_bounds(qc_cfg: dict[str, Any], cfg: dict[str, Any] | None = None) -> tuple[float, float]:
    """Resolve plausible knee frequency bounds (Hz)."""
    knee_min = qc_cfg.get("knee_min_hz")
    knee_max = qc_cfg.get("knee_max_hz")
    if knee_min is None or knee_max is None:
        freq_range = (cfg or {}).get("specparam", {}).get("freq_range", [1.0, 40.0])
        if knee_min is None:
            knee_min = float(freq_range[0])
        if knee_max is None:
            knee_max = float(freq_range[1])
    return float(knee_min), float(knee_max)


def flag_invalid_fits(
    df: pd.DataFrame,
    qc_cfg: dict[str, Any],
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    标记低质量通道拟合。

    规则:
    - r_squared < min_r_squared
    - exponent 超出 [exponent_min, exponent_max]
    - fit_error 位于全样本最高 top_percentile%
    - knee 模式：aperiodic_knee 超出 [knee_min_hz, knee_max_hz]
    """
    out = df.copy()
    min_r2 = qc_cfg.get("min_r_squared", 0.90)
    exp_min = qc_cfg.get("exponent_min", 0.0)
    exp_max = qc_cfg.get("exponent_max", 5.0)
    top_pct = qc_cfg.get("fit_error_top_percentile", 5.0)

    error_thresh = np.nanpercentile(out["fit_error"], 100 - top_pct)

    invalid = (
        (out["r_squared"] < min_r2)
        | (out["aperiodic_exponent"] < exp_min)
        | (out["aperiodic_exponent"] > exp_max)
        | (out["fit_error"] >= error_thresh)
    )
    if "aperiodic_knee" in out.columns:
        knee_min, knee_max = _knee_bounds(qc_cfg, cfg)
        knee = pd.to_numeric(out["aperiodic_knee"], errors="coerce")
        bad_knee = knee.notna() & ((knee < knee_min) | (knee > knee_max))
        invalid = invalid | bad_knee

    out["fit_valid"] = ~invalid
    out["invalid_reason"] = ""
    out.loc[out["r_squared"] < min_r2, "invalid_reason"] += "low_r2;"
    out.loc[
        (out["aperiodic_exponent"] < exp_min) | (out["aperiodic_exponent"] > exp_max),
        "invalid_reason",
    ] += "bad_exponent;"
    out.loc[out["fit_error"] >= error_thresh, "invalid_reason"] += "high_error;"
    if "aperiodic_knee" in out.columns:
        knee_min, knee_max = _knee_bounds(qc_cfg, cfg)
        knee = pd.to_numeric(out["aperiodic_knee"], errors="coerce")
        out.loc[knee.notna() & ((knee < knee_min) | (knee > knee_max)), "invalid_reason"] += "bad_knee;"

    logger.info(
        "QC: %d / %d 通道有效 (%.1f%%)",
        out["fit_valid"].sum(),
        len(out),
        100 * out["fit_valid"].mean(),
    )
    return out


def summarize_subject_qc(
    df: pd.DataFrame,
    max_invalid_ratio: float = 0.20,
) -> pd.DataFrame:
    """按被试汇总 specparam QC。"""
    rows = []
    for sid, sub in df.groupby("subject_id"):
        total = len(sub)
        valid = sub["fit_valid"].sum()
        invalid = total - valid
        ratio = invalid / total if total > 0 else np.nan
        rows.append({
            "subject_id": sid,
            "group": sub["group"].iloc[0],
            "total_channels": total,
            "valid_channels": int(valid),
            "invalid_channels": int(invalid),
            "mean_r_squared": sub["r_squared"].mean(),
            "mean_fit_error": sub["fit_error"].mean(),
            "invalid_channel_ratio": ratio,
            "low_quality_subject": int(ratio > max_invalid_ratio),
        })
    return pd.DataFrame(rows)


def run_specparam_qc(
    channel_results: pd.DataFrame,
    cfg: dict[str, Any],
    out_channel_csv: Path,
    out_subject_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """完整 specparam QC 流程。"""
    qc_cfg = cfg.get("fit_quality", {})
    max_ratio = qc_cfg.get("subject_invalid_channel_ratio_max", 0.20)

    flagged = flag_invalid_fits(channel_results, qc_cfg, cfg=cfg)
    summary = summarize_subject_qc(flagged, max_invalid_ratio=max_ratio)

    save_csv(flagged, out_channel_csv)
    save_csv(summary, out_subject_csv)
    return flagged, summary
