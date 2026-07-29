"""ROI 通道聚合。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_roi_config
from src.io_utils import save_csv

logger = logging.getLogger(__name__)


def get_roi_dict(roi_cfg: dict[str, Any], layout: str | None = None) -> dict[str, list[str]]:
    """获取 ROI -> 通道列表 映射。"""
    layout = layout or roi_cfg.get("default_layout", "channels_32")
    if layout not in roi_cfg:
        raise KeyError(f"ROI 布局 '{layout}' 未在 roi_channels.yaml 中定义")
    return roi_cfg[layout]


def detect_layout(channels: list[str], roi_cfg: dict[str, Any]) -> str:
    """根据可用通道自动选择 ROI 方案（优先 EGI E 编号）。"""
    ch_set = set(channels)
    if any(c.upper().startswith("E") and c[1:].isdigit() for c in ch_set):
        if "channels_egi64" in roi_cfg:
            logger.info("自动选择 ROI 布局: channels_egi64 (EGI E1–E64)")
            return "channels_egi64"
    for layout in ("channels_egi64", "channels_64", "channels_32"):
        if layout not in roi_cfg:
            continue
        roi_dict = roi_cfg[layout]
        all_roi_ch = set()
        for ch_list in roi_dict.values():
            all_roi_ch.update(ch_list)
        overlap = len(ch_set & all_roi_ch)
        if overlap >= 0.5 * len(all_roi_ch):
            logger.info("自动选择 ROI 布局: %s (overlap=%d)", layout, overlap)
            return layout
    return roi_cfg.get("default_layout", "channels_32")


def aggregate_roi_for_subject(
    sub_df: pd.DataFrame,
    roi_dict: dict[str, list[str]],
    min_ratio: float = 0.5,
    value_cols: tuple[str, ...] = ("aperiodic_exponent", "aperiodic_offset"),
) -> dict[str, Any]:
    """
    单被试 ROI 聚合。

    仅使用 fit_valid==True 的通道（若列存在）。
    """
    if "fit_valid" in sub_df.columns:
        valid_df = sub_df[sub_df["fit_valid"]].copy()
    else:
        valid_df = sub_df.copy()

    result: dict[str, Any] = {
        "subject_id": sub_df["subject_id"].iloc[0],
        "group": sub_df["group"].iloc[0],
    }

    # Global
    for col in value_cols:
        short = col.replace("aperiodic_", "")
        result[f"global_{short}"] = valid_df[col].mean() if len(valid_df) else np.nan

    # Per ROI
    long_rows = []
    for roi, ch_list in roi_dict.items():
        roi_ch = [c for c in ch_list if c in valid_df["channel"].values]
        roi_sub = valid_df[valid_df["channel"].isin(roi_ch)]
        n_total = len(ch_list)
        n_valid = len(roi_sub)

        if n_valid < min_ratio * n_total:
            exp_val, off_val = np.nan, np.nan
        else:
            exp_val = roi_sub["aperiodic_exponent"].mean()
            off_val = roi_sub["aperiodic_offset"].mean()

        result[f"{roi}_exponent"] = exp_val
        long_rows.append({
            "subject_id": result["subject_id"],
            "group": result["group"],
            "roi": roi,
            "exponent": exp_val,
            "offset": off_val,
            "n_valid_channels": n_valid,
        })

    return result, pd.DataFrame(long_rows)


def compute_roi_metrics(
    channel_df: pd.DataFrame,
    roi_cfg: dict[str, Any],
    min_ratio: float | None = None,
    layout_override: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """批量计算 ROI 指标。"""
    min_ratio = min_ratio or roi_cfg.get("min_valid_channel_ratio", 0.5)
    channels = channel_df["channel"].unique().tolist()
    layout = layout_override or detect_layout(channels, roi_cfg)
    if layout_override:
        logger.info("ROI 布局由配置指定: %s", layout_override)
    roi_dict = get_roi_dict(roi_cfg, layout)

    wide_rows = []
    long_parts = []
    for sid, sub in channel_df.groupby("subject_id"):
        wide, long_df = aggregate_roi_for_subject(sub, roi_dict, min_ratio)
        wide_rows.append(wide)
        long_parts.append(long_df)

    wide_df = pd.DataFrame(wide_rows)
    long_df = pd.concat(long_parts, ignore_index=True)
    return wide_df, long_df


def run_roi_pipeline(
    channel_qc_csv: Path,
    out_global: Path,
    out_long: Path,
    roi_cfg_path: Path | None = None,
    layout_override: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ROI 流水线入口。"""
    channel_df = pd.read_csv(channel_qc_csv)
    roi_cfg = load_roi_config(roi_cfg_path) if roi_cfg_path else load_roi_config()
    wide, long = compute_roi_metrics(channel_df, roi_cfg, layout_override=layout_override)
    save_csv(wide, out_global)
    save_csv(long, out_long)
    return wide, long
