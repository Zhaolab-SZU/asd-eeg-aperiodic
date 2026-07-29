"""数据读写工具。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def load_participants(path: Path, included_only: bool = True) -> pd.DataFrame:
    """
    加载 participants.csv。

    Parameters
    ----------
    path : Path
        participants 文件路径
    included_only : bool
        若 True，仅保留 included_final == 1 的被试
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 participants 文件: {path}\n"
            "请先填写 data/participants/participants.csv"
        )
    df = pd.read_csv(path)
    required = ["subject_id", "group", "raw_EEG_file"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"participants.csv 缺少必需列: {missing}")

    if included_only and "included_final" in df.columns:
        df = df[df["included_final"].astype(int) == 1].copy()
    df["subject_id"] = df["subject_id"].astype(str)
    return df.reset_index(drop=True)


def load_analysis_participants(
    cfg: dict[str, Any],
    min_epochs: int | None = None,
) -> pd.DataFrame:
    """
    加载纳入 PSD / specparam / 统计 的分析队列。

    条件: included_final == 1，且预处理有效 epoch >= min_usable_epochs。
    优先读取 derivatives/participants_analysis.csv（由 02_preprocess 生成）。
    """
    deriv = Path(cfg["paths"]["derivatives_root"])
    min_epochs = min_epochs if min_epochs is not None else int(
        cfg.get("epochs", {}).get("min_usable_epochs", 60)
    )

    analysis_path = deriv / "participants_analysis.csv"
    if analysis_path.exists():
        df = pd.read_csv(analysis_path)
        df["subject_id"] = df["subject_id"].astype(str)
        if "usable_epochs" in df.columns:
            n_before = len(df)
            df = df[df["usable_epochs"] >= min_epochs].copy()
            if len(df) < n_before:
                logger.info(
                    "按 epoch 阈值再筛: %d -> %d (>= %d)",
                    n_before, len(df), min_epochs,
                )
        logger.info("分析队列 %d 名 ← participants_analysis.csv", len(df))
        return df.reset_index(drop=True)

    participants = load_participants(Path(cfg["paths"]["participants_file"]), included_only=True)
    summary_path = deriv / "qc" / "preproc_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            "未找到 preproc_summary.csv，请先运行 scripts/02_preprocess_eeg.py"
        )

    summary = pd.read_csv(summary_path)
    summary["subject_id"] = summary["subject_id"].astype(str)
    passed = summary.loc[summary["usable_epochs"] >= min_epochs, "subject_id"]
    df = participants[participants["subject_id"].isin(passed)].copy()
    df = df.merge(
        summary[["subject_id", "usable_epochs", "usable_seconds"]],
        on="subject_id",
        how="left",
    )
    logger.info(
        "分析队列 %d / %d 名 (epoch >= %d，来自 preproc_summary)",
        len(df),
        len(participants),
        min_epochs,
    )
    return df.reset_index(drop=True)


def attach_usable_epochs(df: pd.DataFrame, deriv_root: Path) -> pd.DataFrame:
    """
    确保 DataFrame 含可用的 usable_epochs 列。

    避免重复 merge 产生 usable_epochs_x/y，以及用全空的 EEG_usable_epochs 覆盖真实值。
    """
    df = df.copy()
    if "usable_epochs_x" in df.columns:
        df["usable_epochs"] = df["usable_epochs_x"]
        if "usable_epochs_y" in df.columns:
            df["usable_epochs"] = df["usable_epochs"].fillna(df["usable_epochs_y"])
        df = df.drop(columns=[c for c in ("usable_epochs_x", "usable_epochs_y") if c in df.columns])
        return df

    if "usable_epochs" in df.columns:
        return df

    preproc = Path(deriv_root) / "qc" / "preproc_summary.csv"
    if preproc.exists():
        summ = pd.read_csv(preproc)
        summ["subject_id"] = summ["subject_id"].astype(str)
        df["subject_id"] = df["subject_id"].astype(str)
        df = df.merge(summ[["subject_id", "usable_epochs"]], on="subject_id", how="left")

    if "usable_epochs" not in df.columns and "EEG_usable_epochs" in df.columns:
        ue = pd.to_numeric(df["EEG_usable_epochs"], errors="coerce")
        if ue.notna().any():
            df["usable_epochs"] = ue
    return df


def exclude_specparam_low_quality(df: pd.DataFrame, deriv_root: Path) -> pd.DataFrame:
    """排除 specparam QC 标记的低质量被试。"""
    qc_path = Path(deriv_root) / "specparam" / "specparam_qc_summary_subject.csv"
    if not qc_path.exists():
        return df
    qc = pd.read_csv(qc_path)
    bad = qc.loc[qc["low_quality_subject"] == 1, "subject_id"].astype(str)
    out = df[~df["subject_id"].astype(str).isin(bad)].copy()
    if len(out) < len(df):
        logger.info("排除 specparam 低质量被试: %d -> %d", len(df), len(out))
    return out


def save_json(data: dict[str, Any], path: Path) -> None:
    """保存 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    """加载 JSON。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    """保存 CSV。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)
    logger.info("已保存: %s (%d 行)", path, len(df))


def load_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    """加载 CSV。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {path}")
    return pd.read_csv(path, **kwargs)


def ensure_dir(path: Path) -> Path:
    """确保目录存在。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_epochs_fif_path(epochs_dir: Path, subject_id: str) -> Path:
    """查找被试 epochs 文件（兼容新旧命名）。"""
    epochs_dir = Path(epochs_dir)
    for name in (f"{subject_id}-epo.fif", f"{subject_id}_epochs.fif"):
        p = epochs_dir / name
        if p.exists():
            return p
    return epochs_dir / f"{subject_id}-epo.fif"


def get_raw_fif_path(preproc_dir: Path, subject_id: str) -> Path:
    """查找被试清洗后 raw 文件（兼容新旧命名）。"""
    preproc_dir = Path(preproc_dir)
    for name in (f"{subject_id}-raw.fif", f"{subject_id}_raw_clean.fif"):
        p = preproc_dir / name
        if p.exists():
            return p
    return preproc_dir / f"{subject_id}-raw.fif"
