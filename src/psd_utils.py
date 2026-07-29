"""Welch PSD 计算与 QC 图。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from src.io_utils import ensure_dir, save_csv

logger = logging.getLogger(__name__)


def _validate_sliding_window(window_sec: float, step_sec: float) -> None:
    if window_sec <= 0:
        raise ValueError("sliding_window.window_sec 必须 > 0")
    if step_sec <= 0:
        raise ValueError("sliding_window.step_sec 必须 > 0")
    if step_sec > window_sec:
        raise ValueError("sliding_window.step_sec 不能大于 window_sec")


def compute_psd_matrix_from_epochs(
    epochs: mne.Epochs,
    fmin: float,
    fmax: float,
    welch_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """返回逐 epoch PSD: (n_epochs, n_channels, n_freqs)。"""
    n_fft = welch_cfg.get("n_fft", 500)
    n_overlap = welch_cfg.get("n_overlap", 250)
    method = welch_cfg.get("window", "hamming")

    spectrum = epochs.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        n_overlap=n_overlap,
        window=method,
        verbose=False,
    )
    freqs = spectrum.freqs
    psd = spectrum.get_data()
    ch_names = spectrum.ch_names
    return freqs, psd, ch_names


def compute_psd_from_epochs(
    epochs: mne.Epochs,
    fmin: float,
    fmax: float,
    welch_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    使用 Welch 方法计算各通道 PSD。

    Returns
    -------
    freqs, psd (n_channels, n_freqs), ch_names
    """
    freqs, psd_matrix, ch_names = compute_psd_matrix_from_epochs(
        epochs=epochs, fmin=fmin, fmax=fmax, welch_cfg=welch_cfg
    )
    psd = psd_matrix.mean(axis=0)
    return freqs, psd, ch_names


def psd_to_long_df(
    subject_id: str,
    group: str,
    freqs: np.ndarray,
    psd: np.ndarray,
    ch_names: list[str],
) -> pd.DataFrame:
    """将 PSD 转为长表 CSV 格式。"""
    rows = []
    for ch_i, ch in enumerate(ch_names):
        for fi, f in enumerate(freqs):
            rows.append(
                {
                    "subject_id": subject_id,
                    "group": group,
                    "channel": ch,
                    "frequency": float(f),
                    "power": float(psd[ch_i, fi]),
                }
            )
    return pd.DataFrame(rows)


def psd_windows_to_wide_df(
    subject_id: str,
    group: str,
    freqs: np.ndarray,
    psd_windows: np.ndarray,
    ch_names: list[str],
    window_starts: np.ndarray,
    window_ends: np.ndarray,
) -> pd.DataFrame:
    """将滑窗 PSD 存为宽表，避免长表体积爆炸。"""
    rows = []
    freq_cols = [f"power_{f:.2f}" for f in freqs]
    for wi in range(psd_windows.shape[0]):
        for ci, ch in enumerate(ch_names):
            row = {
                "subject_id": subject_id,
                "group": group,
                "window_index": int(wi),
                "window_start_sec": float(window_starts[wi]),
                "window_end_sec": float(window_ends[wi]),
                "channel": ch,
            }
            row.update({
                col: float(val) for col, val in zip(freq_cols, psd_windows[wi, ci, :], strict=True)
            })
            rows.append(row)
    return pd.DataFrame(rows)


def dynamic_psd_to_mean_long_df(psd_dynamic_df: pd.DataFrame) -> pd.DataFrame:
    """将滑窗宽表按窗口平均后转成长表（用于 QC 出图）。"""
    freq_cols = [c for c in psd_dynamic_df.columns if c.startswith("power_")]
    if not freq_cols:
        raise ValueError("未检测到 power_* 频率列，无法生成 QC 均值 PSD")

    meta_cols = ["subject_id", "group", "channel"]
    mean_df = psd_dynamic_df.groupby(meta_cols, as_index=False)[freq_cols].mean()

    rows = []
    for _, row in mean_df.iterrows():
        for col in freq_cols:
            freq = float(col.replace("power_", ""))
            rows.append(
                {
                    "subject_id": row["subject_id"],
                    "group": row["group"],
                    "channel": row["channel"],
                    "frequency": freq,
                    "power": float(row[col]),
                }
            )
    return pd.DataFrame(rows)


def compute_subject_psd(
    subject_id: str,
    group: str,
    epochs_path: Path,
    out_csv: Path,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """单被试 PSD 计算并保存。"""
    epochs_path = Path(epochs_path)
    if not epochs_path.exists():
        raise FileNotFoundError(f"未找到 epochs 文件: {epochs_path}")

    psd_cfg = cfg["psd"]
    welch_cfg = psd_cfg.get("welch", {})
    fmin = psd_cfg.get("freq_min_hz", welch_cfg.get("fmin", 1.0))
    fmax = psd_cfg.get("freq_max_hz", welch_cfg.get("fmax", 40.0))

    epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
    freqs, psd, ch_names = compute_psd_from_epochs(epochs, fmin, fmax, welch_cfg)
    df = psd_to_long_df(subject_id, group, freqs, psd, ch_names)
    save_csv(df, out_csv)
    return df


def compute_subject_sliding_psd(
    subject_id: str,
    group: str,
    raw_path: Path,
    out_csv: Path,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """单被试滑窗 PSD（用于时变 specparam）。"""
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"未找到预处理后 raw 文件: {raw_path}")

    psd_cfg = cfg["psd"]
    welch_cfg = psd_cfg.get("welch", {})
    fmin = psd_cfg.get("freq_min_hz", welch_cfg.get("fmin", 1.0))
    fmax = psd_cfg.get("freq_max_hz", welch_cfg.get("fmax", 40.0))
    sw_cfg = psd_cfg.get("sliding_window", {})
    window_sec = float(sw_cfg.get("window_sec", 2.0))
    step_sec = float(sw_cfg.get("step_sec", 0.5))
    _validate_sliding_window(window_sec, step_sec)

    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    overlap = max(0.0, window_sec - step_sec)
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=window_sec,
        overlap=overlap,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )
    if len(epochs) == 0:
        raise RuntimeError(f"{subject_id}: 未生成任何滑窗 epoch，请检查 raw 时长与参数")

    freqs, psd_windows, ch_names = compute_psd_matrix_from_epochs(epochs, fmin, fmax, welch_cfg)
    sfreq = raw.info["sfreq"]
    starts = (epochs.events[:, 0] - raw.first_samp) / sfreq
    ends = starts + window_sec

    df = psd_windows_to_wide_df(
        subject_id=subject_id,
        group=group,
        freqs=freqs,
        psd_windows=psd_windows,
        ch_names=ch_names,
        window_starts=starts,
        window_ends=ends,
    )
    save_csv(df, out_csv)
    return df


def plot_subject_psd(
    df: pd.DataFrame,
    out_path: Path,
    title: str | None = None,
) -> None:
    """绘制单被试全通道 PSD。"""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(8, 5))
    for ch, sub in df.groupby("channel"):
        ax.semilogy(sub["frequency"], sub["power"], alpha=0.5, linewidth=0.8, label=ch)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(title or df["subject_id"].iloc[0])
    ax.set_xlim(df["frequency"].min(), df["frequency"].max())
    # 通道过多时不显示图例
    if df["channel"].nunique() <= 20:
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_group_mean_psd(
    psd_dir: Path,
    participants: pd.DataFrame,
    cfg: dict[str, Any],
    out_path: Path,
) -> None:
    """ASD vs TD 平均 PSD 对比图。"""
    groups = cfg.get("groups", {})
    asd_label = groups.get("asd_label", "ASD")
    td_label = groups.get("td_label", "TD")

    all_dfs = []
    for _, row in participants.iterrows():
        sid = row["subject_id"]
        csv_path = psd_dir / f"{sid}_psd.csv"
        if csv_path.exists():
            all_dfs.append(pd.read_csv(csv_path))
    if not all_dfs:
        logger.warning("无 PSD 文件，跳过组平均图")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for grp, color in [(asd_label, "C0"), (td_label, "C1")]:
        sub = combined[combined["group"] == grp]
        if sub.empty:
            continue
        mean_psd = sub.groupby("frequency")["power"].mean()
        ax.semilogy(mean_psd.index, mean_psd.values, label=grp, color=color, linewidth=2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean power (log scale)")
    ax.set_title("Group mean PSD")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    logger.info("已保存组平均 PSD: %s", out_path)


def plot_group_mean_psd_from_df(
    combined: pd.DataFrame,
    cfg: dict[str, Any],
    out_path: Path,
) -> None:
    """从合并好的长表直接绘制 ASD vs TD 组均值 PSD。"""
    groups = cfg.get("groups", {})
    asd_label = groups.get("asd_label", "ASD")
    td_label = groups.get("td_label", "TD")
    if combined.empty:
        logger.warning("组均值 PSD 输入为空，跳过绘图")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for grp, color in [(asd_label, "C0"), (td_label, "C1")]:
        sub = combined[combined["group"] == grp]
        if sub.empty:
            continue
        mean_psd = sub.groupby("frequency")["power"].mean()
        ax.semilogy(mean_psd.index, mean_psd.values, label=grp, color=color, linewidth=2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean power (log scale)")
    ax.set_title("Group mean PSD")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    logger.info("已保存组平均 PSD: %s", out_path)


def compute_band_power_from_psd(
    psd_df: pd.DataFrame,
    bands: dict[str, list[float]],
) -> pd.DataFrame:
    """从 PSD 长表计算传统频段功率（梯形积分）。"""
    rows = []
    for (sid, grp, ch), sub in psd_df.groupby(["subject_id", "group", "channel"]):
        sub = sub.sort_values("frequency")
        for band_name, (lo, hi) in bands.items():
            mask = (sub["frequency"] >= lo) & (sub["frequency"] <= hi)
            band_sub = sub.loc[mask]
            if len(band_sub) < 2:
                power = np.nan
            else:
                power = float(np.trapz(band_sub["power"], band_sub["frequency"]))
            rows.append(
                {
                    "subject_id": sid,
                    "group": grp,
                    "channel": ch,
                    "band": band_name,
                    "band_power": power,
                }
            )
    return pd.DataFrame(rows)
