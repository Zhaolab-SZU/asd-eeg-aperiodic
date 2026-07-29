"""EEG 预处理：读取、滤波、ICA、分段、质控。"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from src.config import get_project_root
from src.io_utils import ensure_dir, save_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 多格式读取
# ---------------------------------------------------------------------------

def read_raw_eeg(filepath: Path, preload: bool = True) -> mne.io.BaseRaw:
    """
    根据扩展名读取 EEG 原始数据。

    支持: .mff (EGI), .set, .vhdr, .edf, .bdf, .fif
    """
    filepath = Path(filepath)
    if not filepath.is_dir() and not filepath.exists():
        raise FileNotFoundError(f"EEG 文件不存在: {filepath}")

    suffix = filepath.suffix.lower()
    if suffix == ".mff":
        logger.info("读取 EGI .mff: %s", filepath)
        return mne.io.read_raw_egi(str(filepath), preload=preload, verbose=False)

    readers = {
        ".set": lambda p: mne.io.read_raw_eeglab(p, preload=preload),
        ".vhdr": lambda p: mne.io.read_raw_brainvision(p, preload=preload),
        ".edf": lambda p: mne.io.read_raw_edf(p, preload=preload),
        ".bdf": lambda p: mne.io.read_raw_bdf(p, preload=preload),
        ".fif": lambda p: mne.io.read_raw_fif(p, preload=preload),
    }
    if suffix not in readers:
        raise ValueError(
            f"不支持的文件格式: {suffix} ({filepath})\n"
            f"支持格式: .mff, {list(readers.keys())}"
        )
    logger.info("读取 %s", filepath)
    return readers[suffix](filepath)


def crop_movie_segment_by_marker(
    raw: mne.io.BaseRaw,
    marker: str,
    duration_sec: float,
) -> mne.io.BaseRaw:
    """
    以首个 marker 为起点裁剪固定时长片段（用于 movie 任务对齐）。

    Parameters
    ----------
    raw : mne.io.BaseRaw
        原始连续数据
    marker : str
        事件标记文本（大小写不敏感）
    duration_sec : float
        从 marker 起裁剪时长（秒）
    """
    if len(raw.annotations) == 0:
        raise RuntimeError("无 annotations，无法按 marker 裁剪 movie 片段")
    marker_upper = str(marker).upper()
    desc = np.array(raw.annotations.description, dtype=str)
    onsets = np.array(raw.annotations.onset, dtype=float)
    idx = [i for i, d in enumerate(desc) if marker_upper in d.upper()]
    if not idx:
        raise RuntimeError(f"未找到 marker: {marker}")

    onset = float(onsets[idx[0]])
    sfreq = float(raw.info["sfreq"])
    total_dur = float(raw.n_times / sfreq)
    if total_dur - onset < duration_sec:
        raise RuntimeError(
            f"marker 后时长不足: after={total_dur - onset:.3f}s < required={duration_sec:.3f}s"
        )

    tmin = onset
    tmax = min(onset + float(duration_sec), total_dur - (1.0 / sfreq))
    raw = raw.copy().crop(tmin=tmin, tmax=tmax)
    logger.info("按 marker 裁剪: marker=%s, tmin=%.3f, tmax=%.3f", marker, tmin, tmax)
    return raw


def _eeg_channel_keep_range(eeg_cfg: dict[str, Any]) -> tuple[int, int]:
    """解析 keep_channels_eeg，如 E1-E64 / E1-E128。"""
    spec = str(eeg_cfg.get("keep_channels_eeg", "E1-E64")).strip().upper()
    m = re.match(r"^E(\d+)\s*-\s*E(\d+)$", spec)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi
    return 1, 64


def drop_reference_and_non_scalp_channels(
    raw: mne.io.BaseRaw,
    eeg_cfg: dict[str, Any],
) -> tuple[mne.io.BaseRaw, list[str]]:
    """
    剔除 Cz/VREF/E65 等参考通道，仅保留配置范围内的头皮电极（默认 E1–E64）。

    EGI .mff 读入后参考通道通常名为 VREF（coordinates.xml 中 sensor 65）。
    """
    exclude = {str(c).strip() for c in eeg_cfg.get("exclude_channels", ["VREF", "Cz", "E65"])}
    exclude_upper = {c.upper() for c in exclude}
    ch_lo, ch_hi = _eeg_channel_keep_range(eeg_cfg)

    to_drop: list[str] = []
    for ch in raw.ch_names:
        ch_up = ch.upper()
        if ch_up in exclude_upper or ch in exclude:
            to_drop.append(ch)
            continue
        m = re.match(r"^E(\d+)$", ch, re.I)
        if m:
            if not (ch_lo <= int(m.group(1)) <= ch_hi):
                to_drop.append(ch)
        elif raw.get_channel_types([ch])[0] == "eeg":
            # 非 E 编号的 EEG 通道（fiducial 等）剔除
            to_drop.append(ch)

    to_drop = sorted(set(to_drop))
    if to_drop:
        logger.info("剔除参考/非分析通道: %s", to_drop)
        raw.drop_channels(to_drop)

    logger.info("保留 %d 个头皮通道: %s ... %s", len(raw.ch_names),
                raw.ch_names[0] if raw.ch_names else "-",
                raw.ch_names[-1] if raw.ch_names else "-")
    return raw, to_drop


def set_montage(raw: mne.io.BaseRaw, montage_name: str = "GSN-HydroCel-64_1.0") -> mne.io.BaseRaw:
    """设置电极 montage（EGI GSN-64 使用 GSN-HydroCel-64_1.0）。"""
    try:
        montage = mne.channels.make_standard_montage(montage_name)
        raw.set_montage(montage, on_missing="warn")
    except Exception as exc:
        logger.warning("设置 montage 失败 (%s): %s", montage_name, exc)
    return raw


# ---------------------------------------------------------------------------
# 滤波与重采样
# ---------------------------------------------------------------------------

def apply_filters(
    raw: mne.io.BaseRaw,
    l_freq: float,
    h_freq: float,
    notch_freq: float | None,
    notch_enabled: bool,
) -> mne.io.BaseRaw:
    """高通、低通、陷波滤波。"""
    logger.info("滤波: HP=%.2f Hz, LP=%.2f Hz", l_freq, h_freq)
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", verbose=False)
    if notch_enabled and notch_freq:
        logger.info("陷波: %.1f Hz", notch_freq)
        raw.notch_filter(freqs=notch_freq, picks="eeg", verbose=False)
    return raw


def resample_if_needed(raw: mne.io.BaseRaw, sfreq_target: float) -> mne.io.BaseRaw:
    """降采样至目标采样率。"""
    if raw.info["sfreq"] != sfreq_target:
        logger.info("重采样: %.1f -> %.1f Hz", raw.info["sfreq"], sfreq_target)
        raw.resample(sfreq_target, verbose=False)
    return raw


# ---------------------------------------------------------------------------
# 坏导、ICA、插值、参考
# ---------------------------------------------------------------------------

def mark_bad_channels_automatic(raw: mne.io.BaseRaw, flat_criterion: float = 1e-15) -> list[str]:
    """
    自动标记坏导（模板逻辑，建议后续人工复核）。

    策略：平坦通道 + 极端高方差通道（全样本 z > 3）。
    """
    data, _ = raw.get_data(picks="eeg", return_times=True)
    ch_names = raw.copy().pick("eeg").ch_names
    bads: list[str] = []

    # 平坦通道
    peak_to_peak = data.max(axis=1) - data.min(axis=1)
    for i, name in enumerate(ch_names):
        if peak_to_peak[i] < flat_criterion:
            bads.append(name)

    # 高方差
    variances = np.var(data, axis=1)
    z = (variances - variances.mean()) / (variances.std() + 1e-30)
    for i, name in enumerate(ch_names):
        if z[i] > 3 and name not in bads:
            bads.append(name)

    raw.info["bads"] = list(set(raw.info.get("bads", []) + bads))
    logger.info("自动标记坏导: %s", raw.info["bads"])
    return raw.info["bads"]


def run_ica(
    raw: mne.io.BaseRaw,
    n_components: int | None,
    method: str,
    random_state: int,
    ica_labels_path: Path | None = None,
    manual_review: bool = False,
) -> tuple[mne.preprocessing.ICA, list[int]]:
    """
    运行 ICA 并排除伪迹成分。

    若存在 ica_labels JSON（{"exclude": [0,1,...]}），则直接应用。
    manual_review=True 时仅拟合 ICA，不自动排除（供人工检查）。
    """
    n_ch = len(raw.ch_names) - len(raw.info.get("bads", []))
    if n_components is None:
        n_comp = min(30, max(1, n_ch - 1))
    elif isinstance(n_components, float) and n_components < 1:
        n_comp = n_components
    else:
        n_comp = int(min(n_components, n_ch - 1))

    ica = mne.preprocessing.ICA(
        n_components=n_comp,
        method=method,
        random_state=random_state,
        max_iter="auto",
    )
    filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    try:
        ica.fit(filt_raw, verbose=False)
    except RuntimeError as exc:
        logger.warning("ICA 拟合失败，跳过 ICA 步骤: %s", exc)
        return ica, []

    exclude: list[int] = []
    if ica_labels_path and Path(ica_labels_path).exists():
        with Path(ica_labels_path).open(encoding="utf-8") as f:
            labels = json.load(f)
        exclude = labels.get("exclude", [])
        logger.info("从标签文件加载 ICA 排除成分: %s", exclude)
    elif manual_review:
        logger.warning(
            "manual_review=True：ICA 已拟合但未自动排除成分。"
            "请人工检查后写入 ica_labels JSON，再重新运行。"
        )
    else:
        # 模板：基于 EOG/肌电相关成分的自动检测（可能不完整）
        try:
            eog_inds, _ = ica.find_bads_eog(raw, verbose=False)
            exclude.extend(eog_inds)
        except Exception:
            pass
        try:
            muscle_inds, _ = ica.find_bads_muscle(raw, verbose=False)
            exclude.extend(muscle_inds)
        except Exception:
            pass
        exclude = sorted(set(exclude))

    if exclude and not manual_review:
        ica.exclude = exclude
        ica.apply(raw, verbose=False)
        logger.info("ICA 已应用，排除成分: %s", exclude)

    return ica, exclude


def interpolate_bads(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """插值坏导。"""
    if raw.info["bads"]:
        logger.info("插值坏导: %s", raw.info["bads"])
        raw.interpolate_bads(reset_bads=True, verbose=False)
    return raw


def set_reference(raw: mne.io.BaseRaw, method: str = "average") -> mne.io.BaseRaw:
    """重参考。"""
    if method == "average":
        raw.set_eeg_reference("average", projection=False, verbose=False)
    else:
        logger.warning("未知参考方式 '%s'，跳过", method)
    return raw


# ---------------------------------------------------------------------------
# Epoch 与自动剔除
# ---------------------------------------------------------------------------

def make_epochs(
    raw: mne.io.BaseRaw,
    duration: float,
    overlap: float,
    reject_uv: float,
) -> mne.Epochs:
    """切成固定长度 epoch 并自动剔除振幅过大的 epoch。"""
    events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)
    reject = {"eeg": reject_uv * 1e-6}  # MNE 使用 V
    epochs = mne.Epochs(
        raw,
        events,
        event_id=None,
        tmin=0.0,
        tmax=duration - 1.0 / raw.info["sfreq"],
        baseline=None,
        preload=True,
        reject=reject,
        verbose=False,
    )
    logger.info(
        "Epochs: %d / %d 保留 (reject > %.0f uV)",
        len(epochs),
        len(events),
        reject_uv,
    )
    return epochs


# ---------------------------------------------------------------------------
# 单被试完整流水线
# ---------------------------------------------------------------------------

def preprocess_subject(
    subject_id: str,
    raw_path: Path,
    cfg: dict[str, Any],
    out_preprocessed_dir: Path,
    out_epochs_dir: Path,
    out_qc_dir: Path,
    ica_labels_path: Path | None = None,
) -> dict[str, Any]:
    """
    单被试预处理流水线。

    Returns
    -------
    dict
        QC 记录字典
    """
    eeg_cfg = cfg["eeg"]
    filt = cfg["filter"]
    ref = cfg["reference"]
    ep_cfg = cfg["epochs"]
    ica_cfg = cfg["ica"]

    project_root = Path(cfg["_project_root"])
    raw_file = Path(raw_path)
    if not raw_file.is_absolute():
        raw_file = project_root / raw_file

    raw = read_raw_eeg(raw_file)

    # movie 任务：按 VID+ 对齐并裁剪固定 300s 片段，避免跨被试时序错位
    task_cond = str(eeg_cfg.get("task_condition", "")).lower()
    if "movie" in task_cond:
        marker = str(eeg_cfg.get("movie_start_marker", "VID+"))
        crop_sec = float(eeg_cfg.get("movie_crop_duration_sec", eeg_cfg.get("recording_duration_sec", 300)))
        raw = crop_movie_segment_by_marker(raw, marker=marker, duration_sec=crop_sec)

    raw, dropped_ref = drop_reference_and_non_scalp_channels(raw, eeg_cfg)
    raw = set_montage(raw, eeg_cfg.get("montage", "GSN-HydroCel-64_1.0"))
    raw = apply_filters(
        raw,
        l_freq=filt["highpass_hz"],
        h_freq=filt["lowpass_hz"],
        notch_freq=filt.get("notch_hz"),
        notch_enabled=filt.get("notch_enabled", True),
    )
    raw = resample_if_needed(raw, eeg_cfg["sampling_rate_target"])

    bads = mark_bad_channels_automatic(raw)
    ica, ica_exclude = run_ica(
        raw,
        n_components=ica_cfg.get("n_components"),
        method=ica_cfg.get("method", "fastica"),
        random_state=ica_cfg.get("random_state", 42),
        ica_labels_path=ica_labels_path,
        manual_review=ica_cfg.get("manual_review", False),
    )
    raw = interpolate_bads(raw)
    raw = set_reference(raw, ref.get("method", "average"))

    epochs = make_epochs(
        raw,
        duration=ep_cfg["duration_sec"],
        overlap=ep_cfg.get("overlap_sec", 0.0),
        reject_uv=ep_cfg.get("reject_amplitude_uv", 150.0),
    )

    usable_epochs = len(epochs)
    usable_seconds = usable_epochs * ep_cfg["duration_sec"]

    # 保存
    ensure_dir(out_preprocessed_dir)
    ensure_dir(out_epochs_dir)
    ensure_dir(out_qc_dir)

    raw_out = out_preprocessed_dir / f"{subject_id}-raw.fif"
    ep_out = out_epochs_dir / f"{subject_id}-epo.fif"
    raw.save(raw_out, overwrite=True, verbose=False)
    epochs.save(ep_out, overwrite=True, verbose=False)

    qc = {
        "subject_id": subject_id,
        "raw_file": str(raw_file),
        "eeg_system": eeg_cfg.get("system", ""),
        "resting_condition": eeg_cfg.get("resting_condition", ""),
        "dropped_reference_channels": dropped_ref,
        "bad_channel_count": len(bads),
        "bad_channels": bads,
        "ica_removed_components": ica_exclude,
        "usable_epochs": usable_epochs,
        "usable_seconds": usable_seconds,
        "sampling_rate": raw.info["sfreq"],
        "n_channels": len(raw.ch_names),
    }
    save_json(qc, out_qc_dir / f"{subject_id}_preproc_qc.json")
    logger.info(
        "%s 完成: epochs=%d, bad_ch=%d",
        subject_id,
        usable_epochs,
        len(bads),
    )
    return qc


def batch_preprocess(
    participants: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    批量预处理所有被试。

    Returns
    -------
    summary : 成功被试的 QC 表
    failures : 失败被试记录
    """
    deriv = Path(cfg["paths"]["derivatives_root"])
    preproc_dir = deriv / "preprocessed"
    epochs_dir = deriv / "epochs"
    qc_dir = deriv / "qc"
    ica_labels_dir = Path(cfg.get("ica", {}).get("ica_labels_dir", deriv / "preprocessed" / "ica_labels"))
    if not Path(ica_labels_dir).is_absolute():
        ica_labels_dir = Path(cfg["_project_root"]) / ica_labels_dir

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    n_total = len(participants)

    for i, row in participants.iterrows():
        sid = str(row["subject_id"])
        raw_path = Path(row["raw_EEG_file"])
        labels_path = ica_labels_dir / f"{sid}_ica.json"
        logger.info("预处理 [%d/%d] %s", len(records) + len(failures) + 1, n_total, sid)
        try:
            qc = preprocess_subject(
                sid,
                raw_path,
                cfg,
                preproc_dir,
                epochs_dir,
                qc_dir,
                ica_labels_path=labels_path if labels_path.exists() else None,
            )
            qc["group"] = row.get("group", "")
            qc["status"] = "ok"
            records.append(qc)
        except FileNotFoundError as exc:
            logger.error("%s: %s", sid, exc)
            failures.append({
                "subject_id": sid,
                "group": row.get("group", ""),
                "status": "file_not_found",
                "error": str(exc),
            })
        except Exception as exc:
            logger.exception("%s 预处理失败: %s", sid, exc)
            failures.append({
                "subject_id": sid,
                "group": row.get("group", ""),
                "status": "error",
                "error": str(exc),
            })

    return pd.DataFrame(records), pd.DataFrame(failures)
