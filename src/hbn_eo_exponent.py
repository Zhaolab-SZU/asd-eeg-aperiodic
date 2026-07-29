"""HBN 静息态 eyes-open 后枕叶 aperiodic exponent 提取。"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from src.eeg_preprocessing import (
    apply_filters,
    drop_reference_and_non_scalp_channels,
    read_raw_eeg,
    resample_if_needed,
    set_reference,
)
from src.hbn_external import (
    EO_DEFAULT,
    _compile_patterns,
    _scale_hbn_voltage,
    find_resting_events_file,
    find_resting_set_file,
    parse_eyes_open_intervals,
)
from src.io_utils import ensure_dir, save_csv
from src.psd_utils import compute_psd_from_epochs
from src.specparam_utils import fit_specparam_channel

logger = logging.getLogger(__name__)

# AutoReject 为可选重型依赖，仅在 nuclear.enabled=true 时加载
try:
    from autoreject import AutoReject as _AutoReject
except ImportError:
    _AutoReject = None

POSTERIOR_ROI_DEFAULT = ("E67", "E72", "E75", "E77")


def _normalize_channel_name(name: str) -> str:
    m = re.match(r"^E(\d+)$", str(name).strip(), re.I)
    if m:
        return f"E{int(m.group(1))}"
    return str(name).strip()


def trim_eo_intervals(
    intervals: list[tuple[float, float]],
    trim_start_sec: float = 2.0,
    trim_end_sec: float = 1.0,
    min_duration_sec: float = 0.5,
) -> list[tuple[float, float]]:
    """在每个 EO 段首尾切除边缘时间，避免状态切换伪迹。"""
    trimmed: list[tuple[float, float]] = []
    for t0, t1 in intervals:
        t0_new = t0 + trim_start_sec
        t1_new = t1 - trim_end_sec
        if t1_new - t0_new >= min_duration_sec:
            trimmed.append((t0_new, t1_new))
    return trimmed


def parse_eyes_open_intervals_from_annotations(
    raw: mne.io.BaseRaw,
    eo_pattern: re.Pattern[str] | None = None,
    fallback_duration_sec: float = 20.0,
) -> list[tuple[float, float]]:
    """从 Raw annotations 解析 eyes-open 时间段。"""
    eo_pattern = eo_pattern or EO_DEFAULT
    if len(raw.annotations) == 0:
        return []

    onsets = np.asarray(raw.annotations.onset, dtype=float)
    durations = np.asarray(raw.annotations.duration, dtype=float)
    descriptions = [str(d) for d in raw.annotations.description]
    total_dur = float(raw.times[-1])

    intervals: list[tuple[float, float]] = []
    for i, desc in enumerate(descriptions):
        if not eo_pattern.search(desc):
            continue
        t0 = float(onsets[i])
        if i + 1 < len(onsets) and np.isfinite(onsets[i + 1]) and onsets[i + 1] > t0:
            t1 = float(onsets[i + 1])
        elif durations[i] > 0:
            t1 = t0 + float(durations[i])
        else:
            t1 = t0 + fallback_duration_sec
        t1 = min(t1, total_dur)
        if t1 - t0 >= 0.5:
            intervals.append((t0, t1))
    return intervals


def find_eyes_open_intervals(
    raw: mne.io.BaseRaw,
    events_path: Path | None = None,
    eo_pattern: re.Pattern[str] | None = None,
    fallback_duration_sec: float = 20.0,
    trim_start_sec: float = 2.0,
    trim_end_sec: float = 1.0,
    min_duration_sec: float = 0.5,
) -> list[tuple[float, float]]:
    """优先 events.tsv，其次 annotations；返回已切除边缘的 EO 时间段。"""
    intervals: list[tuple[float, float]] = []
    if events_path is not None and Path(events_path).exists():
        intervals = parse_eyes_open_intervals(events_path, eo_pattern, fallback_duration_sec)
        source = "events.tsv"
    else:
        intervals = parse_eyes_open_intervals_from_annotations(
            raw, eo_pattern, fallback_duration_sec,
        )
        source = "annotations"

    if not intervals:
        raise RuntimeError("未找到 eyes-open 时间段（events.tsv 与 annotations 均为空）")

    trimmed = trim_eo_intervals(
        intervals,
        trim_start_sec=trim_start_sec,
        trim_end_sec=trim_end_sec,
        min_duration_sec=min_duration_sec,
    )
    if not trimmed:
        raise RuntimeError(
            f"EO 段在切除首尾 {trim_start_sec}s/{trim_end_sec}s 后无可用数据"
        )
    logger.info(
        "EO 段: %d → %d（来源=%s，切除 %ss/%ss）",
        len(intervals), len(trimmed), source, trim_start_sec, trim_end_sec,
    )
    return trimmed


def concat_eyes_open_raw(
    raw: mne.io.BaseRaw,
    intervals: list[tuple[float, float]],
) -> mne.io.BaseRaw:
    """拼接多个 EO 片段。"""
    if not intervals:
        raise RuntimeError("无 eyes-open 时间段")
    segments = []
    total_dur = raw.times[-1]
    for t0, t1 in intervals:
        t0 = max(0.0, t0)
        t1 = min(total_dur, t1)
        if t1 - t0 >= 0.5:
            segments.append(raw.copy().crop(tmin=t0, tmax=t1))
    if not segments:
        raise RuntimeError("eyes-open 段裁剪后为空")
    if len(segments) == 1:
        return segments[0]
    return mne.concatenate_raws(segments, preload=True)


def preprocess_hbn_raw(raw: mne.io.BaseRaw, cfg: dict[str, Any]) -> mne.io.BaseRaw:
    """HBN 128 导标准预处理：电压缩放、通道筛选、滤波、重采样、平均参考。"""
    raw = _scale_hbn_voltage(raw)
    raw, _ = drop_reference_and_non_scalp_channels(raw, cfg["eeg"])
    flt = cfg["filter"]
    raw = apply_filters(
        raw,
        l_freq=flt["highpass_hz"],
        h_freq=flt["lowpass_hz"],
        notch_freq=flt.get("notch_hz"),
        notch_enabled=flt.get("notch_enabled", True),
    )
    raw = resample_if_needed(raw, cfg["eeg"]["sampling_rate_target"])
    raw = set_reference(raw, cfg["reference"]["method"])
    return raw


def ensure_hbn_montage(
    inst: mne.io.BaseRaw | mne.Epochs,
    cfg: dict[str, Any],
) -> mne.io.BaseRaw | mne.Epochs:
    """
    为 Raw/Epochs 设置 EGI 标准电极坐标。

    CSD 与 AutoReject 的空间插值均依赖有效 montage；按候选列表依次尝试，
    直到匹配通道数达到阈值。
    """
    nuclear_cfg = cfg.get("nuclear", {})
    csd_cfg = nuclear_cfg.get("csd", {})
    eeg_cfg = cfg.get("eeg", {})

    candidates: list[str] = list(csd_cfg.get("montage_candidates", []))
    primary = str(eeg_cfg.get("montage", "")).strip()
    if primary and primary not in candidates:
        candidates.insert(0, primary)
    if not candidates:
        candidates = ["GSN-HydroCel-128", "GSN-HydroCel-129"]

    min_match = int(csd_cfg.get("min_matched_channels", 64))
    eeg_names = {
        _normalize_channel_name(ch)
        for ch in inst.ch_names
        if inst.get_channel_types([ch])[0] == "eeg"
    }

    last_exc: Exception | None = None
    for montage_name in candidates:
        try:
            montage = mne.channels.make_standard_montage(montage_name)
            montage_names = {_normalize_channel_name(ch) for ch in montage.ch_names}
            overlap = len(eeg_names & montage_names)
            if overlap < min_match:
                logger.warning(
                    "Montage %s 仅匹配 %d/%d 通道 (< %d)，尝试下一个",
                    montage_name, overlap, len(eeg_names), min_match,
                )
                continue
            inst = inst.copy().set_montage(montage, on_missing="warn")
            logger.info("已设置 montage: %s（匹配 %d 通道）", montage_name, overlap)
            return inst
        except Exception as exc:
            last_exc = exc
            logger.warning("Montage %s 设置失败: %s", montage_name, exc)

    raise RuntimeError(
        f"无法为 HBN 数据设置有效 montage（候选: {candidates}）"
    ) from last_exc


def make_eo_epochs(raw_eo: mne.io.BaseRaw, cfg: dict[str, Any]) -> mne.Epochs:
    """将拼接后的 EO 连续数据切成固定长度 epoch（默认 2 s，无重叠）。"""
    ep_cfg = cfg.get("epochs", {})
    duration = float(ep_cfg.get("duration_sec", 2.0))
    overlap = float(ep_cfg.get("overlap_sec", 0.0))

    epochs = mne.make_fixed_length_epochs(
        raw_eo,
        duration=duration,
        overlap=overlap,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )
    if len(epochs) == 0:
        raise RuntimeError("EO 拼接数据未能生成任何 epoch，请检查时长与分段参数")
    logger.info("EO epoch 切分: %d 段 × %.1fs", len(epochs), duration)
    return epochs


def autoreject_clean_epochs(
    epochs: mne.Epochs,
    cfg: dict[str, Any],
) -> tuple[mne.Epochs, dict[str, Any]]:
    """
    数据驱动的坏段/坏导修复（AutoReject）。

    在 2 s epoch 上自动搜索最优阈值，剔除肌肉微颤与眼动伪迹，并插值坏通道。
    """
    if _AutoReject is None:
        raise ImportError(
            "nuclear.autoreject 需要 autoreject 库：pip install autoreject"
        )

    ar_cfg = cfg.get("nuclear", {}).get("autoreject", {})
    n_before = len(epochs)

    ar = _AutoReject(
        n_jobs=int(ar_cfg.get("n_jobs", -1)),
        random_state=int(ar_cfg.get("random_state", 42)),
        verbose=bool(ar_cfg.get("verbose", False)),
    )
    # fit_transform 同时完成阈值搜索与清洗；return_log 便于 QC
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    n_after = len(epochs_clean)
    n_rejected = int(np.sum(reject_log.bad_epochs)) if hasattr(reject_log, "bad_epochs") else n_before - n_after

    min_keep = int(cfg.get("nuclear", {}).get("min_epochs_after_autoreject", 10))
    if n_after < min_keep:
        raise RuntimeError(
            f"AutoReject 后可用 epoch 过少: {n_after} < {min_keep}"
        )

    qc = {
        "n_epochs_before_autoreject": n_before,
        "n_epochs_after_autoreject": n_after,
        "n_epochs_rejected": n_rejected,
    }
    logger.info(
        "AutoReject: %d → %d epochs（剔除 %d）",
        n_before, n_after, n_rejected,
    )
    return epochs_clean, qc


def apply_csd_to_epochs(epochs: mne.Epochs, cfg: dict[str, Any]) -> mne.Epochs:
    """
    电流源密度（CSD）空间滤波。

    必须在 AutoReject 清洗之后、Welch PSD 之前执行，以降低容积传导并提升
    后枕叶局部频谱的空间特异性。
    """
    if not cfg.get("nuclear", {}).get("csd", {}).get("enabled", True):
        logger.info("CSD 已禁用，跳过")
        return epochs

    # CSD 强依赖电极坐标；防错：若尚未设置 montage 则在此补设
    if epochs.get_montage() is None:
        logger.warning("Epochs 缺少 montage，CSD 前自动补设")
        epochs = ensure_hbn_montage(epochs, cfg)  # type: ignore[assignment]

    epochs_csd = mne.preprocessing.compute_current_source_density(epochs.copy())
    logger.info("CSD 完成: %d 通道 × %d epochs", len(epochs_csd.ch_names), len(epochs_csd))
    return epochs_csd


def _pick_roi_channels(
    ch_names: list[str],
    roi_channels: tuple[str, ...],
) -> list[str]:
    """在通道列表中解析 ROI 名称（大小写/空格容错）。"""
    ch_map = {_normalize_channel_name(c): c for c in ch_names}
    picks = [ch_map[c] for c in roi_channels if c in ch_map]
    if not picks:
        raise RuntimeError(
            f"ROI 通道均不存在: {roi_channels}; 可用示例={ch_names[:8]}"
        )
    return picks


def compute_roi_welch_psd_from_epochs(
    epochs: mne.Epochs,
    roi_channels: tuple[str, ...],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """对 ROI 通道从 epoch 计算 Welch PSD（跨 epoch 平均），返回 freqs, psd, ch_names。"""
    picks = _pick_roi_channels(epochs.ch_names, roi_channels)
    epochs_roi = epochs.copy().pick(picks)

    psd_cfg = cfg["psd"]
    welch_cfg = psd_cfg.get("welch", {})
    fmin = psd_cfg.get("freq_min_hz", welch_cfg.get("fmin", 1.0))
    fmax = psd_cfg.get("freq_max_hz", welch_cfg.get("fmax", 40.0))

    freqs, psd, ch_names = compute_psd_from_epochs(
        epochs_roi, fmin=fmin, fmax=fmax, welch_cfg=welch_cfg,
    )
    return freqs, psd, ch_names


def compute_roi_welch_psd(
    raw: mne.io.BaseRaw,
    roi_channels: tuple[str, ...],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    对 ROI 通道从连续 Raw 计算 Welch PSD（legacy 路径，nuclear.enabled=false 时使用）。
    """
    picks = _pick_roi_channels(raw.ch_names, roi_channels)
    psd_cfg = cfg["psd"]
    welch_cfg = psd_cfg.get("welch", {})
    fmin = psd_cfg.get("freq_min_hz", welch_cfg.get("fmin", 1.0))
    fmax = psd_cfg.get("freq_max_hz", welch_cfg.get("fmax", 40.0))

    spectrum = raw.copy().pick(picks).compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        n_fft=welch_cfg.get("n_fft", 500),
        n_overlap=welch_cfg.get("n_overlap", 250),
        window=welch_cfg.get("window", "hamming"),
        verbose=False,
    )
    return spectrum.freqs, spectrum.get_data(), spectrum.ch_names


def run_nuclear_signal_pipeline(
    raw_eo: mne.io.BaseRaw,
    roi_channels: tuple[str, ...],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """
    Nuclear 极限去噪主链路：
      EO 连续数据 → 2s Epoch → Montage → AutoReject → CSD → ROI Welch PSD
    """
    qc: dict[str, Any] = {"pipeline": "nuclear"}

    epochs = make_eo_epochs(raw_eo, cfg)
    qc["n_epochs_initial"] = len(epochs)

    epochs = ensure_hbn_montage(epochs, cfg)  # type: ignore[assignment]
    epochs, ar_qc = autoreject_clean_epochs(epochs, cfg)
    qc.update(ar_qc)

    epochs = apply_csd_to_epochs(epochs, cfg)
    freqs, psd, ch_names = compute_roi_welch_psd_from_epochs(epochs, roi_channels, cfg)
    return freqs, psd, ch_names, qc


def fit_posterior_roi_exponent(
    freqs: np.ndarray,
    psd: np.ndarray,
    ch_names: list[str],
    sp_cfg: dict[str, Any],
) -> tuple[float, pd.DataFrame]:
    """
    对 ROI 各通道拟合 specparam，返回通道均值 exponent 与通道级结果表。
    """
    rows: list[dict[str, Any]] = []
    exponents: list[float] = []
    for i, ch in enumerate(ch_names):
        try:
            fit = fit_specparam_channel(freqs, psd[i], sp_cfg)
            exp = float(fit["aperiodic_exponent"])
        except Exception as exc:
            logger.warning("通道 %s specparam 拟合失败: %s", ch, exc)
            fit = {"aperiodic_exponent": np.nan, "r_squared": np.nan, "fit_error": np.nan}
            exp = np.nan
        if np.isfinite(exp):
            exponents.append(exp)
        rows.append({"channel": ch, **fit})

    roi_exponent = float(np.mean(exponents)) if exponents else np.nan
    return roi_exponent, pd.DataFrame(rows)


def process_subject_eo_posterior_exponent(
    subject_id: str,
    raw_path: Path,
    cfg: dict[str, Any],
    events_path: Path | None = None,
) -> dict[str, Any]:
    """
    单被试完整流程。

    nuclear.enabled=true（默认）:
      读 raw → EO 拼接 → 2s Epoch → AutoReject → CSD → Welch PSD → specparam(knee)

    nuclear.enabled=false:
      读 raw → EO 拼接 → 连续 Raw Welch PSD → specparam（legacy）
    """
    hbn_cfg = cfg.get("hbn", {})
    nuclear_on = bool(cfg.get("nuclear", {}).get("enabled", False))
    sp_cfg = cfg["specparam"]
    eo_pat = _compile_patterns(
        hbn_cfg.get("eyes_open_patterns", ["instructed_toOpenEyes"])
    )
    roi_channels = tuple(
        _normalize_channel_name(c)
        for c in hbn_cfg.get("roi_channels", list(POSTERIOR_ROI_DEFAULT))
    )

    raw = read_raw_eeg(Path(raw_path), preload=True)
    intervals = find_eyes_open_intervals(
        raw,
        events_path=Path(events_path) if events_path else None,
        eo_pattern=eo_pat,
        fallback_duration_sec=float(hbn_cfg.get("eo_fallback_duration_sec", 20.0)),
        trim_start_sec=float(hbn_cfg.get("eo_trim_start_sec", 2.0)),
        trim_end_sec=float(hbn_cfg.get("eo_trim_end_sec", 1.0)),
        min_duration_sec=float(hbn_cfg.get("eo_min_segment_sec", 0.5)),
    )
    raw = preprocess_hbn_raw(raw, cfg)
    raw_eo = concat_eyes_open_raw(raw, intervals)

    pipeline_qc: dict[str, Any] = {}
    if nuclear_on:
        freqs, psd, ch_names, pipeline_qc = run_nuclear_signal_pipeline(
            raw_eo, roi_channels, cfg,
        )
    else:
        freqs, psd, ch_names = compute_roi_welch_psd(raw_eo, roi_channels, cfg)
        pipeline_qc = {"pipeline": "legacy"}

    roi_exponent, channel_df = fit_posterior_roi_exponent(
        freqs, psd, ch_names, sp_cfg,
    )

    # knee 模式下同步导出 knee 频率（通道均值）
    knee_mean = np.nan
    if sp_cfg.get("aperiodic_mode", "fixed") != "fixed" and "aperiodic_knee" in channel_df.columns:
        knee_vals = channel_df["aperiodic_knee"].dropna()
        knee_mean = float(knee_vals.mean()) if len(knee_vals) else np.nan

    return {
        "subject_id": subject_id,
        "EO_posterior_exponent": roi_exponent,
        "aperiodic_mode": sp_cfg.get("aperiodic_mode", "fixed"),
        "aperiodic_knee_mean": knee_mean,
        "n_eo_intervals": len(intervals),
        "total_eo_sec": float(raw_eo.times[-1]),
        "n_roi_channels": len(ch_names),
        "status": "ok" if np.isfinite(roi_exponent) else "fit_failed",
        "channel_results": channel_df,
        **pipeline_qc,
    }


def resolve_hbn_file_path(
    row: pd.Series,
    bids_root: Path,
    file_kind: str = "set",
) -> Path | None:
    """
    解析被试 raw/events 本地路径。

    participants.csv 可能仍指向其他机器上的绝对路径；优先用 BIDS 规则重建。
    """
    bids_root = Path(bids_root)
    sid = str(row["subject_id"])
    sid_bids = str(row.get("subject_id_bids", f"sub-{sid}"))
    release = str(row.get("release_id", ""))
    col = "raw_EEG_file" if file_kind == "set" else "events_file"
    raw_val = row.get(col, "")
    if pd.notna(raw_val) and str(raw_val).strip():
        p = Path(str(raw_val))
        if p.exists():
            return p
        parts = p.parts
        for i, part in enumerate(parts):
            if part.startswith("cmi_bids_R"):
                candidate = bids_root / Path(*parts[i:])
                if candidate.exists():
                    return candidate

    if not release:
        return None
    eeg_dir = bids_root / f"cmi_bids_{release}" / sid_bids / "eeg"
    if file_kind == "set":
        cands = sorted(eeg_dir.glob(f"{sid_bids}_task-RestingState_eeg.set"))
    else:
        cands = sorted(eeg_dir.glob(f"{sid_bids}_task-RestingState_events.tsv"))
    return cands[0] if cands else None


def fix_hbn_participant_paths(
    participants: pd.DataFrame,
    bids_root: Path,
) -> pd.DataFrame:
    """将 participants 表中的 raw/events 路径重写为当前机器可访问的本地路径。"""
    df = participants.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    raw_paths: list[str] = []
    event_paths: list[str] = []
    for _, row in df.iterrows():
        set_p = resolve_hbn_file_path(row, bids_root, "set")
        ev_p = resolve_hbn_file_path(row, bids_root, "events")
        raw_paths.append(str(set_p) if set_p else "")
        event_paths.append(str(ev_p) if ev_p else "")
    df["raw_EEG_file"] = raw_paths
    df["events_file"] = event_paths
    n_ok = int(df["raw_EEG_file"].astype(bool).sum())
    logger.info("路径修复: %d/%d 被试找到本地 .set", n_ok, len(df))
    return df


def discover_hbn_subjects_all_releases(bids_root: Path) -> pd.DataFrame:
    """扫描 bids_root 下全部 cmi_bids_R* release。"""
    bids_root = Path(bids_root)
    frames: list[pd.DataFrame] = []
    for rel_dir in sorted(bids_root.glob("cmi_bids_R*")):
        if not rel_dir.is_dir():
            continue
        release_id = rel_dir.name.replace("cmi_bids_", "")
        part = discover_hbn_subjects_from_bids(bids_root, release_id)
        if not part.empty:
            frames.append(part)
    if not frames:
        raise FileNotFoundError(f"未在 {bids_root} 发现任何 cmi_bids_R* 目录")
    return pd.concat(frames, ignore_index=True).drop_duplicates("subject_id")


def extract_eo_posterior_from_derivatives(
    specparam_qc_csv: Path,
    roi_channels: tuple[str, ...] = POSTERIOR_ROI_DEFAULT,
    require_fit_valid: bool = True,
) -> pd.DataFrame:
    """从已有 specparam QC 表提取 EO 后枕叶 exponent（无需重跑预处理）。"""
    specparam_qc_csv = Path(specparam_qc_csv)
    if not specparam_qc_csv.exists():
        raise FileNotFoundError(f"未找到 specparam QC 文件: {specparam_qc_csv}")

    roi_set = {_normalize_channel_name(c) for c in roi_channels}
    df = pd.read_csv(specparam_qc_csv)
    df["subject_id"] = df["subject_id"].astype(str)
    df["channel"] = df["channel"].map(_normalize_channel_name)
    sub = df[df["channel"].isin(roi_set)].copy()
    if require_fit_valid and "fit_valid" in sub.columns:
        sub = sub[sub["fit_valid"].astype(bool)]

    rows: list[dict[str, Any]] = []
    for sid, grp in sub.groupby("subject_id"):
        vals = grp["aperiodic_exponent"].dropna()
        rows.append({
            "subject_id": sid,
            "EO_posterior_exponent": float(vals.mean()) if len(vals) else np.nan,
            "n_roi_channels": int(len(grp)),
            "n_valid_channels": int(len(vals)),
        })
    return pd.DataFrame(rows)


def discover_hbn_subjects_from_bids(
    bids_root: Path,
    release_id: str,
    subject_ids: list[str] | None = None,
) -> pd.DataFrame:
    """从 BIDS 目录发现静息态 .set 与 events.tsv。"""
    bids_root = Path(bids_root)
    rel_dir = bids_root / f"cmi_bids_{release_id}"
    if not rel_dir.exists():
        raise FileNotFoundError(f"BIDS 目录不存在: {rel_dir}")

    rows: list[dict[str, str]] = []
    sub_dirs = sorted(p for p in rel_dir.glob("sub-*") if p.is_dir())
    for sub_dir in sub_dirs:
        sid_bids = sub_dir.name
        if subject_ids and sid_bids not in subject_ids and sid_bids.replace("sub-", "") not in subject_ids:
            continue
        set_path = find_resting_set_file(bids_root, release_id, sid_bids)
        ev_path = find_resting_events_file(bids_root, release_id, sid_bids)
        if set_path is None:
            continue
        rows.append({
            "subject_id": sid_bids.replace("sub-", ""),
            "subject_id_bids": sid_bids,
            "release_id": release_id,
            "raw_EEG_file": str(set_path),
            "events_file": str(ev_path) if ev_path else "",
        })
    return pd.DataFrame(rows)


def run_hbn_eo_posterior_batch(
    subjects: pd.DataFrame,
    cfg: dict[str, Any],
    out_csv: Path,
    channel_detail_dir: Path | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """批量计算 EO 后枕叶 exponent 并写出 CSV。"""
    out_csv = Path(out_csv)
    ensure_dir(out_csv.parent)
    if channel_detail_dir is not None:
        ensure_dir(channel_detail_dir)

    if out_csv.exists() and not overwrite:
        logger.info("输出已存在，跳过: %s", out_csv)
        return pd.read_csv(out_csv)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    bids_root = Path(cfg["paths"].get("bids_root", ""))

    for _, row in subjects.iterrows():
        sid = str(row["subject_id"])
        raw_path = resolve_hbn_file_path(row, bids_root, "set")
        events_path = resolve_hbn_file_path(row, bids_root, "events")
        if raw_path is None or not raw_path.exists():
            failures.append({"subject_id": sid, "error": f"文件不存在: {raw_path}"})
            continue
        try:
            rec = process_subject_eo_posterior_exponent(
                sid, raw_path, cfg, events_path=events_path,
            )
            channel_df = rec.pop("channel_results")
            if channel_detail_dir is not None:
                save_csv(channel_df, channel_detail_dir / f"{sid}_posterior_specparam.csv")
            results.append(rec)
            logger.info(
                "%s: EO_posterior_exponent=%.4f [%s] (%d EO段, %.1fs, %s)",
                sid,
                rec["EO_posterior_exponent"],
                rec.get("aperiodic_mode", "fixed"),
                rec["n_eo_intervals"],
                rec["total_eo_sec"],
                rec.get("pipeline", "legacy"),
            )
        except Exception as exc:
            logger.warning("%s 处理失败: %s", sid, exc)
            failures.append({"subject_id": sid, "error": str(exc)})

    out_df = pd.DataFrame(results)
    if not out_df.empty:
        # 主表保留被试 ID 与 exponent；附带模式与 QC 列便于远端服务器核查
        export_cols = [
            c for c in (
                "subject_id",
                "EO_posterior_exponent",
                "aperiodic_mode",
                "aperiodic_knee_mean",
                "pipeline",
                "n_epochs_before_autoreject",
                "n_epochs_after_autoreject",
                "n_epochs_rejected",
            )
            if c in out_df.columns
        ]
        out_df = out_df[export_cols]
    save_csv(out_df, out_csv)

    if failures:
        fail_path = out_csv.with_name(out_csv.stem + "_failures.csv")
        save_csv(pd.DataFrame(failures), fail_path)
        logger.warning("失败 %d 名被试，详见 %s", len(failures), fail_path)

    return out_df
