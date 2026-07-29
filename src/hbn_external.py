"""HBN-EEG 外部复现 pipeline 工具。"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_roi_config
from src.eeg_preprocessing import (
    apply_filters,
    drop_reference_and_non_scalp_channels,
    make_epochs,
    read_raw_eeg,
    resample_if_needed,
    set_reference,
)
from src.io_utils import ensure_dir, save_csv
from src.psd_utils import compute_psd_from_epochs, psd_to_long_df
from src.qc_utils import run_specparam_qc
from src.roi_utils import run_roi_pipeline
from src.specparam_utils import fit_subject_specparam
from src.stats_utils import descriptive_table, model_results_to_row, run_ols

logger = logging.getLogger(__name__)

EO_DEFAULT = re.compile(r"instructed_toOpenEyes|eyes[\s\-_]?open", re.IGNORECASE)


def resolve_hbn_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """解析 HBN 配置中的相对路径。"""
    paths = cfg.get("paths", {})
    out: dict[str, Path] = {}
    for key in (
        "bids_root", "manifest_file", "participants_file",
        "derivatives_root", "outputs_root", "logs_dir",
    ):
        if key in paths and paths[key]:
            p = Path(paths[key])
            if not p.is_absolute():
                p = (PROJECT_ROOT / p).resolve()
            out[key] = p
    return out


def _compile_patterns(patterns: list[str]) -> re.Pattern[str]:
    return re.compile("|".join(re.escape(p) for p in patterns), re.IGNORECASE)


def find_resting_set_file(bids_root: Path, release_id: str, subject_bids: str) -> Path | None:
    rel_dir = bids_root / f"cmi_bids_{release_id}" / subject_bids / "eeg"
    if not rel_dir.exists():
        return None
    cands = [
        p for p in rel_dir.glob(f"{subject_bids}_task-RestingState_eeg.set")
        if not p.name.startswith("._")
    ]
    return cands[0] if cands else None


def find_resting_events_file(bids_root: Path, release_id: str, subject_bids: str) -> Path | None:
    rel_dir = bids_root / f"cmi_bids_{release_id}" / subject_bids / "eeg"
    if not rel_dir.exists():
        return None
    cands = [
        p for p in rel_dir.glob(f"{subject_bids}_task-RestingState_events.tsv")
        if not p.name.startswith("._")
    ]
    return cands[0] if cands else None


def find_thepresent_set_file(bids_root: Path, release_id: str, subject_bids: str) -> Path | None:
    rel_dir = bids_root / f"cmi_bids_{release_id}" / subject_bids / "eeg"
    if not rel_dir.exists():
        return None
    cands = [
        p for p in rel_dir.glob(f"{subject_bids}_task-ThePresent_eeg.set")
        if not p.name.startswith("._")
    ]
    return cands[0] if cands else None


def find_thepresent_events_file(bids_root: Path, release_id: str, subject_bids: str) -> Path | None:
    rel_dir = bids_root / f"cmi_bids_{release_id}" / subject_bids / "eeg"
    if not rel_dir.exists():
        return None
    cands = [
        p for p in rel_dir.glob(f"{subject_bids}_task-ThePresent_events.tsv")
        if not p.name.startswith("._")
    ]
    return cands[0] if cands else None


def parse_thepresent_interval(
    events_path: Path,
    start_pattern: re.Pattern[str],
    stop_pattern: re.Pattern[str],
    fallback_duration_sec: float = 206.0,
) -> tuple[float, float]:
    """从 ThePresent events.tsv 解析 video_start → video_stop 区间。"""
    df = pd.read_csv(events_path, sep="\t")
    label_cols = [c for c in df.columns if c.lower() in {"value", "trial_type", "condition", "hed"}]
    if not label_cols:
        label_cols = [c for c in df.columns if c not in {"onset", "duration", "sample", "event_code"}]
    onsets = pd.to_numeric(df["onset"], errors="coerce").values

    t_start: float | None = None
    t_stop: float | None = None
    for i in range(len(df)):
        text = " | ".join(str(df.iloc[i][c]) for c in label_cols if pd.notna(df.iloc[i][c]))
        onset = float(onsets[i])
        if start_pattern.search(text):
            t_start = onset if t_start is None else min(t_start, onset)
        if stop_pattern.search(text):
            t_stop = onset if t_stop is None else max(t_stop, onset)

    if t_start is None:
        raise RuntimeError("无 video_start 事件")
    if t_stop is None:
        t_stop = t_start + fallback_duration_sec
    if t_stop - t_start < 0.5:
        raise RuntimeError("ThePresent 视频段过短")
    return t_start, t_stop


def parse_eyes_open_intervals(
    events_path: Path,
    eo_pattern: re.Pattern[str] | None = None,
    fallback_duration_sec: float = 20.0,
) -> list[tuple[float, float]]:
    """从 BIDS events.tsv 解析 eyes-open 时间段 (tmin, tmax)。"""
    eo_pattern = eo_pattern or EO_DEFAULT
    df = pd.read_csv(events_path, sep="\t")
    label_cols = [c for c in df.columns if c.lower() in {"value", "trial_type", "condition", "hed"}]
    if not label_cols:
        label_cols = [c for c in df.columns if c != "onset"]
    onsets = pd.to_numeric(df["onset"], errors="coerce").values
    intervals: list[tuple[float, float]] = []
    for i in range(len(df)):
        text = " | ".join(str(df.iloc[i][c]) for c in label_cols if pd.notna(df.iloc[i][c]))
        if not eo_pattern.search(text):
            continue
        t0 = float(onsets[i])
        if i + 1 < len(onsets) and np.isfinite(onsets[i + 1]):
            t1 = float(onsets[i + 1])
        else:
            t1 = t0 + fallback_duration_sec
        if t1 - t0 >= 0.5:
            intervals.append((t0, t1))
    return intervals


def build_hbn_participants(cfg: dict[str, Any]) -> pd.DataFrame:
    """从 Manifest A 生成 HBN participants.csv。"""
    paths = resolve_hbn_paths(cfg)
    manifest = pd.read_csv(paths["manifest_file"])
    bids_root = paths["bids_root"]
    asd = cfg["groups"]["asd_label"]
    td = cfg["groups"]["td_label"]

    rows: list[dict[str, Any]] = []
    for _, r in manifest.iterrows():
        sid = str(r["subject_id_std"])
        sid_bids = str(r["subject_id_bids"])
        rel = str(r["release_id"])
        grp_raw = str(r["group_asd_vs_control"])
        if grp_raw == "ASD":
            group = asd
        elif grp_raw == "TD_like":
            group = td
        else:
            continue

        set_path = find_resting_set_file(bids_root, rel, sid_bids)
        events_path = find_resting_events_file(bids_root, rel, sid_bids)
        rows.append({
            "subject_id": sid,
            "subject_id_bids": sid_bids,
            "release_id": rel,
            "group": group,
            "group_hbn_raw": grp_raw,
            "age_months": r.get("age_months"),
            "sex": r.get("sex_std"),
            "IQ_total": r.get("IQ_best_available"),
            "SRS_total": r.get("SRS_total_best"),
            "SCQ_total": r.get("SCQ_total_best"),
            "raw_EEG_file": str(set_path) if set_path else "",
            "events_file": str(events_path) if events_path else "",
            "included_final": int(set_path is not None and events_path is not None),
        })

    df = pd.DataFrame(rows)
    out_path = paths["participants_file"]
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("HBN participants: %d 行 → %s", len(df), out_path)
    return df


def run_hbn_inventory(cfg: dict[str, Any]) -> pd.DataFrame:
    """扫描下载完整性 + eyes-open 段信息。"""
    paths = resolve_hbn_paths(cfg)
    participants = build_hbn_participants(cfg)
    hbn_cfg = cfg.get("hbn", {})
    eo_pat = _compile_patterns(hbn_cfg.get("eyes_open_patterns", ["instructed_toOpenEyes"]))
    fallback = float(hbn_cfg.get("eo_fallback_duration_sec", 20.0))

    inv_rows: list[dict[str, Any]] = []
    for _, row in participants.iterrows():
        set_path = Path(row["raw_EEG_file"]) if row["raw_EEG_file"] else None
        ev_path = Path(row["events_file"]) if row["events_file"] else None
        rec: dict[str, Any] = {
            "subject_id": row["subject_id"],
            "group": row["group"],
            "release_id": row["release_id"],
            "has_set": bool(set_path and set_path.exists()),
            "has_events": bool(ev_path and ev_path.exists()),
            "set_size_mb": round(set_path.stat().st_size / 1e6, 2) if set_path and set_path.exists() else np.nan,
            "n_eo_intervals": 0,
            "total_eo_sec": 0.0,
            "ready": False,
        }
        if ev_path and ev_path.exists():
            try:
                intervals = parse_eyes_open_intervals(ev_path, eo_pat, fallback)
                rec["n_eo_intervals"] = len(intervals)
                rec["total_eo_sec"] = sum(t1 - t0 for t0, t1 in intervals)
            except Exception as exc:
                rec["error"] = str(exc)
        rec["ready"] = rec["has_set"] and rec["has_events"] and rec["n_eo_intervals"] > 0
        inv_rows.append(rec)

    inv = pd.DataFrame(inv_rows)
    out_dir = paths["outputs_root"]
    ensure_dir(out_dir)
    save_csv(inv, out_dir / "hbn_inventory.csv")
    summary = {
        "n_manifest": len(participants),
        "n_has_set": int(inv["has_set"].sum()),
        "n_has_events": int(inv["has_events"].sum()),
        "n_ready": int(inv["ready"].sum()),
        "n_ASD": int((inv["group"] == cfg["groups"]["asd_label"]).sum()),
        "n_TD": int((inv["group"] == cfg["groups"]["td_label"]).sum()),
    }
    (out_dir / "hbn_inventory_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Inventory: %s", summary)
    return inv


def _concat_eyes_open_raw(
    raw: mne.io.BaseRaw,
    intervals: list[tuple[float, float]],
) -> mne.io.BaseRaw:
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


def _scale_hbn_voltage(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """HBN EEGLAB .set 数值量级通常为 mV（MNE 标记为 V），转为 SI 伏特。"""
    data = raw.get_data(picks="eeg")
    ptp = float(np.ptp(data, axis=1).max())
    if ptp < 5.0:
        logger.info("HBN 电压缩放 ×1e-3 (mV→V), peak-to-peak=%.4f", ptp)
        raw.apply_function(lambda x: x * 1e-3, picks="eeg")
    return raw


def preprocess_hbn_subject(
    row: pd.Series,
    cfg: dict[str, Any],
    deriv_root: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """单被试：读 .set → 裁 eyes-open → 滤波 → epoch。"""
    subject_id = str(row["subject_id"])
    group = str(row["group"])
    epochs_path = deriv_root / "epochs" / f"{subject_id}-epo.fif"
    if epochs_path.exists() and not overwrite:
        epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
        return {
            "subject_id": subject_id,
            "group": group,
            "status": "cached",
            "usable_epochs": len(epochs),
            "usable_seconds": len(epochs) * cfg["epochs"]["duration_sec"],
            "n_eo_intervals": np.nan,
        }

    set_path = Path(row["raw_EEG_file"])
    ev_path = Path(row["events_file"])
    hbn_cfg = cfg.get("hbn", {})
    eo_pat = _compile_patterns(hbn_cfg.get("eyes_open_patterns", ["instructed_toOpenEyes"]))
    fallback = float(hbn_cfg.get("eo_fallback_duration_sec", 20.0))

    intervals = parse_eyes_open_intervals(ev_path, eo_pat, fallback)
    raw = read_raw_eeg(set_path, preload=True)
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
    raw_eo = _concat_eyes_open_raw(raw, intervals)

    ep_cfg = cfg["epochs"]
    epochs = make_epochs(
        raw_eo,
        duration=ep_cfg["duration_sec"],
        overlap=ep_cfg["overlap_sec"],
        reject_uv=ep_cfg["reject_amplitude_uv"],
    )

    ensure_dir(epochs_path.parent)
    epochs.save(epochs_path, overwrite=True)

    qc = {
        "subject_id": subject_id,
        "group": group,
        "status": "ok",
        "usable_epochs": len(epochs),
        "usable_seconds": len(epochs) * ep_cfg["duration_sec"],
        "n_eo_intervals": len(intervals),
        "total_eo_sec": sum(t1 - t0 for t0, t1 in intervals),
        "n_channels": len(epochs.ch_names),
        "sfreq": epochs.info["sfreq"],
    }
    qc_path = deriv_root / "qc" / f"{subject_id}_preproc_qc.json"
    ensure_dir(qc_path.parent)
    qc_path.write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")
    return qc


def batch_preprocess_hbn(
    cfg: dict[str, Any],
    limit: int | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    paths = resolve_hbn_paths(cfg)
    deriv = paths["derivatives_root"]
    participants = pd.read_csv(paths["participants_file"])
    participants = participants[participants["included_final"] == 1].copy()
    if limit:
        participants = participants.head(limit)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    min_epochs = int(cfg["epochs"]["min_usable_epochs"])

    for _, row in participants.iterrows():
        sid = row["subject_id"]
        try:
            qc = preprocess_hbn_subject(row, cfg, deriv, overwrite=overwrite)
            qc["passes_epoch_threshold"] = int(qc["usable_epochs"] >= min_epochs)
            rows.append(qc)
            logger.info("%s: %d epochs", sid, qc["usable_epochs"])
        except Exception as exc:
            logger.warning("%s 预处理失败: %s", sid, exc)
            failures.append({"subject_id": sid, "error": str(exc)})

    summary = pd.DataFrame(rows)
    ensure_dir(deriv / "qc")
    save_csv(summary, deriv / "qc" / "preproc_summary.csv")
    if failures:
        save_csv(pd.DataFrame(failures), deriv / "qc" / "preproc_failures.csv")

    summary_slim = summary.drop(columns=["group"], errors="ignore")
    analysis = participants.merge(summary_slim, on="subject_id", how="left")
    analysis = analysis[analysis["passes_epoch_threshold"] == 1].copy()
    save_csv(analysis, deriv / "participants_analysis.csv")
    return summary


def build_hbn_thepresent_participants(cfg: dict[str, Any]) -> pd.DataFrame:
    """从 Manifest A 生成 ThePresent participants.csv（仅本地有 .set + events 的被试）。"""
    paths = resolve_hbn_paths(cfg)
    manifest = pd.read_csv(paths["manifest_file"])
    bids_root = paths["bids_root"]
    asd = cfg["groups"]["asd_label"]
    td = cfg["groups"]["td_label"]

    rows: list[dict[str, Any]] = []
    for _, r in manifest.iterrows():
        sid = str(r["subject_id_std"])
        sid_bids = str(r["subject_id_bids"])
        rel = str(r["release_id"])
        grp_raw = str(r["group_asd_vs_control"])
        if grp_raw == "ASD":
            group = asd
        elif grp_raw == "TD_like":
            group = td
        else:
            continue

        set_path = find_thepresent_set_file(bids_root, rel, sid_bids)
        events_path = find_thepresent_events_file(bids_root, rel, sid_bids)
        rows.append({
            "subject_id": sid,
            "subject_id_bids": sid_bids,
            "release_id": rel,
            "group": group,
            "group_hbn_raw": grp_raw,
            "age_months": r.get("age_months"),
            "sex": r.get("sex_std"),
            "IQ_total": r.get("IQ_best_available"),
            "SRS_total": r.get("SRS_total_best"),
            "SCQ_total": r.get("SCQ_total_best"),
            "raw_EEG_file": str(set_path) if set_path else "",
            "events_file": str(events_path) if events_path else "",
            "task": "ThePresent",
            "included_final": int(set_path is not None and events_path is not None),
        })

    df = pd.DataFrame(rows)
    out_path = paths["participants_file"]
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    n_ready = int(df["included_final"].sum())
    logger.info("HBN ThePresent participants: %d ready / %d → %s", n_ready, len(df), out_path)
    return df


def preprocess_hbn_thepresent_subject(
    row: pd.Series,
    cfg: dict[str, Any],
    deriv_root: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """单被试：读 ThePresent .set → 裁 video 段 → 滤波 → epoch。"""
    subject_id = str(row["subject_id"])
    group = str(row["group"])
    epochs_path = deriv_root / "epochs" / f"{subject_id}-epo.fif"
    ep_cfg = cfg["epochs"]
    if epochs_path.exists() and not overwrite:
        epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
        return {
            "subject_id": subject_id,
            "group": group,
            "status": "cached",
            "usable_epochs": len(epochs),
            "usable_seconds": len(epochs) * ep_cfg["duration_sec"],
            "movie_duration_sec": np.nan,
        }

    set_path = Path(row["raw_EEG_file"])
    ev_path = Path(row["events_file"])
    tp_cfg = cfg.get("hbn", {}).get("thepresent", {})
    start_pat = _compile_patterns(tp_cfg.get("video_start_patterns", ["video_start"]))
    stop_pat = _compile_patterns(tp_cfg.get("video_stop_patterns", ["video_stop"]))
    fallback = float(tp_cfg.get("fallback_duration_sec", 206.0))

    t0, t1 = parse_thepresent_interval(ev_path, start_pat, stop_pat, fallback)
    raw = read_raw_eeg(set_path, preload=True)
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

    total_dur = raw.times[-1]
    t0 = max(0.0, t0)
    t1 = min(total_dur, t1)
    raw_movie = raw.copy().crop(tmin=t0, tmax=t1)

    epochs = make_epochs(
        raw_movie,
        duration=ep_cfg["duration_sec"],
        overlap=ep_cfg["overlap_sec"],
        reject_uv=ep_cfg["reject_amplitude_uv"],
    )

    ensure_dir(epochs_path.parent)
    epochs.save(epochs_path, overwrite=True)

    movie_dur = t1 - t0
    qc = {
        "subject_id": subject_id,
        "group": group,
        "status": "ok",
        "task": "ThePresent",
        "usable_epochs": len(epochs),
        "usable_seconds": len(epochs) * ep_cfg["duration_sec"],
        "movie_tmin_sec": t0,
        "movie_tmax_sec": t1,
        "movie_duration_sec": movie_dur,
        "n_channels": len(epochs.ch_names),
        "sfreq": epochs.info["sfreq"],
    }
    qc_path = deriv_root / "qc" / f"{subject_id}_preproc_qc.json"
    ensure_dir(qc_path.parent)
    qc_path.write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")
    return qc


def batch_preprocess_hbn_thepresent(
    cfg: dict[str, Any],
    limit: int | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    paths = resolve_hbn_paths(cfg)
    deriv = paths["derivatives_root"]
    participants = pd.read_csv(paths["participants_file"])
    participants = participants[participants["included_final"] == 1].copy()
    if limit:
        participants = participants.head(limit)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    min_epochs = int(cfg["epochs"]["min_usable_epochs"])

    for _, row in participants.iterrows():
        sid = row["subject_id"]
        try:
            qc = preprocess_hbn_thepresent_subject(row, cfg, deriv, overwrite=overwrite)
            qc["passes_epoch_threshold"] = int(qc["usable_epochs"] >= min_epochs)
            rows.append(qc)
            logger.info("%s: %d epochs (%.0fs movie)", sid, qc["usable_epochs"], qc.get("movie_duration_sec", np.nan))
        except Exception as exc:
            logger.warning("%s 预处理失败: %s", sid, exc)
            failures.append({"subject_id": sid, "error": str(exc)})

    summary = pd.DataFrame(rows)
    ensure_dir(deriv / "qc")
    save_csv(summary, deriv / "qc" / "preproc_summary.csv")
    if failures:
        save_csv(pd.DataFrame(failures), deriv / "qc" / "preproc_failures.csv")

    summary_slim = summary.drop(columns=["group"], errors="ignore")
    analysis = participants.merge(summary_slim, on="subject_id", how="left")
    analysis = analysis[analysis["passes_epoch_threshold"] == 1].copy()
    save_csv(analysis, deriv / "participants_analysis.csv")
    return summary


def run_hbn_psd_specparam(
    cfg: dict[str, Any],
    limit: int | None = None,
    overwrite: bool = False,
) -> None:
    paths = resolve_hbn_paths(cfg)
    deriv = paths["derivatives_root"]
    analysis_path = deriv / "participants_analysis.csv"
    if not analysis_path.exists():
        raise FileNotFoundError("请先运行 101_hbn_preprocess_resting.py")

    participants = pd.read_csv(analysis_path)
    if limit:
        participants = participants.head(limit)

    psd_dir = deriv / "psd"
    sp_dir = deriv / "specparam"
    ensure_dir(psd_dir)
    ensure_dir(sp_dir)

    all_sp: list[pd.DataFrame] = []
    for _, row in participants.iterrows():
        sid = str(row["subject_id"])
        group = str(row["group"])
        psd_path = psd_dir / f"{sid}_psd.csv"
        epochs_path = deriv / "epochs" / f"{sid}-epo.fif"

        if psd_path.exists() and not overwrite:
            psd_df = pd.read_csv(psd_path)
        else:
            epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
            welch = cfg["psd"]["welch"]
            freqs, psd, ch_names = compute_psd_from_epochs(
                epochs,
                fmin=welch["fmin"],
                fmax=welch["fmax"],
                welch_cfg=welch,
            )
            psd_df = psd_to_long_df(sid, group, freqs, psd, ch_names)
            save_csv(psd_df, psd_path)

        sp_df = fit_subject_specparam(psd_df, cfg["specparam"])
        all_sp.append(sp_df)

    channel_df = pd.concat(all_sp, ignore_index=True)
    save_csv(channel_df, sp_dir / "specparam_channel_results.csv")
    run_specparam_qc(
        channel_df,
        cfg,
        sp_dir / "specparam_channel_results_qc.csv",
        sp_dir / "specparam_qc_summary_subject.csv",
    )


def run_hbn_roi_aggregation(cfg: dict[str, Any]) -> None:
    paths = resolve_hbn_paths(cfg)
    deriv = paths["derivatives_root"]
    hbn_cfg = cfg.get("hbn", {})
    roi_path = PROJECT_ROOT / hbn_cfg.get("roi_config", "config/roi_channels_hbn129.yaml")
    layout = hbn_cfg.get("roi_layout", "channels_hbn129")

    in_csv = deriv / "specparam" / "specparam_channel_results_qc.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"未找到 {in_csv}")

    run_roi_pipeline(
        in_csv,
        deriv / "roi" / "specparam_subject_global.csv",
        deriv / "roi" / "specparam_subject_roi_long.csv",
        roi_cfg_path=roi_path,
        layout_override=layout,
    )


def run_hbn_external_validation(cfg: dict[str, Any]) -> Path:
    """外部复现总报告：confirmatory (106) 为主，spatial (105) 为探索。"""
    paths = resolve_hbn_paths(cfg)
    deriv = paths["derivatives_root"]
    out_root = paths["outputs_root"]
    ensure_dir(out_root)

    confirmatory_models = out_root / "tables" / "confirmatory_models.csv"
    confirmatory_report = out_root / "confirmatory_replication_report_zh.md"
    spatial_summary = deriv / "stats" / "spatial" / "spatial_inference_summary.csv"
    spatial_report = out_root / "spatial_stats_report_zh.md"

    lines = [
        "# HBN-EEG 外部复现报告",
        "",
        "## 定位",
        "独立 confirmatory replication：HBN Manifest A，复现主研究 **静息 eyes-open posterior/global exponent** 与 **group×age**。",
        "",
    ]

    participants = pd.read_csv(deriv / "participants_analysis.csv")
    n_asd = int((participants["group"] == cfg["groups"]["asd_label"]).sum())
    n_td = int((participants["group"] == cfg["groups"]["td_label"]).sum())
    lines.extend([
        f"- Manifest A 被试: **{len(participants)}** (ASD={n_asd}, TD={n_td})",
        "",
    ])

    if confirmatory_models.exists():
        models = pd.read_csv(confirmatory_models)
        lines.extend([
            "## 主推断：Confirmatory replication（106）",
            "",
            "**Primary posterior ROI**: homologous E33/E36/E37/E38 → E67/E72/E75/E77",
            "",
            "### Primary age-matched (72–131 mo)",
            "",
        ])
        for outcome in ["posterior_homologous_exponent", "global_exponent"]:
            for model, term_kind, label in [
                ("group_main", "group", "group (TD vs ASD)"),
                ("group_x_age", "interaction", "group×age"),
            ]:
                sub = models[
                    (models["cohort"] == "primary_age_matched")
                    & (models["outcome"] == outcome)
                    & (models["model"] == model)
                ]
                if term_kind == "group":
                    sub = sub[sub["term"].astype(str).str.contains(r"C\(group.*\[T\.TD\]", regex=True) & ~sub["term"].astype(str).str.contains("age_months")]
                else:
                    sub = sub[sub["term"].astype(str).str.contains(r":age_months", regex=False)]
                if len(sub):
                    r = sub.iloc[0]
                    lines.append(
                        f"- **{outcome}** {label}: β={r['coef']:.4f}, p={r['pvalue']:.4f}, n={int(r['n_obs'])}"
                    )
        lines.extend(["", "### Age strata (>72 mo)", ""])
        for outcome in ["posterior_homologous_exponent", "global_exponent"]:
            sub = models[
                (models["cohort"] == "age_gt_72")
                & (models["outcome"] == outcome)
                & (models["model"] == "group_main")
            ]
            sub = sub[sub["term"].astype(str).str.contains(r"C\(group.*\[T\.TD\]", regex=True) & ~sub["term"].astype(str).str.contains("age_months")]
            if len(sub):
                r = sub.iloc[0]
                lines.append(
                    f"- **{outcome}**: β={r['coef']:.4f}, p={r['pvalue']:.4f}, n={int(r['n_obs'])}"
                )
        lines.extend([
            "",
            "完整表格: `tables/confirmatory_models.csv`",
            f"详见: `{confirmatory_report.name}`",
            "",
        ])
    else:
        lines.extend([
            "## 主推断",
            "",
            "尚未运行 `106_hbn_confirmatory_replication.py`。",
            "",
        ])

    if spatial_summary.exists():
        sp = pd.read_csv(spatial_summary)
        lines.extend(["## 探索：空间统计（105）", ""])
        for _, r in sp.iterrows():
            p = r.get("p_cluster_global", np.nan)
            p_str = f"{p:.4f}" if pd.notna(p) else "NA"
            lines.append(f"- **{r['method']}**: stat={r['max_cluster_stat']:.4f}, p={p_str}")
        lines.extend(["", f"详见: `{spatial_report.name}`", ""])

    lines.extend([
        "## 解释口径",
        "",
        "- HBN TD = TD_like（transdiagnostic community control）。",
        "- EO 为多段 20 s 拼接；128 导非 64 导同名电极。",
        "- 主研究预期：TD posterior/global exponent > ASD；>72 月层效应更明显。",
        "- 任务态 ISC / JR 建模不在本 external replication 范围。",
        "",
    ])
    report_path = out_root / "validation_report_zh.md"
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("报告: %s", report_path)
    return report_path
