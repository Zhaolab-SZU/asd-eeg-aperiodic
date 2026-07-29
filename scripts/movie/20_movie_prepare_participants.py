# Ideal release name: 20_movie_prepare_participants.py
# Original path: scripts/31_prepare_movie_participants.py
# Note: Movie participant preparation
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
31_prepare_movie_participants.py
-------------------------------
扫描 movie EEG（.set），检查 VID+ marker 与可用时长，生成任务态 participants 清单。

输出:
- data/participants/participants_task_movie.csv
- derivatives_task_movie/qc/movie_marker_screening.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import save_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 movie 任务 participants 文件")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_task_movie.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--movie-root",
        type=str,
        default="data/Movie",
        help="movie 数据根目录（下含 ASD/TD）",
    )
    parser.add_argument(
        "--resting-participants",
        type=str,
        default="data/participants/participants.csv",
        help="用于合并人口学信息的原 participants.csv",
    )
    parser.add_argument(
        "--marker",
        type=str,
        default="VID+",
        help="视频开始 marker 文本（默认 VID+）",
    )
    parser.add_argument(
        "--required-duration-sec",
        type=float,
        default=300.0,
        help="从 marker 起要求的最短时长（秒）",
    )
    return parser.parse_args()


def extract_subject_id(name: str) -> str:
    stem = Path(name).stem
    m = re.match(r"^([ST]\d+)", stem, flags=re.IGNORECASE)
    if not m:
        return stem
    sid = m.group(1).upper()
    if sid.startswith("T") and sid[1:].isdigit():
        return f"T{int(sid[1:]):03d}"
    if sid.startswith("S") and sid[1:].isdigit():
        return f"S{int(sid[1:]):03d}"
    return sid


def pick_marker_onset(raw: mne.io.BaseRaw, marker: str) -> tuple[float | None, int]:
    if len(raw.annotations) == 0:
        return None, 0
    desc = np.array(raw.annotations.description, dtype=str)
    onsets = np.array(raw.annotations.onset, dtype=float)
    marker_upper = marker.upper()
    idx = [i for i, d in enumerate(desc) if marker_upper in d.upper()]
    if not idx:
        return None, 0
    return float(onsets[idx[0]]), len(idx)


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    log = setup_logging(cfg, name="prepare_movie_participants")

    movie_root = (PROJECT_ROOT / args.movie_root).resolve()
    rest_path = (PROJECT_ROOT / args.resting_participants).resolve()
    out_participants = PROJECT_ROOT / "data" / "participants" / "participants_task_movie.csv"
    deriv = Path(cfg["paths"]["derivatives_root"])
    out_screening = deriv / "qc" / "movie_marker_screening.csv"
    out_screening.parent.mkdir(parents=True, exist_ok=True)

    resting_df = pd.read_csv(rest_path)
    resting_df["subject_id"] = resting_df["subject_id"].astype(str).str.upper()
    if "group" in resting_df.columns:
        resting_df["group"] = resting_df["group"].astype(str).str.upper()

    records: list[dict[str, Any]] = []
    for group in ("ASD", "TD"):
        gdir = movie_root / group
        if not gdir.exists():
            log.warning("目录不存在，跳过: %s", gdir)
            continue
        files = sorted(gdir.glob("*.set"))
        log.info("%s: 检测到 %d 个 .set 文件", group, len(files))
        for fp in files:
            sid = extract_subject_id(fp.name)
            rec: dict[str, Any] = {
                "subject_id": sid,
                "group": group,
                "movie_set_file": str(fp.resolve()),
                "marker_name": args.marker,
                "read_ok": 0,
                "marker_found": 0,
                "marker_count": 0,
                "vid_start_sec": np.nan,
                "recording_duration_sec": np.nan,
                "duration_after_vid_sec": np.nan,
                "meets_duration_criterion": 0,
                "included_final": 0,
                "exclusion_reason": "",
            }
            try:
                raw = mne.io.read_raw_eeglab(fp, preload=False, verbose=False)
                rec["read_ok"] = 1
                total_dur = float(raw.n_times / raw.info["sfreq"])
                rec["recording_duration_sec"] = total_dur
                onset, cnt = pick_marker_onset(raw, args.marker)
                rec["marker_count"] = int(cnt)
                if onset is None:
                    rec["exclusion_reason"] = f"marker_{args.marker}_missing"
                else:
                    rec["marker_found"] = 1
                    rec["vid_start_sec"] = float(onset)
                    after = total_dur - float(onset)
                    rec["duration_after_vid_sec"] = after
                    if after >= args.required_duration_sec:
                        rec["meets_duration_criterion"] = 1
                        rec["included_final"] = 1
                    else:
                        rec["exclusion_reason"] = "duration_after_vid_insufficient"
            except Exception as exc:
                rec["exclusion_reason"] = f"read_error:{exc}"
            records.append(rec)

    screening = pd.DataFrame(records).sort_values(["group", "subject_id"]).reset_index(drop=True)
    save_csv(screening, out_screening)

    base_cols = [
        "subject_id", "group", "age_months", "sex", "IQ_total", "IQ_verbal", "IQ_nonverbal",
        "ADOS_total", "ADOS_SA", "ADOS_RRB", "SRS_total", "CARS_total", "ABC_total",
        "language_score", "medication_status",
    ]
    available_base_cols = [c for c in base_cols if c in resting_df.columns]
    out_df = screening[["subject_id", "group"]].merge(
        resting_df[available_base_cols].drop_duplicates(["subject_id", "group"]),
        on=["subject_id", "group"],
        how="left",
    )
    out_df["raw_EEG_file"] = screening["movie_set_file"]
    out_df["EEG_usable_seconds"] = np.nan
    out_df["EEG_usable_epochs"] = np.nan
    out_df["included_final"] = screening["included_final"].astype(int)
    out_df["exclusion_reason"] = screening["exclusion_reason"]
    save_csv(out_df, out_participants)

    n_total = len(screening)
    n_marker = int(screening["marker_found"].sum())
    n_dur = int(screening["meets_duration_criterion"].sum())
    n_in = int(screening["included_final"].sum())
    log.info(
        "movie 筛查完成: 总=%d, marker命中=%d, 时长达标=%d, included_final=1=%d",
        n_total, n_marker, n_dur, n_in,
    )
    log.info("screening: %s", out_screening)
    log.info("participants: %s", out_participants)


if __name__ == "__main__":
    main()
