# Ideal release name: 02_preprocess_eeg.py
# Original path: scripts/02_preprocess_eeg.py
# Note: Resting EEG preprocess (import, filter, bad-channel, ICA, epoch). Maps to idealized 00_raw_import / 01_bad_channel / 02_ica / 03_epoch steps.
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
02_preprocess_eeg.py
--------------------
批量 EEG 预处理：滤波、ICA、分段、质控。

仅处理 participants.csv 中 included_final == 1 的被试。

输入: data/participants/participants.csv
      data/raw/**/*.mff
输出: derivatives/preprocessed/{subject_id}-raw.fif
      derivatives/epochs/{subject_id}-epo.fif
      derivatives/qc/{subject_id}_preproc_qc.json
      derivatives/qc/preproc_summary.csv
      derivatives/qc/preproc_failures.csv
      derivatives/qc/preproc_low_epoch_subjects.csv
      derivatives/participants_analysis.csv   # 仅 epoch 达标，供 03 及之后使用
      derivatives/qc/participants_excluded_epochs.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.eeg_preprocessing import batch_preprocess  # noqa: E402
from src.io_utils import load_participants, save_csv  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量 EEG 预处理")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（例如 config/config_task_movie.yaml）",
    )
    parser.add_argument(
        "--limit-subjects",
        type=int,
        default=None,
        help="仅处理前 N 名被试（pilot 调试）",
    )
    return parser.parse_args()


def build_preproc_summary(
    qc_df: pd.DataFrame,
    participants: pd.DataFrame,
    min_epochs: int,
) -> pd.DataFrame:
    """合并人口学信息并标记 epoch 不足的被试。"""
    if qc_df.empty:
        return qc_df

    merge_cols = ["subject_id", "group", "age_months", "sex"]
    merge_cols = [c for c in merge_cols if c in participants.columns]
    base = participants[merge_cols].drop_duplicates("subject_id")

    summary = qc_df.merge(base, on=["subject_id", "group"], how="left", suffixes=("", "_meta"))
    if "group_meta" in summary.columns:
        summary["group"] = summary["group"].fillna(summary["group_meta"])
        summary = summary.drop(columns=["group_meta"])

    summary["min_epochs_required"] = min_epochs
    summary["meets_epoch_criterion"] = summary["usable_epochs"] >= min_epochs
    summary["ica_n_removed"] = summary["ica_removed_components"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    summary["dropped_ref_n"] = summary["dropped_reference_channels"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    col_order = [
        "subject_id", "group", "age_months", "sex", "status",
        "usable_epochs", "usable_seconds", "meets_epoch_criterion",
        "min_epochs_required", "n_channels", "sampling_rate",
        "bad_channel_count", "ica_n_removed", "dropped_ref_n",
        "bad_channels", "ica_removed_components", "raw_file",
        "eeg_system", "resting_condition",
    ]
    cols = [c for c in col_order if c in summary.columns]
    extra = [c for c in summary.columns if c not in cols]
    return summary[cols + extra]


def main() -> None:
    args = _parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    log = setup_logging(cfg, name="preprocess_eeg")
    deriv = Path(cfg["paths"]["derivatives_root"])
    qc_dir = deriv / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    part_path = Path(cfg["paths"]["participants_file"])
    participants = load_participants(part_path, included_only=True)
    if args.limit_subjects is not None:
        participants = participants.head(max(0, args.limit_subjects)).copy()
        log.info("已启用 --limit-subjects=%d，当前处理 %d 名", args.limit_subjects, len(participants))

    if participants.empty:
        log.error("无被试可处理。请检查 participants.csv 中 included_final == 1。")
        sys.exit(1)

    min_epochs = int(cfg.get("epochs", {}).get("min_usable_epochs", 60))
    log.info(
        "开始预处理: %d 名被试 (included_final=1), 最少有效 epoch=%d",
        len(participants),
        min_epochs,
    )

    summary_raw, failures = batch_preprocess(participants, cfg)

    if summary_raw.empty and failures.empty:
        log.error("未处理任何被试。")
        sys.exit(1)

    if not failures.empty:
        fail_path = qc_dir / "preproc_failures.csv"
        save_csv(failures, fail_path)
        log.warning("失败 %d 名，见 %s", len(failures), fail_path)

    if summary_raw.empty:
        log.error("预处理未成功任何被试。")
        sys.exit(1)

    summary = build_preproc_summary(summary_raw, participants, min_epochs)
    save_csv(summary, qc_dir / "preproc_summary.csv")

    low_epoch = summary[~summary["meets_epoch_criterion"]]
    if not low_epoch.empty:
        save_csv(low_epoch, qc_dir / "preproc_low_epoch_subjects.csv")
        save_csv(low_epoch, qc_dir / "participants_excluded_epochs.csv")
        log.warning(
            "有效 epoch < %d: %d 名已排除后续分析，见 participants_excluded_epochs.csv",
            min_epochs,
            len(low_epoch),
        )

    passed_ids = summary.loc[summary["meets_epoch_criterion"], "subject_id"]
    analysis_df = participants[participants["subject_id"].isin(passed_ids)].copy()
    analysis_df = analysis_df.merge(
        summary[["subject_id", "usable_epochs", "usable_seconds", "meets_epoch_criterion"]],
        on="subject_id",
        how="left",
    )
    save_csv(analysis_df, deriv / "participants_analysis.csv")
    log.info("分析队列已写入 participants_analysis.csv: %d 名", len(analysis_df))

    n_ok = int(summary["meets_epoch_criterion"].sum())
    log.info(
        "预处理完成: 成功 %d 名，满足 epoch 标准 %d 名，失败 %d 名",
        len(summary),
        n_ok,
        len(failures),
    )
    log.info("摘要: %s", qc_dir / "preproc_summary.csv")


if __name__ == "__main__":
    main()
