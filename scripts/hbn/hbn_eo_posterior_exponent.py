# Ideal release name: hbn_eo_posterior_exponent.py
# Original path: scripts/hbn_eo_posterior_exponent.py
# Note: HBN EO posterior exponent helper
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
hbn_eo_posterior_exponent.py
----------------------------
HBN 静息态 eyes-open 后枕叶 aperiodic exponent 提取。

流程 (nuclear.enabled=true，默认):
  1. 读取 .set / .raw → EO 定位 + 首尾 2s/1s 切除 → 拼接
  2. 2 s Epoch → Montage → AutoReject → CSD
  3. ROI Welch PSD → specparam (knee) → EO_posterior_exponent

示例:
  # Nuclear 全量（远端服务器推荐）
  python scripts/hbn_eo_posterior_exponent.py --config config/config_hbn_resting.yaml --overwrite

  # 匹配队列 pilot
  python scripts/hbn_eo_posterior_exponent.py \\
    --participants outputs/hbn_external/participants_thepresent_matched_resting.csv \\
    --output outputs/hbn_nuclear/eo_posterior_exponent_matched.csv --limit-subjects 3

  # 复用旧 derivatives（legacy，无 AutoReject/CSD/knee）
  python scripts/hbn_eo_posterior_exponent.py --from-derivatives
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.hbn_eo_exponent import (  # noqa: E402
    POSTERIOR_ROI_DEFAULT,
    discover_hbn_subjects_from_bids,
    extract_eo_posterior_from_derivatives,
    fix_hbn_participant_paths,
    run_hbn_eo_posterior_batch,
)
from src.io_utils import ensure_dir, save_csv  # noqa: E402
from src.io_utils import load_participants  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HBN 静息态 EO 后枕叶 aperiodic exponent",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_hbn_resting.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--participants",
        type=str,
        default=None,
        help="被试表 CSV（需含 subject_id, raw_EEG_file；可选 events_file）",
    )
    parser.add_argument(
        "--bids-root",
        type=str,
        default=None,
        help="HBN BIDS 根目录（与 --release 联用自动发现被试）",
    )
    parser.add_argument(
        "--release",
        type=str,
        default="R11",
        help="HBN release ID，如 R11",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 CSV 路径（默认 outputs/hbn_resting/eo_posterior_exponent.csv）",
    )
    parser.add_argument(
        "--limit-subjects",
        type=int,
        default=None,
        help="仅处理前 N 名被试（调试）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有输出",
    )
    parser.add_argument(
        "--from-derivatives",
        action="store_true",
        help="从 derivatives/hbn_external 已有 specparam 提取（不重跑预处理）",
    )
    parser.add_argument(
        "--no-fix-paths",
        action="store_true",
        help="不自动修复 participants 中的跨机器路径",
    )
    return parser.parse_args()


def _load_subjects(args: argparse.Namespace, cfg: dict) -> "pd.DataFrame":
    import pandas as pd

    if args.participants:
        path = Path(args.participants)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        df = pd.read_csv(path)
        if "raw_EEG_file" not in df.columns:
            raise ValueError("participants 需包含 raw_EEG_file 列")
        df["subject_id"] = df["subject_id"].astype(str)
        return df

    if args.bids_root:
        bids_root = Path(args.bids_root)
        return discover_hbn_subjects_from_bids(bids_root, args.release)

    part_path = Path(cfg["paths"].get("participants_file", ""))
    if part_path.exists():
        try:
            return load_participants(part_path, included_only=False)
        except ValueError:
            import pandas as pd
            df = pd.read_csv(part_path)
            if "raw_EEG_file" in df.columns:
                df["subject_id"] = df["subject_id"].astype(str)
                return df

    raise FileNotFoundError(
        "请通过 --participants、--bids-root 或 config paths.participants_file 指定被试列表"
    )


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = load_config(cfg_path)
    log = setup_logging(cfg, name="hbn_eo_posterior")

    out_root = Path(cfg["paths"]["outputs_root"])
    out_csv = Path(args.output) if args.output else out_root / "eo_posterior_exponent.csv"
    if not out_csv.is_absolute():
        out_csv = PROJECT_ROOT / out_csv
    ensure_dir(out_csv.parent)
    overwrite = args.overwrite or bool(cfg.get("processing", {}).get("overwrite", False))

    if args.from_derivatives:
        deriv = Path(cfg["paths"]["derivatives_root"])
        spec_csv = deriv / "specparam" / "specparam_channel_results_qc.csv"
        roi = tuple(cfg.get("hbn", {}).get("roi_channels", list(POSTERIOR_ROI_DEFAULT)))
        if out_csv.exists() and not overwrite:
            log.info("输出已存在，跳过: %s", out_csv)
            return
        out_df = extract_eo_posterior_from_derivatives(spec_csv, roi)
        save_csv(out_df[["subject_id", "EO_posterior_exponent"]], out_csv)
        log.info(
            "从已有 derivatives 提取: %d 名被试 → %s（无 EO 2s/1s 边缘切除）",
            len(out_df), out_csv,
        )
        return

    subjects = _load_subjects(args, cfg)
    if not args.no_fix_paths:
        bids_root = Path(cfg["paths"]["bids_root"])
        subjects = fix_hbn_participant_paths(subjects, bids_root)
    if args.limit_subjects:
        subjects = subjects.head(args.limit_subjects)

    channel_dir = Path(cfg["paths"]["derivatives_root"]) / "specparam" / "posterior_channels"

    log.info("被试数: %d", len(subjects))
    out_df = run_hbn_eo_posterior_batch(
        subjects=subjects,
        cfg=cfg,
        out_csv=out_csv,
        channel_detail_dir=channel_dir,
        overwrite=overwrite,
    )
    log.info("完成: %d 名被试 → %s", len(out_df), out_csv)


if __name__ == "__main__":
    main()
