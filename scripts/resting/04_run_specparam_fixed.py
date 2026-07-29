# Ideal release name: 04_run_specparam_fixed.py
# Original path: scripts/04_run_specparam.py
# Note: Primary fixed-mode specparam
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
04_run_specparam.py
-------------------
对每个被试 × 通道拟合 specparam（支持滑窗时变拟合）。

输入: derivatives/psd/{subject_id}_psd.csv (静态)
      derivatives/psd/{subject_id}_psd_sliding.csv (滑窗)
输出: derivatives/specparam/specparam_channel_results.csv (静态)
      derivatives/specparam/specparam_channel_time_resolved_results.csv (滑窗)
      derivatives/specparam/specparam_exponent_timeseries_global.csv (滑窗汇总)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import load_analysis_participants  # noqa: E402
from src.specparam_utils import run_specparam_batch  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行静态或时变 specparam 拟合")
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


def main() -> None:
    args = _parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    log = setup_logging(cfg, name="run_specparam")

    participants = load_analysis_participants(cfg)
    if args.limit_subjects is not None:
        participants = participants.head(max(0, args.limit_subjects)).copy()
        log.info("已启用 --limit-subjects=%d，当前处理 %d 名", args.limit_subjects, len(participants))

    deriv = Path(cfg["paths"]["derivatives_root"])
    psd_dir = deriv / "psd"
    out_dir = deriv / "specparam"
    out_dir.mkdir(parents=True, exist_ok=True)
    dynamic_mode = bool(cfg.get("psd", {}).get("sliding_window", {}).get("enabled", False))
    out_csv = (
        out_dir / "specparam_channel_time_resolved_results.csv"
        if dynamic_mode
        else out_dir / "specparam_channel_results.csv"
    )
    summary_csv = (
        out_dir / "specparam_exponent_timeseries_global.csv"
        if dynamic_mode
        else None
    )
    psd_suffix = "_psd_sliding.csv" if dynamic_mode else "_psd.csv"

    try:
        run_specparam_batch(
            participants=participants,
            psd_dir=psd_dir,
            out_csv=out_csv,
            cfg=cfg,
            psd_suffix=psd_suffix,
            dynamic_summary_out_csv=summary_csv,
        )
        log.info("specparam 拟合完成: %s", out_csv)
        if summary_csv is not None:
            log.info("动态 exponent 时间序列已保存: %s", summary_csv)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
