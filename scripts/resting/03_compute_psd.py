# Ideal release name: 03_compute_psd.py
# Original path: scripts/03_compute_psd.py
# Note: Welch PSD
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
03_compute_psd.py
-----------------
计算 Welch PSD（静态平均或滑窗时变）。

输入: derivatives/epochs/{subject_id}_epochs.fif (静态模式)
      derivatives/preprocessed/{subject_id}-raw.fif (滑窗模式)
      participants 表（group 信息）
输出: derivatives/psd/{subject_id}_psd.csv (静态)
      derivatives/psd/{subject_id}_psd_sliding.csv (滑窗宽表)
      derivatives/qc/psd/*.png
      outputs/figures/qc_group_mean_psd*
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mne

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import get_epochs_fif_path, get_raw_fif_path, load_analysis_participants  # noqa: E402
from src.psd_utils import (  # noqa: E402
    compute_subject_sliding_psd,
    compute_subject_psd,
    dynamic_psd_to_mean_long_df,
    plot_group_mean_psd,
    plot_group_mean_psd_from_df,
    plot_subject_psd,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算静态或滑窗 PSD")
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
    log = setup_logging(cfg, name="compute_psd")

    participants = load_analysis_participants(cfg)
    if args.limit_subjects is not None:
        participants = participants.head(max(0, args.limit_subjects)).copy()
        log.info("已启用 --limit-subjects=%d，当前处理 %d 名", args.limit_subjects, len(participants))

    min_epochs = int(cfg.get("epochs", {}).get("min_usable_epochs", 60))
    deriv = Path(cfg["paths"]["derivatives_root"])
    epochs_dir = deriv / "epochs"
    preproc_dir = deriv / "preprocessed"
    psd_dir = deriv / "psd"
    psd_qc_dir = deriv / "qc" / "psd"
    psd_dir.mkdir(parents=True, exist_ok=True)
    psd_qc_dir.mkdir(parents=True, exist_ok=True)
    sw_cfg = cfg.get("psd", {}).get("sliding_window", {})
    dynamic_mode = bool(sw_cfg.get("enabled", False))
    log.info("PSD 模式: %s", "sliding_window" if dynamic_mode else "static")

    n_ok = 0
    mean_psd_dfs = []
    for _, row in participants.iterrows():
        sid = row["subject_id"]
        group = row["group"]

        try:
            if dynamic_mode:
                raw_path = get_raw_fif_path(preproc_dir, sid)
                out_csv = psd_dir / f"{sid}_psd_sliding.csv"
                df = compute_subject_sliding_psd(sid, group, raw_path, out_csv, cfg)
                mean_df = dynamic_psd_to_mean_long_df(df)
                plot_subject_psd(mean_df, psd_qc_dir / f"{sid}_psd_sliding_mean", title=f"{sid} Sliding PSD mean")
                mean_psd_dfs.append(mean_df)
            else:
                ep_path = get_epochs_fif_path(epochs_dir, sid)
                out_csv = psd_dir / f"{sid}_psd.csv"
                if not ep_path.exists():
                    log.warning("%s: 未找到 epochs 文件 %s，请先运行 02_preprocess_eeg.py", sid, ep_path)
                    continue
                epochs = mne.read_epochs(ep_path, preload=True, verbose=False)
                if len(epochs) < min_epochs:
                    log.warning(
                        "%s: 仅 %d 个 epoch (< %d)，跳过 PSD（应不在分析队列中）",
                        sid, len(epochs), min_epochs,
                    )
                    continue
                df = compute_subject_psd(sid, group, ep_path, out_csv, cfg)
                plot_subject_psd(df, psd_qc_dir / f"{sid}_psd", title=f"{sid} PSD")
                mean_psd_dfs.append(df)
            n_ok += 1
        except Exception as exc:
            log.exception("%s PSD 计算失败: %s", sid, exc)

    if n_ok == 0:
        log.error("未成功计算任何 PSD。请检查 epochs 文件是否存在。")
        sys.exit(1)

    fig_out = Path(cfg["paths"]["outputs_root"]) / "figures" / "qc_group_mean_psd"
    if dynamic_mode:
        if mean_psd_dfs:
            import pandas as pd

            all_df = pd.concat(mean_psd_dfs, ignore_index=True)
            plot_group_mean_psd_from_df(
                all_df,
                cfg,
                fig_out.with_name("qc_group_mean_psd_sliding"),
            )
    else:
        plot_group_mean_psd(psd_dir, participants, cfg, fig_out)
    log.info("PSD 计算完成: %d 名被试", n_ok)


if __name__ == "__main__":
    main()
