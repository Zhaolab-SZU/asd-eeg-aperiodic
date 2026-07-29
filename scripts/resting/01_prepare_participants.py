# Ideal release name: 01_prepare_participants.py
# Original path: scripts/01_prepare_participants.py
# Note: Participant table preparation
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
01_prepare_participants.py
--------------------------
从 Resting_info.xlsx 与 data/raw 同步 participants.csv，并生成分析用列表。

输入: data/participants/Resting_info.xlsx, data/raw/**/*.mff
输出: data/participants/participants.csv
      data/participants/participants_excluded_no_eeg_data.csv
      data/participants/participants_data_without_info.csv
      derivatives/participants_analysis.csv
      derivatives/participants_excluded.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import load_participants, save_csv  # noqa: E402

REQUIRED_COLS = [
    "subject_id", "group", "age_months", "sex",
    "raw_EEG_file", "included_final",
]
OPTIONAL_CLINICAL = [
    "IQ_total", "IQ_verbal", "IQ_nonverbal",
    "ADOS_total", "ADOS_SA", "ADOS_RRB",
    "SRS_total", "language_score",
]


def validate_participants(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """基本字段验证。"""
    groups = cfg.get("groups", {})
    valid_groups = {groups.get("asd_label", "ASD"), groups.get("td_label", "TD")}
    unknown = set(df["group"].dropna().unique()) - valid_groups
    if unknown:
        raise ValueError(f"未知 group 标签: {unknown}，期望: {valid_groups}")
    if df["subject_id"].duplicated().any():
        dup = df.loc[df["subject_id"].duplicated(), "subject_id"].tolist()
        raise ValueError(f"重复 subject_id: {dup}")
    return df


def main() -> None:
    cfg = load_config()
    log = setup_logging(cfg, name="prepare_participants")

    resting_info = PROJECT_ROOT / "data" / "participants" / "Resting_info.xlsx"
    part_path = Path(cfg["paths"]["participants_file"])

    if resting_info.exists():
        from scripts.sync_participants_from_resting_info import build_participants

        participants, excluded_no_data, data_no_info, report = build_participants()
        out_dir = part_path.parent
        try:
            participants.to_csv(part_path, index=False, encoding="utf-8-sig")
        except PermissionError:
            alt = out_dir / "participants_new.csv"
            participants.to_csv(alt, index=False, encoding="utf-8-sig")
            log.warning(
                "无法写入 %s（文件可能被占用），已写入 %s。请关闭 CSV 后重命名覆盖。",
                part_path,
                alt,
            )
            part_path = alt
        excluded_no_data.to_csv(
            out_dir / "participants_excluded_no_eeg_data.csv", index=False, encoding="utf-8-sig",
        )
        data_no_info.to_csv(
            out_dir / "participants_data_without_info.csv", index=False, encoding="utf-8-sig",
        )
        (out_dir / "participants_sync_report.txt").write_text(report, encoding="utf-8")
        log.info("\n%s", report)

    df_all = pd.read_csv(part_path)
    log.info("加载 %d 名被试", len(df_all))

    missing = [c for c in REQUIRED_COLS if c not in df_all.columns]
    if missing:
        raise ValueError(f"participants.csv 缺少必需列: {missing}")

    for col in OPTIONAL_CLINICAL:
        if col not in df_all.columns:
            log.warning("缺少可选临床列: %s", col)

    df_all = validate_participants(df_all, cfg)

    deriv = Path(cfg["paths"]["derivatives_root"])
    included = df_all[df_all["included_final"].astype(int) == 1].copy()
    excluded = df_all[df_all["included_final"].astype(int) != 1].copy()

    save_csv(included, deriv / "participants_analysis.csv")
    save_csv(excluded, deriv / "participants_excluded.csv")
    log.info("纳入分析: %d 名，排除: %d 名", len(included), len(excluded))


if __name__ == "__main__":
    main()
