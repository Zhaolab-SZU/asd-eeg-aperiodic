# Ideal release name: 13_traditional_band_power.py
# Original path: scripts/13_traditional_band_power.py
# Note: Traditional band-power controls
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
13_traditional_band_power.py
----------------------------
计算传统频段功率，并与 specparam peak power 对照。

输入: derivatives/psd/, specparam 结果
输出: derivatives/stats/traditional_band_power.csv
      derivatives/stats/band_power_comparison.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import load_analysis_participants, save_csv  # noqa: E402
from src.psd_utils import compute_band_power_from_psd  # noqa: E402
from src.stats_utils import spearman_correlation  # noqa: E402


def main() -> None:
    cfg = load_config()
    log = setup_logging(cfg, name="traditional_band_power")

    participants = load_analysis_participants(cfg)
    deriv = Path(cfg["paths"]["derivatives_root"])
    psd_dir = deriv / "psd"
    bands = cfg.get("bands", {})

    all_psd = []
    for _, row in participants.iterrows():
        sid = row["subject_id"]
        p = psd_dir / f"{sid}_psd.csv"
        if p.exists():
            all_psd.append(pd.read_csv(p))
    if not all_psd:
        log.error("未找到 PSD 文件")
        sys.exit(1)

    psd_combined = pd.concat(all_psd, ignore_index=True)
    band_power = compute_band_power_from_psd(psd_combined, bands)
    stats_dir = deriv / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    save_csv(band_power, stats_dir / "traditional_band_power.csv")

    # 与 specparam 对照
    sp_path = deriv / "specparam" / "specparam_channel_results_qc.csv"
    if not sp_path.exists():
        log.warning("无 specparam 结果，跳过对照")
        return

    sp_df = pd.read_csv(sp_path)
    if "fit_valid" in sp_df.columns:
        sp_df = sp_df[sp_df["fit_valid"]]

    comparison_rows = []
    for band in bands:
        trad_sub = band_power[band_power["band"] == band].groupby("subject_id")["band_power"].mean()
        peak_col = f"{band}_pw"
        if peak_col not in sp_df.columns:
            continue
        spec_sub = sp_df.groupby("subject_id")[peak_col].mean()
        common = trad_sub.index.intersection(spec_sub.index)
        for sid in common:
            comparison_rows.append({
                "subject_id": sid,
                "band": band,
                "traditional_power": trad_sub[sid],
                "specparam_peak_power": spec_sub[sid],
            })
        if len(common) >= 3:
            res = spearman_correlation(trad_sub[common], spec_sub[common])
            log.info("%s 传统 vs specparam Spearman rho=%.3f p=%.4f", band, res["rho"], res["pvalue"])

    if comparison_rows:
        save_csv(pd.DataFrame(comparison_rows), stats_dir / "band_power_comparison.csv")
    log.info("传统频段功率分析完成")


if __name__ == "__main__":
    main()
