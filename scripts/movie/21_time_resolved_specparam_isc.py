# Ideal release name: 21_time_resolved_specparam_isc.py
# Original path: scripts/97_posterior_movie_specparam_isc.py
# Note: Time-resolved posterior specparam + ISC pipeline
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
97_posterior_movie_specparam_isc.py
-----------------------------------
阶段 2 实证对齐：后枕四导（E33/E36/E37/E38）观影滑窗 specparam + Aperiodic-ISC 重算。

输入:
  derivatives_task_movie/psd/*_psd_sliding.csv
  derivatives_task_movie/stats/movie_event_aligned_timeseries_qc_valid.csv（事件标签）
输出:
  outputs/jr_modelling/posterior_movie_isc/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.aperiodic_isc import compute_cohort_isc, isc_group_tests  # noqa: E402
from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import save_csv  # noqa: E402
from src.spectral_maturation_analysis import POSTERIOR_CORE  # noqa: E402
from src.jansen_rit import loglog_aperiodic_exponent  # noqa: E402
from src.specparam_utils import _extract_freq_columns  # noqa: E402

POSTERIOR_CHANNELS = POSTERIOR_CORE
EVENT_TYPES = ("mental", "pain", "neutral")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="后枕四导观影滑窗 specparam + ISC")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--movie-deriv", type=str, default="derivatives_task_movie")
    p.add_argument("--max-subjects", type=int, default=None)
    return p.parse_args()


def _fit_sliding_posterior(psd_path: Path, fmin: float, fmax: float) -> pd.DataFrame:
    """滑窗 PSD → log-log exponent（与 JR 仿真指标一致，速度快）。"""
    df = pd.read_csv(psd_path)
    df = df[df["channel"].isin(POSTERIOR_CHANNELS)].copy()
    if df.empty:
        return pd.DataFrame()
    freq_cols, freqs = _extract_freq_columns(df)
    rows = []
    group_cols = ["subject_id", "group", "window_index", "window_start_sec", "window_end_sec", "channel"]
    for keys, sub in df.groupby(group_cols):
        sid, grp, widx, wstart, wend, ch = keys
        power = sub.iloc[0][freq_cols].to_numpy(dtype=float)
        exp = loglog_aperiodic_exponent(freqs, power, fmin=fmin, fmax=fmax)
        valid = np.isfinite(exp)
        rows.append(
            {
                "subject_id": str(sid),
                "group": grp,
                "window_index": int(widx),
                "window_start_sec": float(wstart),
                "window_end_sec": float(wend),
                "channel": ch,
                "aperiodic_exponent": exp,
                "fit_valid": bool(valid),
            }
        )
    return pd.DataFrame(rows)


def _attach_event_labels(fit_df: pd.DataFrame, labels_path: Path) -> pd.DataFrame:
    if not labels_path.exists():
        fit_df["event_type"] = "neutral"
        return fit_df
    lab = pd.read_csv(labels_path)
    lab = lab[["subject_id", "window_index", "event_type"]].drop_duplicates()
    lab["subject_id"] = lab["subject_id"].astype(str)
    fit_df["subject_id"] = fit_df["subject_id"].astype(str)
    merged = fit_df.merge(lab, on=["subject_id", "window_index"], how="left")
    merged["event_type"] = merged["event_type"].fillna("neutral")
    return merged


def main() -> None:
    args = _parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    log = setup_logging(cfg, name="posterior_movie_isc")

    movie_deriv = PROJECT_ROOT / args.movie_deriv
    psd_dir = movie_deriv / "psd"
    labels_path = movie_deriv / "stats" / "movie_event_aligned_timeseries_qc_valid.csv"
    if not psd_dir.exists():
        raise FileNotFoundError(f"未找到 {psd_dir}")

    out_dir = Path(cfg["paths"]["outputs_root"]) / "jr_modelling" / "posterior_movie_isc"
    out_dir.mkdir(parents=True, exist_ok=True)

    sp_cfg = cfg.get("specparam", {})
    fmin, fmax = sp_cfg.get("freq_range", [1.0, 40.0])

    psd_files = sorted(psd_dir.glob("*_psd_sliding.csv"))
    if args.max_subjects:
        psd_files = psd_files[: args.max_subjects]
    log.info("处理 %d 名被试滑窗 PSD", len(psd_files))

    all_fit = []
    for psd_path in tqdm(psd_files, desc="posterior specparam"):
        sub_fit = _fit_sliding_posterior(psd_path, fmin=float(fmin), fmax=float(fmax))
        if not sub_fit.empty:
            all_fit.append(sub_fit)
    if not all_fit:
        raise RuntimeError("无有效后枕滑窗拟合结果")

    fit_df = pd.concat(all_fit, ignore_index=True)
    fit_df = _attach_event_labels(fit_df, labels_path)
    save_csv(fit_df, out_dir / "posterior_sliding_specparam_channel.csv")

    valid = fit_df[fit_df["fit_valid"]].copy()
    # 四导均值
    mean_df = (
        valid.groupby(
            ["subject_id", "group", "window_index", "window_start_sec", "window_end_sec", "event_type"],
            as_index=False,
        )["aperiodic_exponent"]
        .mean()
        .rename(columns={"aperiodic_exponent": "posterior_exponent"})
    )
    save_csv(mean_df, out_dir / "posterior_sliding_exponent_timeseries.csv")

    isc_input = mean_df.rename(columns={"posterior_exponent": "exponent"})
    isc_df = compute_cohort_isc(isc_input, value_col="exponent", event_col="event_type")
    save_csv(isc_df, out_dir / "posterior_isc_subject_values.csv")

    isc_stats = isc_group_tests(isc_df)
    save_csv(isc_stats, out_dir / "posterior_isc_group_stats.csv")

    # 与历史单通道 ISC 对照（若存在）
    legacy_path = movie_deriv / "stats" / "movie_isc_group_stats.csv"
    compare_lines = ["## 与历史 pipeline 对照", ""]
    if legacy_path.exists():
        leg = pd.read_csv(legacy_path)
        merged = isc_stats.merge(leg, on="event_type", how="outer", suffixes=("_posterior4", "_legacy"))
        save_csv(merged, out_dir / "isc_legacy_comparison.csv")
        for _, r in merged.iterrows():
            compare_lines.append(
                f"- **{r['event_type']}**: 四导 posterior4 p={r.get('p_value_posterior4', np.nan):.4f}, "
                f"legacy p={r.get('p_value', np.nan):.4f}"
            )
    else:
        compare_lines.append("- 未找到 legacy movie_isc_group_stats.csv")

    lines = [
        "# 后枕四导 Aperiodic-ISC 重算报告（阶段 2 实证对齐）",
        "",
        f"- 电极: {', '.join(POSTERIOR_CHANNELS)}",
        f"- 被试数: {mean_df['subject_id'].nunique()}",
        f"- 滑窗拟合有效行: {len(valid)} / {len(fit_df)}",
        "",
        "## 组间 ISC（Fisher z）",
        "",
        "| event | n_ASD | n_TD | ASD mean(z) | TD mean(z) | t | p |",
        "|-------|-------|------|-------------|------------|---|---|",
    ]
    for _, r in isc_stats.iterrows():
        lines.append(
            f"| {r['event_type']} | {int(r['n_asd'])} | {int(r['n_td'])} | "
            f"{r['asd_mean_z']:.4f} | {r['td_mean_z']:.4f} | {r['t_stat']:.3f} | {r['p_value']:.4f} |"
        )
    lines.extend(["", *compare_lines, "", "> ISC 计算：片段内时序重采样至 80 点 + TD leave-one-out 模板。"])
    (out_dir / "posterior_isc_report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("完成 → %s", out_dir)


if __name__ == "__main__":
    main()
