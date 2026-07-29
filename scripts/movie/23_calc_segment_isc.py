# Ideal release name: 23_calc_segment_isc.py
# Original path: scripts/33_compute_segment_isc.py
# Note: Segment-level ISC helpers
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
33_compute_segment_isc.py
-------------------------
基于电影事件片段计算被试间相关（ISC）：
1) 从 roi_mean 动态 exponent 序列提取 mental / pain 事件片段
2) 同类事件片段按时间顺序拼接为每个被试的事件序列
3) 计算 TD-template ISC（TD 留一模板 + ASD 对全 TD 模板）
4) Fisher z 转换、组间 t 检验、分组箱线图
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import save_csv  # noqa: E402
from src.stats_utils import fdr_correction  # noqa: E402


EVENT_TYPES_BASE = ("mental", "pain")
EVENT_TYPES_ALL = ("mental", "pain", "neutral")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ISC on concatenated movie event segments")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_task_movie.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--events_csv",
        type=str,
        default="data/movie_events.csv",
        help="事件标签CSV路径",
    )
    parser.add_argument(
        "--time_bin_sec",
        type=float,
        default=0.5,
        help="时间步长（保持 0.5s）",
    )
    parser.add_argument(
        "--min_overlap_points",
        type=int,
        default=10,
        help="计算相关所需最小重叠点数",
    )
    parser.add_argument(
        "--movie_analysis_csv",
        type=str,
        default="derivatives_task_movie/participants_analysis.csv",
        help="movie 分析队列 CSV（用于统一样本口径）",
    )
    parser.add_argument(
        "--movie_specparam_qc_csv",
        type=str,
        default="derivatives_task_movie/specparam/specparam_qc_summary_subject.csv",
        help="movie 被试级 specparam QC CSV（用于剔除 low_quality_subject）",
    )
    parser.add_argument(
        "--neutral_stats_csv",
        type=str,
        default="derivatives_task_movie/stats/movie_isc_group_stats_with_neutral.csv",
        help="保留参数（当前版本不使用，仅兼容旧命令）",
    )
    parser.add_argument(
        "--isc-mode",
        type=str,
        choices=("segmented", "overall"),
        default=None,
        help="segmented=按 movie_events 片段；overall=全片时间轴（无事件标签，用于外部 ThePresent）",
    )
    return parser.parse_args()


def _required_cols(df: pd.DataFrame, cols: set[str], name: str) -> None:
    miss = cols - set(df.columns)
    if miss:
        raise ValueError(f"{name} 缺少列: {sorted(miss)}")


def safe_corr(x: np.ndarray, y: np.ndarray, min_points: int) -> tuple[float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < min_points:
        return np.nan, n
    xv = x[mask]
    yv = y[mask]
    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return np.nan, n
    return float(np.corrcoef(xv, yv)[0, 1]), n


def fisher_z(r: float) -> float:
    if pd.isna(r):
        return np.nan
    r = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(r))


def build_concat_keys(
    ts_bins: np.ndarray,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """将所有时间 bin 标注为 mental / pain / neutral。"""
    rows = []
    events = events.copy()
    events["event_type"] = events["event_type"].astype(str).str.strip().str.lower()
    events = events[events["event_type"].isin(EVENT_TYPES_BASE)].copy()
    events["start"] = pd.to_numeric(events["onset_sec"], errors="coerce")
    events["end"] = pd.to_numeric(events["onset_sec"], errors="coerce") + pd.to_numeric(
        events["duration_sec"], errors="coerce"
    )
    events = events.dropna(subset=["start", "end"]).sort_values("start").reset_index(drop=True)

    for b in np.sort(np.unique(np.round(ts_bins, 6))):
        labels = []
        for _, ev in events.iterrows():
            if float(ev["start"]) <= float(b) < float(ev["end"]):
                labels.append(str(ev["event_type"]))
        if "mental" in labels:
            event_type = "mental"
        elif "pain" in labels:
            event_type = "pain"
        else:
            event_type = "neutral"
        rows.append(
            {
                "center_sec": float(b),
                "event_type": event_type,
            }
        )

    out = pd.DataFrame(rows)
    for ev in EVENT_TYPES_ALL:
        idx = out.index[out["event_type"] == ev].to_numpy()
        out.loc[out["event_type"] == ev, "concat_index"] = np.arange(len(idx), dtype=int)
    out["concat_index"] = out["concat_index"].astype(int)
    return out


def build_concat_keys_overall(ts_bins: np.ndarray) -> pd.DataFrame:
    """全片时间轴 ISC：不依赖事件标签，所有时间点归为 overall。"""
    bins = np.sort(np.unique(np.round(ts_bins, 6)))
    return pd.DataFrame(
        {
            "center_sec": bins.astype(float),
            "event_type": "overall",
            "concat_index": np.arange(len(bins), dtype=int),
        }
    )


def build_subject_concat_series(
    ts: pd.DataFrame,
    keys_df: pd.DataFrame,
) -> pd.DataFrame:
    out_rows = []
    for (sid, grp), sub in ts.groupby(["subject_id", "group"], sort=False):
        sub2 = (
            sub[["center_sec", "exponent_mean"]]
            .groupby("center_sec", as_index=False)
            .mean()
            .copy()
        )
        merged = keys_df.merge(sub2, on="center_sec", how="left")
        merged["subject_id"] = sid
        merged["group"] = grp
        out_rows.append(
            merged[
                [
                    "subject_id",
                    "group",
                    "event_type",
                    "concat_index",
                    "center_sec",
                    "exponent_mean",
                ]
            ]
        )
    return pd.concat(out_rows, ignore_index=True)


def compute_isc(
    subject_concat: pd.DataFrame,
    min_overlap_points: int,
    event_types: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for ev in event_types:
        sub_ev = subject_concat[subject_concat["event_type"] == ev].copy()
        mat = sub_ev.pivot_table(
            index=["subject_id", "group"],
            columns="concat_index",
            values="exponent_mean",
            aggfunc="mean",
        )
        mat = mat.sort_index(axis=1)
        idx_df = mat.index.to_frame(index=False)
        idx_df.columns = ["subject_id", "group"]

        td_mask = idx_df["group"] == "TD"
        asd_mask = idx_df["group"] == "ASD"
        td_vals = mat[td_mask.to_numpy()]
        asd_vals = mat[asd_mask.to_numpy()]

        if td_vals.shape[0] < 2:
            raise RuntimeError(f"{ev}: TD 样本不足（至少需要2名用于留一模板）")

        # TD 留一模板
        td_subjects = idx_df.loc[td_mask, "subject_id"].tolist()
        for i, sid in enumerate(td_subjects):
            x = td_vals.iloc[i].to_numpy(dtype=float)
            others = td_vals.drop(td_vals.index[i])
            tmpl = others.mean(axis=0, skipna=True).to_numpy(dtype=float)
            r, n_overlap = safe_corr(x, tmpl, min_points=min_overlap_points)
            rows.append({
                "subject_id": sid,
                "group": "TD",
                "event_type": ev,
                "isc_r": r,
                "isc_z": fisher_z(r),
                "n_overlap_points": n_overlap,
                "template_type": "TD_LOO",
            })

        # ASD 对 full TD 模板
        td_template = td_vals.mean(axis=0, skipna=True).to_numpy(dtype=float)
        asd_subjects = idx_df.loc[asd_mask, "subject_id"].tolist()
        for i, sid in enumerate(asd_subjects):
            x = asd_vals.iloc[i].to_numpy(dtype=float)
            r, n_overlap = safe_corr(x, td_template, min_points=min_overlap_points)
            rows.append({
                "subject_id": sid,
                "group": "ASD",
                "event_type": ev,
                "isc_r": r,
                "isc_z": fisher_z(r),
                "n_overlap_points": n_overlap,
                "template_type": "TD_FULL",
            })
    return pd.DataFrame(rows)


def _resolve_isc_mode(args: argparse.Namespace, cfg: dict) -> str:
    if args.isc_mode is not None:
        return str(args.isc_mode).strip().lower()
    return str(cfg.get("movie", {}).get("isc_mode", "segmented")).strip().lower()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    log = setup_logging(cfg, name="compute_segment_isc")
    isc_mode = _resolve_isc_mode(args, cfg)
    if isc_mode not in {"segmented", "overall"}:
        raise ValueError(f"未知 isc_mode: {isc_mode}")

    deriv = Path(cfg["paths"]["derivatives_root"])
    out_root = Path(cfg["paths"]["outputs_root"])
    ts_path = deriv / "specparam" / "specparam_exponent_timeseries_global.csv"
    events_path = (PROJECT_ROOT / args.events_csv).resolve()
    movie_analysis_path = (PROJECT_ROOT / args.movie_analysis_csv).resolve()
    movie_qc_path = (PROJECT_ROOT / args.movie_specparam_qc_csv).resolve()

    if not ts_path.exists():
        raise FileNotFoundError(f"未找到动态序列: {ts_path}")
    if isc_mode == "segmented" and not events_path.exists():
        raise FileNotFoundError(f"未找到事件文件: {events_path}")

    ts = pd.read_csv(ts_path)
    _required_cols(ts, {"subject_id", "group", "window_start_sec", "window_end_sec", "exponent_mean"}, "timeseries")
    ts = ts.copy()
    ts["subject_id"] = ts["subject_id"].astype(str)
    ts["group"] = ts["group"].astype(str).str.upper()
    ts["center_sec"] = np.round((ts["window_start_sec"] + ts["window_end_sec"]) / 2.0, 6)

    # 统一 movie 分析队列（participants_analysis ∩ 非 low_quality）
    if movie_analysis_path.exists():
        movie_analysis = pd.read_csv(movie_analysis_path)
        _required_cols(movie_analysis, {"subject_id", "group"}, "movie_analysis_csv")
        movie_analysis["subject_id"] = movie_analysis["subject_id"].astype(str)
        movie_analysis["group"] = movie_analysis["group"].astype(str).str.upper()
        allowed_pairs = movie_analysis[["subject_id", "group"]].drop_duplicates()
        ts = ts.merge(allowed_pairs, on=["subject_id", "group"], how="inner")

    if movie_qc_path.exists():
        movie_qc = pd.read_csv(movie_qc_path)
        _required_cols(movie_qc, {"subject_id", "low_quality_subject"}, "movie_specparam_qc_csv")
        movie_qc["subject_id"] = movie_qc["subject_id"].astype(str)
        bad_ids = set(
            movie_qc.loc[pd.to_numeric(movie_qc["low_quality_subject"], errors="coerce") == 1, "subject_id"].tolist()
        )
        if bad_ids:
            ts = ts[~ts["subject_id"].isin(bad_ids)].copy()

    ts_bins = np.sort(ts["center_sec"].unique())
    if isc_mode == "overall":
        log.info("ISC 模式: overall（全片时间轴，不使用事件标签）")
        concat_keys = build_concat_keys_overall(ts_bins=ts_bins)
        event_types_for_isc = ("overall",)
    else:
        log.info("ISC 模式: segmented（mental/pain/neutral 事件片段）")
        events = pd.read_csv(events_path)
        _required_cols(events, {"onset_sec", "duration_sec", "event_type"}, "events")
        events = events.copy()
        events["event_type"] = events["event_type"].astype(str).str.strip().str.lower()
        events = events[events["event_type"].isin(EVENT_TYPES_BASE)].reset_index(drop=True)
        concat_keys = build_concat_keys(ts_bins=ts_bins, events=events)
        event_types_for_isc = EVENT_TYPES_ALL

    if concat_keys.empty:
        raise RuntimeError("无可用时间 bin，请检查事件时间或动态序列")
    subject_concat = build_subject_concat_series(ts=ts, keys_df=concat_keys)

    isc_df = compute_isc(
        subject_concat=subject_concat,
        min_overlap_points=args.min_overlap_points,
        event_types=event_types_for_isc,
    )

    stats_rows = []
    for ev in event_types_for_isc:
        sub = isc_df[(isc_df["event_type"] == ev)].copy()
        asd = sub.loc[sub["group"] == "ASD", "isc_z"].dropna().to_numpy()
        td = sub.loc[sub["group"] == "TD", "isc_z"].dropna().to_numpy()
        if len(asd) < 2 or len(td) < 2:
            t_stat, p_val = np.nan, np.nan
        else:
            res = ttest_ind(asd, td, equal_var=False, nan_policy="omit")
            t_stat, p_val = float(res.statistic), float(res.pvalue)
        stats_rows.append({
            "event_type": ev,
            "n_asd": int(len(asd)),
            "n_td": int(len(td)),
            "asd_mean_z": float(np.nanmean(asd)) if len(asd) else np.nan,
            "td_mean_z": float(np.nanmean(td)) if len(td) else np.nan,
            "asd_mean_r": float(np.tanh(np.nanmean(asd))) if len(asd) else np.nan,
            "td_mean_r": float(np.tanh(np.nanmean(td))) if len(td) else np.nan,
            "t_stat": t_stat,
            "p_value": p_val,
        })
    stats_df = pd.DataFrame(stats_rows)

    stats_dir = deriv / "stats"
    fig_dir = out_root / "figures"
    stats_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if isc_mode == "overall":
        save_csv(concat_keys, stats_dir / "movie_isc_concat_keys_overall.csv")
        save_csv(subject_concat, stats_dir / "movie_isc_subject_concat_timeseries_overall.csv")
        save_csv(isc_df, stats_dir / "movie_isc_subject_values_overall.csv")
        save_csv(stats_df, stats_dir / "movie_isc_group_stats_overall.csv")
        # 供 71_resting_to_movie_coupling_overall_isc 等脚本读取
        save_csv(isc_df, stats_dir / "movie_isc_subject_values_with_neutral.csv")
        save_csv(stats_df, stats_dir / "movie_isc_group_stats_with_neutral.csv")
        fig_stem = "movie_isc_group_boxplot_overall"
        plot_title = "TD-template ISC (full movie timeline, no event labels)"
        x_label = "Condition"
    else:
        save_csv(concat_keys, stats_dir / "movie_isc_concat_keys_with_neutral.csv")
        save_csv(subject_concat, stats_dir / "movie_isc_subject_concat_timeseries_with_neutral.csv")
        save_csv(isc_df, stats_dir / "movie_isc_subject_values_with_neutral.csv")
        save_csv(stats_df, stats_dir / "movie_isc_group_stats_with_neutral.csv")
        isc_base_df = isc_df[isc_df["event_type"].isin(EVENT_TYPES_BASE)].copy()
        stats_base_df = stats_df[stats_df["event_type"].isin(EVENT_TYPES_BASE)].copy()
        save_csv(isc_base_df, stats_dir / "movie_isc_subject_values.csv")
        save_csv(stats_base_df, stats_dir / "movie_isc_group_stats.csv")
        fig_stem = "movie_isc_group_boxplot"
        plot_title = "TD-template ISC by Group and Event Type"
        x_label = "Event Type"

        family_rows = []
        for _, row in stats_df.iterrows():
            family_rows.append(
                {
                    "event_type": str(row["event_type"]).lower(),
                    "n_asd": row["n_asd"],
                    "n_td": row["n_td"],
                    "t_stat": row["t_stat"],
                    "raw_p": row["p_value"],
                    "source": "movie_isc_group_stats_with_neutral.csv",
                }
            )
        family_df = pd.DataFrame(family_rows)
        if not family_df.empty:
            reject, p_adj = fdr_correction(
                pd.to_numeric(family_df["raw_p"], errors="coerce").to_numpy(dtype=float)
            )
            family_df["fdr_p"] = p_adj
            family_df["significant_fdr"] = reject
        save_csv(family_df, stats_dir / "movie_isc_family_fdr.csv")

    plt.figure(figsize=(9, 5.5))
    plot_df = isc_df.copy()
    sns.boxplot(
        data=plot_df,
        x="event_type",
        y="isc_z",
        hue="group",
        showfliers=False,
        width=0.6,
    )
    sns.stripplot(
        data=plot_df,
        x="event_type",
        y="isc_z",
        hue="group",
        dodge=True,
        alpha=0.55,
        size=3.5,
        linewidth=0,
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], title="Group", frameon=True)
    plt.xlabel(x_label)
    plt.ylabel("ISC (Fisher z)")
    plt.title(plot_title)

    txt_lines = []
    for _, r in stats_df.iterrows():
        p_txt = "NA" if pd.isna(r["p_value"]) else (f"{r['p_value']:.4g}")
        txt_lines.append(f"{r['event_type']}: p={p_txt}")
    plt.gca().text(
        0.98, 0.98, "\n".join(txt_lines),
        transform=plt.gca().transAxes,
        ha="right", va="top", fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#666"},
    )

    plt.tight_layout()
    fig_path = fig_dir / fig_stem
    plt.savefig(fig_path.with_suffix(".png"), dpi=180)
    plt.savefig(fig_path.with_suffix(".pdf"))
    plt.close()

    if isc_mode == "overall":
        log.info("ISC 被试值(overall): %s", stats_dir / "movie_isc_subject_values_overall.csv")
        log.info("ISC 组间统计(overall): %s", stats_dir / "movie_isc_group_stats_overall.csv")
    else:
        log.info("ISC 被试值(含neutral): %s", stats_dir / "movie_isc_subject_values_with_neutral.csv")
        log.info("ISC 组间统计(含neutral): %s", stats_dir / "movie_isc_group_stats_with_neutral.csv")
        log.info("ISC 被试值(mental/pain): %s", stats_dir / "movie_isc_subject_values.csv")
        log.info("ISC 组间统计(mental/pain): %s", stats_dir / "movie_isc_group_stats.csv")
        log.info("ISC family FDR: %s", stats_dir / "movie_isc_family_fdr.csv")
    log.info("ISC 图: %s", fig_path.with_suffix(".png"))


if __name__ == "__main__":
    main()
