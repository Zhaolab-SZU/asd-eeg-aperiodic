"""HBN 静息 EO：非周期 exponent 被试间相关（ISC），对齐主研究 movie TD-template 流程。"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from scipy.stats import spearmanr, ttest_ind

from src.config import PROJECT_ROOT, load_roi_config
from src.hbn_confirmatory_replication import _load_participants_qc
from src.hbn_dimensional_replication import build_dimensional_table, build_enriched_cohort
from src.hbn_main_matched_cohort import (
    apply_covariate_match_filters,
    greedy_match_asd_td,
)
from src.hbn_external import resolve_hbn_paths
from src.io_utils import ensure_dir, save_csv
from src.roi_utils import get_roi_dict
from src.specparam_utils import fit_specparam_channel

logger = logging.getLogger(__name__)

WINDOW_SEC = 2.0
STEP_SEC = 0.5
MIN_OVERLAP_POINTS = 10


def _homologous_channels(cfg: dict[str, Any]) -> tuple[str, ...]:
    roi_path = PROJECT_ROOT / cfg.get("hbn", {}).get("roi_config", "config/roi_channels_hbn129.yaml")
    roi_dict = get_roi_dict(load_roi_config(roi_path), cfg.get("hbn", {}).get("roi_layout", "channels_hbn129"))
    hom = roi_dict.get("homologous_four", ("E67", "E72", "E75", "E77"))
    return tuple(str(c) for c in hom)


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


def _welch_segment(
    sig: np.ndarray,
    sfreq: float,
    welch_cfg: dict[str, Any],
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_fft = int(welch_cfg.get("n_fft", 500))
    n_overlap = int(welch_cfg.get("n_overlap", 250))
    n_fft = min(n_fft, len(sig))
    if n_fft < 16:
        return np.array([]), np.array([])
    freqs, psd = signal.welch(
        sig,
        fs=sfreq,
        nperseg=n_fft,
        noverlap=min(n_overlap, max(0, n_fft - 1)),
        window=welch_cfg.get("window", "hamming"),
    )
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], psd[mask]


def _channel_indices(epochs: mne.BaseEpochs, homologous: tuple[str, ...], scope: str) -> list[int]:
    ch_names = list(epochs.ch_names)
    if scope == "global":
        return list(range(len(ch_names)))
    return [ch_names.index(c) for c in homologous if c in ch_names]


def build_sliding_exponent_timeseries(
    epochs_path: Path,
    cfg: dict[str, Any],
    scope: str = "posterior",
    homologous: tuple[str, ...] | None = None,
    window_sec: float = WINDOW_SEC,
    step_sec: float = STEP_SEC,
) -> pd.DataFrame:
    """拼接 epoch 后做滑窗 specparam；scope=posterior（ROI 均值）或 global（全通道均值）。"""
    homologous = homologous or ()
    epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
    try:
        idx = _channel_indices(epochs, homologous, scope)
        if not idx:
            return pd.DataFrame()

        psd_cfg = cfg.get("psd", {})
        welch = psd_cfg.get("welch", {})
        fmin = float(welch.get("fmin", psd_cfg.get("freq_min_hz", 1.0)))
        fmax = float(welch.get("fmax", psd_cfg.get("freq_max_hz", 40.0)))
        sp_cfg = cfg.get("specparam", {})
        sfreq = float(epochs.info["sfreq"])

        data = epochs.get_data().astype(np.float32, copy=False)[:, idx, :]
        combined = data.mean(axis=1).reshape(-1)
        win_n = max(1, int(round(window_sec * sfreq)))
        step_n = max(1, int(round(step_sec * sfreq)))

        rows: list[dict[str, Any]] = []
        start = 0
        widx = 0
        while start + win_n <= len(combined):
            seg = combined[start : start + win_n]
            freqs, power = _welch_segment(seg, sfreq, welch, fmin, fmax)
            if len(freqs) < 5:
                exp = np.nan
            else:
                try:
                    fit = fit_specparam_channel(freqs, power, sp_cfg)
                    exp = fit["aperiodic_exponent"]
                except Exception:
                    exp = np.nan
            rows.append(
                {
                    "window_index": widx,
                    "window_start_sec": start / sfreq,
                    "window_end_sec": (start + win_n) / sfreq,
                    "center_sec": (start + win_n / 2.0) / sfreq,
                    "exponent_mean": exp,
                }
            )
            widx += 1
            start += step_n
        return pd.DataFrame(rows)
    finally:
        del epochs
        gc.collect()


def build_sliding_posterior_timeseries(
    epochs_path: Path,
    homologous: tuple[str, ...],
    cfg: dict[str, Any],
    window_sec: float = WINDOW_SEC,
    step_sec: float = STEP_SEC,
) -> pd.DataFrame:
    """拼接 EO epoch 后做 2s/0.5s 滑窗 specparam（posterior ROI mean）。"""
    return build_sliding_exponent_timeseries(
        epochs_path, cfg, scope="posterior", homologous=homologous,
        window_sec=window_sec, step_sec=step_sec,
    )


def _fit_epoch_exponent_series(
    epochs_path: Path,
    cfg: dict[str, Any],
    scope: str = "posterior",
    homologous: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """逐 epoch specparam；全通道或 ROI 内通道均值。"""
    from src.psd_utils import compute_psd_matrix_from_epochs

    homologous = homologous or ()
    epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
    try:
        psd_cfg = cfg.get("psd", {})
        welch = psd_cfg.get("welch", {})
        fmin = float(welch.get("fmin", psd_cfg.get("freq_min_hz", 1.0)))
        fmax = float(welch.get("fmax", psd_cfg.get("freq_max_hz", 40.0)))
        freqs, psd_mat, ch_names = compute_psd_matrix_from_epochs(epochs, fmin, fmax, welch)
        idx = _channel_indices(epochs, homologous, scope)
        if not idx:
            return pd.DataFrame()

        sp_cfg = cfg.get("specparam", {})
        col = "global_exponent" if scope == "global" else "posterior_exponent"
        rows = []
        for ei in range(psd_mat.shape[0]):
            power = psd_mat[ei, idx, :].mean(axis=0)
            try:
                fit = fit_specparam_channel(freqs, power, sp_cfg)
                exp = fit["aperiodic_exponent"]
            except Exception:
                exp = np.nan
            rows.append({"epoch_index": ei, col: exp})
        return pd.DataFrame(rows)
    finally:
        del epochs
        gc.collect()


def build_epoch_exponent_timeseries(cache_path: Path, scope: str = "posterior") -> pd.DataFrame:
    """读取 epoch-wise exponent 缓存。"""
    ts = pd.read_csv(cache_path)
    col = "global_exponent" if scope == "global" else "posterior_exponent"
    if col not in ts.columns:
        return pd.DataFrame()
    out = ts.copy()
    if "epoch_index" in out.columns:
        dur = 2.0
        out["center_sec"] = (out["epoch_index"].astype(float) + 0.5) * dur
        out["window_start_sec"] = out["epoch_index"].astype(float) * dur
        out["window_end_sec"] = out["window_start_sec"] + dur
    out = out.rename(columns={col: "exponent_mean"})
    return out[["center_sec", "window_start_sec", "window_end_sec", "exponent_mean"]]


def build_epoch_posterior_timeseries(cache_path: Path) -> pd.DataFrame:
    """复用 112 缓存的 epoch-wise posterior exponent（2s 非重叠）。"""
    return build_epoch_exponent_timeseries(cache_path, scope="posterior")


def _collect_timeseries(
    cfg: dict[str, Any],
    cohort_df: pd.DataFrame,
    mode: str,
    homologous: tuple[str, ...],
    sliding_require_cache: bool = False,
    epoch_require_cache: bool = False,
    scope: str = "posterior",
) -> pd.DataFrame:
    paths = resolve_hbn_paths(cfg)
    epoch_dir = paths["derivatives_root"] / "epochs"
    cache_dir = paths["derivatives_root"] / "isc"
    temporal_dir = paths["derivatives_root"] / "temporal"
    ensure_dir(cache_dir)

    scope_tag = "global" if scope == "global" else "posterior"
    rows: list[pd.DataFrame] = []
    for i, (_, row) in enumerate(cohort_df.iterrows()):
        sid = str(row["subject_id"])
        grp = str(row["group"]).upper()
        if mode == "sliding":
            cache = cache_dir / f"{sid}_sliding_{scope_tag}.csv"
            if cache.exists():
                ts = pd.read_csv(cache)
            elif sliding_require_cache:
                continue
            else:
                epo = epoch_dir / f"{sid}-epo.fif"
                if not epo.exists():
                    continue
                try:
                    ts = build_sliding_exponent_timeseries(
                        epo, cfg, scope=scope, homologous=homologous,
                    )
                except MemoryError:
                    logger.warning("内存不足，跳过 %s sliding (%s)", sid, scope_tag)
                    continue
                if ts.empty:
                    continue
                ts.to_csv(cache, index=False)
        else:
            cache = temporal_dir / f"{sid}_epoch_{scope_tag}.csv"
            if cache.exists():
                ts = build_epoch_exponent_timeseries(cache, scope=scope)
            elif epoch_require_cache:
                continue
            else:
                epo = epoch_dir / f"{sid}-epo.fif"
                if not epo.exists():
                    continue
                try:
                    ts_raw = _fit_epoch_exponent_series(
                        epo, cfg, scope=scope, homologous=homologous,
                    )
                except MemoryError:
                    logger.warning("内存不足，跳过 %s epoch (%s)", sid, scope_tag)
                    continue
                if ts_raw.empty:
                    continue
                ensure_dir(temporal_dir)
                ts_raw.to_csv(cache, index=False)
                ts = build_epoch_exponent_timeseries(cache, scope=scope)
        ts = ts.copy()
        ts["subject_id"] = sid
        ts["group"] = grp
        rows.append(ts)
        if (i + 1) % 50 == 0:
            logger.info("  timeseries progress (%s): %d / %d", scope_tag, i + 1, len(cohort_df))
            gc.collect()
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _align_timeseries_grid(ts: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按公共时间网格对齐（等同主研究 concat_keys）。"""
    if mode == "sliding":
        grid = np.sort(ts["center_sec"].round(6).unique())
    else:
        # epoch 模式：按相对窗口序号对齐，避免各被试绝对 center_sec 微差
        ts = ts.copy()
        ts["concat_index"] = ts.groupby("subject_id").cumcount()
        grid = np.arange(int(ts["concat_index"].max()) + 1)
        keys = pd.DataFrame({"concat_index": grid, "event_type": "overall"})
        aligned_rows: list[pd.DataFrame] = []
        for (sid, grp), sub in ts.groupby(["subject_id", "group"], sort=False):
            sub = sub.sort_values("concat_index")
            merged = keys.merge(
                sub[["concat_index", "exponent_mean", "center_sec"]],
                on="concat_index",
                how="left",
            )
            merged["subject_id"] = sid
            merged["group"] = grp
            aligned_rows.append(merged)
        return pd.concat(aligned_rows, ignore_index=True), keys

    keys = pd.DataFrame(
        {
            "center_sec": grid.astype(float),
            "concat_index": np.arange(len(grid), dtype=int),
            "event_type": "overall",
        }
    )
    aligned_rows: list[pd.DataFrame] = []
    for (sid, grp), sub in ts.groupby(["subject_id", "group"], sort=False):
        sub2 = sub.groupby("center_sec", as_index=False)["exponent_mean"].mean()
        sub2["center_sec"] = sub2["center_sec"].round(6)
        merged = keys.merge(sub2, on="center_sec", how="left")
        merged["subject_id"] = sid
        merged["group"] = grp
        aligned_rows.append(merged)
    return pd.concat(aligned_rows, ignore_index=True), keys


def compute_td_template_isc(
    ts: pd.DataFrame,
    min_overlap_points: int = MIN_OVERLAP_POINTS,
    align_col: str = "concat_index",
) -> pd.DataFrame:
    """TD 留一 / ASD 对 full TD 模板 ISC（Fisher z）。"""
    mat = ts.pivot_table(
        index=["subject_id", "group"],
        columns=align_col,
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
        raise RuntimeError("TD 样本不足（至少需要 2 名用于留一模板）")

    rows: list[dict[str, Any]] = []
    td_subjects = idx_df.loc[td_mask, "subject_id"].tolist()
    for i, sid in enumerate(td_subjects):
        x = td_vals.iloc[i].to_numpy(dtype=float)
        others = td_vals.drop(td_vals.index[i])
        tmpl = others.mean(axis=0, skipna=True).to_numpy(dtype=float)
        r, n_overlap = safe_corr(x, tmpl, min_points=min_overlap_points)
        rows.append(
            {
                "subject_id": sid,
                "group": "TD",
                "isc_r": r,
                "isc_z": fisher_z(r),
                "n_overlap_points": n_overlap,
                "template_type": "TD_LOO",
            }
        )

    td_template = td_vals.mean(axis=0, skipna=True).to_numpy(dtype=float)
    asd_subjects = idx_df.loc[asd_mask, "subject_id"].tolist()
    for i, sid in enumerate(asd_subjects):
        x = asd_vals.iloc[i].to_numpy(dtype=float)
        r, n_overlap = safe_corr(x, td_template, min_points=min_overlap_points)
        rows.append(
            {
                "subject_id": sid,
                "group": "ASD",
                "isc_r": r,
                "isc_z": fisher_z(r),
                "n_overlap_points": n_overlap,
                "template_type": "TD_FULL",
            }
        )
    return pd.DataFrame(rows)


def _group_isc_stats(isc_df: pd.DataFrame) -> dict[str, Any]:
    asd = isc_df.loc[isc_df["group"] == "ASD", "isc_z"].dropna().to_numpy()
    td = isc_df.loc[isc_df["group"] == "TD", "isc_z"].dropna().to_numpy()
    if len(asd) < 2 or len(td) < 2:
        return {
            "n_asd": int(len(asd)),
            "n_td": int(len(td)),
            "asd_mean_z": float(np.nanmean(asd)) if len(asd) else np.nan,
            "td_mean_z": float(np.nanmean(td)) if len(td) else np.nan,
            "asd_mean_r": float(np.tanh(np.nanmean(asd))) if len(asd) else np.nan,
            "td_mean_r": float(np.tanh(np.nanmean(td))) if len(td) else np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "mean_diff_asd_minus_td_z": np.nan,
        }
    res = ttest_ind(asd, td, equal_var=False, nan_policy="omit")
    return {
        "n_asd": int(len(asd)),
        "n_td": int(len(td)),
        "asd_mean_z": float(np.nanmean(asd)),
        "td_mean_z": float(np.nanmean(td)),
        "asd_mean_r": float(np.tanh(np.nanmean(asd))),
        "td_mean_r": float(np.tanh(np.nanmean(td))),
        "t_stat": float(res.statistic),
        "p_value": float(res.pvalue),
        "mean_diff_asd_minus_td_z": float(np.nanmean(asd) - np.nanmean(td)),
    }


def _symptom_correlations(isc_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    merged = isc_df.merge(meta, on="subject_id", how="left")
    asd = merged[merged["group"] == "ASD"].copy()
    rows: list[dict[str, Any]] = []
    for sym in ["SRS_total", "SCQ_total"]:
        sub = asd.dropna(subset=["isc_z", sym])
        if len(sub) < 10:
            continue
        rho, p = spearmanr(sub["isc_z"], sub[sym])
        rows.append(
            {
                "symptom": sym,
                "n_asd": len(sub),
                "spearman_rho": float(rho),
                "p_value": float(p),
            }
        )
    return pd.DataFrame(rows)


def plot_isc_boxplot(isc_df: pd.DataFrame, stats: dict[str, Any], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=isc_df, x="group", y="isc_z", order=["TD", "ASD"], ax=ax, width=0.5, showfliers=False)
    sns.stripplot(
        data=isc_df,
        x="group",
        y="isc_z",
        order=["TD", "ASD"],
        ax=ax,
        color="black",
        alpha=0.45,
        size=3.5,
        jitter=0.15,
    )
    p_txt = "NA" if pd.isna(stats.get("p_value")) else f"{stats['p_value']:.4g}"
    ax.set_title(title)
    ax.set_xlabel("Group")
    ax.set_ylabel("ISC (Fisher z, TD template)")
    ax.text(
        0.98,
        0.98,
        f"ASD n={stats.get('n_asd', 'NA')}, TD n={stats.get('n_td', 'NA')}\np={p_txt}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#666"},
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_isc_report(
    results: dict[str, dict[str, Any]],
    main_ref: dict[str, float],
    *,
    data_label: str = "HBN 静息 **eyes-open**（无 movie 事件标签，故为全段 overall ISC）",
    signal_label: str = "homologous posterior（E67/E72/E75/E77，对应主研究 E33/E36/E37/E38）",
    interpretation_extra: list[str] | None = None,
) -> str:
    lines = [
        "# HBN 非周期 exponent 同步性（ISC）分析报告",
        "",
        "## 方法（对齐主研究 movie ISC）",
        "",
        f"- **信号**：{signal_label}",
        "- **模板**：TD-template（TD 留一法；ASD 对 full TD 模板）",
        "- **数据**：" + data_label,
        f"- **主研究参考**（movie mental/pain）：TD ISC > ASD，mental p≈{main_ref.get('mental_p', 0.022):.3g}，"
        f"pain p≈{main_ref.get('pain_p', 0.0014):.3g}",
        "",
        "## 结果",
        "",
    ]
    for label, block in results.items():
        st = block["stats"]
        lines.append(f"### {label}")
        lines.append(
            f"- N: ASD={st['n_asd']}, TD={st['n_td']} "
            f"(目标队列 {int(st.get('n_cohort_target', st['n_asd']+st['n_td']))}, "
            f"有效 {int(st.get('n_with_timeseries', st['n_asd']+st['n_td']))})"
        )
        lines.append(
            f"- ISC (z): ASD={st['asd_mean_z']:.3f}, TD={st['td_mean_z']:.3f} "
            f"(r≈{st['asd_mean_r']:.3f} vs {st['td_mean_r']:.3f})"
        )
        lines.append(
            f"- Welch t (ASD vs TD): t={st['t_stat']:.3f}, p={st['p_value']:.4g} "
            f"(Δz ASD−TD={st['mean_diff_asd_minus_td_z']:.3f})"
        )
        sym = block.get("symptom_corr")
        if sym is not None and not sym.empty:
            lines.append("- ASD 内症状相关（Spearman）：")
            for _, r in sym.iterrows():
                lines.append(f"  - {r['symptom']}: ρ={r['spearman_rho']:.3f}, p={r['p_value']:.3g}, n={int(r['n_asd'])}")
        lines.append("")

    lines.extend(["## 解读", ""])
    if interpretation_extra:
        lines.extend(interpretation_extra)
    else:
        lines.extend(
            [
                "- **主研究 movie**：mental/pain 片段 ISC 为 **TD > ASD**（ASD 与 TD 模板同步性更低）。",
                "- **HBN 限制**：无 movie 任务 / 事件标签，仅为静息 EO **overall ISC**；滑窗 0.5s 分析目前仅覆盖部分被试（内存约束下复用已缓存序列）。",
                "- **epoch 2s 模式**：方向与主研究一致（TD 略高）但 **p≈.22，不显著**。",
                "- **滑窗 0.5s confirmatory**：出现 **ASD > TD**（p≈.014），与主研究 **方向相反**；考虑 partial N 与静息态无刺激语境，不宜作 confirmatory 复现。",
                "",
            ]
        )
    return "\n".join(lines)


def _load_thepresent_cohort(cfg: dict[str, Any]) -> pd.DataFrame:
    paths = resolve_hbn_paths(cfg)
    pa_path = paths["derivatives_root"] / "participants_analysis.csv"
    if not pa_path.exists():
        raise FileNotFoundError(f"请先运行 101_hbn_preprocess_thepresent.py → {pa_path}")
    df = pd.read_csv(pa_path)
    df["subject_id"] = df["subject_id"].astype(str)
    epoch_dir = paths["derivatives_root"] / "epochs"
    df = df[df["subject_id"].apply(lambda s: (epoch_dir / f"{s}-epo.fif").exists())].copy()
    return df.reset_index(drop=True)


def build_thepresent_matched_cohort(
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ThePresent：硬筛选 + ASD:TD 1:1 年龄/IQ/性别匹配。"""
    cohort = _load_thepresent_cohort(cfg)
    filtered, filter_log = apply_covariate_match_filters(
        cohort,
        cfg,
        cfg_key="main_matched",
        strict_td=True,
        extra_required=None,
    )
    rep_cfg = cfg.get("hbn", {}).get("main_matched", {})
    matched, match_table = greedy_match_asd_td(
        filtered,
        caliper_age=float(rep_cfg.get("match_caliper_age_months", 12)),
        caliper_iq=float(rep_cfg.get("match_caliper_iq", 15)),
        require_same_sex=bool(rep_cfg.get("require_same_sex", True)),
        seed=int(cfg.get("project", {}).get("random_seed", 42)),
    )
    return filtered, matched, filter_log, match_table


def _run_thepresent_isc_cohort_blocks(
    cfg: dict[str, Any],
    cohort_specs: list[tuple[str, pd.DataFrame, str]],
    *,
    rep_dir: Path,
    fig_dir: Path,
    tbl_dir: Path,
    summary_path: Path,
    report_path: Path,
    report_title: str,
    data_label: str,
    interpretation_extra: list[str],
    scope: str = "posterior",
    signal_label: str | None = None,
) -> dict[str, dict[str, Any]]:
    homologous = _homologous_channels(cfg)
    scope_label = "global 全通道均值" if scope == "global" else "posterior ROI"
    if signal_label is None:
        signal_label = (
            "全通道均值 exponent（128 导 EEG 波形先平均，再 Welch + specparam；无 ROI）"
            if scope == "global"
            else "homologous posterior（E67/E72/E75/E77，对应主研究 E33/E36/E37/E38）"
        )
    results: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []

    for label, sub_cohort, mode in cohort_specs:
        if sub_cohort.empty:
            continue
        logger.info("ThePresent ISC [%s] mode=%s scope=%s n=%d", label, mode, scope, len(sub_cohort))
        ts = _collect_timeseries(
            cfg, sub_cohort, mode=mode, homologous=homologous, scope=scope,
        )
        if ts.empty:
            logger.warning("跳过 %s: 无时间序列", label)
            continue
        ts_aligned, keys = _align_timeseries_grid(ts, mode=mode)
        save_csv(keys, rep_dir / f"concat_keys_{label}.csv")
        save_csv(ts_aligned, rep_dir / f"timeseries_{label}.csv")

        isc_df = compute_td_template_isc(ts_aligned, min_overlap_points=MIN_OVERLAP_POINTS)
        save_csv(isc_df, rep_dir / f"isc_subject_values_{label}.csv")

        stats = _group_isc_stats(isc_df)
        stats["cohort"] = label
        stats["isc_mode"] = mode
        stats["channel_scope"] = scope
        stats["n_cohort_target"] = len(sub_cohort)
        stats["n_with_timeseries"] = int(isc_df["subject_id"].nunique())
        summary_rows.append(stats)

        meta_cols = [c for c in ["SRS_total", "SCQ_total", "age_months", "IQ_total"] if c in sub_cohort.columns]
        sym_corr = _symptom_correlations(isc_df, sub_cohort[["subject_id"] + meta_cols])
        if not sym_corr.empty:
            sym_corr["cohort"] = label
            save_csv(sym_corr, rep_dir / f"isc_symptom_corr_{label}.csv")

        plot_isc_boxplot(
            isc_df,
            stats,
            fig_dir / f"fig_isc_{label}.png",
            title=f"HBN ThePresent ISC ({scope_label}, {label})",
        )
        results[label] = {"stats": stats, "symptom_corr": sym_corr}

    summary_df = pd.DataFrame(summary_rows)
    save_csv(summary_df, summary_path)

    main_ref = {"mental_p": 0.0223, "pain_p": 0.00139}
    report = build_isc_report(
        results,
        main_ref,
        data_label=data_label,
        signal_label=signal_label,
        interpretation_extra=interpretation_extra,
    )
    report_path.write_text(
        report.replace(
            "# HBN 非周期 exponent 同步性（ISC）分析报告",
            report_title,
            1,
        ),
        encoding="utf-8",
    )
    return results


def run_hbn_thepresent_aperiodic_isc(
    cfg: dict[str, Any],
    limit: int | None = None,
) -> dict[str, Path]:
    """HBN ThePresent 自然观看：overall aperiodic ISC（对齐主研究 TD-template）。"""
    paths = resolve_hbn_paths(cfg)
    rep_dir = paths["derivatives_root"] / "replication" / "isc"
    fig_dir = paths["outputs_root"] / "figures"
    tbl_dir = paths["outputs_root"] / "tables"
    ensure_dir(rep_dir)
    ensure_dir(fig_dir)
    ensure_dir(tbl_dir)

    cohort = _load_thepresent_cohort(cfg)
    if limit:
        cohort = cohort.head(limit)

    _run_thepresent_isc_cohort_blocks(
        cfg,
        [
            ("thepresent_sliding_0p5s", cohort, "sliding"),
            ("thepresent_epoch_2s", cohort, "epoch"),
        ],
        rep_dir=rep_dir,
        fig_dir=fig_dir,
        tbl_dir=tbl_dir,
        summary_path=tbl_dir / "isc_group_stats.csv",
        report_path=paths["outputs_root"] / "isc_report_zh.md",
        report_title="# HBN ThePresent 非周期 exponent 同步性（ISC）分析报告",
        data_label="HBN **ThePresent** 自然观看（~200 s，无 mental/pain 分段 → **overall ISC**）",
        interpretation_extra=[
            "- **主研究 movie**：mental/pain 分段 ISC 为 **TD > ASD**；HBN ThePresent 为不同影片，仅可比 **全片 overall** 模式。",
            "- **无事件标签**：不能复现 mental/pain/neutral 分段 ISC。",
            "- 若 overall 模式仍不显著或方向与主研究不一致，不宜作为外部 confirmatory 证据。",
            "",
        ],
    )

    return {
        "summary": tbl_dir / "isc_group_stats.csv",
        "report": paths["outputs_root"] / "isc_report_zh.md",
        "replication_dir": rep_dir,
    }


def run_hbn_thepresent_matched_aperiodic_isc(cfg: dict[str, Any]) -> dict[str, Path]:
    """ThePresent：年龄/性别/IQ 1:1 匹配队列上的 overall Aperiodic ISC。"""
    paths = resolve_hbn_paths(cfg)
    rep_dir = paths["derivatives_root"] / "replication" / "isc_matched"
    match_dir = paths["derivatives_root"] / "replication" / "matched"
    fig_dir = paths["outputs_root"] / "figures"
    tbl_dir = paths["outputs_root"] / "tables"
    ensure_dir(rep_dir)
    ensure_dir(match_dir)
    ensure_dir(fig_dir)
    ensure_dir(tbl_dir)

    filtered, matched, filter_log, match_table = build_thepresent_matched_cohort(cfg)
    save_csv(filter_log, match_dir / "filter_log.csv")
    save_csv(match_table, match_dir / "match_table.csv")
    save_csv(matched, match_dir / "participants_matched.csv")

    n_pairs = len(match_table)
    logger.info(
        "匹配队列: 筛选 n=%d → 匹配 %d 对 (ASD/TD 各 %d)",
        len(filtered),
        n_pairs,
        n_pairs,
    )

    rep_cfg = cfg.get("hbn", {}).get("main_matched", {})
    cal_age = rep_cfg.get("match_caliper_age_months", 12)
    cal_iq = rep_cfg.get("match_caliper_iq", 15)
    same_sex = rep_cfg.get("require_same_sex", True)

    _run_thepresent_isc_cohort_blocks(
        cfg,
        [
            ("thepresent_matched_sliding_0p5s", matched, "sliding"),
            ("thepresent_matched_epoch_2s", matched, "epoch"),
        ],
        rep_dir=rep_dir,
        fig_dir=fig_dir,
        tbl_dir=tbl_dir,
        summary_path=tbl_dir / "isc_group_stats_matched.csv",
        report_path=paths["outputs_root"] / "isc_report_matched_zh.md",
        report_title="# HBN ThePresent 匹配队列 Aperiodic ISC 分析报告",
        data_label=(
            f"HBN **ThePresent** 匹配队列（1:1 年龄±{cal_age} mo、IQ±{cal_iq}、"
            f"{'同性别' if same_sex else '不限性别'}；strict TD SCQ/SRS）"
        ),
        interpretation_extra=[
            f"- **匹配**：{n_pairs} 对 ASD:TD（自 `participants_analysis.csv` 硬筛选后贪婪匹配）。",
            "- **模板**：TD-template 仅基于匹配后 TD 子集构建。",
            "- 与全样本 ThePresent ISC 对照，检验组间差异是否受 age/IQ/性别混杂驱动。",
            "",
        ],
    )

    return {
        "summary": tbl_dir / "isc_group_stats_matched.csv",
        "report": paths["outputs_root"] / "isc_report_matched_zh.md",
        "replication_dir": rep_dir,
        "match_dir": match_dir,
        "n_pairs": n_pairs,
    }


def run_hbn_thepresent_matched_global_aperiodic_isc(cfg: dict[str, Any]) -> dict[str, Path]:
    """ThePresent 匹配队列：全通道均值 exponent 的 overall Aperiodic ISC（无 ROI）。"""
    paths = resolve_hbn_paths(cfg)
    rep_dir = paths["derivatives_root"] / "replication" / "isc_matched_global"
    match_dir = paths["derivatives_root"] / "replication" / "matched"
    fig_dir = paths["outputs_root"] / "figures"
    tbl_dir = paths["outputs_root"] / "tables"
    ensure_dir(rep_dir)
    ensure_dir(fig_dir)
    ensure_dir(tbl_dir)

    filtered, matched, filter_log, match_table = build_thepresent_matched_cohort(cfg)
    if not (match_dir / "match_table.csv").exists():
        ensure_dir(match_dir)
        save_csv(filter_log, match_dir / "filter_log.csv")
        save_csv(match_table, match_dir / "match_table.csv")
        save_csv(matched, match_dir / "participants_matched.csv")

    n_pairs = len(match_table)
    logger.info("全脑 global ISC：匹配 %d 对", n_pairs)

    rep_cfg = cfg.get("hbn", {}).get("main_matched", {})
    cal_age = rep_cfg.get("match_caliper_age_months", 12)
    cal_iq = rep_cfg.get("match_caliper_iq", 15)
    same_sex = rep_cfg.get("require_same_sex", True)

    _run_thepresent_isc_cohort_blocks(
        cfg,
        [
            ("thepresent_matched_global_sliding_0p5s", matched, "sliding"),
            ("thepresent_matched_global_epoch_2s", matched, "epoch"),
        ],
        rep_dir=rep_dir,
        fig_dir=fig_dir,
        tbl_dir=tbl_dir,
        summary_path=tbl_dir / "isc_group_stats_matched_global.csv",
        report_path=paths["outputs_root"] / "isc_report_matched_global_zh.md",
        report_title="# HBN ThePresent 匹配队列 全脑 Global Aperiodic ISC 分析报告",
        data_label=(
            f"HBN **ThePresent** 匹配队列（{n_pairs} 对）；**全通道均值 exponent**（128 导，无 ROI）；"
            f"1:1 年龄±{cal_age} mo、IQ±{cal_iq}、{'同性别' if same_sex else '不限性别'}"
        ),
        interpretation_extra=[
            f"- **全脑定义**：每时间窗对 **全部 EEG 通道** 波形先平均，再 Welch + specparam 得 exponent（非逐通道 ISC 再平均）。",
            f"- **匹配**：{n_pairs} 对 ASD:TD；TD-template 基于匹配 TD 子集。",
            "- 与 posterior homologous-four ROI ISC 对照，检验效应是否局限于枕叶。",
            "",
        ],
        scope="global",
    )

    return {
        "summary": tbl_dir / "isc_group_stats_matched_global.csv",
        "report": paths["outputs_root"] / "isc_report_matched_global_zh.md",
        "replication_dir": rep_dir,
        "n_pairs": n_pairs,
    }


def run_hbn_aperiodic_isc(cfg: dict[str, Any], limit: int | None = None) -> dict[str, Path]:
    paths = resolve_hbn_paths(cfg)
    rep_dir = paths["derivatives_root"] / "replication" / "isc"
    fig_dir = paths["outputs_root"] / "figures"
    tbl_dir = paths["outputs_root"] / "tables"
    ensure_dir(rep_dir)
    ensure_dir(fig_dir)
    ensure_dir(tbl_dir)

    homologous = _homologous_channels(cfg)
    confirmatory = _load_participants_qc(cfg)
    dim_table = build_dimensional_table(cfg)
    dimensional, _, _ = build_enriched_cohort(dim_table, cfg)

    cohorts: list[tuple[str, pd.DataFrame, str, bool, bool]] = [
        ("dimensional_epoch_2s", dimensional, "epoch", False, False),
        ("dimensional_sliding_0p5s", dimensional, "sliding", True, False),
        ("confirmatory_sliding_0p5s", confirmatory, "sliding", True, False),
        ("confirmatory_epoch_2s", confirmatory, "epoch", False, True),
    ]

    results: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []

    for label, cohort, mode, slide_cache_only, epoch_cache_only in cohorts:
        if cohort.empty:
            continue
        logger.info("ISC [%s] mode=%s n=%d", label, mode, len(cohort))
        ts = _collect_timeseries(
            cfg, cohort, mode=mode, homologous=homologous,
            sliding_require_cache=slide_cache_only,
            epoch_require_cache=epoch_cache_only,
        )
        if ts.empty:
            logger.warning("跳过 %s: 无时间序列", label)
            continue
        ts_aligned, keys = _align_timeseries_grid(ts, mode=mode)
        save_csv(keys, rep_dir / f"concat_keys_{label}.csv")
        save_csv(ts_aligned, rep_dir / f"timeseries_{label}.csv")

        isc_df = compute_td_template_isc(ts_aligned, min_overlap_points=MIN_OVERLAP_POINTS)
        save_csv(isc_df, rep_dir / f"isc_subject_values_{label}.csv")

        stats = _group_isc_stats(isc_df)
        stats["cohort"] = label
        stats["isc_mode"] = mode
        stats["n_cohort_target"] = len(cohort)
        stats["n_with_timeseries"] = int(isc_df["subject_id"].nunique())
        summary_rows.append(stats)

        meta_cols = [c for c in ["SRS_total", "SCQ_total", "age_months", "IQ_total"] if c in cohort.columns]
        sym_corr = _symptom_correlations(isc_df, cohort[["subject_id"] + meta_cols])
        if not sym_corr.empty:
            sym_corr["cohort"] = label
            save_csv(sym_corr, rep_dir / f"isc_symptom_corr_{label}.csv")

        plot_isc_boxplot(
            isc_df,
            stats,
            fig_dir / f"fig_isc_{label}.png",
            title=f"HBN resting EO ISC ({label})",
        )
        results[label] = {"stats": stats, "symptom_corr": sym_corr}

    summary_df = pd.DataFrame(summary_rows)
    save_csv(summary_df, tbl_dir / "isc_group_stats.csv")

    main_ref = {"mental_p": 0.0223, "pain_p": 0.00139}
    report = build_isc_report(results, main_ref)
    report_path = paths["outputs_root"] / "isc_report_zh.md"
    report_path.write_text(report, encoding="utf-8")

    return {
        "summary": tbl_dir / "isc_group_stats.csv",
        "report": report_path,
        "replication_dir": rep_dir,
    }
