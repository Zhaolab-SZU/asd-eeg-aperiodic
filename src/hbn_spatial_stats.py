"""HBN 通道级空间统计：cluster permutation + TFCE + 被试随机效应混合模型。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.stats.cluster_level import _find_clusters
from scipy import sparse

from src.config import PROJECT_ROOT
from src.hbn_external import resolve_hbn_paths
from src.io_utils import ensure_dir, save_csv
from src.stats_utils import model_results_to_row, run_mixedlm

logger = logging.getLogger(__name__)

GROUP_FORMULA = (
    "aperiodic_exponent ~ C(group, Treatment(reference='TD')) "
    "+ age_months + C(sex) + IQ_total + usable_epochs"
)


def build_hbn_adjacency(montage_name: str = "GSN-HydroCel-128") -> tuple[sparse.csr_matrix, list[str], mne.Info]:
    """129 导 HBN 使用 E1–E128，与 GSN-HydroCel-128 对齐。"""
    montage = mne.channels.make_standard_montage(montage_name)
    info = mne.create_info(montage.ch_names, sfreq=250.0, ch_types="eeg")
    info.set_montage(montage)
    adjacency, ch_names = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    return adjacency, list(ch_names), info


def _load_spatial_cohort(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = resolve_hbn_paths(cfg)
    deriv = paths["derivatives_root"]
    ch_path = deriv / "specparam" / "specparam_channel_results_qc.csv"
    if not ch_path.exists():
        raise FileNotFoundError(f"未找到 {ch_path}，请先运行 102_hbn_specparam.py")

    channel_df = pd.read_csv(ch_path)
    if "fit_valid" in channel_df.columns:
        channel_df = channel_df[channel_df["fit_valid"].astype(bool)].copy()

    participants = pd.read_csv(deriv / "participants_analysis.csv")
    sp_qc = deriv / "specparam" / "specparam_qc_summary_subject.csv"
    if sp_qc.exists():
        bad = pd.read_csv(sp_qc).loc[
            lambda d: d["low_quality_subject"] == 1, "subject_id"
        ].astype(str)
        participants = participants[~participants["subject_id"].astype(str).isin(bad)]

    channel_df["subject_id"] = channel_df["subject_id"].astype(str)
    participants["subject_id"] = participants["subject_id"].astype(str)
    merged = channel_df.merge(
        participants[
            ["subject_id", "group", "age_months", "sex", "IQ_total", "usable_epochs"]
        ],
        on=["subject_id", "group"],
        how="inner",
    )
    merged = merged.dropna(
        subset=["aperiodic_exponent", "group", "age_months", "sex", "IQ_total", "usable_epochs"]
    )
    return merged, participants


def _build_design_matrix(subjects: pd.DataFrame) -> tuple[np.ndarray, int, list[str]]:
    """设计矩阵: intercept, ASD(vs TD), age, male, IQ, usable_epochs。"""
    group_asd = (subjects["group"] == "ASD").astype(float).values
    age = subjects["age_months"].astype(float).values
    sex_m = subjects["sex"].astype(str).str.upper().eq("M").astype(float).values
    iq = subjects["IQ_total"].astype(float).values
    epochs = subjects["usable_epochs"].astype(float).values
    X = np.column_stack([
        np.ones(len(subjects)),
        group_asd,
        age,
        sex_m,
        iq,
        epochs,
    ])
    return X, 1, ["intercept", "group_ASD", "age_months", "sex_M", "IQ_total", "usable_epochs"]


def _mass_ols_tstats(Y: np.ndarray, X: np.ndarray, coef_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Y: (n_subjects, n_channels). 返回 t 统计量与 ASD 系数。"""
    n, p = X.shape
    pinv = np.linalg.pinv(X)
    beta = pinv @ Y
    resid = Y - X @ beta
    df = max(n - p, 1)
    mse = np.sum(resid ** 2, axis=0) / df
    var_b = np.sum(pinv[coef_idx, :] ** 2)
    se = np.sqrt(mse * var_b)
    coef = beta[coef_idx, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = coef / se
    t[~np.isfinite(t)] = 0.0
    return t, coef


def _prepare_wide_matrix(
    merged: pd.DataFrame,
    channel_order: list[str],
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    subjects = (
        merged.groupby("subject_id", as_index=False)
        .first()[["subject_id", "group", "age_months", "sex", "IQ_total", "usable_epochs"]]
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    wide = merged.pivot_table(
        index="subject_id",
        columns="channel",
        values="aperiodic_exponent",
        aggfunc="first",
    )
    subjects = subjects[subjects["subject_id"].isin(wide.index)].reset_index(drop=True)
    wide = wide.loc[subjects["subject_id"]]

    present = [ch for ch in channel_order if ch in wide.columns]
    Y = wide[present].to_numpy(dtype=float)
    return Y, subjects, present


def _max_cluster_stat(t_map: np.ndarray, adjacency: sparse.csr_matrix, threshold: float) -> tuple[float, list, np.ndarray]:
    clusters, sums = _find_clusters(
        t_map,
        threshold=threshold,
        tail=0,
        adjacency=adjacency,
        t_power=1,
    )
    max_stat = float(np.max(sums)) if len(sums) else 0.0
    return max_stat, clusters, sums


def _tfce_scores(
    t_map: np.ndarray,
    adjacency: sparse.csr_matrix,
    tfce_cfg: dict[str, float],
) -> np.ndarray:
    threshold = {
        "start": float(tfce_cfg["start"]),
        "step": float(tfce_cfg["step"]),
        "h_power": float(tfce_cfg.get("h_power", 2.0)),
        "e_power": float(tfce_cfg.get("e_power", 0.5)),
    }
    _, scores = _find_clusters(
        t_map,
        threshold=threshold,
        tail=0,
        adjacency=adjacency,
        t_power=1,
    )
    return np.asarray(scores, dtype=float)


def run_cluster_permutation(
    Y: np.ndarray,
    X: np.ndarray,
    group_idx: int,
    adjacency: sparse.csr_matrix,
    *,
    n_permutations: int,
    threshold: float,
    seed: int,
) -> dict[str, Any]:
    t_obs, coef_obs = _mass_ols_tstats(Y, X, group_idx)
    max_obs, clusters_obs, sums_obs = _max_cluster_stat(t_obs, adjacency, threshold)

    rng = np.random.default_rng(seed)
    null_max = np.zeros(n_permutations, dtype=float)
    group_col = X[:, group_idx].copy()
    for i in range(n_permutations):
        Xp = X.copy()
        Xp[:, group_idx] = rng.permutation(group_col)
        t_perm, _ = _mass_ols_tstats(Y, Xp, group_idx)
        null_max[i], _, _ = _max_cluster_stat(t_perm, adjacency, threshold)

    cluster_rows: list[dict[str, Any]] = []
    for cl_idx, (cluster, stat) in enumerate(zip(clusters_obs, sums_obs)):
        if isinstance(cluster, slice):
            ch_idx = np.arange(cluster.start, cluster.stop)
        else:
            ch_idx = np.where(cluster)[0]
        p_cluster = float((null_max >= stat).mean())
        cluster_rows.append({
            "cluster_id": cl_idx + 1,
            "cluster_stat": float(stat),
            "p_cluster": p_cluster,
            "n_channels": int(len(ch_idx)),
            "channel_indices": ";".join(map(str, ch_idx.tolist())),
        })

    return {
        "t_obs": t_obs,
        "coef_obs": coef_obs,
        "clusters": clusters_obs,
        "cluster_sums": sums_obs,
        "cluster_table": pd.DataFrame(cluster_rows),
        "max_cluster_stat_obs": max_obs,
        "null_max_cluster": null_max,
        "p_cluster_global": float((null_max >= max_obs).mean()) if max_obs > 0 else np.nan,
    }


def run_tfce_permutation(
    Y: np.ndarray,
    X: np.ndarray,
    group_idx: int,
    adjacency: sparse.csr_matrix,
    tfce_cfg: dict[str, float],
    *,
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    t_obs, _ = _mass_ols_tstats(Y, X, group_idx)
    tfce_obs = _tfce_scores(t_obs, adjacency, tfce_cfg)

    rng = np.random.default_rng(seed + 1)
    group_col = X[:, group_idx].copy()
    null_max = np.zeros(n_permutations, dtype=float)
    for i in range(n_permutations):
        Xp = X.copy()
        Xp[:, group_idx] = rng.permutation(group_col)
        t_perm, _ = _mass_ols_tstats(Y, Xp, group_idx)
        tfce_perm = _tfce_scores(t_perm, adjacency, tfce_cfg)
        null_max[i] = float(np.max(tfce_perm))

    p_map = np.array([(null_max >= s).mean() for s in tfce_obs], dtype=float)
    return {
        "t_obs": t_obs,
        "tfce_obs": tfce_obs,
        "tfce_p_map": p_map,
        "null_max_tfce": null_max,
        "p_tfce_global": float((null_max >= tfce_obs.max()).mean()) if tfce_obs.max() > 0 else np.nan,
    }


def run_spatial_mixed_model(long_df: pd.DataFrame) -> pd.DataFrame:
    """长格式混合模型：被试随机截距，检验整体 group 效应（非空间定位）。"""
    fit = run_mixedlm(GROUP_FORMULA, long_df, groups="subject_id")
    return pd.DataFrame(model_results_to_row(fit, "hbn_spatial_mixedlm", "aperiodic_exponent"))


def _cluster_channel_names(
    clusters: list,
    channels: list[str],
) -> list[str]:
    names: list[str] = []
    for cluster in clusters:
        if isinstance(cluster, slice):
            idx = list(range(cluster.start, cluster.stop))
        else:
            idx = np.where(cluster)[0].tolist()
        names.append(";".join(channels[i] for i in idx if i < len(channels)))
    return names


def _attach_montage_coords(
    channel_table: pd.DataFrame,
    info: mne.Info,
) -> pd.DataFrame:
    pos = np.array([info.get_montage().get_positions()["ch_pos"][ch] for ch in channel_table["channel"]])
    out = channel_table.copy()
    out["x"] = pos[:, 0]
    out["y"] = pos[:, 1]
    out["z"] = pos[:, 2]
    return out


def plot_spatial_maps(
    channels: list[str],
    info: mne.Info,
    t_obs: np.ndarray,
    tfce_obs: np.ndarray | None,
    tfce_p: np.ndarray | None,
    cluster_table: pd.DataFrame,
    clusters: list,
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)
    ch_pos = [info.get_montage().get_positions()["ch_pos"][ch] for ch in channels]
    pos_arr = np.array(ch_pos)[:, :2]

    fig, axes = plt.subplots(1, 2 if tfce_obs is not None else 1, figsize=(10, 4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    mne.viz.plot_topomap(t_obs, pos_arr, names=channels, axes=axes[0], show=False, contours=0)
    axes[0].set_title("Group t-map (ASD vs TD, adjusted)")

    if tfce_obs is not None:
        display = tfce_obs.copy()
        if tfce_p is not None:
            display[tfce_p > 0.05] = 0.0
        mne.viz.plot_topomap(display, pos_arr, names=channels, axes=axes[1], show=False, contours=0)
        axes[1].set_title("TFCE (p<0.05 masked)")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_spatial_group_t_tfce.png", dpi=150)
    plt.close(fig)

    if len(cluster_table):
        sig = cluster_table[cluster_table["p_cluster"] < 0.05].copy()
        if len(sig):
            cl_names = _cluster_channel_names(clusters, channels)
            sig = sig.copy()
            sig["channels"] = [cl_names[i] for i in sig["cluster_id"] - 1]
            sig.to_csv(out_dir / "significant_clusters.csv", index=False)


def run_hbn_spatial_stats(cfg: dict[str, Any]) -> dict[str, Path]:
    """主入口：cluster permutation + TFCE + mixed model。"""
    paths = resolve_hbn_paths(cfg)
    deriv = paths["derivatives_root"]
    out_root = paths["outputs_root"]
    stats_dir = deriv / "stats" / "spatial"
    fig_dir = out_root / "figures"
    ensure_dir(stats_dir)
    ensure_dir(fig_dir)

    sp_cfg = cfg.get("spatial_stats", {})
    montage_name = sp_cfg.get("montage", "GSN-HydroCel-128")
    n_perm = int(sp_cfg.get("n_permutations", 5000))
    threshold = float(sp_cfg.get("cluster_forming_threshold", 2.0))
    seed = int(cfg.get("project", {}).get("random_seed", 42))
    tfce_cfg = sp_cfg.get("tfce", {"enabled": True, "start": 2.0, "step": 0.2})

    adjacency, ch_order, info = build_hbn_adjacency(montage_name)
    merged, _ = _load_spatial_cohort(cfg)
    Y, subjects, channels = _prepare_wide_matrix(merged, ch_order)
    ch_idx = [ch_order.index(ch) for ch in channels]
    if sparse.issparse(adjacency):
        adjacency = adjacency.tocsr()[ch_idx][:, ch_idx].tocoo()
    else:
        adjacency = sparse.coo_matrix(adjacency[ch_idx][:, ch_idx])
    info = mne.pick_info(info, sel=ch_idx, copy=True)
    X, group_idx, pred_names = _build_design_matrix(subjects)

    if len(subjects) < 10:
        raise RuntimeError(f"空间统计样本过小: n={len(subjects)}")

    logger.info("空间统计: n=%d, channels=%d, permutations=%d", len(subjects), len(channels), n_perm)

    cluster_res = run_cluster_permutation(
        Y, X, group_idx, adjacency,
        n_permutations=n_perm,
        threshold=threshold,
        seed=seed,
    )
    cluster_table = cluster_res["cluster_table"].copy()
    if len(cluster_table):
        cl_names = _cluster_channel_names(cluster_res["clusters"], channels)
        cluster_table["channels"] = [cl_names[i] for i in cluster_table["cluster_id"] - 1]

    channel_stats = pd.DataFrame({
        "channel": channels,
        "group_coef_ASD_vs_TD": cluster_res["coef_obs"],
        "t_obs": cluster_res["t_obs"],
    })
    channel_stats = _attach_montage_coords(channel_stats, info)
    save_csv(channel_stats, stats_dir / "channel_mass_univariate.csv")
    save_csv(cluster_table, stats_dir / "cluster_permutation.csv")

    summary_rows = [{
        "method": "cluster_permutation",
        "n_subjects": len(subjects),
        "n_channels": len(channels),
        "cluster_threshold_t": threshold,
        "n_permutations": n_perm,
        "max_cluster_stat": cluster_res["max_cluster_stat_obs"],
        "p_cluster_global": cluster_res["p_cluster_global"],
        "n_significant_clusters": int((cluster_table["p_cluster"] < 0.05).sum()) if len(cluster_table) else 0,
    }]

    tfce_obs = None
    tfce_p = None
    if tfce_cfg.get("enabled", True):
        tfce_res = run_tfce_permutation(
            Y, X, group_idx, adjacency, tfce_cfg,
            n_permutations=n_perm,
            seed=seed,
        )
        tfce_obs = tfce_res["tfce_obs"]
        tfce_p = tfce_res["tfce_p_map"]
        tfce_df = pd.DataFrame({
            "channel": channels,
            "tfce": tfce_obs,
            "p_tfce": tfce_p,
            "significant_p05": tfce_p < 0.05,
        })
        tfce_df = _attach_montage_coords(tfce_df, info)
        save_csv(tfce_df, stats_dir / "tfce_channel.csv")
        summary_rows.append({
            "method": "tfce",
            "n_subjects": len(subjects),
            "n_channels": len(channels),
            "cluster_threshold_t": tfce_cfg.get("start"),
            "n_permutations": n_perm,
            "max_cluster_stat": float(tfce_obs.max()),
            "p_cluster_global": tfce_res["p_tfce_global"],
            "n_significant_clusters": int((tfce_p < 0.05).sum()),
        })

    if sp_cfg.get("mixed_model", True):
        long_df = merged[merged["channel"].isin(channels)].copy()
        long_df["subject_id"] = long_df["subject_id"].astype(str)
        mm = run_spatial_mixed_model(long_df)
        save_csv(mm, stats_dir / "spatial_mixed_model.csv")
        group_row = mm[mm["term"].str.contains("group", case=False, na=False)]
        if len(group_row):
            summary_rows.append({
                "method": "spatial_mixed_model",
                "n_subjects": len(subjects),
                "n_channels": len(channels),
                "cluster_threshold_t": np.nan,
                "n_permutations": 0,
                "max_cluster_stat": float(group_row.iloc[0]["coef"]),
                "p_cluster_global": float(group_row.iloc[0]["pvalue"]),
                "n_significant_clusters": np.nan,
            })

    save_csv(pd.DataFrame(summary_rows), stats_dir / "spatial_inference_summary.csv")

    plot_spatial_maps(
        channels, info, cluster_res["t_obs"], tfce_obs, tfce_p,
        cluster_table, cluster_res["clusters"], fig_dir,
    )

    report = _build_spatial_report(
        cfg, subjects, channels, cluster_table, channel_stats, tfce_p, summary_rows,
    )
    report_path = out_root / "spatial_stats_report_zh.md"
    report_path.write_text(report, encoding="utf-8")

    return {
        "channel_stats": stats_dir / "channel_mass_univariate.csv",
        "cluster_table": stats_dir / "cluster_permutation.csv",
        "report": report_path,
        "figure": fig_dir / "fig_spatial_group_t_tfce.png",
    }


def _build_spatial_report(
    cfg: dict[str, Any],
    subjects: pd.DataFrame,
    channels: list[str],
    cluster_table: pd.DataFrame,
    channel_stats: pd.DataFrame,
    tfce_p: np.ndarray | None,
    summary_rows: list[dict[str, Any]],
) -> str:
    n_asd = int((subjects["group"] == cfg["groups"]["asd_label"]).sum())
    n_td = int((subjects["group"] == cfg["groups"]["td_label"]).sum())
    lines = [
        "# HBN 空间统计报告（无预设 ROI）",
        "",
        "## 方法",
        "",
        "- 通道级 mass-univariate OLS：`aperiodic_exponent ~ group + age + sex + IQ + usable_epochs`",
        "- **Cluster permutation**：MNE 电极邻接（Delaunay），被试级 label 置换",
        "- **TFCE**：MNE threshold-free cluster enhancement",
        "- **Mixed model**：长格式 `(1|subject_id)` 随机截距，整体 group 效应",
        "",
        f"- 被试: **{len(subjects)}** (ASD={n_asd}, TD={n_td})",
        f"- 通道: **{len(channels)}**",
        "",
        "## 全局推断",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"- **{row['method']}**: stat={row['max_cluster_stat']:.4f}, "
            f"p={row['p_cluster_global']:.4f}"
        )

    sig_clusters = cluster_table[cluster_table["p_cluster"] < 0.05] if len(cluster_table) else cluster_table
    lines.extend(["", "## 显著 cluster（p_cluster < 0.05）", ""])
    if len(sig_clusters) == 0:
        lines.append("无 cluster 达到 p < 0.05。")
    else:
        for _, r in sig_clusters.iterrows():
            lines.append(
                f"- Cluster {int(r['cluster_id'])}: p={r['p_cluster']:.4f}, "
                f"n_ch={int(r['n_channels'])}, channels={r.get('channels', '')}"
            )

    if tfce_p is not None:
        n_sig = int((tfce_p < 0.05).sum())
        lines.extend(["", "## TFCE 通道", "", f"- p < 0.05 通道数: **{n_sig}**"])
        if n_sig:
            ch_df = channel_stats.copy()
            ch_df["p_tfce"] = tfce_p
            top = ch_df.nsmallest(min(10, n_sig), "p_tfce")
            for _, r in top.iterrows():
                lines.append(
                    f"  - {r['channel']}: TFCE p={r['p_tfce']:.4f}, "
                    f"t={r['t_obs']:.3f}, y={r['y']:.3f} (y↓=枕部)"
                )

    lines.extend([
        "",
        "## 解读",
        "",
        "- 不再依赖手工 occipital_posterior ROI；显著区域由数据驱动定位。",
        "- 若 cluster 落在 y 较负（枕部）且 TD exponent 更高，则与主研究 posterior 方向一致。",
        "- 混合模型 p 值反映整体 group 差异（跨通道平均），不给出空间位置。",
        "",
    ])
    return "\n".join(lines)
