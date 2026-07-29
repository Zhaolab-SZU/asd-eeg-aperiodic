# Ideal release name: 25b_rest_movie_coupling.py
# Original path: scripts/34_resting_to_movie_coupling.py
# Note: Resting-to-movie coupling models
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
34_resting_to_movie_coupling.py
-------------------------------
静息态-电影加工耦合分析：
1) 读取静息态 posterior exponent 与 movie mental ISC
2) 组内相关（Pearson / Spearman）
3) 回归模型：mental_ISC ~ Resting_Posterior_Exponent * Group + Age
4) 画分组散点 + 回归线 + 置信区间
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import save_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resting-to-movie coupling analysis")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_task_movie.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--resting_csv",
        type=str,
        default="outputs/tables/resting_features_locked.csv",
        help="静息态特征CSV（需含 posterior_exponent）",
    )
    parser.add_argument(
        "--movie_isc_csv",
        type=str,
        default="derivatives_task_movie/stats/movie_isc_subject_values.csv",
        help="movie ISC 被试值CSV",
    )
    parser.add_argument(
        "--participants_csv",
        type=str,
        default="data/participants/participants_task_movie.csv",
        help="movie 被试表（用于读取 age_months/IQ/sex 与 included_final）",
    )
    parser.add_argument(
        "--movie_analysis_csv",
        type=str,
        default="derivatives_task_movie/participants_analysis.csv",
        help="movie 分析队列 CSV（统一样本口径）",
    )
    parser.add_argument(
        "--movie_specparam_qc_csv",
        type=str,
        default="derivatives_task_movie/specparam/specparam_qc_summary_subject.csv",
        help="movie 被试级 specparam QC（剔除 low_quality_subject）",
    )
    parser.add_argument(
        "--preproc_summary_csv",
        type=str,
        default="derivatives_task_movie/qc/preproc_summary.csv",
        help="预处理汇总（用于 usable_epochs）",
    )
    parser.add_argument(
        "--n_resamples",
        type=int,
        default=2000,
        help="Bootstrap 重采样次数",
    )
    parser.add_argument(
        "--older_age_cutoff",
        type=float,
        default=72.0,
        help="大龄亚组阈值（月）",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Bootstrap 随机种子",
    )
    parser.add_argument(
        "--target_event",
        type=str,
        default="mental",
        choices=["mental", "pain", "neutral"],
        help="用于耦合模型的 ISC 事件类型",
    )
    parser.add_argument(
        "--exclude_subject_ids",
        type=str,
        default="",
        help="需要排除的被试ID，逗号分隔（如 S039,S078）",
    )
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, cols: set[str], name: str) -> None:
    miss = cols - set(df.columns)
    if miss:
        raise ValueError(f"{name} 缺少列: {sorted(miss)}")


def safe_corr(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(tmp)
    if n < 3:
        return np.nan, np.nan, n
    if tmp["x"].std() < 1e-12 or tmp["y"].std() < 1e-12:
        return np.nan, np.nan, n
    r, p = pearsonr(tmp["x"], tmp["y"])
    return float(r), float(p), n


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(tmp)
    if n < 3:
        return np.nan, np.nan, n
    r, p = spearmanr(tmp["x"], tmp["y"])
    return float(r), float(p), n


def build_formula(
    base_terms: list[str],
    optional_numeric: list[str],
    optional_categorical: list[str],
    df: pd.DataFrame,
) -> str:
    terms = list(base_terms)
    for col in optional_numeric:
        if col in df.columns:
            terms.append(col)
    for col in optional_categorical:
        if col in df.columns:
            terms.append(f"C({col})")
    return "mental_isc_z ~ " + " + ".join(terms)


def interaction_term_name(model_terms: pd.Series) -> str | None:
    candidates = [
        t for t in model_terms.astype(str).tolist()
        if ("posterior_exponent" in t) and (":C(group)" in t)
    ]
    return candidates[0] if candidates else None


def _extract_interaction_p(coef_df: pd.DataFrame, term_pattern: str = "posterior_exponent:C(group)") -> float:
    terms = coef_df["term"].astype(str).tolist()
    matched = [t for t in terms if term_pattern in t]
    if not matched:
        return np.nan
    row = coef_df[coef_df["term"] == matched[0]].iloc[0]
    for p_col in ("P>|t|", "P>|z|", "pvalue", "p_value"):
        if p_col in coef_df.columns:
            try:
                return float(row[p_col])
            except Exception:
                return np.nan
    return np.nan


def winsorize_series(s: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95) -> pd.Series:
    """按分位数进行缩尾（winsorize）。"""
    s2 = pd.to_numeric(s, errors="coerce").copy()
    finite = s2.dropna()
    if finite.empty:
        return s2
    lo = float(finite.quantile(lower_q))
    hi = float(finite.quantile(upper_q))
    return s2.clip(lower=lo, upper=hi)


def fit_rlm(formula: str, data: pd.DataFrame):
    """HuberT 鲁棒线性模型。"""
    return smf.rlm(formula=formula, data=data, M=sm.robust.norms.HuberT()).fit()


def extract_simple_slopes(result, interaction_term: str | None) -> pd.DataFrame:
    """提取 ASD 与 TD 的 posterior_exponent 简单斜率。"""
    rows = []
    candidates = [
        t for t in result.params.index
        if ("posterior_exponent" in str(t)) and (":" not in str(t))
    ]
    asd_term = candidates[0] if candidates else "posterior_exponent"
    if asd_term in result.params.index:
        beta_asd = float(result.params[asd_term])
        p_asd = float(result.pvalues.get(asd_term, np.nan))
        rows.append({
            "group_slope": "ASD",
            "beta": beta_asd,
            "p_value": p_asd,
            "contrast": asd_term,
        })
    if interaction_term is not None and asd_term in result.params.index:
        try:
            test = result.t_test(f"{asd_term} + {interaction_term} = 0")
            beta_td = float(result.params[asd_term] + result.params[interaction_term])
            p_td = float(np.asarray(test.pvalue).reshape(-1)[0])
        except Exception:
            beta_td = float(result.params[asd_term] + result.params.get(interaction_term, 0.0))
            p_td = np.nan
        rows.append({
            "group_slope": "TD",
            "beta": beta_td,
            "p_value": p_td,
            "contrast": f"{asd_term} + {interaction_term}",
        })
    return pd.DataFrame(rows)


def bootstrap_interaction(
    df: pd.DataFrame,
    formula: str,
    interaction_term: str,
    n_resamples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    n = len(df)
    betas: list[float] = []
    for i in range(n_resamples):
        sample_idx = rng.integers(0, n, size=n)
        sample = df.iloc[sample_idx].copy()
        # 若抽样后某组缺失，模型不可解释交互，记 NaN
        if sample["group"].nunique() < 2:
            betas.append(np.nan)
            continue
        try:
            fit = smf.ols(formula, data=sample).fit()
            beta = fit.params.get(interaction_term, np.nan)
            betas.append(float(beta) if pd.notna(beta) else np.nan)
        except Exception:
            betas.append(np.nan)
    dist = pd.DataFrame({"resample_id": np.arange(n_resamples), "beta_interaction": betas})
    valid = dist["beta_interaction"].dropna().to_numpy()
    if len(valid) == 0:
        summary = pd.DataFrame([{
            "n_resamples": n_resamples,
            "n_valid": 0,
            "beta_mean": np.nan,
            "beta_median": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "p_bootstrap_two_sided": np.nan,
        }])
        return summary, dist
    ci_low, ci_high = np.quantile(valid, [0.025, 0.975])
    p_boot = 2.0 * min(np.mean(valid <= 0), np.mean(valid >= 0))
    summary = pd.DataFrame([{
        "n_resamples": n_resamples,
        "n_valid": int(len(valid)),
        "beta_mean": float(np.mean(valid)),
        "beta_median": float(np.median(valid)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_bootstrap_two_sided": float(min(1.0, p_boot)),
    }])
    return summary, dist


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    log = setup_logging(cfg, name="resting_to_movie_coupling")

    resting_path = (PROJECT_ROOT / args.resting_csv).resolve()
    isc_path = (PROJECT_ROOT / args.movie_isc_csv).resolve()
    part_path = (PROJECT_ROOT / args.participants_csv).resolve()
    movie_analysis_path = (PROJECT_ROOT / args.movie_analysis_csv).resolve()
    movie_qc_path = (PROJECT_ROOT / args.movie_specparam_qc_csv).resolve()
    preproc_path = (PROJECT_ROOT / args.preproc_summary_csv).resolve()

    if not resting_path.exists():
        raise FileNotFoundError(f"未找到 resting CSV: {resting_path}")
    if not isc_path.exists():
        raise FileNotFoundError(f"未找到 movie ISC CSV: {isc_path}")
    if not part_path.exists():
        raise FileNotFoundError(f"未找到 participants CSV: {part_path}")
    if not preproc_path.exists():
        raise FileNotFoundError(f"未找到 preproc_summary CSV: {preproc_path}")

    rest = pd.read_csv(resting_path)
    _require_columns(rest, {"subject_id", "group", "posterior_exponent"}, "resting_csv")
    rest = rest[["subject_id", "group", "posterior_exponent"]].copy()
    rest["subject_id"] = rest["subject_id"].astype(str)
    rest["group"] = rest["group"].astype(str).str.upper()

    target_event = args.target_event.strip().lower()
    excluded_ids = {
        s.strip() for s in str(args.exclude_subject_ids).split(",")
        if s and s.strip()
    }

    isc = pd.read_csv(isc_path)
    _require_columns(isc, {"subject_id", "group", "event_type", "isc_z", "isc_r"}, "movie_isc_csv")
    isc = isc[isc["event_type"].astype(str).str.lower() == target_event].copy()
    isc = isc[["subject_id", "group", "isc_z", "isc_r"]]
    isc["subject_id"] = isc["subject_id"].astype(str)
    isc["group"] = isc["group"].astype(str).str.upper()
    isc = isc.rename(columns={"isc_z": "mental_isc_z", "isc_r": "mental_isc_r"})

    part = pd.read_csv(part_path)
    if "age_months" not in part.columns:
        raise ValueError(f"{part_path} 缺少 age_months 列")
    keep_cols = ["subject_id", "group", "age_months"]
    for c in ("IQ_total", "sex"):
        if c in part.columns:
            keep_cols.append(c)
    if "included_final" in part.columns:
        keep_cols.append("included_final")
    part = part[keep_cols].copy()
    part["subject_id"] = part["subject_id"].astype(str)
    part["group"] = part["group"].astype(str).str.upper()
    if "included_final" in part.columns:
        part["included_final"] = pd.to_numeric(part["included_final"], errors="coerce")
        part = part[part["included_final"] == 1].copy()

    preproc = pd.read_csv(preproc_path)
    if "subject_id" not in preproc.columns or "usable_epochs" not in preproc.columns:
        raise ValueError(f"{preproc_path} 需包含 subject_id 和 usable_epochs")
    preproc = preproc[["subject_id", "usable_epochs"]].copy()
    preproc["subject_id"] = preproc["subject_id"].astype(str)

    merged = isc.merge(rest, on=["subject_id", "group"], how="inner")
    merged = merged.merge(part, on=["subject_id", "group"], how="inner")
    merged = merged.merge(preproc, on="subject_id", how="left")

    # 统一 movie 分析队列：participants_analysis ∩ 非 low_quality
    if movie_analysis_path.exists():
        movie_analysis = pd.read_csv(movie_analysis_path)
        _require_columns(movie_analysis, {"subject_id", "group"}, "movie_analysis_csv")
        movie_analysis["subject_id"] = movie_analysis["subject_id"].astype(str)
        movie_analysis["group"] = movie_analysis["group"].astype(str).str.upper()
        allowed_pairs = movie_analysis[["subject_id", "group"]].drop_duplicates()
        merged = merged.merge(allowed_pairs, on=["subject_id", "group"], how="inner")

    if movie_qc_path.exists():
        movie_qc = pd.read_csv(movie_qc_path)
        _require_columns(movie_qc, {"subject_id", "low_quality_subject"}, "movie_specparam_qc_csv")
        movie_qc["subject_id"] = movie_qc["subject_id"].astype(str)
        bad_ids = set(
            movie_qc.loc[pd.to_numeric(movie_qc["low_quality_subject"], errors="coerce") == 1, "subject_id"].tolist()
        )
        if bad_ids:
            merged = merged[~merged["subject_id"].isin(bad_ids)].copy()
    if excluded_ids:
        merged = merged[~merged["subject_id"].isin(excluded_ids)].copy()
    merged["age_months"] = pd.to_numeric(merged["age_months"], errors="coerce")
    if "IQ_total" in merged.columns:
        merged["IQ_total"] = pd.to_numeric(merged["IQ_total"], errors="coerce")
    merged["posterior_exponent"] = pd.to_numeric(merged["posterior_exponent"], errors="coerce")
    merged["mental_isc_z"] = pd.to_numeric(merged["mental_isc_z"], errors="coerce")
    merged["usable_epochs"] = pd.to_numeric(merged["usable_epochs"], errors="coerce")
    if "sex" in merged.columns:
        merged["sex"] = merged["sex"].astype(str).str.upper().replace({"NAN": np.nan, "": np.nan})

    # 组内相关
    corr_rows = []
    for grp in sorted(merged["group"].dropna().unique().tolist()):
        sub = merged[merged["group"] == grp].copy()
        pear_r, pear_p, n_pear = safe_corr(sub["posterior_exponent"], sub["mental_isc_z"])
        spr_r, spr_p, n_spr = safe_spearman(sub["posterior_exponent"], sub["mental_isc_z"])
        corr_rows.append({
            "group": grp,
            "metric": "pearson",
            "r": pear_r,
            "p_value": pear_p,
            "n": n_pear,
        })
        corr_rows.append({
            "group": grp,
            "metric": "spearman",
            "r": spr_r,
            "p_value": spr_p,
            "n": n_spr,
        })
    corr_df = pd.DataFrame(corr_rows)

    # 全样本稳健模型（加入数据质量与可用临床协变量）
    robust_formula = build_formula(
        base_terms=["posterior_exponent * C(group)", "age_months"],
        optional_numeric=["usable_epochs", "IQ_total"],
        optional_categorical=["sex"],
        df=merged,
    )
    needed_cols = ["mental_isc_z", "posterior_exponent", "group", "age_months", "usable_epochs"]
    if "IQ_total" in merged.columns:
        needed_cols.append("IQ_total")
    if "sex" in merged.columns:
        needed_cols.append("sex")
    model_df = merged.dropna(subset=needed_cols).copy()
    model = smf.ols(robust_formula, data=model_df).fit()
    coef_df = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    coef_df["n_obs"] = int(model.nobs)
    coef_df["r_squared"] = float(model.rsquared)
    coef_df["adj_r_squared"] = float(model.rsquared_adj)
    coef_df["formula"] = robust_formula

    inter_term = interaction_term_name(coef_df["term"])
    boot_summary = pd.DataFrame()
    boot_dist = pd.DataFrame()
    if inter_term is not None:
        boot_summary, boot_dist = bootstrap_interaction(
            df=model_df,
            formula=robust_formula,
            interaction_term=inter_term,
            n_resamples=args.n_resamples,
            random_seed=args.random_seed,
        )
        boot_summary["interaction_term"] = inter_term

    # 大龄亚组模型（>72 月）
    older_df = merged[merged["age_months"] > float(args.older_age_cutoff)].copy()
    older_model_df = older_df.dropna(
        subset=["mental_isc_z", "posterior_exponent", "group", "age_months"]
    ).copy()
    older_formula = "mental_isc_z ~ posterior_exponent * C(group) + age_months"
    older_model = smf.ols(older_formula, data=older_model_df).fit()
    older_coef_df = older_model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    older_coef_df["n_obs"] = int(older_model.nobs)
    older_coef_df["r_squared"] = float(older_model.rsquared)
    older_coef_df["adj_r_squared"] = float(older_model.rsquared_adj)
    older_coef_df["formula"] = older_formula

    # ---------------------------
    # Winsorize + Robust (RLM) 分析
    # ---------------------------
    robust_full_df = model_df.copy()
    robust_full_df["posterior_exponent_w"] = winsorize_series(robust_full_df["posterior_exponent"])
    robust_full_df["mental_isc_z_w"] = winsorize_series(robust_full_df["mental_isc_z"])

    robust_terms = build_formula(
        base_terms=["posterior_exponent_w * C(group)", "age_months"],
        optional_numeric=["usable_epochs", "IQ_total"],
        optional_categorical=["sex"],
        df=robust_full_df,
    )
    robust_formula = robust_terms.replace("mental_isc_z", "mental_isc_z_w")
    robust_model = fit_rlm(robust_formula, robust_full_df)
    robust_coef_df = pd.DataFrame({
        "term": robust_model.params.index,
        "Coef.": robust_model.params.values,
        "Std.Err.": robust_model.bse.values,
        "z": robust_model.tvalues.values,
        "P>|z|": robust_model.pvalues.values,
    })
    robust_coef_df["n_obs"] = int(len(robust_full_df))
    robust_coef_df["formula"] = robust_formula
    robust_inter_term = interaction_term_name(robust_coef_df["term"])
    robust_slopes_df = extract_simple_slopes(robust_model, robust_inter_term)
    robust_slopes_df["model_scope"] = "full"

    robust_older_df = older_model_df.copy()
    robust_older_df["posterior_exponent_w"] = winsorize_series(robust_older_df["posterior_exponent"])
    robust_older_df["mental_isc_z_w"] = winsorize_series(robust_older_df["mental_isc_z"])
    older_robust_terms = build_formula(
        base_terms=["posterior_exponent_w * C(group)", "age_months"],
        optional_numeric=["usable_epochs", "IQ_total"],
        optional_categorical=["sex"],
        df=robust_older_df,
    )
    older_robust_formula = older_robust_terms.replace("mental_isc_z", "mental_isc_z_w")
    robust_older_model = fit_rlm(older_robust_formula, robust_older_df)
    robust_older_coef_df = pd.DataFrame({
        "term": robust_older_model.params.index,
        "Coef.": robust_older_model.params.values,
        "Std.Err.": robust_older_model.bse.values,
        "z": robust_older_model.tvalues.values,
        "P>|z|": robust_older_model.pvalues.values,
    })
    robust_older_coef_df["n_obs"] = int(len(robust_older_df))
    robust_older_coef_df["formula"] = older_robust_formula
    robust_older_inter_term = interaction_term_name(robust_older_coef_df["term"])
    robust_older_slopes_df = extract_simple_slopes(robust_older_model, robust_older_inter_term)
    robust_older_slopes_df["model_scope"] = "older72"

    deriv = Path(cfg["paths"]["derivatives_root"])
    out_root = Path(cfg["paths"]["outputs_root"])
    stats_dir = deriv / "stats"
    fig_dir = out_root / "figures"
    stats_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_suffix = "" if target_event == "mental" else f"_{target_event}"

    save_csv(merged, stats_dir / f"resting_movie_coupling_merged{out_suffix}.csv")
    save_csv(corr_df, stats_dir / f"resting_movie_coupling_group_correlations{out_suffix}.csv")
    save_csv(coef_df, stats_dir / f"resting_movie_coupling_interaction_model{out_suffix}.csv")
    if not boot_summary.empty:
        save_csv(boot_summary, stats_dir / f"resting_movie_coupling_bootstrap_interaction_summary{out_suffix}.csv")
        save_csv(boot_dist, stats_dir / f"resting_movie_coupling_bootstrap_interaction_distribution{out_suffix}.csv")
    save_csv(older_model_df, stats_dir / f"resting_movie_coupling_older72_merged{out_suffix}.csv")
    save_csv(older_coef_df, stats_dir / f"resting_movie_coupling_older72_interaction_model{out_suffix}.csv")
    save_csv(robust_coef_df, stats_dir / f"resting_movie_coupling_interaction_model_rlm_winsor{out_suffix}.csv")
    save_csv(
        robust_older_coef_df,
        stats_dir / f"resting_movie_coupling_older72_interaction_model_rlm_winsor{out_suffix}.csv",
    )
    save_csv(
        pd.concat([robust_slopes_df, robust_older_slopes_df], ignore_index=True),
        stats_dir / f"resting_movie_coupling_simple_slopes_rlm_winsor{out_suffix}.csv",
    )

    # 出图：散点 + 分组回归线（95% CI）
    plt.figure(figsize=(7.6, 5.8))
    palette = {"ASD": "#1f77b4", "TD": "#d62728"}
    sns.scatterplot(
        data=merged,
        x="posterior_exponent",
        y="mental_isc_z",
        hue="group",
        palette=palette,
        alpha=0.72,
        s=34,
        edgecolor=None,
    )
    for grp in ["ASD", "TD"]:
        sub = merged[merged["group"] == grp].copy()
        if len(sub.dropna(subset=["posterior_exponent", "mental_isc_z"])) < 3:
            continue
        sns.regplot(
            data=sub,
            x="posterior_exponent",
            y="mental_isc_z",
            scatter=False,
            ci=95,
            color=palette[grp],
            line_kws={"linewidth": 2},
        )
    plt.xlabel("Resting Posterior Exponent")
    plt.ylabel(f"Movie {target_event.capitalize()} ISC (Fisher z)")
    plt.title(f"State-Processing Coupling: Resting vs {target_event.capitalize()} ISC")

    # 图中注释交互项 p 值（同时报告 OLS 与 RLM）
    p_inter_ols = _extract_interaction_p(coef_df, term_pattern="posterior_exponent:C(group)")
    p_inter_rlm = _extract_interaction_p(robust_coef_df, term_pattern="posterior_exponent_w:C(group)")
    p_ols_txt = "NA" if pd.isna(p_inter_ols) else f"{float(p_inter_ols):.4g}"
    p_rlm_txt = "NA" if pd.isna(p_inter_rlm) else f"{float(p_inter_rlm):.4g}"
    plt.gca().text(
        0.98,
        0.98,
        f"Interaction p (OLS) = {p_ols_txt}\nInteraction p (RLM) = {p_rlm_txt}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#666"},
    )
    plt.tight_layout()
    fig_name = "resting_to_movie_coupling_scatter" if target_event == "mental" else f"resting_to_movie_coupling_scatter_{target_event}"
    fig_path = fig_dir / fig_name
    plt.savefig(fig_path.with_suffix(".png"), dpi=190)
    plt.savefig(fig_path.with_suffix(".pdf"))
    plt.close()

    # 大龄亚组图
    plt.figure(figsize=(7.6, 5.8))
    palette = {"ASD": "#1f77b4", "TD": "#d62728"}
    sns.scatterplot(
        data=older_model_df,
        x="posterior_exponent",
        y="mental_isc_z",
        hue="group",
        palette=palette,
        alpha=0.72,
        s=36,
        edgecolor=None,
    )
    for grp in ["ASD", "TD"]:
        sub = older_model_df[older_model_df["group"] == grp].copy()
        if len(sub.dropna(subset=["posterior_exponent", "mental_isc_z"])) < 3:
            continue
        sns.regplot(
            data=sub,
            x="posterior_exponent",
            y="mental_isc_z",
            scatter=False,
            ci=95,
            color=palette[grp],
            line_kws={"linewidth": 2},
        )
    plt.xlabel("Resting Posterior Exponent")
    plt.ylabel("Movie Mental ISC (Fisher z)")
    plt.title(f"Older Subgroup (> {int(args.older_age_cutoff)} months)")
    older_inter_term = interaction_term_name(older_coef_df["term"])
    if older_inter_term is not None:
        row = older_coef_df[older_coef_df["term"] == older_inter_term].iloc[0]
        p_int = row["P>|t|"] if "P>|t|" in row else np.nan
        p_txt = "NA" if pd.isna(p_int) else f"{float(p_int):.4g}"
        plt.gca().text(
            0.98, 0.98, f"Interaction p = {p_txt}",
            transform=plt.gca().transAxes,
            ha="right", va="top", fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#666"},
        )
    plt.tight_layout()
    older_fig_name = (
        "resting_to_movie_coupling_scatter_older72"
        if target_event == "mental"
        else f"resting_to_movie_coupling_scatter_{target_event}_older72"
    )
    older_fig_path = fig_dir / older_fig_name
    plt.savefig(older_fig_path.with_suffix(".png"), dpi=190)
    plt.savefig(older_fig_path.with_suffix(".pdf"))
    plt.close()

    log.info("融合数据: %s", stats_dir / f"resting_movie_coupling_merged{out_suffix}.csv")
    log.info("组内相关: %s", stats_dir / f"resting_movie_coupling_group_correlations{out_suffix}.csv")
    log.info("交互回归: %s", stats_dir / f"resting_movie_coupling_interaction_model{out_suffix}.csv")
    if not boot_summary.empty:
        log.info(
            "Bootstrap 交互汇总: %s",
            stats_dir / f"resting_movie_coupling_bootstrap_interaction_summary{out_suffix}.csv",
        )
    log.info("大龄亚组交互回归: %s", stats_dir / f"resting_movie_coupling_older72_interaction_model{out_suffix}.csv")
    log.info(
        "RLM+Winsor 全样本: %s",
        stats_dir / f"resting_movie_coupling_interaction_model_rlm_winsor{out_suffix}.csv",
    )
    log.info(
        "RLM+Winsor 大龄亚组: %s",
        stats_dir / f"resting_movie_coupling_older72_interaction_model_rlm_winsor{out_suffix}.csv",
    )
    log.info("RLM+Winsor simple slopes: %s", stats_dir / f"resting_movie_coupling_simple_slopes_rlm_winsor{out_suffix}.csv")
    log.info("耦合图: %s", fig_path.with_suffix(".png"))
    log.info("大龄亚组图: %s", older_fig_path.with_suffix(".png"))
    if excluded_ids:
        log.info("手动排除被试: %s", ",".join(sorted(excluded_ids)))


if __name__ == "__main__":
    main()
