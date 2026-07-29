"""通用统计分析函数。"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import mixedlm, ols
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 描述性统计
# ---------------------------------------------------------------------------

def descriptive_table(
    df: pd.DataFrame,
    group_col: str,
    variables: list[str],
    continuous: list[str] | None = None,
    categorical: list[str] | None = None,
) -> pd.DataFrame:
    """按组生成描述性统计表。"""
    continuous = continuous or variables
    categorical = categorical or []
    rows = []
    for grp, sub in df.groupby(group_col):
        for var in continuous:
            if var not in sub.columns:
                continue
            vals = sub[var].dropna()
            rows.append({
                "group": grp,
                "variable": var,
                "type": "continuous",
                "n": len(vals),
                "mean": vals.mean(),
                "std": vals.std(),
                "median": vals.median(),
                "q25": vals.quantile(0.25),
                "q75": vals.quantile(0.75),
            })
        for var in categorical:
            if var not in sub.columns:
                continue
            counts = sub[var].value_counts()
            for level, cnt in counts.items():
                rows.append({
                    "group": grp,
                    "variable": var,
                    "type": "categorical",
                    "level": level,
                    "n": int(cnt),
                    "percent": 100 * cnt / len(sub),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 组间检验
# ---------------------------------------------------------------------------

def independent_ttest(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """独立样本 t 检验。"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return {"statistic": np.nan, "pvalue": np.nan}
    stat, p = stats.ttest_ind(x, y, equal_var=False)
    return {"statistic": float(stat), "pvalue": float(p)}


def mann_whitney(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Mann-Whitney U 检验。"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 1 or len(y) < 1:
        return {"statistic": np.nan, "pvalue": np.nan}
    stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    return {"statistic": float(stat), "pvalue": float(p)}


def chi_square_or_fisher(
    table: np.ndarray,
) -> dict[str, Any]:
    """卡方检验；2x2 小样本时用 Fisher 精确检验。"""
    table = np.asarray(table)
    if table.shape == (2, 2) and table.sum() < 40:
        odds, p = stats.fisher_exact(table)
        return {"test": "fisher", "statistic": float(odds), "pvalue": float(p)}
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return {"test": "chi2", "statistic": float(chi2), "pvalue": float(p), "dof": int(dof)}


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d（独立样本）。"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled_std = np.sqrt(((n1 - 1) * x.std(ddof=1) ** 2 + (n2 - 1) * y.std(ddof=1) ** 2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return float((x.mean() - y.mean()) / pooled_std)


def fdr_correction(pvalues: np.ndarray, alpha: float = 0.05, method: str = "fdr_bh") -> tuple[np.ndarray, np.ndarray]:
    """FDR 校正，返回 reject 标志与校正后 p 值。"""
    pvalues = np.asarray(pvalues, dtype=float)
    mask = ~np.isnan(pvalues)
    reject = np.full_like(pvalues, False, dtype=bool)
    p_adj = np.full_like(pvalues, np.nan)
    if mask.sum() == 0:
        return reject, p_adj
    reject_mask, p_adj_mask, *_ = multipletests(
        pvalues[mask], alpha=alpha, method=method
    )
    reject[mask] = reject_mask
    p_adj[mask] = p_adj_mask
    return reject, p_adj


# ---------------------------------------------------------------------------
# 回归
# ---------------------------------------------------------------------------

def run_ols(formula: str, data: pd.DataFrame) -> Any:
    """OLS 回归（仅按公式变量剔除缺失，不整表 dropna）。"""
    model = ols(formula, data=data).fit()
    if model.nobs < 10:
        logger.warning("OLS 样本量过小 (n=%d): %s", int(model.nobs), formula)
    return model


def run_robust_regression(formula: str, data: pd.DataFrame) -> Any:
    """稳健回归 (RLM)。"""
    model = RLM.from_formula(formula, data=data).fit()
    return model


def run_mixedlm(
    formula: str,
    data: pd.DataFrame,
    groups: str,
    re_formula: str = "1",
) -> Any:
    """
    线性混合效应模型。

    若拟合失败，自动降级为 OLS 并在日志中提示。
    """
    try:
        model = mixedlm(formula, data=data, groups=data[groups], re_formula=re_formula)
        result = model.fit(method="lbfgs", maxiter=200, reml=False)
        result._used_mixedlm = True  # type: ignore
        return result
    except Exception as exc:
        logger.warning(
            "MixedLM 不稳定，降级为 OLS: %s\n公式: %s",
            exc,
            formula,
        )
        result = run_ols(formula, data)
        result._used_mixedlm = False  # type: ignore
        return result


def spearman_correlation(x: pd.Series, y: pd.Series) -> dict[str, float]:
    """Spearman 相关。"""
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return {"rho": np.nan, "pvalue": np.nan, "n": int(mask.sum())}
    rho, p = stats.spearmanr(x[mask], y[mask])
    return {"rho": float(rho), "pvalue": float(p), "n": int(mask.sum())}


def model_results_to_row(
    model: Any,
    model_name: str,
    outcome: str,
    predictors: list[str] | None = None,
) -> list[dict[str, Any]]:
    """将回归结果转为长表行。"""
    rows = []
    params = model.params
    pvals = model.pvalues
    conf = model.conf_int()
    predictors = predictors or list(params.index)
    for term in predictors:
        if term not in params.index:
            continue
        rows.append({
            "model": model_name,
            "outcome": outcome,
            "term": term,
            "coef": params[term],
            "std_err": model.bse[term] if term in model.bse.index else np.nan,
            "pvalue": pvals[term],
            "ci_low": conf.loc[term, 0] if term in conf.index else np.nan,
            "ci_high": conf.loc[term, 1] if term in conf.index else np.nan,
            "n_obs": int(model.nobs),
            "r_squared": getattr(model, "rsquared", np.nan),
            "used_mixedlm": getattr(model, "_used_mixedlm", None),
        })
    return rows


def residualize_ols(
    y: pd.Series,
    cov_df: pd.DataFrame,
    formula_rhs: str,
) -> np.ndarray:
    """对 y 关于 formula_rhs 做 OLS 残差化。"""
    data = cov_df.copy()
    data["_y"] = pd.to_numeric(y, errors="coerce")
    sub = data.dropna(subset=["_y"])
    if len(sub) < 8:
        return np.full(len(y), np.nan)
    model = ols(f"_y ~ {formula_rhs}", data=sub).fit()
    resid = pd.Series(np.nan, index=y.index, dtype=float)
    resid.loc[sub.index] = model.resid
    return resid.to_numpy(dtype=float)


def partial_correlation_pearson(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    cov_cols: list[str] | None = None,
) -> dict[str, Any]:
    """偏 Pearson 相关：对 y 与 x 分别残差化协变量后求相关。"""
    cov_cols = cov_cols or ["age_months", "IQ_total"]
    req = [y_col, x_col, *cov_cols]
    sub = df.dropna(subset=req).copy()
    if len(sub) < 8:
        return {"partial_r": np.nan, "pvalue": np.nan, "n": len(sub)}
    cov_formula = " + ".join(cov_cols)
    ry = residualize_ols(sub[y_col], sub, cov_formula)
    rx = residualize_ols(sub[x_col], sub, cov_formula)
    mask = np.isfinite(rx) & np.isfinite(ry)
    if mask.sum() < 8:
        return {"partial_r": np.nan, "pvalue": np.nan, "n": int(mask.sum())}
    r, p = stats.pearsonr(rx[mask], ry[mask])
    return {"partial_r": float(r), "pvalue": float(p), "n": int(mask.sum())}


def partial_spearman(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    cov_cols: list[str] | None = None,
) -> dict[str, Any]:
    """偏 Spearman 相关：协变量 OLS 残差化后对残差做 Spearman。"""
    cov_cols = cov_cols or ["age_months", "IQ_total"]
    req = [y_col, x_col, *cov_cols]
    sub = df.dropna(subset=req).copy()
    if len(sub) < 8:
        return {"partial_rho": np.nan, "pvalue": np.nan, "n": len(sub)}
    cov_formula = " + ".join(cov_cols)
    rx = residualize_ols(sub[x_col], sub, cov_formula)
    ry = residualize_ols(sub[y_col], sub, cov_formula)
    mask = np.isfinite(rx) & np.isfinite(ry)
    if mask.sum() < 8:
        return {"partial_rho": np.nan, "pvalue": np.nan, "n": int(mask.sum())}
    rho, p = stats.spearmanr(rx[mask], ry[mask])
    return {"partial_rho": float(rho), "pvalue": float(p), "n": int(mask.sum())}


def bootstrap_partial_spearman(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    cov_cols: list[str] | None = None,
) -> dict[str, Any]:
    """偏 Spearman + bootstrap 95% CI（有放回重抽样）。"""
    cov_cols = cov_cols or ["age_months", "IQ_total"]
    base = partial_spearman(df, y_col, x_col, cov_cols)
    req = [y_col, x_col, *cov_cols]
    sub = df.dropna(subset=req).copy()
    if len(sub) < 10:
        return {
            **base,
            "boot_ci_low": np.nan,
            "boot_ci_high": np.nan,
            "boot_median": np.nan,
            "n_boot_valid": 0,
        }
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    idx = np.arange(len(sub))
    for _ in range(n_boot):
        samp = sub.iloc[rng.choice(idx, size=len(idx), replace=True)].reset_index(drop=True)
        res = partial_spearman(samp, y_col, x_col, cov_cols)
        if np.isfinite(res["partial_rho"]):
            boots.append(res["partial_rho"])
    if not boots:
        return {
            **base,
            "boot_ci_low": np.nan,
            "boot_ci_high": np.nan,
            "boot_median": np.nan,
            "n_boot_valid": 0,
        }
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        **base,
        "boot_ci_low": float(lo),
        "boot_ci_high": float(hi),
        "boot_median": float(np.median(boots)),
        "n_boot_valid": len(boots),
    }


def build_iq_matched_cohort(
    df: pd.DataFrame,
    *,
    iq_caliper: float = 15.0,
    age_caliper: float = 24.0,
    match_sex: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1:1 贪婪 IQ-balanced 匹配（|ΔIQ| ≤ iq_caliper, |Δage| ≤ age_caliper，可选同性别）。

    返回 (matched_df, pair_table)。
    """
    req = ["subject_id", "group", "IQ_total", "age_months", "sex"]
    sub = df.dropna(subset=req).copy()
    sub["subject_id"] = sub["subject_id"].astype(str)
    asd = sub[sub["group"] == "ASD"].copy()
    td = sub[sub["group"] == "TD"].copy()
    asd = asd.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    pairs: list[dict[str, Any]] = []
    used_td: set[str] = set()
    for _, a_row in asd.iterrows():
        cand = td[~td["subject_id"].isin(used_td)].copy()
        if match_sex:
            cand = cand[cand["sex"] == a_row["sex"]]
        if cand.empty:
            continue
        cand["iq_diff"] = (cand["IQ_total"] - a_row["IQ_total"]).abs()
        cand["age_diff"] = (cand["age_months"] - a_row["age_months"]).abs()
        cand = cand[(cand["iq_diff"] <= iq_caliper) & (cand["age_diff"] <= age_caliper)]
        if cand.empty:
            continue
        cand = cand.sort_values(["iq_diff", "age_diff"])
        t_row = cand.iloc[0]
        used_td.add(str(t_row["subject_id"]))
        pairs.append(
            {
                "asd_subject_id": str(a_row["subject_id"]),
                "td_subject_id": str(t_row["subject_id"]),
                "iq_diff": float(t_row["iq_diff"]),
                "age_diff_months": float(t_row["age_diff"]),
                "sex": a_row["sex"],
            }
        )

    pair_df = pd.DataFrame(pairs)
    if pair_df.empty:
        return pd.DataFrame(), pair_df

    matched_ids = set(pair_df["asd_subject_id"]) | set(pair_df["td_subject_id"])
    matched = sub[sub["subject_id"].isin(matched_ids)].copy()
    logger.info(
        "IQ-matched pairs: %d (ASD %d, TD %d)",
        len(pair_df),
        len(pair_df),
        len(pair_df),
    )
    return matched.reset_index(drop=True), pair_df


def compare_groups_on_variable(
    df: pd.DataFrame,
    group_col: str,
    variable: str,
    group_a: str,
    group_b: str,
) -> dict[str, Any]:
    """对连续变量做 t 检验、U 检验与 Cohen's d。"""
    a = df.loc[df[group_col] == group_a, variable]
    b = df.loc[df[group_col] == group_b, variable]
    t_res = independent_ttest(a.values, b.values)
    u_res = mann_whitney(a.values, b.values)
    d = cohens_d(a.values, b.values)
    return {
        "variable": variable,
        "group_a": group_a,
        "group_b": group_b,
        "n_a": int(a.notna().sum()),
        "n_b": int(b.notna().sum()),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "t_stat": t_res["statistic"],
        "t_pvalue": t_res["pvalue"],
        "u_stat": u_res["statistic"],
        "u_pvalue": u_res["pvalue"],
        "cohens_d": d,
    }
