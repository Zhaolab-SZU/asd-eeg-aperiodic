"""审稿防御：posterior ROI、匹配队列、临床相关与模型诊断工具。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols

from src.config import load_roi_config
from src.io_utils import (
    attach_usable_epochs,
    exclude_specparam_low_quality,
    load_analysis_participants,
)
from src.spectral_maturation_analysis import POSTERIOR_CORE
from src.stats_utils import (
    cohens_d,
    compare_groups_on_variable,
    model_results_to_row,
    run_ols,
    spearman_correlation,
)

logger = logging.getLogger(__name__)

GROUP_TERM = "C(group)[T.TD]"
INTERACTION_TERM = "C(group)[T.TD]:age_months"
COVARIATE_FORMULA = " + age_months + C(sex) + IQ_total + usable_epochs"
COVARIATE_FORMULA_EXPONENT = COVARIATE_FORMULA + " + mean_r_squared"


def _find_term(params_index: pd.Index, suffix: str) -> str | None:
    for term in params_index:
        if suffix in term:
            return term
    return None


def _group_term(params_index: pd.Index) -> str | None:
    return _find_term(params_index, "[T.TD]")


def _interaction_term(params_index: pd.Index) -> str | None:
    return _find_term(params_index, "[T.TD]:age_months")


def get_roi_channel_sets(layout: str = "channels_egi64") -> dict[str, list[str]]:
    roi_cfg = load_roi_config()
    layout_dict = roi_cfg[layout]
    occ = list(layout_dict["occipital"])
    par = list(layout_dict["parietal"])
    extended = sorted(set(occ) | set(par))
    return {
        "posterior_core_4": list(POSTERIOR_CORE),
        "occipital_roi_13": occ,
        "parietal_roi_11": par,
        "posterior_extended_24": extended,
    }


def channel_mean(
    ch_df: pd.DataFrame,
    channels: list[str],
    value_col: str = "aperiodic_exponent",
    min_ratio: float = 0.5,
) -> pd.DataFrame:
    df = ch_df.copy()
    if "fit_valid" in df.columns:
        df = df[df["fit_valid"].astype(bool)]
    rows: list[dict[str, Any]] = []
    for (sid, grp), sub in df.groupby(["subject_id", "group"]):
        roi = sub[sub["channel"].isin(channels)]
        n_req = len(channels)
        n_valid = roi[value_col].notna().sum() if value_col in roi.columns else 0
        val = float(roi[value_col].mean()) if n_valid >= min_ratio * n_req else np.nan
        rows.append(
            {
                "subject_id": str(sid),
                "group": grp,
                "value": val,
                "n_valid_channels": int(n_valid),
                "n_required_channels": n_req,
            }
        )
    return pd.DataFrame(rows)


def build_posterior_metric_table(
    ch_df: pd.DataFrame,
    layout: str = "channels_egi64",
    value_col: str = "aperiodic_exponent",
) -> pd.DataFrame:
    roi_sets = get_roi_channel_sets(layout)
    base = None
    for name, channels in roi_sets.items():
        part = channel_mean(ch_df, channels, value_col=value_col).rename(columns={"value": name})
        if base is None:
            base = part[["subject_id", "group", name]]
        else:
            base = base.merge(part[["subject_id", name]], on="subject_id", how="outer")
    return base


def load_primary_cohort(cfg: dict[str, Any]) -> pd.DataFrame:
    deriv = Path(cfg["paths"]["derivatives_root"])
    df = load_analysis_participants(cfg)
    df = exclude_specparam_low_quality(df, deriv)
    df = attach_usable_epochs(df, deriv)
    min_ep = int(cfg.get("epochs", {}).get("min_usable_epochs", 60))
    if "usable_epochs" in df.columns:
        df = df[df["usable_epochs"] >= min_ep].copy()
    roi_path = deriv / "roi" / "specparam_subject_global.csv"
    if roi_path.exists():
        roi = pd.read_csv(roi_path)
        merge_cols = ["subject_id"]
        for c in ("mean_r_squared", "global_exponent", "low_gamma_pw", "invalid_channel_ratio"):
            if c in roi.columns:
                merge_cols.append(c)
        if len(merge_cols) > 1:
            df = df.merge(roi[merge_cols].drop_duplicates("subject_id"), on="subject_id", how="left")
    preproc_path = deriv / "qc" / "preproc_summary.csv"
    if preproc_path.exists():
        pre = pd.read_csv(preproc_path)
        pre_cols = ["subject_id"]
        if "bad_channel_count" in pre.columns:
            pre_cols.append("bad_channel_count")
        df = df.merge(pre[pre_cols], on="subject_id", how="left")
    qc_path = deriv / "specparam" / "specparam_qc_summary_subject.csv"
    if qc_path.exists():
        qc = pd.read_csv(qc_path)
        qc_cols = ["subject_id"]
        for c in ("mean_r_squared", "invalid_channel_ratio"):
            if c in qc.columns:
                qc_cols.append(c)
        if len(qc_cols) > 1:
            df = df.merge(qc[qc_cols], on="subject_id", how="left")
    return df.reset_index(drop=True)


def merge_posterior_metrics(cohort: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    keep = [
        c
        for c in (
            "subject_id",
            "group",
            "age_months",
            "sex",
            "IQ_total",
            "usable_epochs",
            "mean_r_squared",
            "bad_channel_count",
            "ADOS_total",
            "ADOS_SA",
            "ADOS_RRB",
        )
        if c in cohort.columns
    ]
    out = metrics.merge(cohort[keep], on="subject_id", how="inner", suffixes=("", "_cohort"))
    if "group_cohort" in out.columns:
        out["group"] = out["group"].fillna(out["group_cohort"])
        out = out.drop(columns=["group_cohort"])
    return out


def build_iq_matched_cohort(
    df: pd.DataFrame,
    iq_caliper: float = 15.0,
    age_caliper: float = 24.0,
    match_sex: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1:1 最近邻 IQ 匹配（可选同龄、同性别）。"""
    req = ["subject_id", "group", "IQ_total", "age_months", "sex"]
    sub = df.dropna(subset=req).copy()
    asd = sub[sub["group"] == "ASD"].copy()
    td = sub[sub["group"] == "TD"].copy()
    rng = np.random.default_rng(seed)
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
    logger.info("IQ-matched pairs: %d (ASD %d, TD %d)", len(pair_df), len(pair_df), len(pair_df))
    return matched.reset_index(drop=True), pair_df


def build_strict_qc_cohort(df: pd.DataFrame, bad_channel_max: int = 2) -> pd.DataFrame:
    out = df.copy()
    if "bad_channel_count" in out.columns:
        out = out[out["bad_channel_count"].fillna(0) <= bad_channel_max]
    if "mean_r_squared" in out.columns:
        out = out[out["mean_r_squared"] >= 0.90]
    return out.reset_index(drop=True)


def _is_posterior_outcome(outcome: str) -> bool:
    keys = ("exponent", "core_4", "occipital", "parietal", "extended")
    return any(k in outcome for k in keys)


def _covariate_suffix(df: pd.DataFrame, outcome: str) -> str:
    if "mean_r_squared" in df.columns and _is_posterior_outcome(outcome):
        return COVARIATE_FORMULA_EXPONENT
    return COVARIATE_FORMULA


def _group_formula(outcome: str, extra_cov: str = "", df: pd.DataFrame | None = None) -> str:
    cov = _covariate_suffix(df, outcome) if df is not None else (
        COVARIATE_FORMULA_EXPONENT if "exponent" in outcome else COVARIATE_FORMULA
    )
    return f"{outcome} ~ C(group, Treatment(reference='ASD')){cov}{extra_cov}"


def _interaction_formula(outcome: str, extra_cov: str = "", df: pd.DataFrame | None = None) -> str:
    cov = _covariate_suffix(df, outcome) if df is not None else (
        COVARIATE_FORMULA_EXPONENT if "exponent" in outcome else COVARIATE_FORMULA
    )
    return f"{outcome} ~ C(group) * age_months{cov}{extra_cov}"


def fit_group_effect(
    df: pd.DataFrame,
    outcome: str,
    cohort_label: str,
    extra_cov: str = "",
) -> dict[str, Any] | None:
    cov_cols = ["group", "age_months", "sex", "IQ_total", "usable_epochs"]
    if "mean_r_squared" in df.columns and _is_posterior_outcome(outcome):
        cov_cols.append("mean_r_squared")
    sub = df.dropna(subset=[outcome] + cov_cols)
    if len(sub) < 20 or sub["group"].nunique() < 2:
        return None
    formula = _group_formula(outcome, extra_cov, df=sub)
    model = run_ols(formula, sub)
    gterm = _group_term(model.params.index)
    if gterm is None:
        return None
    ci = model.conf_int().loc[gterm]
    asd = sub.loc[sub["group"] == "ASD", outcome]
    td = sub.loc[sub["group"] == "TD", outcome]
    return {
        "cohort": cohort_label,
        "analysis": "group_effect",
        "outcome": outcome,
        "n_total": int(model.nobs),
        "n_ASD": int((sub["group"] == "ASD").sum()),
        "n_TD": int((sub["group"] == "TD").sum()),
        "beta_TD_vs_ASD": float(model.params[gterm]),
        "se": float(model.bse[gterm]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p": float(model.pvalues[gterm]),
        "r_squared": float(model.rsquared),
        "cohens_d": cohens_d(asd.values, td.values),
        "partial_eta_sq_group": partial_eta_squared(model, gterm),
        "cohens_f2_group": cohen_f_squared(model, gterm),
    }


def fit_age_group_interaction(
    df: pd.DataFrame,
    outcome: str,
    cohort_label: str,
) -> list[dict[str, Any]]:
    cov_cols = ["group", "age_months", "sex", "IQ_total", "usable_epochs"]
    if "mean_r_squared" in df.columns and _is_posterior_outcome(outcome):
        cov_cols.append("mean_r_squared")
    sub = df.dropna(subset=[outcome] + cov_cols)
    if len(sub) < 20:
        return []
    formula_int = _interaction_formula(outcome, df=sub)
    formula_red = _group_formula(outcome, df=sub)
    model_int = run_ols(formula_int, sub)
    model_red = run_ols(formula_red, sub)
    delta_r2 = float(model_int.rsquared - model_red.rsquared)
    iterm = _interaction_term(model_int.params.index)
    gterm = _group_term(model_int.params.index)
    rows: list[dict[str, Any]] = []
    for term in (gterm, iterm, "age_months"):
        if term is None or term not in model_int.params.index:
            continue
        ci = model_int.conf_int().loc[term]
        rows.append(
            {
                "cohort": cohort_label,
                "analysis": "age_group_interaction",
                "outcome": outcome,
                "term": term,
                "n_total": int(model_int.nobs),
                "n_ASD": int((sub["group"] == "ASD").sum()),
                "n_TD": int((sub["group"] == "TD").sum()),
                "coef": float(model_int.params[term]),
                "se": float(model_int.bse[term]),
                "ci_low": float(ci[0]),
                "ci_high": float(ci[1]),
                "p": float(model_int.pvalues[term]),
                "r_squared_full": float(model_int.rsquared),
                "r_squared_reduced": float(model_red.rsquared),
                "delta_r_squared_interaction": delta_r2 if term == iterm else np.nan,
                "partial_eta_sq": partial_eta_squared(model_int, term),
            }
        )
    return rows


def partial_eta_squared(model: Any, term: str) -> float:
    if term not in model.params.index:
        return np.nan
    ss_terms = getattr(model, "ssr", np.nan)
    # Type II partial eta-squared via F-test on term
    try:
        f_test = model.f_test(f"{term} = 0")
        f_val = float(np.asarray(f_test.fvalue).ravel()[0])
        df_num = float(np.asarray(f_test.df_num).ravel()[0])
        df_den = float(model.df_resid)
        return float((f_val * df_num) / (f_val * df_num + df_den))
    except Exception:
        return np.nan


def cohen_f_squared(model: Any, term: str) -> float:
    eta2 = partial_eta_squared(model, term)
    if not np.isfinite(eta2) or eta2 >= 1:
        return np.nan
    return float(eta2 / (1 - eta2))


def residualize(y: pd.Series, cov_df: pd.DataFrame, formula_rhs: str) -> np.ndarray:
    data = cov_df.copy()
    data["_y"] = pd.to_numeric(y, errors="coerce")
    sub = data.dropna(subset=["_y"])
    if len(sub) < 8:
        return np.full(len(y), np.nan)
    model = ols(f"_y ~ {formula_rhs}", data=sub).fit()
    resid = pd.Series(np.nan, index=y.index, dtype=float)
    resid.loc[sub.index] = model.resid
    return resid.to_numpy(dtype=float)


def partial_spearman(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    cov_formula: str = "age_months + C(sex) + IQ_total + usable_epochs",
) -> dict[str, Any]:
    sub = df.dropna(subset=[x_col, y_col, "age_months", "sex", "IQ_total", "usable_epochs"]).copy()
    if len(sub) < 8:
        return {"partial_rho": np.nan, "pvalue": np.nan, "n": len(sub)}
    rx = residualize(sub[x_col], sub, cov_formula)
    ry = residualize(sub[y_col], sub, cov_formula)
    mask = np.isfinite(rx) & np.isfinite(ry)
    if mask.sum() < 8:
        return {"partial_rho": np.nan, "pvalue": np.nan, "n": int(mask.sum())}
    rho, p = stats.spearmanr(rx[mask], ry[mask])
    return {"partial_rho": float(rho), "pvalue": float(p), "n": int(mask.sum())}


def bootstrap_partial_spearman(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_boot: int = 1000,
    seed: int = 42,
    cov_formula: str = "age_months + C(sex) + IQ_total + usable_epochs",
) -> dict[str, Any]:
    base = partial_spearman(df, x_col, y_col, cov_formula)
    sub = df.dropna(subset=[x_col, y_col, "age_months", "sex", "IQ_total", "usable_epochs"]).copy()
    if len(sub) < 10:
        return {**base, "boot_ci_low": np.nan, "boot_ci_high": np.nan, "n_boot_valid": 0}
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    idx = np.arange(len(sub))
    for _ in range(n_boot):
        samp = sub.iloc[rng.choice(idx, size=len(idx), replace=True)].reset_index(drop=True)
        res = partial_spearman(samp, x_col, y_col, cov_formula)
        if np.isfinite(res["partial_rho"]):
            boots.append(res["partial_rho"])
    if not boots:
        return {**base, "boot_ci_low": np.nan, "boot_ci_high": np.nan, "n_boot_valid": 0}
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        **base,
        "boot_ci_low": float(lo),
        "boot_ci_high": float(hi),
        "boot_median": float(np.median(boots)),
        "n_boot_valid": len(boots),
    }


def leave_one_subject_sensitivity(
    df: pd.DataFrame,
    outcome: str,
    analysis: str = "group",
) -> pd.DataFrame:
    cov_cols = ["group", "age_months", "sex", "IQ_total", "usable_epochs"]
    if "mean_r_squared" in df.columns and _is_posterior_outcome(outcome):
        cov_cols.append("mean_r_squared")
    sub = df.dropna(subset=[outcome] + cov_cols)
    rows: list[dict[str, Any]] = []
    for sid in sub["subject_id"].unique():
        part = sub[sub["subject_id"] != sid]
        if analysis == "group":
            res = fit_group_effect(part, outcome, cohort_label=f"loo_{sid}")
        else:
            ints = fit_age_group_interaction(part, outcome, cohort_label=f"loo_{sid}")
            res = next((r for r in ints if ":age_months" in str(r.get("term", ""))), None)
        if res is None:
            continue
        rows.append({"dropped_subject_id": sid, **res})
    return pd.DataFrame(rows)


def cooks_distance_diagnostics(
    df: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    cov_cols = ["group", "age_months", "sex", "IQ_total", "usable_epochs"]
    if "mean_r_squared" in df.columns and "exponent" in outcome:
        cov_cols.append("mean_r_squared")
    sub = df.dropna(subset=[outcome] + cov_cols).copy()
    if len(sub) < 20:
        return pd.DataFrame()
    formula = _group_formula(outcome, df=sub)
    model = run_ols(formula, sub)
    influence = model.get_influence()
    cooks = influence.cooks_distance[0]
    out = sub[["subject_id", "group", outcome]].copy()
    out["cooks_d"] = cooks
    out["standardized_resid"] = influence.resid_studentized_internal
    threshold = 4 / len(sub)
    out["flag_influential"] = out["cooks_d"] > threshold
    return out.sort_values("cooks_d", ascending=False)


def leave_one_channel_out(
    ch_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    channels: list[str] | None = None,
) -> pd.DataFrame:
    channels = channels or list(POSTERIOR_CORE)
    rows: list[dict[str, Any]] = []
    for dropped in channels:
        remain = [c for c in channels if c != dropped]
        metrics = channel_mean(ch_df, remain).rename(columns={"value": "posterior_loo_exponent"})
        df = cohort_df.merge(
            metrics[["subject_id", "posterior_loo_exponent"]],
            on="subject_id",
            how="inner",
        )
        label = f"loo_drop_{dropped}"
        g = fit_group_effect(df, "posterior_loo_exponent", label)
        if g:
            g["dropped_channel"] = dropped
            g["remaining_channels"] = ",".join(remain)
            rows.append(g)
        for ir in fit_age_group_interaction(df, "posterior_loo_exponent", label):
            ir["dropped_channel"] = dropped
            ir["remaining_channels"] = ",".join(remain)
            rows.append(ir)
        asd = df[df["group"] == "ASD"].dropna(
            subset=["posterior_loo_exponent", "ADOS_total", "age_months", "sex", "IQ_total", "usable_epochs"]
        )
        if len(asd) >= 10 and "ADOS_total" in asd.columns:
            pc = bootstrap_partial_spearman(asd, "posterior_loo_exponent", "ADOS_total", n_boot=500)
            rows.append(
                {
                    "cohort": label,
                    "analysis": "ados_partial_corr",
                    "outcome": "posterior_loo_exponent",
                    "term": "ADOS_total",
                    "dropped_channel": dropped,
                    "partial_rho": pc["partial_rho"],
                    "p": pc["pvalue"],
                    "boot_ci_low": pc.get("boot_ci_low"),
                    "boot_ci_high": pc.get("boot_ci_high"),
                    "n_total": pc["n"],
                }
            )
    return pd.DataFrame(rows)


def load_iclabel_channel_df(cfg: dict[str, Any], threshold: float = 0.80) -> pd.DataFrame:
    deriv = Path(cfg["paths"]["derivatives_root"])
    tag = f"threshold_{threshold:.2f}".replace(".", "_")
    sp_dir = deriv / "specparam" / "iclabel_cleaned" / tag
    if not sp_dir.exists():
        raise FileNotFoundError(f"ICLabel specparam 目录不存在: {sp_dir}")
    parts = []
    for path in sorted(sp_dir.glob("*_specparam_channel.csv")):
        parts.append(pd.read_csv(path))
    if not parts:
        raise FileNotFoundError(f"未找到 ICLabel 通道文件: {sp_dir}")
    df = pd.concat(parts, ignore_index=True)
    df["subject_id"] = df["subject_id"].astype(str)
    df["fit_valid"] = df["r_squared"].ge(0.90) if "r_squared" in df.columns else True
    return df


def run_cohort_matrix(
    df: pd.DataFrame,
    outcomes: list[str],
    cohort_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        g = fit_group_effect(df, outcome, cohort_label)
        if g:
            rows.append(g)
        rows.extend(fit_age_group_interaction(df, outcome, cohort_label))
    return pd.DataFrame(rows)


def split_half_validation(
    ch_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    n_iter: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    subjects = cohort_df["subject_id"].astype(str).unique()
    rows: list[dict[str, Any]] = []
    for i in range(n_iter):
        mask = rng.random(len(subjects)) < 0.5
        if mask.sum() < 20 or (~mask).sum() < 20:
            continue
        disc_ids = set(subjects[mask])
        val_ids = set(subjects[~mask])
        disc_cohort = cohort_df[cohort_df["subject_id"].isin(disc_ids)]
        val_cohort = cohort_df[cohort_df["subject_id"].isin(val_ids)]
        disc_ch = ch_df[ch_df["subject_id"].isin(disc_ids)]
        # Discovery: FDR-like — pick channels with largest |t| for group in discovery
        channel_ts: list[tuple[str, float]] = []
        for ch in POSTERIOR_CORE:
            m = channel_mean(disc_ch, [ch]).rename(columns={"value": "exp"})
            merged = merge_posterior_metrics(disc_cohort, m.rename(columns={"exp": "posterior_core_4"}))
            merged = merged.rename(columns={"posterior_core_4": "exp"})
            sub = merged.dropna(subset=["exp", "group"])
            if len(sub) < 15:
                continue
            a = sub.loc[sub["group"] == "ASD", "exp"]
            b = sub.loc[sub["group"] == "TD", "exp"]
            if len(a) < 5 or len(b) < 5:
                continue
            t, _ = stats.ttest_ind(a, b, equal_var=False)
            channel_ts.append((ch, abs(float(t))))
        if not channel_ts:
            continue
        channel_ts.sort(key=lambda x: x[1], reverse=True)
        disc_roi = [c for c, _ in channel_ts[: max(1, len(channel_ts) // 2)]]
        val_metrics = channel_mean(ch_df[ch_df["subject_id"].isin(val_ids)], disc_roi).rename(
            columns={"value": "disc_defined_posterior"}
        )
        val_df = merge_posterior_metrics(val_cohort, val_metrics.rename(columns={"disc_defined_posterior": "posterior_core_4"}))
        val_df = val_df.rename(columns={"posterior_core_4": "disc_defined_posterior"})
        g = fit_group_effect(val_df, "disc_defined_posterior", f"split_{i}")
        int_rows = fit_age_group_interaction(val_df, "disc_defined_posterior", f"split_{i}")
        int_row = next((r for r in int_rows if ":age_months" in str(r.get("term", ""))), None)
        rows.append(
            {
                "iteration": i,
                "n_discovery": int(mask.sum()),
                "n_validation": int((~mask).sum()),
                "discovery_channels": ",".join(disc_roi),
                "group_beta": g["beta_TD_vs_ASD"] if g else np.nan,
                "group_p": g["p"] if g else np.nan,
                "group_direction_td_gt_asd": bool(g["beta_TD_vs_ASD"] > 0) if g else False,
                "interaction_coef": int_row["coef"] if int_row else np.nan,
                "interaction_p": int_row["p"] if int_row else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_split_half(split_df: pd.DataFrame) -> dict[str, Any]:
    if split_df.empty:
        return {}
    valid_g = split_df["group_p"].notna()
    valid_i = split_df["interaction_p"].notna()
    return {
        "n_iterations": len(split_df),
        "group_direction_consistency": float(split_df.loc[valid_g, "group_direction_td_gt_asd"].mean()),
        "group_p_lt_005_rate": float((split_df.loc[valid_g, "group_p"] < 0.05).mean()),
        "interaction_sign_consistency_asd_slope_lower": float(
            (split_df.loc[valid_i, "interaction_coef"] < 0).mean()
        ),
    }
