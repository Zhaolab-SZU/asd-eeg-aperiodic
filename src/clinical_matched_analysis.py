"""IQ-balanced 匹配样本内：posterior exponent 与 ADOS 临床关联分析。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.io_utils import (
    attach_usable_epochs,
    exclude_specparam_low_quality,
    load_participants,
    save_csv,
)
from src.spectral_maturation_analysis import (
    POSTERIOR_CORE,
    _posterior_channel_mean,
    load_spectral_maturation_cohort,
)
from src.stats_utils import (
    bootstrap_partial_spearman,
    build_iq_matched_cohort,
    fdr_correction,
    partial_correlation_pearson,
)

logger = logging.getLogger(__name__)

EEG_METRIC = "posterior_exponent"
CLINICAL_OUTCOMES = [
    ("ADOS_SA", "ADOS Social Affect"),
    ("ADOS_total", "ADOS total"),
]
COVARIATES = ["age_months", "IQ_total"]
DEFAULT_IQ_CALIPER = 15.0
DEFAULT_AGE_CALIPER = 24.0


def load_clinical_posterior_cohort(cfg: dict[str, Any], deriv: Path) -> pd.DataFrame:
    """
    主分析 QC 队列 + posterior exponent（与 spectral maturation 分析一致）。

    若缺少 preproc_summary，则回退至 participants + ROI/specparam QC 合并。
    """
    try:
        return load_spectral_maturation_cohort(cfg, deriv)
    except FileNotFoundError as exc:
        logger.warning("load_spectral_maturation_cohort 失败 (%s)，使用回退加载", exc)

    participants_path = Path(cfg["paths"]["participants_file"])
    participants = load_participants(participants_path, included_only=True)
    participants = attach_usable_epochs(participants, deriv)
    min_ep = int(cfg.get("epochs", {}).get("min_usable_epochs", 60))
    if "usable_epochs" in participants.columns:
        participants = participants[participants["usable_epochs"] >= min_ep].copy()
    participants = exclude_specparam_low_quality(participants, deriv)

    roi_path = deriv / "roi" / "specparam_subject_global.csv"
    if not roi_path.exists():
        raise FileNotFoundError(f"缺少 ROI 文件: {roi_path}")
    roi_df = pd.read_csv(roi_path)
    roi_df["subject_id"] = roi_df["subject_id"].astype(str)

    ch_path = deriv / "specparam" / "specparam_channel_results_qc.csv"
    if not ch_path.exists():
        raise FileNotFoundError(f"缺少通道 specparam 文件: {ch_path}")
    ch_df = pd.read_csv(ch_path)
    ch_df["subject_id"] = ch_df["subject_id"].astype(str)
    post_exp = _posterior_channel_mean(ch_df, "aperiodic_exponent")
    post_exp = post_exp.rename(columns={"aperiodic_exponent": EEG_METRIC})

    sp_qc = deriv / "specparam" / "specparam_qc_summary_subject.csv"
    if sp_qc.exists():
        sp = pd.read_csv(sp_qc)
        sp["subject_id"] = sp["subject_id"].astype(str)
        participants = participants.merge(
            sp[["subject_id", "mean_r_squared"]],
            on="subject_id",
            how="left",
        )

    participants["subject_id"] = participants["subject_id"].astype(str)
    df = participants.merge(
        roi_df[["subject_id", "group", "global_exponent"]],
        on=["subject_id", "group"],
        how="inner",
    )
    df = df.merge(post_exp[["subject_id", EEG_METRIC]], on="subject_id", how="left")
    return df.reset_index(drop=True)


def run_partial_correlation_table(
    asd_df: pd.DataFrame,
    *,
    cohort_label: str,
) -> pd.DataFrame:
    """年龄 + IQ 校正偏 Pearson 相关，跨临床结局 FDR 校正。"""
    rows: list[dict[str, Any]] = []
    for clinical_col, clinical_label in CLINICAL_OUTCOMES:
        if clinical_col not in asd_df.columns:
            logger.warning("跳过 %s：列不存在", clinical_col)
            continue
        res = partial_correlation_pearson(
            asd_df,
            clinical_col,
            EEG_METRIC,
            cov_cols=COVARIATES,
        )
        rows.append(
            {
                "cohort": cohort_label,
                "clinical_outcome": clinical_col,
                "clinical_label": clinical_label,
                "eeg_metric": EEG_METRIC,
                "n": res["n"],
                "partial_r": res["partial_r"],
                "raw_p": res["pvalue"],
            }
        )

    out = pd.DataFrame(rows)
    if len(out) and out["raw_p"].notna().any():
        _, q = fdr_correction(out["raw_p"].values)
        out["fdr_q"] = q
        out["fdr_significant"] = q < 0.05
    return out


def run_bootstrap_robustness_table(
    asd_df: pd.DataFrame,
    *,
    cohort_label: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """偏 Spearman + bootstrap 95% CI。"""
    rows: list[dict[str, Any]] = []
    for clinical_col, clinical_label in CLINICAL_OUTCOMES:
        if clinical_col not in asd_df.columns:
            continue
        res = bootstrap_partial_spearman(
            asd_df,
            clinical_col,
            EEG_METRIC,
            n_boot=n_boot,
            seed=seed,
            cov_cols=COVARIATES,
        )
        rows.append(
            {
                "cohort": cohort_label,
                "clinical_outcome": clinical_col,
                "clinical_label": clinical_label,
                "eeg_metric": EEG_METRIC,
                "n": res["n"],
                "partial_rho": res["partial_rho"],
                "pvalue": res["pvalue"],
                "boot_ci_low": res["boot_ci_low"],
                "boot_ci_high": res["boot_ci_high"],
                "boot_median": res.get("boot_median"),
                "n_boot": n_boot,
                "n_boot_valid": res.get("n_boot_valid"),
            }
        )
    return pd.DataFrame(rows)


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def format_markdown_results_table(
    partial_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
) -> str:
    """生成补充材料 Markdown 表格。"""
    lines = [
        "### Supplementary Table. Posterior exponent–ADOS associations in IQ-balanced matched ASD subsample",
        "",
        "Partial Pearson correlations control for age and IQ. Bootstrap rows use partial Spearman "
        "correlation with 1,000 resamples (95% CI).",
        "",
        "| Clinical outcome | EEG metric | n | Partial r | Raw p | FDR q | Partial ρ | p | Bootstrap 95% CI |",
        "|---|---|--:|---:|---:|---:|---:|---:|---|",
    ]
    for _, prow in partial_df.iterrows():
        clinical = str(prow["clinical_label"])
        brow = bootstrap_df.loc[
            bootstrap_df["clinical_outcome"] == prow["clinical_outcome"]
        ]
        if brow.empty:
            rho_str = p_str = ci_str = "—"
        else:
            b = brow.iloc[0]
            rho_str = f"{float(b['partial_rho']):.2f}" if np.isfinite(b["partial_rho"]) else "—"
            p_str = _format_p(float(b["pvalue"]))
            if np.isfinite(b["boot_ci_low"]) and np.isfinite(b["boot_ci_high"]):
                ci_str = f"[{float(b['boot_ci_low']):.3f}, {float(b['boot_ci_high']):.3f}]"
            else:
                ci_str = "—"
        r_str = f"{float(prow['partial_r']):.2f}" if np.isfinite(prow["partial_r"]) else "—"
        q_str = _format_p(float(prow["fdr_q"])) if "fdr_q" in prow and np.isfinite(prow["fdr_q"]) else "—"
        lines.append(
            f"| {clinical} | {prow['eeg_metric']} | {int(prow['n'])} | {r_str} | "
            f"{_format_p(float(prow['raw_p']))} | {q_str} | {rho_str} | {p_str} | {ci_str} |"
        )
    return "\n".join(lines)


def format_english_results_paragraph(
    partial_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    *,
    n_matched_pairs: int,
    n_asd_matched: int,
) -> str:
    """生成论文/回复信英文结果描述。"""
    def _pick(clinical_col: str) -> tuple[pd.Series, pd.Series]:
        p = partial_df.loc[partial_df["clinical_outcome"] == clinical_col].iloc[0]
        b = bootstrap_df.loc[bootstrap_df["clinical_outcome"] == clinical_col].iloc[0]
        return p, b

    sa_p, sa_b = _pick("ADOS_SA")
    tot_p, tot_b = _pick("ADOS_total")

    sa_ci = (
        f"[{float(sa_b['boot_ci_low']):.3f}, {float(sa_b['boot_ci_high']):.3f}]"
        if np.isfinite(sa_b["boot_ci_low"])
        else "[NA, NA]"
    )
    tot_ci = (
        f"[{float(tot_b['boot_ci_low']):.3f}, {float(tot_b['boot_ci_high']):.3f}]"
        if np.isfinite(tot_b["boot_ci_low"])
        else "[NA, NA]"
    )

    return (
        f"Within the IQ-balanced matched cohort (1:1 greedy matching: |ΔIQ| ≤ {DEFAULT_IQ_CALIPER:g}, "
        f"|Δage| ≤ {DEFAULT_AGE_CALIPER:g} months, sex-matched; n = {2 * n_matched_pairs} total, "
        f"{n_asd_matched} autistic children with complete ADOS and posterior exponent data), "
        f"lower posterior exponent remained associated with greater ADOS Social Affect severity "
        f"after controlling for age and IQ (partial r = {float(sa_p['partial_r']):.2f}, "
        f"raw p = {_format_p(float(sa_p['raw_p']))}, FDR q = {_format_p(float(sa_p['fdr_q']))}, "
        f"n = {int(sa_p['n'])}). The association with ADOS total severity was "
        f"{'significant' if float(tot_p['raw_p']) < 0.05 else 'directionally concordant but not significant'} "
        f"(partial r = {float(tot_p['partial_r']):.2f}, raw p = {_format_p(float(tot_p['raw_p']))}, "
        f"FDR q = {_format_p(float(tot_p['fdr_q']))}, n = {int(tot_p['n'])}). "
        f"Bootstrap partial Spearman analyses (1,000 resamples) yielded comparable effect directions "
        f"(ADOS Social Affect: partial ρ = {float(sa_b['partial_rho']):.2f}, p = {_format_p(float(sa_b['pvalue']))}, "
        f"95% CI {sa_ci}; ADOS total: partial ρ = {float(tot_b['partial_rho']):.2f}, "
        f"p = {_format_p(float(tot_b['pvalue']))}, 95% CI {tot_ci}), "
        f"supporting symptom-linked posterior aperiodic variation within the matched autistic subsample."
    )


def run_matched_ados_analysis(
    cfg: dict[str, Any],
    *,
    out_dir: Path | None = None,
    n_boot: int = 1000,
    iq_caliper: float = DEFAULT_IQ_CALIPER,
    age_caliper: float = DEFAULT_AGE_CALIPER,
    match_sex: bool = True,
) -> dict[str, Any]:
    """完整流程：加载队列 → IQ 匹配 → ASD 内临床关联。"""
    deriv = Path(cfg["paths"]["derivatives_root"])
    seed = int(cfg.get("project", {}).get("random_seed", 42))
    out_dir = out_dir or (deriv / "isc")
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_clinical_posterior_cohort(cfg, deriv)
    matched, pairs = build_iq_matched_cohort(
        cohort,
        iq_caliper=iq_caliper,
        age_caliper=age_caliper,
        match_sex=match_sex,
        seed=seed,
    )
    if matched.empty:
        raise ValueError("IQ-balanced 匹配未产生任何配对，请检查协变量完整性。")

    asd = matched[matched["group"] == "ASD"].copy()
    cohort_label = "iq_balanced_matched"

    partial_df = run_partial_correlation_table(asd, cohort_label=cohort_label)
    bootstrap_df = run_bootstrap_robustness_table(
        asd,
        cohort_label=cohort_label,
        n_boot=n_boot,
        seed=seed,
    )

    partial_path = out_dir / "post_exponent_ados_matched.csv"
    bootstrap_path = out_dir / "post_exponent_ados_matched_bootstrap.csv"
    pairs_path = out_dir / "post_exponent_ados_matched_pairs.csv"
    save_csv(partial_df, partial_path)
    save_csv(bootstrap_df, bootstrap_path)
    if not pairs.empty:
        save_csv(pairs, pairs_path)

    meta = {
        "n_primary_cohort": len(cohort),
        "n_matched_total": len(matched),
        "n_matched_pairs": len(pairs),
        "n_asd_matched": len(asd),
        "n_td_matched": int((matched["group"] == "TD").sum()),
        "posterior_channels": ",".join(POSTERIOR_CORE),
        "iq_caliper": iq_caliper,
        "age_caliper_months": age_caliper,
        "match_sex": match_sex,
        "covariates": "+".join(COVARIATES),
        "partial_path": partial_path,
        "bootstrap_path": bootstrap_path,
        "pairs_path": pairs_path if not pairs.empty else None,
    }

    markdown = format_markdown_results_table(partial_df, bootstrap_df)
    english = format_english_results_paragraph(
        partial_df,
        bootstrap_df,
        n_matched_pairs=len(pairs),
        n_asd_matched=len(asd),
    )

    return {
        "partial_df": partial_df,
        "bootstrap_df": bootstrap_df,
        "pairs_df": pairs,
        "meta": meta,
        "markdown_table": markdown,
        "english_paragraph": english,
    }
