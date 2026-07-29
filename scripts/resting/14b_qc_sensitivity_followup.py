# Ideal release name: 14b_qc_sensitivity_followup.py
# Original path: scripts/17_qc_and_sensitivity_followup.py
# Note: QC follow-up sensitivities
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
17_qc_and_sensitivity_followup.py
---------------------------------
结果核查与稳健性补强：样本流失、主模型稳健性、描述统计、临床缺失、显著通道。

输出: outputs/tables/*.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import (  # noqa: E402
    attach_usable_epochs,
    exclude_specparam_low_quality,
    load_analysis_participants,
    save_csv,
)
from src.stats_utils import run_ols  # noqa: E402

logger = logging.getLogger(__name__)

GROUP_TERM = "C(group)[T.TD]"


def _count_by_group(df: pd.DataFrame, group_col: str = "group") -> dict[str, int]:
    if df.empty or group_col not in df.columns:
        return {"n_total": len(df), "n_ASD": 0, "n_TD": 0}
    return {
        "n_total": len(df),
        "n_ASD": int((df[group_col] == "ASD").sum()),
        "n_TD": int((df[group_col] == "TD").sum()),
    }


def _row_stage(stage: str, df: pd.DataFrame, exclusion_reason: str = "") -> dict[str, Any]:
    c = _count_by_group(df)
    return {
        "stage": stage,
        "n_total": c["n_total"],
        "n_ASD": c["n_ASD"],
        "n_TD": c["n_TD"],
        "excluded_total": np.nan,
        "exclusion_reason": exclusion_reason,
    }


def build_sample_inclusion_flow(
    participants: pd.DataFrame,
    preproc: pd.DataFrame | None,
    analysis_df: pd.DataFrame | None,
    sp_qc: pd.DataFrame | None,
    roi_global: pd.DataFrame | None,
    min_epochs: int,
    deriv: Path,
) -> pd.DataFrame:
    """构建样本纳入各阶段计数。"""
    rows: list[dict[str, Any]] = []

    all_p = participants.copy()
    rows.append(_row_stage("participants_total", all_p, ""))

    if "included_final" in all_p.columns:
        inc = all_p[all_p["included_final"].astype(int) == 1].copy()
    else:
        inc = all_p.copy()
    rows.append(_row_stage("included_final", inc, "not included_final=1"))

    if preproc is not None and not preproc.empty:
        pre = preproc.copy()
        pre["subject_id"] = pre["subject_id"].astype(str)
        if "status" in pre.columns:
            ok = pre[pre["status"].astype(str).str.lower().eq("ok")].copy()
        else:
            ok = pre[pre["usable_epochs"].fillna(0) > 0].copy()
        rows.append(_row_stage("preprocessing_success", ok, "preprocessing failed or no epochs"))
    else:
        logger.warning("无 preproc_summary，跳过 preprocessing_success 阶段")

    if preproc is not None:
        ep = preproc[preproc["usable_epochs"].fillna(0) >= min_epochs].copy()
        rows.append(
            _row_stage(
                "min_usable_epochs_pass",
                ep,
                f"usable_epochs < {min_epochs}",
            )
        )

    if sp_qc is not None and not sp_qc.empty:
        sp_qc = sp_qc.copy()
        sp_qc["subject_id"] = sp_qc["subject_id"].astype(str)
        pass_qc = sp_qc[sp_qc["low_quality_subject"] == 0].copy()
        rows.append(
            _row_stage(
                "specparam_subject_qc_pass",
                pass_qc,
                "low_quality_subject (invalid channel ratio > threshold)",
            )
        )

    if roi_global is not None and sp_qc is not None:
        roi_ids = set(roi_global["subject_id"].astype(str))
        good_ids = set(
            sp_qc.loc[sp_qc["low_quality_subject"] == 0, "subject_id"].astype(str)
        )
        if analysis_df is not None:
            base = analysis_df[analysis_df["subject_id"].astype(str).isin(roi_ids & good_ids)]
        elif preproc is not None:
            base = preproc[
                preproc["subject_id"].astype(str).isin(roi_ids & good_ids)
            ]
        else:
            base = pd.DataFrame()
        rows.append(_row_stage("roi_available_after_specparam", base, ""))

    # 主分析完整病例（与 08 一致）
    if roi_global is not None and analysis_df is not None:
        main_df = prepare_main_analysis_df(analysis_df, roi_global, deriv, sp_qc)
        need = ["global_exponent", "group", "age_months", "sex", "IQ_total", "usable_epochs"]
        complete = main_df.dropna(subset=[c for c in need if c in main_df.columns])
        rows.append(
            _row_stage(
                "main_analysis_complete_case",
                complete,
                "missing outcome/covariates or specparam QC fail",
            )
        )

    out = pd.DataFrame(rows)
    if len(out) >= 2:
        for i in range(1, len(out)):
            prev_n = out.loc[i - 1, "n_total"]
            cur_n = out.loc[i, "n_total"]
            if pd.notna(prev_n):
                out.loc[i, "excluded_total"] = int(prev_n - cur_n)
    return out


def prepare_main_analysis_df(
    participants: pd.DataFrame,
    roi_global: pd.DataFrame,
    deriv: Path,
    sp_qc: pd.DataFrame | None,
) -> pd.DataFrame:
    """与 08 主分析一致的分析表，并附加 QC 指标。"""
    df = participants.merge(roi_global, on=["subject_id", "group"], how="inner")
    df = attach_usable_epochs(df, deriv)
    df = exclude_specparam_low_quality(df, deriv)

    preproc = deriv / "qc" / "preproc_summary.csv"
    if preproc.exists():
        pre = pd.read_csv(preproc)
        pre["subject_id"] = pre["subject_id"].astype(str)
        extra = ["bad_channel_count"]
        if sp_qc is not None:
            sp = sp_qc[["subject_id", "mean_r_squared"]].copy()
            sp["subject_id"] = sp["subject_id"].astype(str)
            df = df.merge(sp, on="subject_id", how="left")
        df = df.merge(
            pre[["subject_id"] + [c for c in extra if c in pre.columns]],
            on="subject_id",
            how="left",
        )
    return df


def fit_group_model(
    df: pd.DataFrame,
    formula: str,
    model_name: str,
) -> dict[str, Any]:
    """拟合 OLS 并提取 TD vs ASD 的 group 系数。"""
    covariates = formula.split("~", 1)[1].strip()
    row: dict[str, Any] = {
        "model_name": model_name,
        "covariates": covariates,
        "n": 0,
        "n_ASD": 0,
        "n_TD": 0,
        "group_coef_TD_vs_ASD": np.nan,
        "group_se": np.nan,
        "group_t": np.nan,
        "group_p": np.nan,
        "group_ci_low": np.nan,
        "group_ci_high": np.nan,
        "r_squared": np.nan,
    }
    if "group" not in df.columns:
        return row

    try:
        res = run_ols(formula, df)
    except Exception as exc:
        logger.warning("模型 %s 拟合失败: %s", model_name, exc)
        return row

    if int(res.nobs) < 10:
        return row

    used = df.loc[res.fittedvalues.index]
    counts = _count_by_group(used)
    row.update({"n": counts["n_total"], "n_ASD": counts["n_ASD"], "n_TD": counts["n_TD"]})

    if GROUP_TERM not in res.params.index:
        return row

    conf = res.conf_int()
    row.update({
        "group_coef_TD_vs_ASD": float(res.params[GROUP_TERM]),
        "group_se": float(res.bse[GROUP_TERM]),
        "group_t": float(res.tvalues[GROUP_TERM]),
        "group_p": float(res.pvalues[GROUP_TERM]),
        "group_ci_low": float(conf.loc[GROUP_TERM, 0]),
        "group_ci_high": float(conf.loc[GROUP_TERM, 1]),
        "r_squared": float(getattr(res, "rsquared", np.nan)),
    })
    return row


def build_robustness_models(df: pd.DataFrame) -> pd.DataFrame:
    """主结果稳健性：逐步加协变量 + 排除低 IQ。"""
    y = "global_exponent"
    models = [
        ("model_1", f"{y} ~ C(group)"),
        ("model_2", f"{y} ~ C(group) + age_months + C(sex)"),
        ("model_3", f"{y} ~ C(group) + age_months + C(sex) + IQ_total"),
        (
            "model_4",
            f"{y} ~ C(group) + age_months + C(sex) + IQ_total + usable_epochs",
        ),
        (
            "model_5",
            f"{y} ~ C(group) + age_months + C(sex) + IQ_total + usable_epochs + mean_r_squared",
        ),
        (
            "model_6",
            f"{y} ~ C(group) + age_months + C(sex) + IQ_total + usable_epochs + bad_channel_count",
        ),
    ]
    rows = []
    for name, formula in models:
        rows.append(fit_group_model(df, formula, name))

    if "IQ_total" in df.columns:
        low_iq = df[pd.to_numeric(df["IQ_total"], errors="coerce") >= 70].copy()
        formula4 = (
            f"{y} ~ C(group) + age_months + C(sex) + IQ_total + usable_epochs"
        )
        rows.append(
            fit_group_model(
                low_iq,
                formula4,
                "model_7_exclude_IQ_below_70",
            )
        )
    return pd.DataFrame(rows)


def build_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """按组描述 global_exponent。"""
    rows = []
    for grp, sub in df.groupby("group"):
        x = pd.to_numeric(sub["global_exponent"], errors="coerce").dropna()
        if x.empty:
            continue
        q25, q75 = x.quantile(0.25), x.quantile(0.75)
        rows.append({
            "group": grp,
            "n": len(x),
            "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
            "median": float(x.median()),
            "iqr": float(q75 - q25),
            "min": float(x.min()),
            "max": float(x.max()),
        })
    return pd.DataFrame(rows)


def build_clinical_missingness(participants: pd.DataFrame) -> pd.DataFrame:
    """临床变量缺失率（按组）。"""
    vars_ = [
        "ADOS_total", "ADOS_SA", "ADOS_RRB",
        "SRS_total", "CARS_total", "language_score",
    ]
    rows = []
    for grp in sorted(participants["group"].dropna().unique()):
        sub = participants[participants["group"] == grp]
        n = len(sub)
        for var in vars_:
            if var not in sub.columns:
                continue
            nonmiss = pd.to_numeric(sub[var], errors="coerce").notna().sum()
            rows.append({
                "group": grp,
                "variable": var,
                "n_nonmissing": int(nonmiss),
                "n_missing": int(n - nonmiss),
                "percent_missing": round(100 * (n - nonmiss) / n, 2) if n else np.nan,
            })
    return pd.DataFrame(rows)


def build_significant_channels_fdr(
    channel_stats: pd.DataFrame,
    montage_name: str = "GSN-HydroCel-64_1.0",
) -> pd.DataFrame:
    """FDR 显著通道及 EGI 坐标（若可用）。"""
    p_col = "pvalue_fdr" if "pvalue_fdr" in channel_stats.columns else "fdr_p"
    if p_col not in channel_stats.columns:
        if "significant_fdr" in channel_stats.columns:
            sig = channel_stats[channel_stats["significant_fdr"].astype(bool)].copy()
        else:
            logger.warning("channel_level_analysis 无 FDR 列")
            return pd.DataFrame()
    else:
        sig = channel_stats[pd.to_numeric(channel_stats[p_col], errors="coerce") < 0.05].copy()

    if sig.empty:
        return pd.DataFrame()

    out = sig.copy()
    out["direction"] = np.where(
        pd.to_numeric(out["coef"], errors="coerce") > 0,
        "TD > ASD",
        "TD < ASD",
    )

    positions: dict[str, tuple[float, float, float]] = {}
    try:
        import mne
        montage = mne.channels.make_standard_montage(montage_name)
        positions = {
            name: montage.get_positions()["ch_pos"][name]
            for name in montage.ch_names
            if name in montage.get_positions()["ch_pos"]
        }
    except Exception as exc:
        logger.warning("无法加载 montage %s: %s", montage_name, exc)

    xs, ys, zs, notes = [], [], [], []
    for ch in out["channel"].astype(str):
        if ch in positions:
            x, y, z = positions[ch]
            xs.append(x)
            ys.append(y)
            zs.append(z)
            notes.append("MNE montage")
        else:
            xs.append(np.nan)
            ys.append(np.nan)
            zs.append(np.nan)
            notes.append("verify on HydroCel-64 layout map")
    out["pos_x"] = xs
    out["pos_y"] = ys
    out["pos_z"] = zs
    out["position_note"] = notes
    return out


def build_specparam_sensitivity_table(
    participants: pd.DataFrame,
    deriv: Path,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """从 14 生成的 sens_* 通道结果汇总 global exponent 组效应。"""
    sens_dir = deriv / "specparam"
    sens_files = sorted(sens_dir.glob("sens_freq_*.csv"))
    if not sens_files:
        logger.warning("未找到 sens_freq_*.csv，跳过 specparam 条件敏感性表")
        return pd.DataFrame()

    formula = (
        "global_exponent ~ C(group) + age_months + C(sex) + IQ_total + usable_epochs"
    )
    rows = []
    for path in sens_files:
        label = path.stem.replace("sens_", "")
        ch = pd.read_csv(path)
        if "aperiodic_exponent" not in ch.columns:
            continue
        subj = (
            ch.groupby(["subject_id", "group"], as_index=False)["aperiodic_exponent"]
            .mean()
            .rename(columns={"aperiodic_exponent": "global_exponent"})
        )
        df = participants.merge(subj, on=["subject_id", "group"], how="inner")
        df = attach_usable_epochs(df, deriv)
        fit = fit_group_model(df, formula, label)
        desc = df.groupby("group")["global_exponent"].agg(["mean", "count"])
        fit["analysis"] = label
        fit["ASD_mean"] = desc.loc["ASD", "mean"] if "ASD" in desc.index else np.nan
        fit["TD_mean"] = desc.loc["TD", "mean"] if "TD" in desc.index else np.nan
        rows.append(fit)

    return pd.DataFrame(rows)


def build_clinical_model_check(
    participants: pd.DataFrame,
    roi_global: pd.DataFrame,
    deriv: Path,
    resting_info_path: Path,
) -> pd.DataFrame:
    """核查临床变量来源、含义及主分析一致样本量。"""
    analysis_parts = participants.copy()
    if "included_final" in analysis_parts.columns:
        analysis_parts = analysis_parts[
            analysis_parts["included_final"].astype(int) == 1
        ]

    asd_full = analysis_parts[analysis_parts["group"] == "ASD"]
    asd_main = exclude_specparam_low_quality(
        asd_full.merge(roi_global, on=["subject_id", "group"], how="inner"),
        deriv,
    )
    asd_main = attach_usable_epochs(asd_main, deriv)

    resting_cols = {}
    if resting_info_path.exists():
        asd_info = pd.read_excel(resting_info_path, header=103)
        for c in ["ADOS-2", "SA", "RRB"]:
            if c in asd_info.columns:
                resting_cols[c] = c

    var_meta = [
        ("ADOS_total", "ADOS-2", "ADOS_total", "ADOS-2 总分"),
        ("ADOS_SA", "SA", "ADOS_SA", "ADOS Social Affect (SA) 子分"),
        ("ADOS_RRB", "RRB", "ADOS_RRB", "ADOS Restricted/Repetitive Behavior (RRB) 子分"),
        ("SRS_total", "SRS", "SRS_total", "SRS 总分"),
        ("language_score", "ToMI_Total", "language_score", "ToMI 总分"),
    ]

    rows = []
    for out_col, info_col, part_col, meaning in var_meta:
        n_full = (
            pd.to_numeric(asd_full[out_col], errors="coerce").notna().sum()
            if out_col in asd_full.columns
            else 0
        )
        n_main = (
            pd.to_numeric(asd_main[out_col], errors="coerce").notna().sum()
            if out_col in asd_main.columns
            else 0
        )
        rows.append({
            "clinical_variable": out_col,
            "resting_info_column": info_col if info_col in resting_cols or info_col == "ToMI_Total" else "",
            "participants_column": part_col,
            "intended_construct": meaning,
            "n_asd_epoch_cohort": len(asd_full),
            "n_nonmissing_asd_epoch_cohort": n_full,
            "n_asd_main_qc_cohort": len(asd_main),
            "n_nonmissing_asd_main_qc": n_main,
        })

    cov = ["global_exponent", "age_months", "sex", "IQ_total", "usable_epochs"]
    for outcome, predictor, _formula in [
        ("ADOS_total", "global_exponent", ""),
        ("ADOS_SA", "global_exponent", ""),
        ("ADOS_RRB", "global_exponent", ""),
        ("language_score", "temporal_exponent", ""),
    ]:
        if outcome not in asd_main.columns:
            continue
        sub = asd_main.dropna(subset=[outcome, predictor] + [c for c in cov if c in asd_main.columns])
        rows.append({
            "clinical_variable": f"MODEL:{outcome}",
            "resting_info_column": "",
            "participants_column": outcome,
            "intended_construct": f"OLS 完整病例（与主分析 QC 一致）",
            "n_asd_epoch_cohort": len(asd_full),
            "n_nonmissing_asd_epoch_cohort": np.nan,
            "n_asd_main_qc_cohort": len(asd_main),
            "n_nonmissing_asd_main_qc": len(sub),
        })

    out = pd.DataFrame(rows)
    return out


def _load_optional(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning("未找到 %s: %s", label, path)
        return None
    return pd.read_csv(path)


def main() -> None:
    cfg = load_config()
    log = setup_logging(cfg, name="qc_sensitivity_followup")
    deriv = Path(cfg["paths"]["derivatives_root"])
    tables_dir = Path(cfg["paths"]["outputs_root"]) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    min_epochs = int(cfg.get("epochs", {}).get("min_usable_epochs", 60))
    montage = cfg.get("eeg", {}).get("montage", "GSN-HydroCel-64_1.0")

    part_path = Path(cfg["paths"]["participants_file"])
    if not part_path.exists():
        log.error("未找到 participants.csv")
        sys.exit(1)
    participants = pd.read_csv(part_path)
    participants["subject_id"] = participants["subject_id"].astype(str)

    preproc = _load_optional(deriv / "qc" / "preproc_summary.csv", "preproc_summary")
    analysis_df = _load_optional(deriv / "participants_analysis.csv", "participants_analysis")
    sp_qc = _load_optional(
        deriv / "specparam" / "specparam_qc_summary_subject.csv",
        "specparam_qc_summary",
    )
    roi_global = _load_optional(
        deriv / "roi" / "specparam_subject_global.csv",
        "roi global",
    )
    channel_stats = _load_optional(
        deriv / "stats" / "channel_level_analysis.csv",
        "channel_level_analysis",
    )

    # 1. 样本流失
    flow = build_sample_inclusion_flow(
        participants, preproc, analysis_df, sp_qc, roi_global, min_epochs, deriv,
    )
    save_csv(flow, tables_dir / "sample_inclusion_flow.csv")

    if roi_global is None:
        log.error("需要 specparam_subject_global.csv，请先运行 06")
        sys.exit(1)

    analysis_parts = load_analysis_participants(cfg)
    main_df = prepare_main_analysis_df(analysis_parts, roi_global, deriv, sp_qc)

    # 2. 稳健性模型
    robust = build_robustness_models(main_df)
    save_csv(robust, tables_dir / "global_exponent_robustness_models.csv")

    # 3. 描述统计
    complete = main_df.dropna(
        subset=["global_exponent", "group", "age_months", "sex", "IQ_total", "usable_epochs"],
    )
    save_csv(
        build_descriptives(complete),
        tables_dir / "global_exponent_descriptives.csv",
    )

    # 4. 临床缺失
    save_csv(
        build_clinical_missingness(participants),
        tables_dir / "clinical_variable_missingness.csv",
    )

    # 5. FDR 显著通道
    if channel_stats is not None:
        sig_ch = build_significant_channels_fdr(channel_stats, montage)
        save_csv(sig_ch, tables_dir / "significant_channels_fdr.csv")
        if not sig_ch.empty and sig_ch["pos_x"].isna().all():
            log.info("显著通道坐标需对照 HydroCel-64 布局图人工确认")

    # 6. specparam 条件敏感性（若 14 已生成 sens_*.csv）
    sens_final = build_specparam_sensitivity_table(analysis_parts, deriv, cfg)
    if not sens_final.empty:
        save_csv(sens_final, tables_dir / "sensitivity_analysis_final.csv")

    # 7. 临床变量与模型 n 核查
    resting = PROJECT_ROOT / "data" / "participants" / "Resting_info.xlsx"
    clin_check = build_clinical_model_check(participants, roi_global, deriv, resting)
    save_csv(clin_check, tables_dir / "clinical_model_n_and_variable_check.csv")

    log.info("结果核查与稳健性表已写入 %s", tables_dir)


if __name__ == "__main__":
    main()
