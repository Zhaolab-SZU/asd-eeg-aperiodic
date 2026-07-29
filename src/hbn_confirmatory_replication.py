"""HBN confirmatory replication of main-study resting aperiodic exponent findings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_roi_config
from src.hbn_external import resolve_hbn_paths
from src.io_utils import ensure_dir, save_csv
from src.roi_utils import aggregate_roi_for_subject, get_roi_dict
from src.stats_utils import descriptive_table, model_results_to_row, run_ols

logger = logging.getLogger(__name__)

COV = "age_months + C(sex) + IQ_total + usable_epochs"
PRIMARY_OUTCOMES = (
    "posterior_homologous_exponent",
    "posterior_y18_exponent",
    "global_exponent",
    "global_offset",
)


def _load_channel_qc(cfg: dict[str, Any]) -> pd.DataFrame:
    paths = resolve_hbn_paths(cfg)
    deriv = paths["derivatives_root"]
    ch_path = deriv / "specparam" / "specparam_channel_results_qc.csv"
    if not ch_path.exists():
        raise FileNotFoundError(f"未找到 {ch_path}，请先运行 102_hbn_specparam.py")
    channel_df = pd.read_csv(ch_path)
    if "fit_valid" in channel_df.columns:
        channel_df = channel_df[channel_df["fit_valid"].astype(bool)].copy()
    return channel_df


def _load_participants_qc(cfg: dict[str, Any]) -> pd.DataFrame:
    paths = resolve_hbn_paths(cfg)
    deriv = paths["derivatives_root"]
    participants = pd.read_csv(deriv / "participants_analysis.csv")
    sp_qc = deriv / "specparam" / "specparam_qc_summary_subject.csv"
    if sp_qc.exists():
        bad = pd.read_csv(sp_qc).loc[
            lambda d: d["low_quality_subject"] == 1, "subject_id"
        ].astype(str)
        participants = participants[~participants["subject_id"].astype(str).isin(bad)]
    participants["subject_id"] = participants["subject_id"].astype(str)
    return participants


def build_subject_replication_table(cfg: dict[str, Any]) -> pd.DataFrame:
    """通道级 specparam → 被试级 global / homologous posterior / y18 posterior。"""
    rep_cfg = cfg.get("hbn", {}).get("replication", {})
    roi_path = PROJECT_ROOT / cfg.get("hbn", {}).get("roi_config", "config/roi_channels_hbn129.yaml")
    roi_cfg = load_roi_config(roi_path)
    layout = cfg.get("hbn", {}).get("roi_layout", "channels_hbn129")
    roi_dict = get_roi_dict(roi_cfg, layout)
    min_ratio = float(roi_cfg.get("min_valid_channel_ratio", 0.5))

    channel_df = _load_channel_qc(cfg)
    participants = _load_participants_qc(cfg)
    channel_df["subject_id"] = channel_df["subject_id"].astype(str)

    rows: list[dict[str, Any]] = []
    for sid, sub_ch in channel_df.groupby("subject_id"):
        if sid not in set(participants["subject_id"]):
            continue
        rec, _ = aggregate_roi_for_subject(
            sub_ch.assign(subject_id=sid, group=sub_ch["group"].iloc[0]),
            roi_dict,
            min_ratio=min_ratio,
        )
        if "homologous_four_exponent" in rec:
            rec["posterior_homologous_exponent"] = rec["homologous_four_exponent"]
            rec["posterior_homologous_offset"] = rec.get("homologous_four_offset", np.nan)
        if "posterior_y18_exponent" in rec:
            rec["posterior_y18_offset"] = rec.get("posterior_y18_offset", np.nan)
        rows.append(rec)

    roi_subjects = pd.DataFrame(rows)
    meta_cols = [
        "subject_id", "group", "age_months", "sex", "IQ_total", "usable_epochs",
        "release_id", "group_hbn_raw",
    ]
    meta_cols = [c for c in meta_cols if c in participants.columns]
    out = participants[meta_cols].merge(roi_subjects, on=["subject_id", "group"], how="inner")
    out["posterior_exponent"] = out.get("posterior_homologous_exponent", np.nan)
    return out


def _apply_cohort(df: pd.DataFrame, cohort: str, cfg: dict[str, Any]) -> pd.DataFrame:
    rep = cfg.get("hbn", {}).get("replication", {})
    age_min = float(rep.get("primary_age_min_months", 72))
    age_max = float(rep.get("primary_age_max_months", 131))
    split = float(rep.get("age_split_months", 72))
    sub = df.copy()
    if cohort == "primary_age_matched":
        sub = sub[(sub["age_months"] >= age_min) & (sub["age_months"] <= age_max)]
    elif cohort == "age_le_72":
        sub = sub[sub["age_months"] <= split]
    elif cohort == "age_gt_72":
        sub = sub[sub["age_months"] > split]
    return sub.reset_index(drop=True)


def _fit_group_model(df: pd.DataFrame, outcome: str, analysis: str, cohort: str) -> list[dict[str, Any]]:
    req = [outcome, "group", "age_months", "sex", "IQ_total", "usable_epochs"]
    sub = df.dropna(subset=req)
    if len(sub) < 20 or sub["group"].nunique() < 2:
        return [{
            "analysis": analysis,
            "cohort": cohort,
            "outcome": outcome,
            "model": "group_main",
            "status": "skipped",
            "n_obs": len(sub),
        }]
    formula = f"{outcome} ~ C(group, Treatment(reference='ASD')) + {COV}"
    fit = run_ols(formula, sub)
    rows = model_results_to_row(fit, analysis, outcome)
    for r in rows:
        r["cohort"] = cohort
        r["model"] = "group_main"
    return rows


def _fit_interaction_model(df: pd.DataFrame, outcome: str, analysis: str, cohort: str) -> list[dict[str, Any]]:
    req = [outcome, "group", "age_months", "sex", "IQ_total", "usable_epochs"]
    sub = df.dropna(subset=req)
    if len(sub) < 30:
        return [{
            "analysis": analysis,
            "cohort": cohort,
            "outcome": outcome,
            "model": "group_x_age",
            "status": "skipped",
            "n_obs": len(sub),
        }]
    formula = f"{outcome} ~ C(group, Treatment(reference='ASD')) * age_months + C(sex) + IQ_total + usable_epochs"
    fit = run_ols(formula, sub)
    rows = model_results_to_row(fit, analysis, outcome)
    for r in rows:
        r["cohort"] = cohort
        r["model"] = "group_x_age"
    return rows


def run_confirmatory_models(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    cohorts = [
        ("full_qc", "full_qc"),
        ("primary_age_matched", "primary_age_matched"),
        ("age_le_72", "age_le_72"),
        ("age_gt_72", "age_gt_72"),
    ]
    outcomes = [
        "posterior_homologous_exponent",
        "posterior_y18_exponent",
        "global_exponent",
        "global_offset",
    ]
    rows: list[dict[str, Any]] = []
    for cohort_key, cohort_label in cohorts:
        sub = _apply_cohort(df, cohort_key, cfg)
        for outcome in outcomes:
            if outcome not in sub.columns:
                continue
            rows.extend(_fit_group_model(sub, outcome, "confirmatory_primary", cohort_label))
            if outcome.endswith("_exponent"):
                rows.extend(_fit_interaction_model(sub, outcome, "confirmatory_interaction", cohort_label))
    return pd.DataFrame(rows)


def _group_contrast_at_age(
    df: pd.DataFrame,
    outcome: str,
    age_months: float,
) -> dict[str, Any]:
    """在指定 age 点估计 TD−ASD adjusted contrast（基于 group×age 模型）。"""
    req = [outcome, "group", "age_months", "sex", "IQ_total", "usable_epochs"]
    sub = df.dropna(subset=req)
    if len(sub) < 30:
        return {"outcome": outcome, "age_months": age_months, "status": "skipped", "n_obs": len(sub)}
    formula = f"{outcome} ~ C(group, Treatment(reference='ASD')) * age_months + C(sex) + IQ_total + usable_epochs"
    fit = run_ols(formula, sub)
    b_asd = 0.0
    b_td = float(fit.params.get("C(group)[T.TD]", np.nan))
    b_int = float(fit.params.get("C(group)[T.TD]:age_months", 0.0))
    contrast = b_td + b_int * age_months
    # delta method SE approx via design row — use predict for simplicity
    pred_asd = fit.predict(pd.DataFrame([{
        "group": "ASD", "age_months": age_months, "sex": sub["sex"].mode().iloc[0],
        "IQ_total": sub["IQ_total"].median(), "usable_epochs": sub["usable_epochs"].median(),
    }]))
    pred_td = fit.predict(pd.DataFrame([{
        "group": "TD", "age_months": age_months, "sex": sub["sex"].mode().iloc[0],
        "IQ_total": sub["IQ_total"].median(), "usable_epochs": sub["usable_epochs"].median(),
    }]))
    return {
        "outcome": outcome,
        "age_months": age_months,
        "TD_minus_ASD_adjusted": float(pred_td.iloc[0] - pred_asd.iloc[0]),
        "n_obs": int(fit.nobs),
        "interaction_p": float(fit.pvalues.get("C(group)[T.TD]:age_months", np.nan)),
    }


def compute_age_contrast_curve(df: pd.DataFrame) -> pd.DataFrame:
    ages = np.arange(int(df["age_months"].min()), int(df["age_months"].max()) + 1, 3)
    rows = []
    for outcome in ["posterior_homologous_exponent", "global_exponent"]:
        if outcome not in df.columns:
            continue
        for age in ages:
            rows.append(_group_contrast_at_age(df, outcome, float(age)))
    return pd.DataFrame(rows)


def plot_replication_figures(df: pd.DataFrame, curve: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    colors = {"ASD": "#4C72B0", "TD": "#DD8452"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, outcome, title in zip(
        axes,
        ["posterior_homologous_exponent", "global_exponent"],
        ["Homologous posterior (E67/E72/E75/E77)", "Global exponent"],
    ):
        if outcome not in df.columns:
            ax.set_visible(False)
            continue
        for grp, sub in df.groupby("group"):
            ax.scatter(
                sub["age_months"], sub[outcome],
                c=colors.get(str(grp), "gray"), alpha=0.35, s=18, label=str(grp),
            )
        ax.axvline(72, color="gray", ls=":", lw=1)
        ax.set_xlabel("Age (months)")
        ax.set_ylabel("Aperiodic exponent")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_replication_age_scatter.png", dpi=150)
    plt.close(fig)

    if len(curve):
        fig, ax = plt.subplots(figsize=(6, 4))
        for outcome, label in [
            ("posterior_homologous_exponent", "Posterior homologous"),
            ("global_exponent", "Global"),
        ]:
            sub = curve[(curve["outcome"] == outcome) & curve["TD_minus_ASD_adjusted"].notna()]
            if len(sub):
                ax.plot(sub["age_months"], sub["TD_minus_ASD_adjusted"], lw=2, label=label)
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(72, color="gray", ls=":", lw=1)
        ax.set_xlabel("Age (months)")
        ax.set_ylabel("Adjusted TD − ASD exponent")
        ax.set_title("Group contrast by age (interaction model)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_replication_td_minus_asd_by_age.png", dpi=150)
        plt.close(fig)


def _group_td_term(res_df: pd.DataFrame) -> pd.Series:
    return res_df["term"].astype(str).str.contains(r"C\(group.*\[T\.TD\]", regex=True) & ~res_df["term"].astype(str).str.contains("age_months")


def _group_age_interaction_term(res_df: pd.DataFrame) -> pd.Series:
    return res_df["term"].astype(str).str.contains(r"C\(group.*\[T\.TD\]:age_months", regex=True)


def _format_model_row(res_df: pd.DataFrame, outcome: str, cohort: str, model: str, term_kind: str) -> str | None:
    sub = res_df[
        (res_df.get("outcome") == outcome)
        & (res_df.get("cohort") == cohort)
        & (res_df.get("model") == model)
    ]
    if term_kind == "group":
        sub = sub[_group_td_term(sub)]
    elif term_kind == "interaction":
        sub = sub[_group_age_interaction_term(sub)]
    else:
        sub = sub[sub.get("term") == term_kind]
    if sub.empty or "coef" not in sub.columns:
        return None
    r = sub.iloc[0]
    if r.get("status") == "skipped":
        return f"- **{outcome}** [{cohort}]: skipped (n={r.get('n_obs', 'NA')})"
    return (
        f"- **{outcome}** [{cohort}] {r['term']}: "
        f"β={r['coef']:.4f}, p={r['pvalue']:.4f}, n={int(r['n_obs'])}"
    )


def build_confirmatory_report(
    df: pd.DataFrame,
    models: pd.DataFrame,
    cfg: dict[str, Any],
) -> str:
    n_asd = int((df["group"] == cfg["groups"]["asd_label"]).sum())
    n_td = int((df["group"] == cfg["groups"]["td_label"]).sum())
    primary = _apply_cohort(df, "primary_age_matched", cfg)
    lines = [
        "# HBN Confirmatory 主研究复现报告",
        "",
        "## 设计定位",
        "",
        "Confirmatory external replication of the **main resting eyes-open finding**:",
        "TD > ASD in **posterior / global aperiodic exponent**, with **group × age** follow-up.",
        "",
        "Non-replicated by design: task Aperiodic-ISC, JR E/I inversion, ML classification.",
        "",
        f"- Full QC cohort: **{len(df)}** (ASD={n_asd}, TD={n_td})",
        f"- Primary age-matched (72–131 mo): **{len(primary)}**",
        "",
        "## ROI 定义",
        "",
        "- **Primary posterior ROI**: homologous to main-study E33/E36/E37/E38 → **E67/E72/E75/E77** (~3 mm)",
        "- **Secondary**: y-lowest 18% pool (posterior_y18), global exponent/offset",
        "",
        "## Primary models",
        "",
        "`outcome ~ group + age_months + sex + IQ_total + usable_epochs`",
        "",
    ]
    for outcome in ["posterior_homologous_exponent", "global_exponent"]:
        for cohort in ["primary_age_matched", "full_qc", "age_gt_72", "age_le_72"]:
            line = _format_model_row(models, outcome, cohort, "group_main", "group")
            if line:
                lines.append(line)

    lines.extend([
        "",
        "## Group × age interaction",
        "",
        "`outcome ~ group * age_months + sex + IQ_total + usable_epochs`",
        "",
    ])
    for outcome in ["posterior_homologous_exponent", "global_exponent"]:
        for cohort in ["primary_age_matched", "full_qc"]:
            line = _format_model_row(models, outcome, cohort, "group_x_age", "interaction")
            if line:
                lines.append(line)
            line = _format_model_row(models, outcome, cohort, "group_x_age", "group")
            if line:
                lines.append(line)

    lines.extend([
        "",
        "## 与主研究对照",
        "",
        "| 主研究 (N=138) | HBN confirmatory |",
        "|---|---|",
        "| TD > ASD global exponent, p≈.012 | 见 primary_age_matched global 行 |",
        "| Posterior E33/E36/E37/E38 FDR sig | homologous E67/E72/E75/E77 |",
        "| Group×age p≈.020 | 见 interaction 行 |",
        "| >72 mo layer sig | 见 age_gt_72 cohort |",
        "",
        "## 解读口径",
        "",
        "- TD 来自 HBN TD_like，非 clinic-recruited TD。",
        "- EO 为多段 20 s 拼接，非主研究连续 5 min。",
        "- Null primary replication ≠ 主效应被证伪；报告 effect size 与 age strata。",
        "",
    ])
    return "\n".join(lines)


def run_hbn_confirmatory_replication(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = resolve_hbn_paths(cfg)
    deriv = paths["derivatives_root"]
    out_root = paths["outputs_root"]
    rep_dir = deriv / "replication"
    tab_dir = out_root / "tables"
    fig_dir = out_root / "figures"
    ensure_dir(rep_dir)
    ensure_dir(tab_dir)
    ensure_dir(fig_dir)

    df = build_subject_replication_table(cfg)
    save_csv(df, rep_dir / "subject_replication_metrics.csv")

    desc = descriptive_table(
        df, "group",
        ["posterior_homologous_exponent", "posterior_y18_exponent",
         "global_exponent", "global_offset", "age_months", "IQ_total", "usable_epochs"],
    )
    save_csv(desc, tab_dir / "confirmatory_descriptive_stats.csv")

    models = run_confirmatory_models(df, cfg)
    save_csv(models, tab_dir / "confirmatory_models.csv")

    curve = compute_age_contrast_curve(_apply_cohort(df, "primary_age_matched", cfg))
    save_csv(curve, tab_dir / "confirmatory_td_minus_asd_by_age.csv")

    plot_replication_figures(df, curve, fig_dir)

    report = build_confirmatory_report(df, models, cfg)
    report_path = out_root / "confirmatory_replication_report_zh.md"
    report_path.write_text(report, encoding="utf-8")

    logger.info("Confirmatory replication: n=%d, report=%s", len(df), report_path)
    return {
        "subjects": rep_dir / "subject_replication_metrics.csv",
        "models": tab_dir / "confirmatory_models.csv",
        "report": report_path,
    }
