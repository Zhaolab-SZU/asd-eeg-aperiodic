"""HBN 严格外部复现：临床筛选 + EEG strict QC + geometry-matched posterior ROI。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_roi_config
from src.hbn_confirmatory_replication import _load_channel_qc, _load_participants_qc
from src.hbn_external import resolve_hbn_paths
from src.io_utils import ensure_dir, save_csv
from src.roi_utils import get_roi_dict
from src.stats_utils import descriptive_table, model_results_to_row, run_ols

logger = logging.getLogger(__name__)

HOMOLOGOUS_POSTERIOR = ("E67", "E72", "E75", "E77")
_DX = "Diagnosis_ClinicianConsensus.csv::Diagnosis_ClinicianConsensus,DX_{:02d}"
_DX_CONF = "Diagnosis_ClinicianConsensus.csv::Diagnosis_ClinicianConsensus,DX_{:02d}_Confirmed"

EXCLUSION_DX_KEYWORDS = (
    "epilep",
    "seizure",
    "insomnia",
    "sleep arousal",
    "sleep disorder",
    "narcolep",
    "adhd-hyperactive/impulsive",
)


def _strict_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("hbn", {}).get("strict_replication", {})


def _phenotype_path(cfg: dict[str, Any]) -> Path:
    rel = _strict_cfg(cfg).get(
        "phenotype_merged",
        "data/hbn_metadata/hbn_subject_phenotype_merged.csv",
    )
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _confirmed_dx_text(row: pd.Series, slot: int) -> tuple[str, bool]:
    dx_col = _DX.format(slot)
    cf_col = _DX_CONF.format(slot)
    if cf_col not in row.index or pd.isna(row[cf_col]) or float(row[cf_col]) != 1.0:
        return "", False
    return str(row.get(dx_col, "")).strip(), True


def _is_confirmed_asd(row: pd.Series) -> bool:
    """ADOS 表型未下载时：临床共识 confirmed ASD/Asperger 诊断。"""
    for i in range(1, 11):
        text, ok = _confirmed_dx_text(row, i)
        if not ok:
            continue
        low = text.lower()
        if "autism" in low or "asperger" in low:
            return True
    return False


def _has_clinical_exclusion(row: pd.Series) -> bool:
    for i in range(1, 11):
        text, ok = _confirmed_dx_text(row, i)
        if not ok:
            continue
        low = text.lower()
        if any(k in low for k in EXCLUSION_DX_KEYWORDS):
            return True
    return False


def _is_strict_td(row: pd.Series, scq_max: float, srs_max: float) -> bool:
    scq = pd.to_numeric(row.get("SCQ_total"), errors="coerce")
    srs = pd.to_numeric(row.get("SRS_total"), errors="coerce")
    return pd.notna(scq) and pd.notna(srs) and scq < scq_max and srs < srs_max


def load_phenotype_clinical(cfg: dict[str, Any]) -> pd.DataFrame:
    path = _phenotype_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"未找到 HBN 表型合并表: {path}")
    ph = pd.read_csv(path, low_memory=False)
    ph["subject_id"] = ph["subject_id_std"].astype(str)
    ph["confirmed_asd_dx"] = ph.apply(_is_confirmed_asd, axis=1)
    ph["clinical_exclusion"] = ph.apply(_has_clinical_exclusion, axis=1)
    site_col = "Diagnosis_ClinicianConsensus.csv::Diagnosis_ClinicianConsensus,Site"
    if site_col in ph.columns:
        ph["clinical_site"] = ph[site_col].astype(str)
    else:
        ph["clinical_site"] = np.nan
    return ph


def compute_posterior_emg_ratio(
    subject_ids: list[str],
    deriv_root: Path,
    channels: tuple[str, ...] = HOMOLOGOUS_POSTERIOR,
) -> pd.DataFrame:
    """枕区 30–45 Hz / 1–30 Hz 功率比，作为 residual EMG 代理。"""
    psd_dir = deriv_root / "psd"
    rows: list[dict[str, Any]] = []
    for sid in subject_ids:
        path = psd_dir / f"{sid}_psd.csv"
        if not path.exists():
            continue
        psd = pd.read_csv(path)
        psd = psd[psd["channel"].isin(channels)]
        if psd.empty:
            continue
        freq_col = "frequency" if "frequency" in psd.columns else "freq"
        hf = psd[(psd[freq_col] >= 30) & (psd[freq_col] <= 45)]["power"].mean()
        lf = psd[(psd[freq_col] >= 1) & (psd[freq_col] <= 30)]["power"].mean()
        rows.append({
            "subject_id": sid,
            "emg_ratio_posterior": hf / lf if lf > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def compute_posterior_strict_metrics(
    channel_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """geometry-matched posterior ROI + 通道级 strict specparam QC。"""
    sc = _strict_cfg(cfg)
    min_r2 = float(sc.get("posterior_min_r_squared", 0.95))
    max_err = float(sc.get("posterior_max_fit_error", 0.04))
    min_ch = int(sc.get("posterior_min_strict_channels", 3))

    roi_path = PROJECT_ROOT / cfg.get("hbn", {}).get("roi_config", "config/roi_channels_hbn129.yaml")
    roi_cfg = load_roi_config(roi_path)
    roi_dict = get_roi_dict(roi_cfg, cfg.get("hbn", {}).get("roi_layout", "channels_hbn129"))
    homologous = roi_dict.get("homologous_four", list(HOMOLOGOUS_POSTERIOR))

    ch = channel_df.copy()
    ch["subject_id"] = ch["subject_id"].astype(str)
    ch["strict_posterior"] = (
        ch["channel"].isin(homologous)
        & (ch["r_squared"] > min_r2)
        & (ch["fit_error"] < max_err)
        & ch["fit_valid"].astype(bool)
    )

    rows: list[dict[str, Any]] = []
    for sid, sub in ch.groupby("subject_id"):
        strict = sub[sub["strict_posterior"]]
        rows.append({
            "subject_id": sid,
            "posterior_homologous_exponent": strict["aperiodic_exponent"].mean() if len(strict) else np.nan,
            "posterior_strict_n_channels": int(len(strict)),
            "posterior_strict_mean_r2": strict["r_squared"].mean() if len(strict) else np.nan,
            "posterior_strict_mean_error": strict["fit_error"].mean() if len(strict) else np.nan,
            "posterior_strict_pass": int(len(strict) >= min_ch),
        })
    return pd.DataFrame(rows)


def build_strict_replication_table(cfg: dict[str, Any]) -> pd.DataFrame:
    paths = resolve_hbn_paths(cfg)
    deriv = paths["derivatives_root"]

    participants = _load_participants_qc(cfg)
    phenotype = load_phenotype_clinical(cfg)
    sp_qc = pd.read_csv(deriv / "specparam" / "specparam_qc_summary_subject.csv")
    channel_df = _load_channel_qc(cfg)
    posterior = compute_posterior_strict_metrics(channel_df, cfg)
    emg = compute_posterior_emg_ratio(
        participants["subject_id"].astype(str).tolist(), deriv,
    )

    ph_cols = [
        "subject_id", "confirmed_asd_dx", "clinical_exclusion", "clinical_site",
    ]
    out = participants.merge(phenotype[ph_cols], on="subject_id", how="left")
    out = out.merge(sp_qc, on="subject_id", how="left", suffixes=("", "_spq"))
    out = out.merge(posterior, on="subject_id", how="left")
    out = out.merge(emg, on="subject_id", how="left")
    out["site"] = out["release_id"].astype(str)
    return out


def _filter_step(
    df: pd.DataFrame,
    mask: pd.Series,
    step: str,
    log: list[dict[str, Any]],
) -> pd.DataFrame:
    n_before = len(df)
    sub = df[mask].copy()
    log.append({
        "step": step,
        "n": len(sub),
        "n_asd": int((sub["group"] == "ASD").sum()),
        "n_td": int((sub["group"] == "TD").sum()),
        "dropped": n_before - len(sub),
    })
    return sub.reset_index(drop=True)


def apply_strict_sample_filters(
    df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sc = _strict_cfg(cfg)
    age_min = float(sc.get("age_min_months", 60))
    age_max = float(sc.get("age_max_months", 120))
    iq_min = float(sc.get("iq_min", 80))
    scq_max = float(sc.get("td_scq_max", 11))
    srs_max = float(sc.get("td_srs_max", 65))
    male_only = bool(sc.get("male_only", True))

    log: list[dict[str, Any]] = []
    sub = df[df["group"].isin(["ASD", "TD"])].copy()
    log.append({"step": "ASD_or_TD", "n": len(sub), "n_asd": int((sub["group"] == "ASD").sum()),
                "n_td": int((sub["group"] == "TD").sum()), "dropped": len(df) - len(sub)})

    sub = _filter_step(sub, sub["age_months"].between(age_min, age_max), f"age_{age_min:g}_{age_max:g}mo", log)

    if male_only:
        sub = _filter_step(sub, sub["sex"].astype(str).str.upper() == "M", "male_only", log)

    sub = _filter_step(sub, pd.to_numeric(sub["IQ_total"], errors="coerce") >= iq_min, f"IQ>={iq_min:g}", log)

    asd_keep = (sub["group"] == "ASD") & sub["confirmed_asd_dx"].fillna(False)
    td_keep = (sub["group"] == "TD") & sub.apply(
        lambda r: _is_strict_td(r, scq_max, srs_max), axis=1,
    )
    sub = _filter_step(sub, asd_keep | td_keep, "ASD_confirmed_dx_TD_low_SCQ_SRS", log)
    sub = _filter_step(sub, ~sub["clinical_exclusion"].fillna(False), "exclude_clinical_confounders", log)

    return sub, pd.DataFrame(log)


def apply_strict_eeg_qc(
    df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sc = _strict_cfg(cfg)
    min_epochs = int(sc.get("min_usable_epochs", 46))
    max_interp = float(sc.get("max_invalid_channel_ratio", 0.05))
    emg_quantile = float(sc.get("emg_ratio_max_quantile", 0.75))
    require_posterior = bool(sc.get("require_posterior_strict_pass", True))

    log: list[dict[str, Any]] = []
    sub = df.copy()

    sub = _filter_step(sub, sub["low_quality_subject"] != 1, "specparam_subject_qc", log)
    sub = _filter_step(sub, sub["usable_epochs"] >= min_epochs, f"usable_epochs>={min_epochs}", log)
    sub = _filter_step(
        sub,
        pd.to_numeric(sub["invalid_channel_ratio"], errors="coerce") <= max_interp,
        f"invalid_channel_ratio<={max_interp}",
        log,
    )

    emg_thr = sub["emg_ratio_posterior"].quantile(emg_quantile)
    if pd.notna(emg_thr):
        sub = _filter_step(
            sub,
            pd.to_numeric(sub["emg_ratio_posterior"], errors="coerce") <= emg_thr,
            f"emg_ratio_posterior<={emg_thr:.4f}_q{emg_quantile:.2f}",
            log,
        )

    if require_posterior:
        sub = _filter_step(
            sub,
            (sub["posterior_strict_pass"] == 1)
            & sub["posterior_homologous_exponent"].notna(),
            "posterior_strict_R2_err_channels",
            log,
        )

    return sub, pd.DataFrame(log)


def fit_primary_replication_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Primary: posterior exponent ~ group * age + sex + IQ + site(release)
    """
    outcome = "posterior_homologous_exponent"
    req = [outcome, "group", "age_months", "sex", "IQ_total", "site"]
    sub = df.dropna(subset=req)
    rows: list[dict[str, Any]] = []
    if len(sub) < 20 or sub["group"].nunique() < 2:
        return pd.DataFrame([{
            "analysis": "strict_replication",
            "model": "primary_interaction",
            "outcome": outcome,
            "status": "skipped",
            "n_obs": len(sub),
        }])

    formula_main = (
        f"{outcome} ~ C(group, Treatment(reference='ASD')) + age_months + "
        "C(sex) + IQ_total + C(site)"
    )
    formula_ix = (
        f"{outcome} ~ C(group, Treatment(reference='ASD')) * age_months + "
        "C(sex) + IQ_total + C(site)"
    )

    for model_name, formula in [("group_adjusted", formula_main), ("primary_interaction", formula_ix)]:
        fit = run_ols(formula, sub)
        part = model_results_to_row(fit, "strict_replication", outcome)
        for r in part:
            r["model"] = model_name
            r["n_asd"] = int((sub["group"] == "ASD").sum())
            r["n_td"] = int((sub["group"] == "TD").sum())
        rows.extend(part)

    return pd.DataFrame(rows)


def build_strict_replication_report(
    sample_log: pd.DataFrame,
    eeg_log: pd.DataFrame,
    cohort: pd.DataFrame,
    models: pd.DataFrame,
    cfg: dict[str, Any],
) -> str:
    sc = _strict_cfg(cfg)
    lines = [
        "# HBN 严格外部复现（Step 1–4）",
        "",
        "## 设计摘要",
        "",
        "- **样本**：5–10 岁、male-only（primary）、FSIQ≥80",
        "- **ASD**：临床共识 **confirmed** ASD/Asperger（本地无 ADOS 表 → 见局限）",
        "- **TD**：SCQ/SRS 低分（排除 broad autism phenotype）",
        "- **排除**：confirmed 癫痫/睡眠/ADHD-多动等（见配置关键词）",
        "- **EEG QC**：高 usable epochs、低 invalid 通道比、低枕区 EMG 代理、",
        f"  posterior homologous（E67/E72/E75/E77）通道级 R²>{sc.get('posterior_min_r_squared', 0.95)}、",
        f"  error<{sc.get('posterior_max_fit_error', 0.04)}、≥{sc.get('posterior_min_strict_channels', 3)}/4 通道",
        "- **Primary 模型**：`posterior_homologous_exponent ~ group * age + sex + IQ + site(release)`",
        "- **不做**全头皮 cluster permutation（montage 不可比）",
        "",
        "## Step 1 — 样本筛选",
        "",
    ]
    for _, r in sample_log.iterrows():
        lines.append(
            f"- {r['step']}: n={int(r['n'])} (ASD={int(r['n_asd'])}, TD={int(r['n_td'])})"
            + (f", dropped {int(r['dropped'])}" if pd.notna(r.get("dropped")) else "")
        )

    lines.extend(["", "## Step 2 — EEG strict QC", ""])
    for _, r in eeg_log.iterrows():
        lines.append(
            f"- {r['step']}: n={int(r['n'])} (ASD={int(r['n_asd'])}, TD={int(r['n_td'])})"
            + (f", dropped {int(r['dropped'])}" if pd.notna(r.get("dropped")) else "")
        )

    lines.extend(["", f"## 最终队列 (N={len(cohort)})", ""])
    if cohort.empty:
        lines.append("（空 — 严格筛选后样本不足，见下方局限）")
    else:
        for g, s in cohort.groupby("group"):
            lines.append(
                f"- **{g}** n={len(s)}, age={s['age_months'].mean():.1f}±{s['age_months'].std():.1f}, "
                f"IQ={s['IQ_total'].mean():.1f}, epochs={s['usable_epochs'].mean():.1f}, "
                f"posterior exp={s['posterior_homologous_exponent'].mean():.3f}"
            )

    lines.extend(["", "## Step 4 — Primary 模型（interaction 为复现靶点）", ""])
    if models.empty or "term" not in models.columns:
        status = models.iloc[0].get("status", "insufficient_n") if len(models) else "no_model"
        lines.append(f"（模型未拟合：{status}，最终 n={len(cohort)}）")
    else:
        ix = models[
            (models["model"] == "primary_interaction")
            & models["term"].astype(str).str.contains("group", case=False)
        ]
        for label, pattern in [
            ("group TD vs ASD（主效应）", "TD]"),
            ("**group × age（primary target）**", ":age"),
        ]:
            row = ix[ix["term"].astype(str).str.contains(pattern, regex=False)]
            if len(row):
                r = row.iloc[0]
                lines.append(
                    f"- {label}: β={r['coef']:.4f}, p={r['pvalue']:.4g}, n={int(r['n_obs'])}"
                )

    lines.extend([
        "",
        "## 局限",
        "",
        "1. **ADOS**：本地 phenotype 仅 6 个 CSV，无 ADOS 模块表；ASD 使用 clinician-confirmed DX 代理。",
        "2. **Psychotropic medication**：表型中无用药字段，未能排除 heavy medication。",
        "3. **EMG**：以枕区 30–45/1–30 Hz 功率比代理，非 ICA-muscle 定量。",
        "4. **Interpolation**：HBN pipeline 未标记坏导插值；用 invalid_channel_ratio 代理。",
        "5. **Epoch 上限**：HBN EO 结构 usable_epochs 中位 ~47，无法用主研究 ≥60。",
        "",
    ])
    return "\n".join(lines)


def run_hbn_strict_replication(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = resolve_hbn_paths(cfg)
    out_root = paths["outputs_root"]
    rep_dir = paths["derivatives_root"] / "replication" / "strict"
    ensure_dir(rep_dir)
    ensure_dir(out_root / "tables")

    base = build_strict_replication_table(cfg)
    save_csv(base, rep_dir / "subjects_base_metrics.csv")

    after_sample, sample_log = apply_strict_sample_filters(base, cfg)
    cohort, eeg_log = apply_strict_eeg_qc(after_sample, cfg)

    save_csv(after_sample, rep_dir / "cohort_after_sample_filter.csv")
    save_csv(cohort, rep_dir / "cohort_strict_final.csv")
    save_csv(sample_log, rep_dir / "filter_log_sample.csv")
    save_csv(eeg_log, rep_dir / "filter_log_eeg_qc.csv")

    desc = descriptive_table(
        cohort, "group",
        ["posterior_homologous_exponent", "age_months", "IQ_total", "usable_epochs",
         "emg_ratio_posterior", "posterior_strict_mean_r2"],
    )
    save_csv(desc, out_root / "tables" / "strict_replication_descriptive.csv")

    models = fit_primary_replication_models(cohort)
    save_csv(models, out_root / "tables" / "strict_replication_models.csv")

    report = build_strict_replication_report(sample_log, eeg_log, cohort, models, cfg)
    report_path = out_root / "strict_replication_report_zh.md"
    report_path.write_text(report, encoding="utf-8")

    logger.info("Strict replication: final n=%d (ASD=%d, TD=%d)",
                len(cohort), int((cohort["group"] == "ASD").sum()), int((cohort["group"] == "TD").sum()))

    return {
        "final_n": len(cohort),
        "n_asd": int((cohort["group"] == "ASD").sum()),
        "n_td": int((cohort["group"] == "TD").sum()),
        "report": report_path,
    }
