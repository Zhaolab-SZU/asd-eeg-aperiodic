"""HBN 队列：按主研究 inclusion 规则筛选 + 协变量匹配。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.hbn_confirmatory_replication import (
    build_subject_replication_table,
    _fit_group_model,
    _fit_interaction_model,
)
from src.hbn_external import resolve_hbn_paths
from src.io_utils import ensure_dir, save_csv
from src.stats_utils import descriptive_table

# 主研究 QC 队列 (N=138) 参考值
MAIN_QC_REFERENCE = {
    "n_asd": 61,
    "n_td": 77,
    "age_min": 40,
    "age_max": 131,
    "asd_age_mean": 85.7,
    "td_age_mean": 88.8,
    "asd_iq_mean": 95.0,
    "td_iq_mean": 113.2,
    "asd_male_pct": 92,
    "td_male_pct": 64,
    "min_usable_epochs_main": 60,
}


def apply_covariate_match_filters(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    cfg_key: str = "main_matched",
    strict_td: bool = True,
    extra_required: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """年龄/性别/IQ/epoch 硬筛选 + 可选 strict TD。"""
    rep_cfg = cfg.get("hbn", {}).get(cfg_key, {})
    age_min = float(rep_cfg.get("age_min_months", 40))
    age_max = float(rep_cfg.get("age_max_months", 131))
    min_epochs = int(rep_cfg.get("min_usable_epochs", 42))

    log: list[dict[str, Any]] = []
    n0 = len(df)

    sub = df[df["group"].isin(["ASD", "TD"])].copy()
    log.append({"step": "ASD_or_TD", "n": len(sub), "dropped": n0 - len(sub)})

    sub = sub[(sub["age_months"] >= age_min) & (sub["age_months"] <= age_max)]
    log.append({"step": f"age_{age_min:g}_{age_max:g}", "n": len(sub)})

    req = ["age_months", "sex", "IQ_total", "usable_epochs"]
    if extra_required:
        req.extend(extra_required)
    sub = sub.dropna(subset=req)
    log.append({"step": "complete_covariates", "n": len(sub)})

    sub = sub[sub["usable_epochs"] >= min_epochs]
    log.append({"step": f"usable_epochs>={min_epochs}", "n": len(sub)})

    if strict_td and "SCQ_total" in sub.columns and "SRS_total" in sub.columns:
        scq_max = float(rep_cfg.get("td_scq_max", 11))
        srs_max = float(rep_cfg.get("td_srs_max", 65))
        keep = (sub["group"] == "ASD") | (
            (sub["SCQ_total"] < scq_max) & (sub["SRS_total"] < srs_max)
        )
        n_before = len(sub)
        sub = sub[keep].copy()
        log.append({"step": f"strict_TD_SCQ<{scq_max}_SRS<{srs_max}", "n": len(sub), "dropped": n_before - len(sub)})

    return sub.reset_index(drop=True), pd.DataFrame(log)


def apply_main_study_filters(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    strict_td: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    主研究对齐硬筛选：
    - ASD vs TD
    - age 40–131 mo（HBN Manifest A 实际 ~70+）
    - age/sex/IQ 完整
    - specparam 被试级 QC 已通过（输入 df 应已 QC）
    - usable_epochs >= hbn_min（主研究 60 epoch 在 HBN EO 结构下不可达，见配置说明）
    - 可选：TD 低 SCQ/SRS（更接近 clinic TD）
    """
    filtered, log = apply_covariate_match_filters(
        df,
        cfg,
        cfg_key="main_matched",
        strict_td=strict_td,
        extra_required=["posterior_homologous_exponent"],
    )
    if not log.empty and log.iloc[-1]["step"] == "complete_covariates":
        log = log.copy()
        log.loc[log["step"] == "complete_covariates", "step"] = "complete_covariates_outcomes"
    return filtered, log


def greedy_match_asd_td(
    df: pd.DataFrame,
    *,
    caliper_age: float = 12.0,
    caliper_iq: float = 15.0,
    require_same_sex: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1:1 贪婪匹配：每个 ASD 配一个 TD，最小化 age/IQ 距离（可选同 sex）。
    返回 (matched_df, match_table)。
    """
    rng = np.random.default_rng(seed)
    asd = df[df["group"] == "ASD"].copy()
    td_pool = df[df["group"] == "TD"].copy()
    td_pool = td_pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    pairs: list[dict[str, Any]] = []
    used_td: set[str] = set()

    for _, a in asd.iterrows():
        cands = td_pool[~td_pool["subject_id"].astype(str).isin(used_td)]
        if require_same_sex:
            cands = cands[cands["sex"].astype(str).str.upper() == str(a["sex"]).upper()]
        if cands.empty:
            continue
        age_diff = (cands["age_months"] - float(a["age_months"])).abs()
        iq_diff = (cands["IQ_total"] - float(a["IQ_total"])).abs()
        in_caliper = (age_diff <= caliper_age) & (iq_diff <= caliper_iq)
        if in_caliper.any():
            cands = cands.loc[in_caliper]
            age_diff = age_diff.loc[in_caliper]
            iq_diff = iq_diff.loc[in_caliper]
        dist = (age_diff / caliper_age) ** 2 + (iq_diff / caliper_iq) ** 2
        j = dist.idxmin()
        td_row = cands.loc[j]
        used_td.add(str(td_row["subject_id"]))
        pairs.append({
            "asd_id": a["subject_id"],
            "td_id": td_row["subject_id"],
            "age_diff": float(abs(td_row["age_months"] - a["age_months"])),
            "iq_diff": float(abs(td_row["IQ_total"] - a["IQ_total"])),
            "same_sex": str(a["sex"]).upper() == str(td_row["sex"]).upper(),
        })

    match_df = pd.DataFrame(pairs)
    if match_df.empty:
        return df.iloc[0:0].copy(), match_df

    ids = set(match_df["asd_id"]).union(match_df["td_id"])
    matched = df[df["subject_id"].astype(str).isin(ids)].copy()
    return matched.reset_index(drop=True), match_df


def run_main_matched_models(
    df: pd.DataFrame,
    cohort_label: str,
) -> pd.DataFrame:
    """对指定队列跑与 106 相同的主模型 + 交互。"""
    outcomes = [
        "posterior_homologous_exponent",
        "posterior_y18_exponent",
        "global_exponent",
    ]
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        rows.extend(_fit_group_model(df, outcome, "main_matched", cohort_label))
        if outcome.endswith("_exponent"):
            rows.extend(_fit_interaction_model(df, outcome, "main_matched_interaction", cohort_label))
    return pd.DataFrame(rows)


def build_main_matched_report(
    filtered: pd.DataFrame,
    matched: pd.DataFrame,
    filter_log: pd.DataFrame,
    match_table: pd.DataFrame,
    models_f: pd.DataFrame,
    models_m: pd.DataFrame,
) -> str:
    ref = MAIN_QC_REFERENCE
    lines = [
        "# HBN 主研究对齐筛选 + 匹配复现",
        "",
        "## 主研究参考 (QC N=138)",
        "",
        f"- ASD n={ref['n_asd']}, TD n={ref['n_td']}, age {ref['age_min']}–{ref['age_max']} mo",
        f"- ASD: age≈{ref['asd_age_mean']}, IQ≈{ref['asd_iq_mean']}, {ref['asd_male_pct']}% M",
        f"- TD: age≈{ref['td_age_mean']}, IQ≈{ref['td_iq_mean']}, {ref['td_male_pct']}% M",
        f"- min usable epochs: **{ref['min_usable_epochs_main']}**（HBN EO 结构下不可达，见下）",
        "",
        "## HBN 硬筛选步骤",
        "",
    ]
    for _, r in filter_log.iterrows():
        extra = f", dropped {int(r['dropped'])}" if "dropped" in r and pd.notna(r.get("dropped")) else ""
        lines.append(f"- {r['step']}: n={int(r['n'])}{extra}")

    def _demo_block(label: str, sub: pd.DataFrame) -> list[str]:
        if sub.empty:
            return [f"### {label}", "", "（空）", ""]
        out = [f"### {label} (N={len(sub)})", ""]
        for g, s in sub.groupby("group"):
            out.append(
                f"- **{g}** n={len(s)}, age={s['age_months'].mean():.1f}±{s['age_months'].std():.1f}, "
                f"IQ={s['IQ_total'].mean():.1f}, M%={100*(s['sex'].astype(str).str.upper()=='M').mean():.0f}, "
                f"epochs={s['usable_epochs'].mean():.1f}"
            )
        return out + [""]

    lines.extend(_demo_block("筛选后队列", filtered))
    lines.extend(_demo_block("1:1 匹配队列", matched))

    if len(match_table):
        lines.extend([
            f"- 成功匹配对数: **{len(match_table)}**",
            f"- 年龄差 |Δage| 中位数: **{match_table['age_diff'].median():.1f}** mo",
            f"- IQ 差 |ΔIQ| 中位数: **{match_table['iq_diff'].median():.1f}**",
            "",
        ])

    lines.extend([
        "## 说明：epoch 阈值",
        "",
        "主研究要求 ≥60 个 2 s epoch（~5 min 连续 EO）。HBN 为多段 ~20 s EO，",
        "本队列 usable_epochs 中位数约 **47**，故匹配分析使用 HBN 可行下限（≥42），",
        "并在模型中控制 usable_epochs。",
        "",
        "## 模型结果",
        "",
    ])

    def _append_models(title: str, mdf: pd.DataFrame) -> None:
        lines.extend([f"### {title}", ""])
        if mdf.empty:
            lines.append("（无结果）")
            lines.append("")
            return
        for outcome in ["posterior_homologous_exponent", "global_exponent"]:
            sub_main = mdf[(mdf["outcome"] == outcome) & (mdf["model"] == "group_main")]
            td_row = sub_main[sub_main["term"].astype(str).str.contains("TD", regex=False)]
            if len(td_row):
                r = td_row.iloc[0]
                lines.append(
                    f"- **{outcome}** group TD vs ASD: β={r['coef']:.4f}, p={r['pvalue']:.4f}, n={int(r['n_obs'])}"
                )
            sub_ix = mdf[(mdf["outcome"] == outcome) & (mdf["model"] == "group_x_age")]
            ix_row = sub_ix[sub_ix["term"].astype(str).str.contains(":age", regex=False)]
            if len(ix_row):
                r = ix_row.iloc[0]
                lines.append(
                    f"- **{outcome}** group×age: β={r['coef']:.4f}, p={r['pvalue']:.4f}, n={int(r['n_obs'])}"
                )
        lines.append("")

    _append_models("筛选后（未匹配）", models_f)
    _append_models("1:1 匹配后", models_m)
    return "\n".join(lines)


def run_hbn_main_matched_replication(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = resolve_hbn_paths(cfg)
    out_root = paths["outputs_root"]
    rep_dir = paths["derivatives_root"] / "replication" / "main_matched"
    ensure_dir(rep_dir)
    ensure_dir(out_root / "tables")

    df = build_subject_replication_table(cfg)
    part = pd.read_csv(paths["derivatives_root"] / "participants_analysis.csv")
    extra = [c for c in ("SRS_total", "SCQ_total") if c in part.columns]
    if extra:
        df = df.merge(part[["subject_id"] + extra], on="subject_id", how="left")

    filtered, filter_log = apply_main_study_filters(df, cfg, strict_td=True)
    matched, match_table = greedy_match_asd_td(filtered, seed=int(cfg.get("project", {}).get("random_seed", 42)))

    save_csv(filtered, rep_dir / "cohort_main_filtered.csv")
    save_csv(matched, rep_dir / "cohort_main_matched_1to1.csv")
    save_csv(filter_log, rep_dir / "filter_log.csv")
    save_csv(match_table, rep_dir / "match_pairs.csv")

    desc_f = descriptive_table(
        filtered, "group",
        ["posterior_homologous_exponent", "global_exponent", "age_months", "IQ_total", "usable_epochs"],
    )
    save_csv(desc_f, out_root / "tables" / "main_matched_descriptive_filtered.csv")

    models_f = run_main_matched_models(filtered, "main_filtered")
    models_m = run_main_matched_models(matched, "main_matched_1to1")
    save_csv(models_f, out_root / "tables" / "main_matched_models_filtered.csv")
    save_csv(models_m, out_root / "tables" / "main_matched_models_1to1.csv")

    report = build_main_matched_report(filtered, matched, filter_log, match_table, models_f, models_m)
    report_path = out_root / "main_matched_replication_report_zh.md"
    report_path.write_text(report, encoding="utf-8")

    return {
        "filtered_n": len(filtered),
        "matched_n": len(matched),
        "matched_pairs": len(match_table),
        "report": report_path,
    }
