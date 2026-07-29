"""Standardized betas for ROI mixed-effects model (Table S3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.io_utils import attach_usable_epochs, exclude_specparam_low_quality, load_analysis_participants, save_csv
from src.stats_utils import run_mixedlm


def _predictor_series(df: pd.DataFrame, term_key: str) -> pd.Series:
    """Map Table S3 / statsmodels term to an observed contrast column."""
    g_td = (df["group"] == "TD").astype(float)
    roi = df["roi"].astype(str).str.lower()
    if term_key == "group_td_central":
        return g_td
    if term_key == "group_x_frontal":
        return g_td * (roi == "frontal").astype(float)
    if term_key == "group_x_occipital":
        return g_td * (roi == "occipital").astype(float)
    if term_key == "group_x_parietal":
        return g_td * (roi == "parietal").astype(float)
    if term_key == "group_x_temporal":
        return g_td * (roi == "temporal").astype(float)
    raise KeyError(term_key)


def standardized_beta(df: pd.DataFrame, outcome: str, predictor: pd.Series, coef: float) -> float:
    y = df[outcome].astype(float)
    x = predictor.astype(float)
    sy, sx = y.std(ddof=1), x.std(ddof=1)
    if sy == 0 or sx == 0 or np.isnan(coef):
        return np.nan
    return float(coef * sx / sy)


def build_s3_table_rows(model_df: pd.DataFrame, analysis_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows for Supplementary Table S3 with β_std."""
    term_map = {
        "Group (TD vs ASD), central": "C(group)[T.TD]",
        "Group × frontal": "C(group)[T.TD]:C(roi)[T.frontal]",
        "Group × occipital": "C(group)[T.TD]:C(roi)[T.occipital]",
        "Group × parietal": "C(group)[T.TD]:C(roi)[T.parietal]",
        "Group × temporal": "C(group)[T.TD]:C(roi)[T.temporal]",
    }
    pred_key = {
        "Group (TD vs ASD), central": "group_td_central",
        "Group × frontal": "group_x_frontal",
        "Group × occipital": "group_x_occipital",
        "Group × parietal": "group_x_parietal",
        "Group × temporal": "group_x_temporal",
    }
    rows: list[dict[str, Any]] = []
    for label, sm_term in term_map.items():
        hit = model_df[model_df["term"] == sm_term]
        if hit.empty:
            continue
        r = hit.iloc[0]
        pk = pred_key[label]
        x = _predictor_series(analysis_df, pk)
        b_std = standardized_beta(analysis_df, "exponent", x, float(r["coef"]))
        p = float(r["pvalue"])
        p_str = "< 0.001" if p < 0.001 else f"{p:.3f}"
        rows.append({
            "term_label": label,
            "term": sm_term,
            "beta": float(r["coef"]),
            "beta_std": b_std,
            "se": float(r["std_err"]),
            "p": p,
            "p_display": p_str,
        })
    return rows


def run_roi_s3_effect_sizes(cfg: dict[str, Any]) -> pd.DataFrame:
    deriv = Path(cfg["paths"]["derivatives_root"])
    outputs = Path(cfg["paths"]["outputs_root"])
    if not outputs.is_absolute():
        outputs = Path(__file__).resolve().parents[1] / outputs

    participants = load_analysis_participants(cfg)
    roi_long = pd.read_csv(deriv / "roi" / "specparam_subject_roi_long.csv")
    df = participants.merge(roi_long, on=["subject_id", "group"], how="inner")
    df = attach_usable_epochs(df, deriv)
    df = exclude_specparam_low_quality(df, deriv)
    sub = df.dropna(subset=["exponent", "group", "roi", "age_months", "sex", "IQ_total", "usable_epochs"])

    model_path = deriv / "stats" / "roi_mixed_model.csv"
    if model_path.exists():
        model_df = pd.read_csv(model_path)
        model_df = model_df[model_df["outcome"] == "exponent"]
    else:
        formula = "exponent ~ C(group) * C(roi) + age_months + C(sex) + IQ_total + usable_epochs"
        result = run_mixedlm(formula, sub, groups="subject_id")
        from src.stats_utils import model_results_to_row

        model_df = pd.DataFrame(model_results_to_row(result, "roi_mixed", "exponent"))

    rows = build_s3_table_rows(model_df, sub)
    out = pd.DataFrame(rows)
    tables_dir = outputs / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    save_csv(out, tables_dir / "roi_mixed_model_s3_with_std_beta.csv")
    return out
