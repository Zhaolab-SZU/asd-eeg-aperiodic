#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Matched-estimator sensitivity: rest using same 2s/0.5s sliding mean-of-exponents as movie.

Primary rest-movie analysis uses:
  rest  = average PSD across usable epochs, then one specparam fit (condition-averaged)
  movie = per-window specparam, then mean of window exponents

This sensitivity recomputes rest with the movie-matched estimator on continuous
preprocessed resting FIF (posterior ROI mean E33/E36/E37/E38), then reruns the
state x group models on the same paired n=136 cohort.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf
from scipy import signal, stats

PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "src" / "config.py").is_file():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError("Cannot locate repository root (missing src/config.py)")
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import ensure_dir, save_csv  # noqa: E402
from src.specparam_utils import fit_specparam_channel  # noqa: E402
from src.stats_utils import run_mixedlm  # noqa: E402

POSTERIOR = ("E33", "E36", "E37", "E38")
WINDOW_SEC = 2.0
STEP_SEC = 0.5
MIN_R2 = 0.80
EXP_LO, EXP_HI = 0.0, 5.0

OUT_DIR = PROJECT_ROOT / "data" / "supplementary_source" / "rest_movie_matched_estimator"
CACHE_DIR = PROJECT_ROOT / "derivatives" / "rest_sliding_cache"
RAW_DIR = PROJECT_ROOT / "derivatives" / "preprocessed"

def _first_existing(*candidates: Path) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("None of these files exist:\n" + "\n".join(str(p) for p in candidates))

WIDE_PRIMARY_CANDIDATES = (
    PROJECT_ROOT / "figure_source_data" / "rest_movie_posterior_exponent_wide.csv",
    PROJECT_ROOT / "data" / "figure_source" / "rest_movie_posterior_exponent_wide.csv",
    PROJECT_ROOT / "outputs" / "figure_source_data" / "rest_movie_posterior_exponent_wide.csv",
)
WIDE_PRIMARY = WIDE_PRIMARY_CANDIDATES[0]


def _welch_segment(sig: np.ndarray, sfreq: float, welch_cfg: dict, fmin: float, fmax: float):
    n_fft = int(welch_cfg.get("n_fft", 500))
    n_overlap = int(welch_cfg.get("n_overlap", 0))
    n_fft = min(n_fft, len(sig))
    if n_fft < 16:
        return np.array([]), np.array([])
    freqs, psd = signal.welch(
        sig,
        fs=sfreq,
        nperseg=n_fft,
        noverlap=min(n_overlap, max(0, n_fft - 1)),
        window=welch_cfg.get("window", "hamming"),
    )
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], psd[mask]


def rest_sliding_mean_exponent(sid: str, cfg: dict, force: bool = False) -> dict:
    cache = CACHE_DIR / f"{sid}_rest_sliding_posterior.csv"
    summary_path = CACHE_DIR / f"{sid}_rest_sliding_summary.json"
    if cache.exists() and summary_path.exists() and not force:
        summ = json.loads(summary_path.read_text(encoding="utf-8"))
        return summ

    raw_path = RAW_DIR / f"{sid}-raw.fif"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    sp_cfg = cfg["specparam"]
    psd_cfg = cfg.get("psd", {})
    welch_cfg = psd_cfg.get("welch", {})
    fmin = float(welch_cfg.get("fmin", psd_cfg.get("freq_min_hz", 1.0)))
    fmax = float(welch_cfg.get("fmax", psd_cfg.get("freq_max_hz", 40.0)))

    raw = mne.io.read_raw_fif(str(raw_path), preload=True, verbose=False)
    picks = [c for c in POSTERIOR if c in raw.ch_names]
    if len(picks) < 1:
        raise ValueError(f"{sid}: no posterior channels")
    data = raw.get_data(picks=picks).mean(axis=0)
    sfreq = float(raw.info["sfreq"])
    del raw

    win_n = max(1, int(round(WINDOW_SEC * sfreq)))
    step_n = max(1, int(round(STEP_SEC * sfreq)))

    rows = []
    start = 0
    widx = 0
    while start + win_n <= len(data):
        seg = data[start : start + win_n]
        freqs, power = _welch_segment(seg, sfreq, welch_cfg, fmin, fmax)
        exp = np.nan
        r2 = np.nan
        valid = False
        if len(freqs) >= 5:
            try:
                fit = fit_specparam_channel(freqs, power, sp_cfg)
                exp = float(fit["aperiodic_exponent"])
                r2 = float(fit.get("r_squared", fit.get("r_squared_", np.nan)))
                if np.isfinite(exp) and np.isfinite(r2) and r2 >= MIN_R2 and EXP_LO <= exp <= EXP_HI:
                    valid = True
                else:
                    exp = np.nan
            except Exception:
                exp = np.nan
                r2 = np.nan
        rows.append(
            {
                "subject_id": sid,
                "window_index": widx,
                "window_start_sec": start / sfreq,
                "window_end_sec": (start + win_n) / sfreq,
                "center_sec": (start + win_n / 2.0) / sfreq,
                "exponent": exp if valid else np.nan,
                "r_squared": r2,
                "fit_valid": valid,
            }
        )
        widx += 1
        start += step_n

    ts = pd.DataFrame(rows)
    ensure_dir(CACHE_DIR)
    ts.to_csv(cache, index=False)

    valid_exp = ts.loc[ts["fit_valid"], "exponent"]
    summ = {
        "subject_id": sid,
        "n_windows": int(len(ts)),
        "n_valid_windows": int(valid_exp.shape[0]),
        "valid_fraction": float(valid_exp.shape[0] / max(len(ts), 1)),
        "rest_sliding_mean_exponent": float(valid_exp.mean()) if len(valid_exp) else np.nan,
        "rest_sliding_median_exponent": float(valid_exp.median()) if len(valid_exp) else np.nan,
        "mean_r_squared_valid": float(ts.loc[ts["fit_valid"], "r_squared"].mean()) if len(valid_exp) else np.nan,
        "n_posterior_channels": int(len(picks)),
        "duration_sec": float(len(data) / sfreq),
    }
    summary_path.write_text(json.dumps(summ, indent=2), encoding="utf-8")
    return summ


def _fit_models(long: pd.DataFrame) -> pd.DataFrame:
    cov_cols = ["posterior_exponent", "group", "state", "age_months", "sex", "IQ_total", "usable_epochs"]
    sub = long.dropna(subset=cov_cols)
    rows: list[dict] = []
    for name, formula in [
        ("mixed_no_cov", "posterior_exponent ~ C(state)*C(group)"),
        (
            "mixed_covariates",
            "posterior_exponent ~ C(state)*C(group) + age_months + C(sex) + IQ_total + usable_epochs",
        ),
    ]:
        res = run_mixedlm(formula, sub, groups="subject_id")
        for term in res.params.index:
            if term == "Group Var":
                continue
            rows.append(
                {
                    "model": name,
                    "term": term,
                    "coef": float(res.params[term]),
                    "se": float(res.bse[term]),
                    "pvalue": float(res.pvalues[term]),
                    "n_obs": int(res.nobs),
                }
            )
    ols = smf.ols(
        "posterior_exponent ~ C(state)*C(group) + age_months + C(sex) + IQ_total + usable_epochs",
        data=sub,
    ).fit(cov_type="cluster", cov_kwds={"groups": sub["subject_id"]})
    for term in ols.params.index:
        rows.append(
            {
                "model": "ols_cluster_robust",
                "term": term,
                "coef": float(ols.params[term]),
                "se": float(ols.bse[term]),
                "pvalue": float(ols.pvalues[term]),
                "n_obs": int(ols.nobs),
            }
        )
    return pd.DataFrame(rows)


def _group_desc(wide: pd.DataFrame, rest_col: str, movie_col: str) -> pd.DataFrame:
    rows = []
    for g, gg in wide.groupby("group"):
        t = stats.ttest_ind(
            wide.loc[wide["group"] == "TD", rest_col].dropna(),
            wide.loc[wide["group"] == "ASD", rest_col].dropna(),
            equal_var=False,
        )
        # defer group contrast to later; per-group means first
        rows.append(
            {
                "estimator": rest_col,
                "group": g,
                "n": len(gg),
                "rest_mean": float(gg[rest_col].mean()),
                "rest_sd": float(gg[rest_col].std(ddof=1)),
                "movie_mean": float(gg[movie_col].mean()),
                "movie_sd": float(gg[movie_col].std(ddof=1)),
                "delta_mean": float((gg[movie_col] - gg[rest_col]).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    log = setup_logging(cfg, name="rest_movie_matched_estimator")
    ensure_dir(OUT_DIR)
    ensure_dir(CACHE_DIR)

    primary = pd.read_csv(_first_existing(*WIDE_PRIMARY_CANDIDATES))
    ids = primary["subject_id"].astype(str).tolist()
    log.info("Paired cohort n=%d; computing resting sliding estimates...", len(ids))

    summaries = []
    t0 = time.time()
    for i, sid in enumerate(ids, 1):
        summ = rest_sliding_mean_exponent(sid, cfg, force=False)
        summaries.append(summ)
        if i % 20 == 0 or i == len(ids):
            log.info("[%d/%d] elapsed=%.1fs last=%s valid=%s/%s",
                     i, len(ids), time.time() - t0, sid,
                     summ.get("n_valid_windows"), summ.get("n_windows"))

    rest_slide = pd.DataFrame(summaries)
    save_csv(rest_slide, OUT_DIR / "rest_sliding_subject_summary.csv")

    wide = primary.merge(
        rest_slide[
            [
                "subject_id",
                "rest_sliding_mean_exponent",
                "n_windows",
                "n_valid_windows",
                "valid_fraction",
                "mean_r_squared_valid",
            ]
        ],
        on="subject_id",
        how="left",
    )
    wide["delta_movie_rest_primary"] = wide["movie_posterior_exponent"] - wide["rest_posterior_exponent"]
    wide["delta_movie_rest_matched"] = wide["movie_posterior_exponent"] - wide["rest_sliding_mean_exponent"]
    save_csv(wide, OUT_DIR / "rest_movie_matched_estimator_wide.csv")

    # Compare primary vs matched rest estimates
    r_rest = float(
        wide[["rest_posterior_exponent", "rest_sliding_mean_exponent"]].corr().iloc[0, 1]
    )
    mae = float((wide["rest_sliding_mean_exponent"] - wide["rest_posterior_exponent"]).abs().mean())

    # Long for matched
    long_rows = []
    for _, r in wide.iterrows():
        for state, val in [
            ("rest", r.rest_sliding_mean_exponent),
            ("movie", r.movie_posterior_exponent),
        ]:
            long_rows.append(
                {
                    "subject_id": r.subject_id,
                    "group": r.group,
                    "state": state,
                    "posterior_exponent": val,
                    "age_months": r.age_months,
                    "sex": r.sex,
                    "IQ_total": r.IQ_total,
                    "usable_epochs": r.usable_epochs,
                    "estimator": "matched_sliding",
                }
            )
    long = pd.DataFrame(long_rows)
    long["state"] = pd.Categorical(long["state"], categories=["rest", "movie"], ordered=True)
    save_csv(long, OUT_DIR / "rest_movie_matched_estimator_long.csv")

    aov = pg.mixed_anova(
        data=long, dv="posterior_exponent", within="state", between="group", subject="subject_id"
    )
    models = _fit_models(long)
    save_csv(aov, OUT_DIR / "rest_movie_matched_mixed_anova.csv")
    save_csv(models, OUT_DIR / "rest_movie_matched_state_group_models.csv")

    # Group contrasts
    def welch_gap(col: str):
        asd = wide.loc[wide["group"] == "ASD", col].dropna()
        td = wide.loc[wide["group"] == "TD", col].dropna()
        t = stats.ttest_ind(td, asd, equal_var=False)
        return {
            "outcome": col,
            "asd_mean": float(asd.mean()),
            "td_mean": float(td.mean()),
            "td_minus_asd": float(td.mean() - asd.mean()),
            "welch_t": float(t.statistic),
            "welch_p": float(t.pvalue),
            "n_asd": int(asd.shape[0]),
            "n_td": int(td.shape[0]),
        }

    contrasts = pd.DataFrame(
        [
            welch_gap("rest_posterior_exponent"),
            welch_gap("rest_sliding_mean_exponent"),
            welch_gap("movie_posterior_exponent"),
            welch_gap("delta_movie_rest_primary"),
            welch_gap("delta_movie_rest_matched"),
        ]
    )
    save_csv(contrasts, OUT_DIR / "rest_movie_matched_group_contrasts.csv")

    gap_rest_p = contrasts.loc[contrasts.outcome == "rest_posterior_exponent", "td_minus_asd"].iloc[0]
    gap_rest_m = contrasts.loc[contrasts.outcome == "rest_sliding_mean_exponent", "td_minus_asd"].iloc[0]
    gap_movie = contrasts.loc[contrasts.outcome == "movie_posterior_exponent", "td_minus_asd"].iloc[0]

    inter = aov.loc[aov["Source"] == "Interaction"].iloc[0]
    ols_inter = models.loc[
        (models.model == "ols_cluster_robust")
        & (models.term == "C(state)[T.movie]:C(group)[T.TD]")
    ].iloc[0]

    headline = {
        "n": int(len(wide)),
        "n_asd": int((wide.group == "ASD").sum()),
        "n_td": int((wide.group == "TD").sum()),
        "corr_primary_vs_sliding_rest": r_rest,
        "mae_primary_vs_sliding_rest": mae,
        "gap_rest_primary": float(gap_rest_p),
        "gap_rest_matched": float(gap_rest_m),
        "gap_movie": float(gap_movie),
        "gap_convergence_primary": float(gap_rest_p - gap_movie),
        "gap_convergence_matched": float(gap_rest_m - gap_movie),
        "mixed_anova_interaction_F": float(inter["F"]),
        "mixed_anova_interaction_p": float(inter["p_unc"]),
        "mixed_anova_partial_eta2": float(inter["np2"]),
        "ols_state_x_td_beta": float(ols_inter["coef"]),
        "ols_state_x_td_se": float(ols_inter["se"]),
        "ols_state_x_td_p": float(ols_inter["pvalue"]),
        "mean_valid_fraction": float(wide["valid_fraction"].mean()),
        "min_valid_windows": int(wide["n_valid_windows"].min()),
    }
    (OUT_DIR / "matched_estimator_headline.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8"
    )
    pd.DataFrame([headline]).to_csv(OUT_DIR / "matched_estimator_headline.csv", index=False)
    log.info("Headline: %s", json.dumps(headline, indent=2))
    log.info("Wrote %s", OUT_DIR)


if __name__ == "__main__":
    main()
