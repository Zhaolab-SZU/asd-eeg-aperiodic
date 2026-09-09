# Public path: scripts/movie/25_rest_movie_state_analysis.py
# Source: scripts/35_rest_movie_posterior_state_analysis.py

#!/usr/bin/env python
"""
35_rest_movie_posterior_state_analysis.py
-----------------------------------------
Resting vs movie within-subject posterior exponent 对比，
检验 state × group 交互。

队列：primary resting cohort（N=138）与 movie sliding-window exponent
均有数据的被试（N=136；ASD=61，TD=75）。

Movie posterior exponent：E33/E36/E37/E38 滑动窗 specparam 全片均值
（jr_remote_bundle/.../posterior_sliding_exponent_timeseries.csv）。

输出 outputs/figure_source_data/：
  rest_movie_posterior_exponent_wide.csv
  rest_movie_posterior_exponent_long.csv
  rest_movie_posterior_state_group_models.csv
  rest_movie_posterior_mixed_anova.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "src" / "config.py").is_file():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError("Cannot locate repository root (missing src/config.py)")
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import ensure_dir, save_csv  # noqa: E402
from src.stats_utils import independent_ttest, run_mixedlm  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs" / "figure_source_data"
MOVIE_TS = (
    PROJECT_ROOT
    / "jr_remote_bundle/outputs/jr_modelling/posterior_movie_isc/posterior_sliding_exponent_timeseries.csv"
)
# Prefer continuous Partly Cloudy exponent timeseries from the active ISC pipeline.
MOVIE_TS_ACTIVE = (
    PROJECT_ROOT
    / "derivatives/derivatives_task_movie/stats/partly_cloudy_isc/partly_cloudy_exponent_concat_timeseries.csv"
)


def _build_wide() -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = pd.read_csv(PROJECT_ROOT / "outputs/tables/main_cohort_subject_list.csv")
    rest = pd.read_csv(PROJECT_ROOT / "outputs/tables/resting_features_locked.csv")

    if MOVIE_TS_ACTIVE.exists():
        movie_ts = pd.read_csv(MOVIE_TS_ACTIVE)
        movie_full = (
            movie_ts.groupby("subject_id", as_index=False)
            .agg(movie_posterior_exponent=("exponent_mean", "mean"))
        )
    elif MOVIE_TS.exists():
        movie_ts = pd.read_csv(MOVIE_TS)
        movie_full = movie_ts.groupby("subject_id", as_index=False).agg(
            movie_posterior_exponent=("posterior_exponent", "mean"),
        )
    else:
        raise FileNotFoundError(
            "[ANALYSIS BLOCKED: missing Partly Cloudy exponent timeseries for rest-movie state analysis]"
        )

    wide = cohort.merge(rest[["subject_id", "posterior_exponent"]], on="subject_id", how="inner")
    wide = wide.merge(movie_full, on="subject_id", how="inner")
    wide["delta_movie_rest"] = wide["movie_posterior_exponent"] - wide["posterior_exponent"]

    long_rows: list[dict] = []
    for _, r in wide.iterrows():
        for state, val in [("rest", r.posterior_exponent), ("movie", r.movie_posterior_exponent)]:
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
                }
            )
    long = pd.DataFrame(long_rows)
    long["state"] = pd.Categorical(long["state"], categories=["rest", "movie"], ordered=True)
    return wide, long


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
                    "used_mixedlm": getattr(res, "_used_mixedlm", None),
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
                "used_mixedlm": False,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    log = setup_logging(cfg, name="rest_movie_posterior")
    wide, long = _build_wide()
    n_asd = int((wide["group"] == "ASD").sum())
    n_td = int((wide["group"] == "TD").sum())
    log.info("Cohort N=%d (ASD=%d, TD=%d)", len(wide), n_asd, n_td)

    aov = pg.mixed_anova(data=long, dv="posterior_exponent", within="state", between="group", subject="subject_id")
    models = _fit_models(long)

    ensure_dir(OUT_DIR)
    export_wide = wide[
        [
            "subject_id",
            "group",
            "age_months",
            "sex",
            "IQ_total",
            "usable_epochs",
            "posterior_exponent",
            "movie_posterior_exponent",
            "delta_movie_rest",
        ]
    ].rename(columns={"posterior_exponent": "rest_posterior_exponent"})
    save_csv(export_wide, OUT_DIR / "rest_movie_posterior_exponent_wide.csv")
    save_csv(long, OUT_DIR / "rest_movie_posterior_exponent_long.csv")
    save_csv(models, OUT_DIR / "rest_movie_posterior_state_group_models.csv")
    save_csv(aov, OUT_DIR / "rest_movie_posterior_mixed_anova.csv")

    # Remove obsolete segment-delta output if present from prior runs.
    obsolete = OUT_DIR / "rest_movie_posterior_segment_delta.csv"
    if obsolete.exists():
        obsolete.unlink()

    inter = aov.loc[aov["Source"] == "Interaction", "p_unc"].iloc[0]
    log.info("Mixed ANOVA state×group interaction p=%.4f", inter)
    log.info("Wrote outputs to %s", OUT_DIR)


if __name__ == "__main__":
    main()
