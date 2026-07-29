# Ideal release name: 25_rest_movie_state_analysis.py
# Original path: scripts/35_rest_movie_posterior_state_analysis.py
# Note: Rest–movie posterior exponent modulation
# This file is a copy for the public github_release/ bundle.

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

输出 figure_source_data/：
  rest_movie_posterior_exponent_wide.csv
  rest_movie_posterior_exponent_long.csv
  rest_movie_posterior_state_group_models.csv
  rest_movie_posterior_mixed_anova.csv
  rest_movie_posterior_segment_delta.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import ensure_dir, save_csv  # noqa: E402
from src.stats_utils import independent_ttest, run_mixedlm  # noqa: E402

OUT_DIR = PROJECT_ROOT / "figure_source_data"
MOVIE_TS = (
    PROJECT_ROOT
    / "jr_remote_bundle/outputs/jr_modelling/posterior_movie_isc/posterior_sliding_exponent_timeseries.csv"
)


def _build_wide() -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = pd.read_csv(PROJECT_ROOT / "outputs/tables/main_cohort_subject_list.csv")
    rest = pd.read_csv(PROJECT_ROOT / "outputs/tables/resting_features_locked.csv")
    movie_ts = pd.read_csv(MOVIE_TS)

    movie_full = movie_ts.groupby("subject_id", as_index=False).agg(
        movie_posterior_exponent=("posterior_exponent", "mean"),
    )
    movie_seg = movie_ts.groupby(["subject_id", "event_type"], as_index=False)["posterior_exponent"].mean()
    movie_seg = movie_seg.pivot(index="subject_id", columns="event_type", values="posterior_exponent")
    movie_seg.columns = [f"movie_{c}_exponent" for c in movie_seg.columns]
    movie_seg = movie_seg.reset_index()

    wide = cohort.merge(rest[["subject_id", "posterior_exponent"]], on="subject_id", how="inner")
    wide = wide.merge(movie_full, on="subject_id", how="inner")
    wide = wide.merge(movie_seg, on="subject_id", how="left")
    wide = wide[wide["subject_id"].isin(set(movie_ts["subject_id"].astype(str)))]
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


def _segment_deltas(wide: pd.DataFrame) -> pd.DataFrame:
    seg_rows = []
    for seg in ["mental", "pain", "neutral"]:
        col = f"movie_{seg}_exponent"
        if col not in wide.columns:
            continue
        w = wide.dropna(subset=[col]).copy()
        w[f"delta_{seg}"] = w[col] - w["posterior_exponent"]
        tt_seg = independent_ttest(
            w.loc[w["group"] == "ASD", f"delta_{seg}"].to_numpy(),
            w.loc[w["group"] == "TD", f"delta_{seg}"].to_numpy(),
        )
        seg_rows.append(
            {
                "segment": seg,
                "asd_mean_delta": float(w.loc[w["group"] == "ASD", f"delta_{seg}"].mean()),
                "td_mean_delta": float(w.loc[w["group"] == "TD", f"delta_{seg}"].mean()),
                "p_group_diff": float(tt_seg["pvalue"]),
                "n": len(w),
            }
        )
    return pd.DataFrame(seg_rows)


def main() -> None:
    cfg = load_config()
    log = setup_logging(cfg, name="rest_movie_posterior")
    wide, long = _build_wide()
    n_asd = int((wide["group"] == "ASD").sum())
    n_td = int((wide["group"] == "TD").sum())
    log.info("Cohort N=%d (ASD=%d, TD=%d)", len(wide), n_asd, n_td)

    aov = pg.mixed_anova(data=long, dv="posterior_exponent", within="state", between="group", subject="subject_id")
    models = _fit_models(long)
    seg = _segment_deltas(wide)

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
            "movie_mental_exponent",
            "movie_pain_exponent",
            "movie_neutral_exponent",
        ]
    ].rename(columns={"posterior_exponent": "rest_posterior_exponent"})
    save_csv(export_wide, OUT_DIR / "rest_movie_posterior_exponent_wide.csv")
    save_csv(long, OUT_DIR / "rest_movie_posterior_exponent_long.csv")
    save_csv(models, OUT_DIR / "rest_movie_posterior_state_group_models.csv")
    save_csv(aov, OUT_DIR / "rest_movie_posterior_mixed_anova.csv")
    save_csv(seg, OUT_DIR / "rest_movie_posterior_segment_delta.csv")

    inter = aov.loc[aov["Source"] == "Interaction", "p_unc"].iloc[0]
    log.info("Mixed ANOVA state×group interaction p=%.4f", inter)
    log.info("Wrote outputs to %s", OUT_DIR)


if __name__ == "__main__":
    main()
