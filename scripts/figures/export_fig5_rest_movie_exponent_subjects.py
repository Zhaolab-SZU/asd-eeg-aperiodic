# Ideal release name: export_fig5_rest_movie_exponent_subjects.py
# Original path: scripts/export_fig5_rest_movie_exponent_subjects.py
# Note: Fig.5A rest–movie subject table
# This file is a copy for the public github_release/ bundle.

"""Export Fig.5A subject-level Rest vs Movie posterior exponent long table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "working" / "final_figures" / "figure5_naturalistic_redesign"
CSV_OUT = OUT_DIR / "fig5_rest_movie_exponent_subjects.csv"
CHECK_OUT = OUT_DIR / "fig5_rest_movie_exponent_subjects_check.txt"

REST_PATH = PROJECT_ROOT / "figure_source_data" / "roi_subject_wide_primary.csv"
MOVIE_TS_PATH = (
    PROJECT_ROOT
    / "jr_remote_bundle"
    / "outputs"
    / "jr_modelling"
    / "posterior_movie_isc"
    / "posterior_sliding_exponent_timeseries.csv"
)
REF_WIDE_PATH = PROJECT_ROOT / "figure_source_data" / "rest_movie_posterior_exponent_wide.csv"
REF_LONG_PATH = PROJECT_ROOT / "figure_source_data" / "rest_movie_posterior_exponent_long.csv"

EXPECTED = {
    "N": 136,
    "ASD": 61,
    "TD": 75,
    "ASD Rest": 1.87,
    "ASD Movie": 1.95,
    "ASD Delta": 0.087,
    "TD Rest": 1.99,
    "TD Movie": 1.90,
    "TD Delta": -0.087,
}
DESC_TOL = 0.01


def build_wide() -> pd.DataFrame:
    rest = pd.read_csv(REST_PATH)[["subject_id", "group", "posterior_exponent"]].rename(
        columns={"posterior_exponent": "rest_posterior_exponent"}
    )
    movie_ts = pd.read_csv(MOVIE_TS_PATH)
    movie = (
        movie_ts.groupby(["subject_id", "group"], as_index=False)["posterior_exponent"]
        .mean()
        .rename(columns={"posterior_exponent": "movie_posterior_exponent"})
    )
    wide = rest.merge(movie, on=["subject_id", "group"], how="inner")
    return wide.dropna(subset=["rest_posterior_exponent", "movie_posterior_exponent"])


def build_long(wide: pd.DataFrame) -> pd.DataFrame:
    rest_long = wide[["subject_id", "group", "rest_posterior_exponent"]].rename(
        columns={"rest_posterior_exponent": "posterior_exponent"}
    )
    rest_long["state"] = "Rest"
    movie_long = wide[["subject_id", "group", "movie_posterior_exponent"]].rename(
        columns={"movie_posterior_exponent": "posterior_exponent"}
    )
    movie_long["state"] = "Movie"
    long = pd.concat([rest_long, movie_long], ignore_index=True)
    long = long[["subject_id", "group", "state", "posterior_exponent"]]
    long["state"] = pd.Categorical(long["state"], categories=["Rest", "Movie"], ordered=True)
    return long.sort_values(["group", "subject_id", "state"]).reset_index(drop=True)


def write_check(wide: pd.DataFrame, long: pd.DataFrame) -> bool:
    n_total = int(wide["subject_id"].nunique())
    n_asd = int(wide.loc[wide["group"] == "ASD", "subject_id"].nunique())
    n_td = int(wide.loc[wide["group"] == "TD", "subject_id"].nunique())

    lines = [
        "Fig.5A rest–movie posterior exponent subject-level export check",
        "=" * 72,
        "",
        f"Output file: {CSV_OUT.relative_to(PROJECT_ROOT)}",
        "",
        "Data sources",
        f"- Rest posterior exponent (E33/E36/E37/E38 mean): {REST_PATH.relative_to(PROJECT_ROOT)}",
        f"- Movie posterior exponent (full-film sliding-window specparam mean): {MOVIE_TS_PATH.relative_to(PROJECT_ROOT)}",
        f"- Cross-reference wide table: {REF_WIDE_PATH.relative_to(PROJECT_ROOT)}",
        "- Note: working/figure_source_data_latest/fig1_resting_primary_subjects.csv not found in repo;",
        "  Rest values match figure_source_data/roi_subject_wide_primary.csv posterior_exponent exactly.",
        "",
        "1. Subject counts",
        f"   Total N = {n_total} (expected {EXPECTED['N']})",
        f"   ASD = {n_asd} (expected {EXPECTED['ASD']})",
        f"   TD  = {n_td} (expected {EXPECTED['TD']})",
        "",
        "2. Long-format row count",
        f"   Rows = {len(long)} (expected 2 × N = {2 * n_total})",
        f"   Rest rows = {(long['state'] == 'Rest').sum()} ; Movie rows = {(long['state'] == 'Movie').sum()}",
        "",
        "3. Group × state mean ± SD",
    ]
    for group in ["ASD", "TD"]:
        for state in ["Rest", "Movie"]:
            values = long.loc[(long["group"] == group) & (long["state"] == state), "posterior_exponent"]
            lines.append(
                f"   {group} {state}: {values.mean():.6f} ± {values.std(ddof=1):.6f} (n={len(values)})"
            )

    lines.extend(["", "4. Δ = Movie − Rest (group means)"])
    delta_obs: dict[str, float] = {}
    for group in ["ASD", "TD"]:
        sub = wide.loc[wide["group"] == group]
        delta = sub["movie_posterior_exponent"] - sub["rest_posterior_exponent"]
        delta_obs[group] = float(delta.mean())
        lines.append(f"   {group}: {delta_obs[group]:+.6f} (SD={delta.std(ddof=1):.6f})")

    lines.extend(["", "5. Manuscript-style descriptive check"])
    desc_checks = [
        ("ASD Rest", EXPECTED["ASD Rest"], float(long.loc[(long["group"] == "ASD") & (long["state"] == "Rest"), "posterior_exponent"].mean())),
        ("ASD Movie", EXPECTED["ASD Movie"], float(long.loc[(long["group"] == "ASD") & (long["state"] == "Movie"), "posterior_exponent"].mean())),
        ("ASD Delta", EXPECTED["ASD Delta"], delta_obs["ASD"]),
        ("TD Rest", EXPECTED["TD Rest"], float(long.loc[(long["group"] == "TD") & (long["state"] == "Rest"), "posterior_exponent"].mean())),
        ("TD Movie", EXPECTED["TD Movie"], float(long.loc[(long["group"] == "TD") & (long["state"] == "Movie"), "posterior_exponent"].mean())),
        ("TD Delta", EXPECTED["TD Delta"], delta_obs["TD"]),
    ]
    desc_ok = True
    for label, expected, observed in desc_checks:
        diff = observed - expected
        ok = abs(diff) <= DESC_TOL
        desc_ok = desc_ok and ok
        lines.append(
            f"   {label}: expected ≈ {expected:+.3f}, observed {observed:.3f}, diff {diff:+.3f} -> "
            f"{'PASS' if ok else 'FAIL'} (tol ±{DESC_TOL})"
        )

    lines.extend(["", "6. Cross-validation vs existing rest×movie tables"])
    ref_wide = pd.read_csv(REF_WIDE_PATH).dropna(subset=["rest_posterior_exponent", "movie_posterior_exponent"])
    ref_long = pd.read_csv(REF_LONG_PATH)
    set_ok = set(wide["subject_id"]) == set(ref_wide["subject_id"])
    wide_idx = wide.set_index("subject_id")
    ref_idx = ref_wide.set_index("subject_id")
    rest_diff = float((wide_idx["rest_posterior_exponent"] - ref_idx["rest_posterior_exponent"]).abs().max())
    movie_diff = float((wide_idx["movie_posterior_exponent"] - ref_idx["movie_posterior_exponent"]).abs().max())
    ref_cmp = ref_long.copy()
    ref_cmp["state"] = ref_cmp["state"].str.capitalize()
    cmp = long.merge(ref_cmp, on=["subject_id", "group", "state"], suffixes=("_new", "_ref"))
    max_long_diff = float((cmp["posterior_exponent_new"] - cmp["posterior_exponent_ref"]).abs().max())

    lines.extend(
        [
            f"   Subject set matches rest_movie_posterior_exponent_wide.csv: {set_ok}",
            f"   Max |Rest diff| vs wide table: {rest_diff:.3e}",
            f"   Max |Movie diff| vs wide table: {movie_diff:.3e}",
            f"   Max |posterior_exponent diff| vs rest_movie_posterior_exponent_long.csv: {max_long_diff:.3e}",
            "",
        ]
    )

    passed = (
        n_total == EXPECTED["N"]
        and n_asd == EXPECTED["ASD"]
        and n_td == EXPECTED["TD"]
        and len(long) == 2 * n_total
        and desc_ok
        and set_ok
        and rest_diff < 1e-12
        and movie_diff < 1e-12
        and max_long_diff < 1e-12
    )
    lines.append(f"Overall verdict: {'PASS' if passed else 'FAIL'}")
    CHECK_OUT.write_text("\n".join(lines), encoding="utf-8")
    return passed


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wide = build_wide()
    long = build_long(wide)
    long.to_csv(CSV_OUT, index=False)
    ok = write_check(wide, long)
    print(f"Wrote {CSV_OUT} ({len(long)} rows)")
    print(f"Wrote {CHECK_OUT}")
    print(f"Validation: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
