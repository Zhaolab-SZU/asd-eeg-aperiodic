#!/usr/bin/env python3
"""Permutation *engine* for Partly Cloudy Aperiodic-ISC (reconstructed TD template).

Each permutation reassigns diagnostic labels, rebuilds the pseudo-TD template,
and recomputes subject-level ISC. Does not shuffle precomputed ISC values.

This script writes T_obs and the null {T_b}. The column ``p_emp_two_sided``
is the legacy count ``(|Tb| >= |Tobs|)`` and is **not** the manuscript
two-sided *p* (the reconstructed-template null is shifted).

Manuscript-facing *p* values (one-sided TD>ASD and equal-tail two-sided)
are computed from the saved null by ``27b_recalibrate_permutation_p.py``.

Does not re-run EEG / specparam, HBN, cross-fitting, or bootstrap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / "src" / "config.py").is_file():
    if REPO_ROOT.parent == REPO_ROOT:
        raise RuntimeError("Cannot locate repository root (missing src/config.py)")
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from src.aperiodic_isc_analysis import (  # noqa: E402
    FILM_PARTLY_CLOUDY,
    FILM_TYPES_PARTLY_CLOUDY,
    build_film_timeline_keys,
    build_subject_concat_series,
    compute_td_template_isc,
    fisher_z,
    load_exponent_timeseries,
    summarize_group_isc_tests,
)

SEED = 20260908
N_PERM_VALID_TARGET = 10_000
MAX_ATTEMPTS = 50_000
N_ASD_LOCKED = 58
N_TD_LOCKED = 78
MIN_OVERLAP_POINTS = 50
MEAN_Z_TOL = 1e-4
T_TOL = 0.01

OUT_DIR = REPO_ROOT / "data" / "supplementary_source" / "td_template_permutation"

def _first_existing(*candidates: Path) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("None of these files exist:\n" + "\n".join(str(p) for p in candidates))



def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def find_timeseries() -> Path:
    named = [
        REPO_ROOT / "derivatives" / "derivatives_task_movie" / "specparam" / "specparam_exponent_timeseries_global.csv",
        REPO_ROOT / "derivatives" / "specparam" / "specparam_exponent_timeseries_global.csv",
        REPO_ROOT / "outputs" / "jr_modelling" / "posterior_movie_isc" / "posterior_sliding_exponent_timeseries.csv",
        REPO_ROOT
        / "jr_remote_bundle"
        / "outputs"
        / "jr_modelling"
        / "posterior_movie_isc"
        / "posterior_sliding_exponent_timeseries.csv",
    ]
    for p in named:
        if p.is_file():
            return p
    search_roots = [
        REPO_ROOT / "derivatives" / "derivatives_task_movie" / "specparam",
        REPO_ROOT / "derivatives" / "specparam",
        REPO_ROOT / "derivatives" / "derivatives_task_movie" / "stats",
        REPO_ROOT / "outputs",
    ]
    hits: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        hits.extend(root.glob("specparam_exponent_timeseries*.csv"))
        hits.extend(root.glob("**/posterior_sliding_exponent_timeseries.csv"))
        hits.extend(root.glob("**/partly_cloudy_exponent_concat_timeseries.csv"))
    hits = [p for p in hits if p.is_file() and "pre_qcvalid" not in p.name]
    if not hits:
        raise FileNotFoundError(
            "No window-level exponent trajectory found. Need a long table with "
            "subject_id, group, window start/end or center_sec, and exponent_mean "
            "or posterior_exponent (e.g. specparam_exponent_timeseries*.csv or "
            "posterior_sliding_exponent_timeseries.csv)."
        )
    hits.sort(key=lambda p: (0 if "specparam_exponent_timeseries" in p.name else 1, len(p.parts)))
    return hits[0]


def load_locked() -> tuple[pd.DataFrame, pd.Series]:
    subj_path = _first_existing(
        REPO_ROOT / "figure_source_data" / "fig5_partly_cloudy_aperiodic_isc_subjects.csv",
        REPO_ROOT / "data" / "figure_source" / "fig5_partly_cloudy_aperiodic_isc_subjects.csv",
        REPO_ROOT / "outputs" / "figure_source_data" / "fig5_partly_cloudy_aperiodic_isc_subjects.csv",
    )
    stats_path = _first_existing(
        REPO_ROOT / "figure_source_data" / "fig5_partly_cloudy_aperiodic_isc_stats.csv",
        REPO_ROOT / "data" / "figure_source" / "fig5_partly_cloudy_aperiodic_isc_stats.csv",
        REPO_ROOT / "outputs" / "figure_source_data" / "fig5_partly_cloudy_aperiodic_isc_stats.csv",
    )
    subj = pd.read_csv(subj_path)
    subj["subject_id"] = subj["subject_id"].astype(str)
    subj["group"] = subj["group"].astype(str).str.upper()
    n_asd = int((subj["group"] == "ASD").sum())
    n_td = int((subj["group"] == "TD").sum())
    if n_asd != N_ASD_LOCKED or n_td != N_TD_LOCKED:
        raise RuntimeError(f"Locked sample is {n_asd} ASD / {n_td} TD, expected 58/78")
    gold = pd.read_csv(stats_path).iloc[0]
    return subj, gold


def load_covariates(locked_ids: pd.Index) -> pd.DataFrame:
    """Join age/sex/FSIQ; fill remaining from the movie participant table."""
    fig5 = pd.read_csv(_first_existing(
        REPO_ROOT / "figure_source_data" / "fig5_rest_partly_cloudy_exponent.csv",
        REPO_ROOT / "data" / "figure_source" / "fig5_rest_partly_cloudy_exponent.csv",
        REPO_ROOT / "outputs" / "figure_source_data" / "fig5_rest_partly_cloudy_exponent.csv",
    ))
    fig1 = pd.read_csv(_first_existing(
        REPO_ROOT / "figure_source_data" / "fig1_resting_primary_subjects.csv",
        REPO_ROOT / "data" / "figure_source" / "fig1_resting_primary_subjects.csv",
        REPO_ROOT / "outputs" / "figure_source_data" / "fig1_resting_primary_subjects.csv",
    ))
    movie_path = REPO_ROOT / "data" / "participants" / "participants_task_movie.csv"
    movie = pd.read_csv(movie_path) if movie_path.is_file() else pd.DataFrame(columns=["subject_id"])

    def _std(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["subject_id"] = out["subject_id"].astype(str)
        colmap = {}
        if "FSIQ" in out.columns and "IQ_total" not in out.columns:
            colmap["FSIQ"] = "IQ_total"
        out = out.rename(columns=colmap)
        keep = [c for c in ["subject_id", "age_months", "sex", "IQ_total"] if c in out.columns]
        return out[keep].drop_duplicates("subject_id")

    base = pd.DataFrame({"subject_id": locked_ids.astype(str)})
    for src, suffix in ((_std(fig5), "_fig5"), (_std(fig1), "_fig1"), (_std(movie), "_movie")):
        src = src.rename(columns={c: f"{c}{suffix}" for c in src.columns if c != "subject_id"})
        base = base.merge(src, on="subject_id", how="left")
    out = pd.DataFrame({"subject_id": base["subject_id"]})
    out["age_months"] = base["age_months_fig5"].combine_first(base["age_months_fig1"]).combine_first(
        base["age_months_movie"]
    )
    out["sex"] = (
        base["sex_fig5"].combine_first(base["sex_fig1"]).combine_first(base["sex_movie"]).astype(str).str.upper()
    )
    out["IQ_total"] = base["IQ_total_fig5"].combine_first(base["IQ_total_fig1"]).combine_first(base["IQ_total_movie"])
    if out[["age_months", "sex", "IQ_total"]].isna().any().any():
        miss = out[out[["age_months", "sex", "IQ_total"]].isna().any(axis=1)]["subject_id"].tolist()
        raise RuntimeError(f"Missing covariates for locked subjects: {miss}")
    return out


def build_concat(ts_path: Path, locked: pd.DataFrame) -> pd.DataFrame:
    ids = set(locked["subject_id"].astype(str))
    gmap = locked.set_index("subject_id")["group"].to_dict()
    ts = load_exponent_timeseries(ts_path, movie_analysis_path=None, movie_qc_path=None)
    ts["subject_id"] = ts["subject_id"].astype(str)
    ts = ts[ts["subject_id"].isin(ids)].copy()
    if ts["subject_id"].nunique() != 136:
        raise RuntimeError(
            f"Trajectory covers {ts['subject_id'].nunique()} of 136 locked subjects after ID filter"
        )
    ts["group"] = ts["subject_id"].map(gmap)
    keys = build_film_timeline_keys(np.sort(ts["center_sec"].unique()), film_id=FILM_PARTLY_CLOUDY)
    concat = build_subject_concat_series(ts, keys)
    concat["group"] = concat["subject_id"].map(gmap)
    return concat


def pivot_matrix(concat: pd.DataFrame, locked: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (n_subj × n_time), subject_ids, true groups in locked-table order."""
    order = locked["subject_id"].astype(str).tolist()
    sub = concat[concat["event_type"] == FILM_PARTLY_CLOUDY].copy()
    mat = sub.pivot_table(
        index="subject_id",
        columns="concat_index",
        values="exponent_mean",
        aggfunc="mean",
    )
    mat = mat.reindex(order)
    if mat.isna().all(axis=1).any():
        missing = mat.index[mat.isna().all(axis=1)].tolist()
        raise RuntimeError(f"No trajectory rows for locked subjects: {missing}")
    X = mat.to_numpy(dtype=float)
    groups = locked["group"].astype(str).str.upper().to_numpy()
    sids = np.asarray(order, dtype=object)
    return X, sids, groups


def templates_from_labels(X: np.ndarray, is_td: np.ndarray) -> np.ndarray:
    """Leave-one-pseudo-TD-out template for TD rows; full pseudo-TD nanmean for ASD rows."""
    finite = np.isfinite(X)
    td_mask = is_td[:, None] & finite
    td_sum = np.where(td_mask, X, 0.0).sum(axis=0)
    td_cnt = td_mask.sum(axis=0).astype(float)
    full = np.divide(td_sum, td_cnt, out=np.full_like(td_sum, np.nan), where=td_cnt > 0)
    tmpl = np.repeat(full[None, :], X.shape[0], axis=0)
    td_idx = np.flatnonzero(is_td)
    if td_idx.size:
        x_td = X[td_idx]
        fin_td = finite[td_idx]
        loo_sum = td_sum[None, :] - np.where(fin_td, x_td, 0.0)
        loo_cnt = td_cnt[None, :] - fin_td.astype(float)
        loo = np.divide(loo_sum, loo_cnt, out=np.full_like(loo_sum, np.nan), where=loo_cnt > 0)
        tmpl[td_idx] = loo
    return tmpl


def row_pearson(X: np.ndarray, T: np.ndarray, min_points: int) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(X) & np.isfinite(T)
    n = mask.sum(axis=1).astype(int)
    X0 = np.where(mask, X, 0.0)
    T0 = np.where(mask, T, 0.0)
    denom_n = np.maximum(n, 1)
    mx = X0.sum(axis=1) / denom_n
    mt = T0.sum(axis=1) / denom_n
    xc = np.where(mask, X - mx[:, None], 0.0)
    tc = np.where(mask, T - mt[:, None], 0.0)
    num = (xc * tc).sum(axis=1)
    sx = np.sqrt((xc * xc).sum(axis=1))
    st = np.sqrt((tc * tc).sum(axis=1))
    den = sx * st
    r = np.full(X.shape[0], np.nan)
    ok = (n >= min_points) & (den > 1e-12)
    r[ok] = num[ok] / den[ok]
    return r, n


def isc_z_from_labels(X: np.ndarray, is_td: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tmpl = templates_from_labels(X, is_td)
    r, n_overlap = row_pearson(X, tmpl, MIN_OVERLAP_POINTS)
    z = np.full(r.shape, np.nan, dtype=float)
    ok = np.isfinite(r)
    z[ok] = np.arctanh(np.clip(r[ok], -0.999999, 0.999999))
    return z, n_overlap


def welch_t(z: np.ndarray, is_td: np.ndarray) -> tuple[float, int, int]:
    asd = z[~is_td]
    td = z[is_td]
    asd = asd[np.isfinite(asd)]
    td = td[np.isfinite(td)]
    n_asd, n_td = int(asd.size), int(td.size)
    if n_asd < 2 or n_td < 2:
        return np.nan, n_asd, n_td
    res = ttest_ind(asd, td, equal_var=False, nan_policy="omit")
    return float(res.statistic), n_asd, n_td


def make_strata(cov: pd.DataFrame, locked: pd.DataFrame) -> np.ndarray:
    df = locked[["subject_id"]].merge(cov, on="subject_id", how="left")
    sex = df["sex"].astype(str).str.upper().to_numpy()
    age_bin = pd.qcut(df["age_months"].astype(float), 3, labels=["A1", "A2", "A3"], duplicates="drop")
    iq_bin = pd.qcut(df["IQ_total"].astype(float), 3, labels=["Q1", "Q2", "Q3"], duplicates="drop")
    return np.array([f"{s}|{a}|{q}" for s, a, q in zip(sex, age_bin.astype(str), iq_bin.astype(str))], dtype=object)


def permute_labels(groups: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = groups.copy()
    for key in np.unique(strata):
        idx = np.flatnonzero(strata == key)
        out[idx] = rng.permutation(out[idx])
    return out


def qc_against_gold(isc: pd.DataFrame, gold: pd.Series, gold_subj: pd.DataFrame) -> dict:
    valid = isc[np.isfinite(isc["isc_z"])].copy()
    st = summarize_group_isc_tests(valid, FILM_TYPES_PARTLY_CLOUDY).iloc[0]
    merged = gold_subj.merge(valid, on="subject_id", suffixes=("_gold", "_re"))
    row = {
        "n_asd_recomputed": int(st["n_asd"]),
        "n_td_recomputed": int(st["n_td"]),
        "n_asd_gold": int(gold["n_asd"]),
        "n_td_gold": int(gold["n_td"]),
        "asd_mean_z_recomputed": float(st["asd_mean_z"]),
        "td_mean_z_recomputed": float(st["td_mean_z"]),
        "asd_mean_z_gold": float(gold["asd_mean_z"]),
        "td_mean_z_gold": float(gold["td_mean_z"]),
        "abs_diff_asd_mean_z": abs(float(st["asd_mean_z"]) - float(gold["asd_mean_z"])),
        "abs_diff_td_mean_z": abs(float(st["td_mean_z"]) - float(gold["td_mean_z"])),
        "t_recomputed": float(st["t_stat"]),
        "t_gold": float(gold["t_stat"]),
        "abs_diff_t": abs(float(st["t_stat"]) - float(gold["t_stat"])),
        "p_recomputed": float(st["p_value"]),
        "p_gold": float(gold["p_value"]),
        "max_abs_subject_z_diff": float((merged["isc_z_gold"] - merged["isc_z_re"]).abs().max())
        if len(merged)
        else np.nan,
        "n_subjects_merged": int(len(merged)),
        "passed_mean_z_tol": bool(
            abs(float(st["asd_mean_z"]) - float(gold["asd_mean_z"])) < MEAN_Z_TOL
            and abs(float(st["td_mean_z"]) - float(gold["td_mean_z"])) < MEAN_Z_TOL
        ),
        "passed_t_tol": bool(abs(float(st["t_stat"]) - float(gold["t_stat"])) < T_TOL),
    }
    row["passed"] = bool(row["passed_mean_z_tol"] and row["passed_t_tol"] and row["n_asd_recomputed"] == 58 and row["n_td_recomputed"] == 78)
    return row, st


def write_null_figure(tb: np.ndarray, t_obs: float, out_stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(tb, bins=60, color="#4B81B5", edgecolor="white", linewidth=0.4)
    ax.axvline(t_obs, color="#C0392B", lw=1.6, label=f"T_obs = {t_obs:.3f}")
    ax.axvline(-t_obs, color="#C0392B", lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel("Welch t (ASD vs TD)")
    ax.set_ylabel("Count")
    ax.set_title("TD-template reconstructed permutation null")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=300)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    locked, gold = load_locked()
    ts_path = find_timeseries()
    print(f"Trajectory: {_rel(ts_path)}")

    concat = build_concat(ts_path, locked)
    isc_ref = compute_td_template_isc(concat, MIN_OVERLAP_POINTS, FILM_TYPES_PARTLY_CLOUDY)
    qc_row, st = qc_against_gold(isc_ref, gold, locked)
    pd.DataFrame([qc_row]).to_csv(OUT_DIR / "qc_reproduce_welch.csv", index=False)
    print(
        f"QC recompute t={qc_row['t_recomputed']:.6f} p={qc_row['p_recomputed']:.6e} "
        f"passed={qc_row['passed']}"
    )
    if not qc_row["passed"]:
        print("STOP: recomputed Welch does not match the gold-standard fig5 stats. No permutation run.")
        return 2

    X, sids, groups = pivot_matrix(concat, locked)
    is_td_obs = groups == "TD"
    z_vec, n_ov = isc_z_from_labels(X, is_td_obs)
    t_vec, n_asd_v, n_td_v = welch_t(z_vec, is_td_obs)
    if abs(t_vec - float(st["t_stat"])) > 1e-8:
        print("STOP: vectorized ISC Welch does not match pandas compute_td_template_isc.")
        print("t_vec", t_vec, "t_pandas", float(st["t_stat"]))
        return 3

    t_obs = float(st["t_stat"])
    cov = load_covariates(locked["subject_id"])
    cov = locked[["subject_id"]].merge(cov, on="subject_id", how="left")
    strata = make_strata(cov, locked)
    n_strata = int(np.unique(strata).size)

    td_finite = np.isfinite(X[is_td_obs])
    td_n_per_window = td_finite.sum(axis=0)
    td_n_per_window = td_n_per_window[td_n_per_window > 0]

    locked_out = locked.merge(cov, on="subject_id", how="left")
    locked_out["stratum"] = strata
    locked_out.to_csv(OUT_DIR / "locked_participant_list.csv", index=False)

    rng = np.random.default_rng(SEED)
    tb_list: list[float] = []
    invalid_rows: list[dict] = []
    n_valid = 0
    n_invalid = 0
    attempt = 0
    while n_valid < N_PERM_VALID_TARGET and attempt < MAX_ATTEMPTS:
        attempt += 1
        new_g = permute_labels(groups, strata, rng)
        n_asd_lab = int((new_g == "ASD").sum())
        n_td_lab = int((new_g == "TD").sum())
        if n_asd_lab != N_ASD_LOCKED or n_td_lab != N_TD_LOCKED:
            n_invalid += 1
            invalid_rows.append(
                {
                    "attempt": attempt,
                    "reason": "label_counts",
                    "n_asd": n_asd_lab,
                    "n_td": n_td_lab,
                    "t_b": np.nan,
                }
            )
            continue
        is_td = new_g == "TD"
        z_b, _ = isc_z_from_labels(X, is_td)
        t_b, n_asd_b, n_td_b = welch_t(z_b, is_td)
        if n_asd_b != N_ASD_LOCKED or n_td_b != N_TD_LOCKED or not np.isfinite(t_b):
            n_invalid += 1
            invalid_rows.append(
                {
                    "attempt": attempt,
                    "reason": "isc_complete_case",
                    "n_asd": n_asd_b,
                    "n_td": n_td_b,
                    "t_b": t_b,
                }
            )
            continue
        tb_list.append(float(t_b))
        n_valid += 1
        if n_valid % 1000 == 0:
            print(f"  valid {n_valid}/{N_PERM_VALID_TARGET}")

    tb = np.asarray(tb_list, dtype=float)
    count_ge = int(np.sum(np.abs(tb) >= abs(t_obs))) if tb.size else 0
    p_emp = (1.0 + count_ge) / (n_valid + 1.0) if n_valid else np.nan
    direction_same = bool(np.sign(t_obs) == np.sign(np.nanmean(tb))) if tb.size else False
    if np.isfinite(p_emp) and p_emp < 0.05 and np.sign(t_obs) < 0:
        verdict = "robust"
        conclusion = (
            "Template-reconstructed permutation remains significant (p_emp<0.05) "
            "with the same TD>ASD direction; the Welch inference is robust to TD-template overlap."
        )
    elif np.isfinite(p_emp) and p_emp < 0.10 and np.sign(t_obs) < 0:
        verdict = "weakened"
        conclusion = (
            "Direction is unchanged (TD>ASD) but 0.05<=p_emp<0.10; "
            "evidence is weakened once template membership is permuted."
        )
    else:
        verdict = "not_significant"
        conclusion = (
            "Template-reconstructed permutation is not significant; "
            "the strong Welch claim must not be retained on this test."
        )

    summary = pd.DataFrame(
        [
            {
                "T_obs": t_obs,
                "p_welch_observed": float(st["p_value"]),
                "asd_mean_z": float(st["asd_mean_z"]),
                "td_mean_z": float(st["td_mean_z"]),
                "n_asd": N_ASD_LOCKED,
                "n_td": N_TD_LOCKED,
                "B_valid": n_valid,
                "B_invalid": n_invalid,
                "n_attempts": attempt,
                "n_extreme_abs_t": count_ge,
                "p_emp_two_sided": p_emp,
                "seed": SEED,
                "stratification": "sex x age_tertile x FSIQ_tertile (qcut, duplicates=drop); labels shuffled within stratum",
                "n_strata": n_strata,
                "td_contributors_per_window_min": int(td_n_per_window.min()) if td_n_per_window.size else np.nan,
                "td_contributors_per_window_median": float(np.median(td_n_per_window)) if td_n_per_window.size else np.nan,
                "min_overlap_points": MIN_OVERLAP_POINTS,
                "timeseries_relative_path": _rel(ts_path),
                "verdict": verdict,
                "direction_td_gt_asd": bool(t_obs < 0),
            }
        ]
    )
    summary.to_csv(OUT_DIR / "permutation_summary.csv", index=False)
    pd.DataFrame({"perm_index": np.arange(1, n_valid + 1), "T_b": tb}).to_csv(
        OUT_DIR / "permutation_null.csv", index=False
    )
    pd.DataFrame(
        invalid_rows,
        columns=["attempt", "reason", "n_asd", "n_td", "t_b"],
    ).to_csv(OUT_DIR / "invalid_permutation_log.csv", index=False)
    write_null_figure(tb, t_obs, OUT_DIR / "null_distribution")

    readme = f"""TD-template reconstructed permutation — Partly Cloudy Aperiodic-ISC

Trajectory (relative to repo root):
  {_rel(ts_path)}

Command:
  python3 scripts/run_td_template_permutation.py

Seed: {SEED}
Valid permutations B_valid={n_valid}; invalid B_invalid={n_invalid}
Stratification: within sex × age tertile × FSIQ tertile
T_obs (Welch ASD vs TD, equal_var=False) = {t_obs:.6f}
p_emp = (1 + count of |T_b| >= |T_obs|) / (B_valid + 1) = {p_emp:.6g}

One-sentence conclusion:
  {conclusion}
"""
    (OUT_DIR / "README.txt").write_text(readme, encoding="utf-8")
    print(readme)
    print(f"Wrote {_rel(OUT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
