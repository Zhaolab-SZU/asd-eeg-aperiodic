#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Recompute TD-template permutation p-values from saved null (no EEG rerun)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve()
while not (ROOT / "src" / "config.py").is_file():
    if ROOT.parent == ROOT:
        raise RuntimeError("Cannot locate repository root (missing src/config.py)")
    ROOT = ROOT.parent
PERM = ROOT / "data" / "supplementary_source" / "td_template_permutation"
OUT = PERM / "p_value_recalibration"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(PERM / "permutation_summary.csv").iloc[0]
    null = pd.read_csv(PERM / "permutation_null.csv")
    locked = pd.read_csv(PERM / "locked_participant_list.csv")

    t_obs = float(summary["T_obs"])
    tb = null["T_b"].to_numpy(dtype=float)
    b = int(tb.size)
    assert b == int(summary["B_valid"])

    # Direction: Welch is ASD vs TD; TD>ASD means more negative t (ASD mean lower).
    # One-sided for pre-specified TD > ASD: count Tb <= Tobs
    n_le = int(np.sum(tb <= t_obs))
    n_ge = int(np.sum(tb >= t_obs))
    n_abs = int(np.sum(np.abs(tb) >= abs(t_obs)))

    p_one_sided_td_gt_asd = (1.0 + n_le) / (b + 1.0)
    p_one_sided_asd_gt_td = (1.0 + n_ge) / (b + 1.0)
    p_abs = (1.0 + n_abs) / (b + 1.0)  # old (problematic if null shifted)
    p_eq_tail = min(1.0, 2.0 * min(p_one_sided_td_gt_asd, p_one_sided_asd_gt_td))

    # Center-relative (median)
    med = float(np.median(tb))
    n_cent = int(np.sum(np.abs(tb - med) >= abs(t_obs - med)))
    p_center = (1.0 + n_cent) / (b + 1.0)

    q = np.quantile(tb, [0.025, 0.25, 0.5, 0.75, 0.975])

    # Stratum exchangeability
    strata = locked.copy()
    strata["stratum"] = strata["stratum"].astype(str)
    rows = []
    for key, g in strata.groupby("stratum"):
        n = len(g)
        n_asd = int((g["group"] == "ASD").sum())
        n_td = int((g["group"] == "TD").sum())
        # exchangeable if both labels present and at least 2 subjects
        exchangeable = bool(n >= 2 and n_asd >= 1 and n_td >= 1)
        single_label = bool(n_asd == 0 or n_td == 0)
        rows.append(
            {
                "stratum": key,
                "n": n,
                "n_asd": n_asd,
                "n_td": n_td,
                "exchangeable": exchangeable,
                "single_diagnosis_only": single_label,
                "too_small_n_lt_2": n < 2,
            }
        )
    strata_df = pd.DataFrame(rows).sort_values("stratum")
    n_strata = len(strata_df)
    n_exch = int(strata_df["exchangeable"].sum())
    n_single = int(strata_df["single_diagnosis_only"].sum())
    n_tiny = int(strata_df["too_small_n_lt_2"].sum())

    calib = {
        "T_obs": t_obs,
        "B_valid": b,
        "seed": int(summary["seed"]),
        "null_mean": float(np.mean(tb)),
        "null_median": med,
        "null_sd": float(np.std(tb, ddof=1)),
        "null_p2_5": float(q[0]),
        "null_p25": float(q[1]),
        "null_p50": float(q[2]),
        "null_p75": float(q[3]),
        "null_p97_5": float(q[4]),
        "n_Tb_le_Tobs": n_le,
        "n_Tb_ge_Tobs": n_ge,
        "n_abs_Tb_ge_abs_Tobs": n_abs,
        "n_center_abs_ge": n_cent,
        "p_one_sided_td_gt_asd": p_one_sided_td_gt_asd,
        "p_one_sided_asd_gt_td": p_one_sided_asd_gt_td,
        "p_two_sided_equal_tail": p_eq_tail,
        "p_two_sided_abs_legacy": p_abs,
        "p_two_sided_center_median": p_center,
        "n_strata_total": n_strata,
        "n_strata_exchangeable": n_exch,
        "n_strata_single_diagnosis_only": n_single,
        "n_strata_n_lt_2": n_tiny,
        "note": (
            "Welch t is ASD vs TD; TD>ASD corresponds to negative t. "
            "Equal-tail two-sided recommended when null is shifted from 0."
        ),
    }
    pd.DataFrame([calib]).to_csv(OUT / "permutation_p_recalibration.csv", index=False)
    (OUT / "permutation_p_recalibration.json").write_text(
        json.dumps(calib, indent=2), encoding="utf-8"
    )
    strata_df.to_csv(OUT / "stratum_exchangeability.csv", index=False)

    print(json.dumps(calib, indent=2))
    print("\nStrata exchangeable:", n_exch, "/", n_strata)
    print(strata_df.to_string(index=False))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
