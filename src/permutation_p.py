"""Manuscript p-value definitions for the reconstructed-template permutation.

``scripts/movie/27_td_template_permutation.py`` is the permutation *engine*:
it rebuilds the TD template under stratified label shuffles and writes T_obs
and the null {T_b}. The ``p_emp_two_sided`` column it writes is the legacy
``(|Tb| >= |Tobs|)`` count and is **not** the manuscript two-sided p.

``scripts/movie/27b_recalibrate_permutation_p.py`` applies the functions
below to the shipped null. Those values are manuscript-facing:

- one-sided TD > ASD: P(Tb <= Tobs)
- two-sided equal-tail: 2 * min(left, right), capped at 1

Welch t is ASD vs TD, so TD > ASD is a more negative t.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def permutation_p_from_null(tb: np.ndarray, t_obs: float) -> dict[str, Any]:
    tb = np.asarray(tb, dtype=float)
    b = int(tb.size)
    if b < 1:
        raise ValueError("Empty permutation null")

    n_le = int(np.sum(tb <= t_obs))
    n_ge = int(np.sum(tb >= t_obs))
    n_abs = int(np.sum(np.abs(tb) >= abs(t_obs)))
    p_one_sided_td_gt_asd = (1.0 + n_le) / (b + 1.0)
    p_one_sided_asd_gt_td = (1.0 + n_ge) / (b + 1.0)
    p_abs = (1.0 + n_abs) / (b + 1.0)
    p_eq_tail = min(1.0, 2.0 * min(p_one_sided_td_gt_asd, p_one_sided_asd_gt_td))

    med = float(np.median(tb))
    n_cent = int(np.sum(np.abs(tb - med) >= abs(t_obs - med)))
    p_center = (1.0 + n_cent) / (b + 1.0)
    q = np.quantile(tb, [0.025, 0.25, 0.5, 0.75, 0.975])

    return {
        "T_obs": float(t_obs),
        "B_valid": b,
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
        "note": (
            "Welch t is ASD vs TD; TD>ASD corresponds to negative t. "
            "Equal-tail two-sided recommended when null is shifted from 0."
        ),
    }


def stratum_exchangeability(locked: pd.DataFrame) -> pd.DataFrame:
    strata = locked.copy()
    strata["stratum"] = strata["stratum"].astype(str)
    rows = []
    for key, g in strata.groupby("stratum"):
        n = len(g)
        n_asd = int((g["group"] == "ASD").sum())
        n_td = int((g["group"] == "TD").sum())
        rows.append(
            {
                "stratum": key,
                "n": n,
                "n_asd": n_asd,
                "n_td": n_td,
                "exchangeable": bool(n >= 2 and n_asd >= 1 and n_td >= 1),
                "single_diagnosis_only": bool(n_asd == 0 or n_td == 0),
                "too_small_n_lt_2": n < 2,
            }
        )
    return pd.DataFrame(rows).sort_values("stratum")
