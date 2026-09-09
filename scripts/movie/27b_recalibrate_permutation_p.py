#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Manuscript-facing permutation *p* values from the saved reconstructed-template null.

This script does **not** re-run EEG or the 10,000 permutations. It reads
``data/supplementary_source/td_template_permutation/permutation_null.csv``
written by ``27_td_template_permutation.py`` and applies the equal-tail /
one-sided definitions in ``src/permutation_p.py``.

The engine script's ``p_emp_two_sided`` column is the legacy
``|Tb| >= |Tobs|`` count and is not the manuscript two-sided *p*.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve()
while not (ROOT / "src" / "config.py").is_file():
    if ROOT.parent == ROOT:
        raise RuntimeError("Cannot locate repository root (missing src/config.py)")
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.permutation_p import permutation_p_from_null, stratum_exchangeability  # noqa: E402

PERM = ROOT / "data" / "supplementary_source" / "td_template_permutation"
OUT = PERM / "p_value_recalibration"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(PERM / "permutation_summary.csv").iloc[0]
    null = pd.read_csv(PERM / "permutation_null.csv")
    locked = pd.read_csv(PERM / "locked_participant_list.csv")

    t_obs = float(summary["T_obs"])
    tb = null["T_b"].to_numpy(dtype=float)
    assert int(tb.size) == int(summary["B_valid"])

    calib = permutation_p_from_null(tb, t_obs)
    calib["seed"] = int(summary["seed"])
    strata_df = stratum_exchangeability(locked)
    calib["n_strata_total"] = int(len(strata_df))
    calib["n_strata_exchangeable"] = int(strata_df["exchangeable"].sum())
    calib["n_strata_single_diagnosis_only"] = int(strata_df["single_diagnosis_only"].sum())
    calib["n_strata_n_lt_2"] = int(strata_df["too_small_n_lt_2"].sum())

    pd.DataFrame([calib]).to_csv(OUT / "permutation_p_recalibration.csv", index=False)
    (OUT / "permutation_p_recalibration.json").write_text(
        json.dumps(calib, indent=2), encoding="utf-8"
    )
    strata_df.to_csv(OUT / "stratum_exchangeability.csv", index=False)

    print(json.dumps(calib, indent=2))
    print("\nStrata exchangeable:", calib["n_strata_exchangeable"], "/", calib["n_strata_total"])
    print(strata_df.to_string(index=False))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
