"""Number locks for the Nature Mental Health submitted manuscript.

These tests use shipped de-identified tables only. They fail if helper behavior,
cohort counts or submitted manuscript values drift silently.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aperiodic_isc_analysis import fisher_z, mean_fisher_z, z_of_mean_r  # noqa: E402
from src.permutation_p import permutation_p_from_null  # noqa: E402

PERM = ROOT / "data" / "supplementary_source" / "td_template_permutation"
FIG = ROOT / "figure_source_data"
S1 = FIG / "supplementary" / "s1_cohort_flow.csv"
TABLES = ROOT / "data" / "manuscript_tables"


def test_imports():
    import src.aperiodic_isc_analysis  # noqa: F401
    import src.permutation_p  # noqa: F401
    import yaml  # noqa: F401
    import src.config  # noqa: F401


def test_fisher_z_subject_level_then_mean():
    rs = np.array([0.10, 0.20, 0.30])
    zs = np.array([fisher_z(r) for r in rs])
    assert mean_fisher_z(rs) == pytest.approx(float(zs.mean()))
    assert mean_fisher_z(rs) != pytest.approx(z_of_mean_r(rs))
    assert z_of_mean_r(rs) == pytest.approx(fisher_z(float(rs.mean())))


def test_partly_cloudy_welch_lock():
    stats = pd.read_csv(FIG / "fig5_partly_cloudy_aperiodic_isc_stats.csv").iloc[0]
    subj = pd.read_csv(FIG / "fig5_partly_cloudy_aperiodic_isc_subjects.csv")
    assert int(stats["n_asd"]) == 58
    assert int(stats["n_td"]) == 78
    assert int((subj["group"] == "ASD").sum()) == 58
    assert int((subj["group"] == "TD").sum()) == 78
    assert float(stats["t_stat"]) == pytest.approx(-5.5756125733008375, abs=1e-8)
    assert float(stats["p_value"]) == pytest.approx(1.9477003661847367e-07, rel=1e-6)
    assert (
        subj.loc[subj["group"] == "TD", "isc_z"].mean()
        - subj.loc[subj["group"] == "ASD", "isc_z"].mean()
    ) == pytest.approx(0.0550737044298426, abs=1e-10)


def test_permutation_equal_tail_from_shipped_null():
    summary = pd.read_csv(PERM / "permutation_summary.csv").iloc[0]
    null = pd.read_csv(PERM / "permutation_null.csv")
    calibrated = permutation_p_from_null(
        null["T_b"].to_numpy(dtype=float), float(summary["T_obs"])
    )
    assert int(summary["n_asd"]) == 58
    assert int(summary["n_td"]) == 78
    assert int(summary["B_valid"]) == 10_000
    assert int(summary["seed"]) == 20260908
    assert calibrated["p_one_sided_td_gt_asd"] == pytest.approx(
        9.999000099990002e-05, abs=1e-12
    )
    assert calibrated["p_two_sided_equal_tail"] == pytest.approx(
        0.0001999800019998, abs=1e-12
    )
    assert calibrated["p_two_sided_abs_legacy"] != pytest.approx(
        calibrated["p_two_sided_equal_tail"]
    )


def test_s1_same_subject_overlap():
    s1 = pd.read_csv(S1)
    row = s1.loc[s1["stage_order"] == 5].iloc[0]
    assert (int(row["n_total"]), int(row["n_asd"]), int(row["n_td"])) == (104, 45, 59)


def test_hbn_submitted_template_correction_lock():
    table = pd.read_csv(TABLES / "table_s13_hbn_template_dependency.csv")
    primary = table.loc[
        table["Analysis"].eq("The Present sliding-window Aperiodic-ISC (primary)")
    ].iloc[0]
    reconstructed = table.loc[
        table["Analysis"].eq("Sliding reconstructed-template permutation")
    ].iloc[0]
    crossfit = table.loc[
        table["Analysis"].eq("Strict five-fold cross-fitted sliding-window")
    ].iloc[0]
    assert primary["n (ASD/TD)"] == "119/119"
    assert float(primary["p"]) == pytest.approx(0.00994)
    assert reconstructed["p"] == "equal-tail p = 0.032"
    assert crossfit["p"] == "equal-tail p = 0.018"


def test_submitted_group_by_age_locks():
    table = pd.read_csv(TABLES / "table_s18_group_by_age_models.csv")
    primary = table.loc[table["Analysis"].eq("Primary HC3 (A0)")].iloc[0]
    bootstrap = table.loc[
        table["Analysis"].eq("Template-reconstruction bootstrap")
    ].iloc[0]
    hbn = table.loc[table["Analysis"].eq("HBN full-sample HC3")].iloc[0]
    assert int(primary["N"]) == 136
    assert primary["Interaction β/year"] == "−0.0121"
    assert float(primary["p"]) == pytest.approx(0.0495)
    assert bootstrap["95% CI"] == "[−0.0233, 0.0019]"
    assert int(hbn["N"]) == 238
    assert hbn["Interaction β/year"] == "−0.0020"
    assert float(hbn["p"]) == pytest.approx(0.856)


def test_submitted_trajectory_adjustment_lock():
    table = pd.read_csv(TABLES / "table_s17b_staged_models.csv")
    m4 = table.loc[table["Model"].eq("M4")].iloc[0]
    assert float(m4["Raw TD−ASD β"]) == pytest.approx(0.0545)
    assert m4["95% CI"] == "[0.0306, 0.0785]"
    assert float(m4["Standardized β"]) == pytest.approx(0.895)
