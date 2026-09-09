"""Tiny tests that lock manuscript numbers to shipped tables / helpers.

These do not require EEG, MNE, or specparam. They fail if a helper drift
silently changes Welch t, permutation p, sample size, or Fisher-z averaging.
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
FIG5 = ROOT / "figure_source_data"
S1 = ROOT / "figure_source_data" / "supplementary" / "s1_cohort_flow.csv"
HEADLINE = ROOT / "data" / "supplementary_source" / "rest_movie_matched_estimator" / "matched_estimator_headline.csv"


def test_imports():
    import src.aperiodic_isc_analysis  # noqa: F401
    import src.permutation_p  # noqa: F401
    import yaml  # CI dependency; config loader needs it
    import src.config  # noqa: F401


def test_fisher_z_subject_level_then_mean():
    rs = np.array([0.10, 0.20, 0.30])
    zs = np.array([fisher_z(r) for r in rs])
    assert mean_fisher_z(rs) == pytest.approx(float(zs.mean()))
    assert mean_fisher_z(rs) != pytest.approx(z_of_mean_r(rs))
    assert z_of_mean_r(rs) == pytest.approx(fisher_z(float(rs.mean())))


def test_fig5_partly_cloudy_welch_lock():
    stats = pd.read_csv(FIG5 / "fig5_partly_cloudy_aperiodic_isc_stats.csv").iloc[0]
    subj = pd.read_csv(FIG5 / "fig5_partly_cloudy_aperiodic_isc_subjects.csv")
    assert int(stats["n_asd"]) == 58
    assert int(stats["n_td"]) == 78
    assert int((subj["group"] == "ASD").sum()) == 58
    assert int((subj["group"] == "TD").sum()) == 78
    assert float(stats["t_stat"]) == pytest.approx(-5.5756125733008375, rel=0, abs=1e-8)
    assert float(stats["p_value"]) == pytest.approx(1.9477003661847367e-07, rel=1e-6)
    asd_z = subj.loc[subj["group"] == "ASD", "isc_z"].mean()
    td_z = subj.loc[subj["group"] == "TD", "isc_z"].mean()
    assert float(stats["asd_mean_z"]) == pytest.approx(asd_z, rel=0, abs=1e-10)
    assert float(stats["td_mean_z"]) == pytest.approx(td_z, rel=0, abs=1e-10)


def test_permutation_equal_tail_from_shipped_null():
    summary = pd.read_csv(PERM / "permutation_summary.csv").iloc[0]
    null = pd.read_csv(PERM / "permutation_null.csv")
    calib = permutation_p_from_null(null["T_b"].to_numpy(dtype=float), float(summary["T_obs"]))
    assert int(summary["n_asd"]) == 58
    assert int(summary["n_td"]) == 78
    assert int(summary["B_valid"]) == 10_000
    assert int(summary["seed"]) == 20260908
    assert calib["p_one_sided_td_gt_asd"] == pytest.approx(9.999000099990002e-05, rel=0, abs=1e-12)
    assert calib["p_two_sided_equal_tail"] == pytest.approx(0.0001999800019998, rel=0, abs=1e-12)
    shipped = pd.read_csv(PERM / "p_value_recalibration" / "permutation_p_recalibration.csv").iloc[0]
    assert float(shipped["p_two_sided_equal_tail"]) == pytest.approx(
        calib["p_two_sided_equal_tail"], rel=0, abs=1e-12
    )
    # Legacy |Tb|>=|Tobs| is not the manuscript two-sided p here.
    assert calib["p_two_sided_abs_legacy"] != pytest.approx(calib["p_two_sided_equal_tail"])


def test_s1_overlap_node_and_matched_estimator():
    s1 = pd.read_csv(S1)
    row = s1.loc[s1["stage_order"] == 5].iloc[0]
    assert int(row["n_total"]) == 104
    assert int(row["n_asd"]) == 45
    assert int(row["n_td"]) == 59
    h = pd.read_csv(HEADLINE).iloc[0]
    assert int(h["n"]) == 136
    assert float(h["mixed_anova_interaction_F"]) == pytest.approx(4.515417210442791, rel=1e-8)
    assert float(h["mixed_anova_interaction_p"]) == pytest.approx(0.035426246645072684, rel=1e-8)
