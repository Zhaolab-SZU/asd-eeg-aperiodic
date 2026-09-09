TD-template reconstructed permutation — Partly Cloudy Aperiodic-ISC

Locked cohort: 58 ASD / 78 TD
Trajectory (not shipped; needed only to re-run the 10,000 permutations):
  derivatives/derivatives_task_movie/specparam/specparam_exponent_timeseries_global.csv

Scripts (from repository root):
  python scripts/movie/27_td_template_permutation.py
  python scripts/movie/27b_recalibrate_permutation_p.py

The null CSV is shipped, so `27b_recalibrate_permutation_p.py` can be run without EEG.

Seed: 20260908
Valid permutations B_valid=10000; invalid B_invalid=0
Stratification: within sex × age tertile × FSIQ tertile
  18 strata; 14 exchangeable; 4 TD-only (no within-stratum label swaps)
T_obs (Welch ASD vs TD, equal_var=False) = -5.575613
Observed Welch p = 1.95e-7

Primary permutation p values (manuscript):
  one-sided TD>ASD (left tail) = 9.999e-05
  equal-tail two-sided         = 0.00020
Do not use |Tb| >= |Tobs| as the primary two-sided p (null median ≈ -1.04).
That legacy absolute-value p remains 9.999e-05 in permutation_summary.csv.

Recalibrated table:
  permutation_summary_recalibrated.csv
  p_value_recalibration/permutation_p_recalibration.csv
  p_value_recalibration/stratum_exchangeability.csv
