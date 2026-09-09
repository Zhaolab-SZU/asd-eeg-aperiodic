# Movie scripts

## Manuscript-facing (use these)

| Script | Role | Needs EEG? |
|--------|------|------------|
| `20_movie_prepare_participants.py` | Movie cohort table | local participants file |
| `20b_align_exponent_movie_labels.py` | Align resting exponents with movie IDs | derivatives |
| `21_time_resolved_specparam_isc.py` | Sliding-window specparam trajectories | movie EEG |
| `22_calc_aperiodic_isc.py` | **Primary** film-level TD-template Aperiodic-ISC (`min_overlap=50`) | trajectories |
| `24_sync_control_classic_isc.py` | Envelope ISC + within-group alpha PLV | movie EEG |
| `25_rest_movie_state_analysis.py` | Primary rest–movie mean-level state × group | rest+movie exponents |
| `25d_rest_movie_matched_estimator.py` | Table S12 estimator-matched sensitivity | resting FIF |
| `26b_partly_cloudy_isc_models.py` | Age/sex/FSIQ and data-quality ISC models | ISC + covariates |
| `27_td_template_permutation.py` | Permutation **engine** (rebuild template; writes null) | trajectories |
| `27b_recalibrate_permutation_p.py` | Manuscript **p-values** from the saved null | shipped null CSV |

`27_` vs `27b`: `27_` produces T_obs and {T_b}. `27b` is the paper's p-value definition (one-sided TD>ASD and equal-tail two-sided). The engine's `p_emp_two_sided` (`|Tb| ≥ |Tobs|`) is not the manuscript two-sided p.

## Not manuscript-facing

Moved to `scripts/legacy/movie/`: event-segment ISC, older rest–movie coupling, segment covariate models, envelope-partial ISC.
