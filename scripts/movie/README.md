# Movie scripts

## Submitted primary path

| Script | Role | Needs EEG/trajectories? |
|---|---|---|
| `20_movie_prepare_participants.py` | Movie cohort table | participant registry |
| `20b_align_exponent_movie_labels.py` | Align resting exponents and movie IDs | derivatives |
| `21_time_resolved_specparam_isc.py` | Sliding-window exponent trajectories | movie EEG |
| `22_calc_aperiodic_isc.py` | Film-level TD-template Aperiodic-ISC | trajectories |
| `24_sync_control_classic_isc.py` | Envelope ISC and alpha PLV controls | movie EEG |
| `26b_partly_cloudy_isc_models.py` | Covariate/data-quality models | derived values |
| `27_td_template_permutation.py` | Reconstructed-template permutation engine | trajectories |
| `27b_recalibrate_permutation_p.py` | Equal-tail and one-sided manuscript p values | shipped null |

The permutation engine produces the observed statistic and null draws. The manuscript-facing two-sided p value is the equal-tail value computed by `27b`; the legacy absolute-statistic p value is not interchangeable with it.

## Submitted secondary/exploratory analyses awaiting v6 transfer

| Required script | Role |
|---|---|
| `film_level_isc_ados_exploratory.py` | Aperiodic-ISC–ADOS analysis |
| `aperiodic_isc_mechanism_v2.py` | Temporal-shift, lag and trajectory controls |
| `film_level_isc_age_v3.py` | Primary-cohort Group × Age |

Rest-to-movie state analyses are retained as exploratory Supplementary material and are not part of the principal main-text narrative. Event-segment ISC and older coupling analyses remain under `scripts/legacy/movie/`.

See `analysis_requests/ANALYSIS_WAREHOUSE_COMPLETION_v6.md` for commands, seeds, expected outputs and numerical locks.
