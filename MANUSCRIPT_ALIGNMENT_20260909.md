# Manuscript alignment — 2026-09-09

This note records how the public repository was updated against the submitted
Nature Communications manuscript and supplementary file
(`Manuscript/Supplementary_aperiodic_EEG_dynamics_20260909_rev`).

## Updated to match the submitted manuscript

| Item | Public repo now |
|------|-----------------|
| Partly Cloudy Aperiodic-ISC | Film-level; default `min_overlap_points=50` (`scripts/movie/22_calc_aperiodic_isc.py`) |
| Classic synchrony | Film-level envelope TD-template + within-group alpha PLV (`24_…`) |
| Permutation | `27_td_template_permutation.py` + `27b_recalibrate_permutation_p.py`; equal-tail *p* = 0.00020 shipped |
| Rest–movie matched estimator | `25d_rest_movie_matched_estimator.py`; Table S12 headline CSVs shipped (F = 4.52, *p* = 0.035) |
| Figures 1–5 / S1–S5 | `scripts/figures/main` and `supplementary`; firebrick/navy; Fig. 5B is film-level ISC |
| HBN The Present QC config | No 20% invalid-channel subject filter; movie-window R² ≥ 0.80; Welch `n_overlap: 0` |
| Nested script roots | Walk upward to `src/config.py` (previously `parents[1]` pointed at `scripts/`) |

## Provenance-only (not manuscript-facing)

These files remain but are not the submitted primary path:

- `scripts/movie/23_calc_segment_isc.py` — event-segment ISC (mentalizing / pain / neutral)
- `scripts/movie/25b_rest_movie_coupling.py`, `25c_…` — rest–movie coupling SI from an earlier draft
- `scripts/hbn/hbn_preprocess_resting.py`, `hbn_rest_eyes_open_matched.py`, `hbn_eo_posterior_exponent.py`
- `data/supplementary_source/s7_gaze_sensitivity.csv` — gaze was removed from the manuscript
- `scripts/figures/_archived_20260812/` — August figure scripts (segment violins, old SI numbering)

## Remaining wording / config nits (not silent changes)

1. **Resting `config/config.yaml` Welch `n_overlap`** may still list 250. For a 2-s epoch of 500 samples this still yields a single segment; SI Method 3 states `n_overlap` is unused. Movie and HBN The Present YAMLs were set to 0.
2. **HBN ISC overlap** remains **10** pairwise-valid points in `src/hbn_aperiodic_isc.py`, matching SI Method 5. Primary Partly Cloudy uses **50**.
3. **SI source filenames** still use historical stems (`s8_hbn_movie_subjects.csv`, `s4_development_*.csv`, `s5_iaf_*.csv`). Figure S numbering in the manuscript is S1–S5; the CSV names were not renamed so existing tables stay valid.
4. **Table S11** in the manuscript is the primary Partly Cloudy naturalistic / permutation table. The August file `statistical_tables/movie_isc_covariate_sensitivity.csv` is the older segment-level covariate table and is not Table S11.
5. **`data/participants/`** is gitignored. Permutation covariate loading uses shipped `fig5` / `fig1` tables and only optionally reads a local participant file.
6. **Supplementary Figure S1** uses the submitted overlap node **N = 104 (ASD=45, TD=59)** (`Same-subject ISC ∩ state`). The older 1:1 rest+movie matched N=92 row is no longer the S1 node.
7. **`scripts/figures/plot_supplementary_figures.py`** and the August SI plotters live under `scripts/figures/_archived_20260812/`. Use `scripts/figures/main/` and `scripts/figures/supplementary/` for the submitted figures.
8. **`src/aperiodic_isc.py`** still defaults `min_overlap=10` (legacy helper). The manuscript Partly Cloudy path is `src/aperiodic_isc_analysis.py` via `22_calc_aperiodic_isc.py` (`min_overlap_points=50`).

## What is not in this repository

- Raw EEG / `*.fif`
- Window-level exponent trajectories needed to *recompute* the 10,000 permutations
- Resting FIF files needed to *recompute* the matched-estimator sliding rest fits

Headline CSVs for those two analyses are shipped so the manuscript numbers can be inspected without EEG.
