# asd-eeg-aperiodic

Code and de-identified derived data for posterior aperiodic EEG dynamics in autistic and typically developing children during resting-state and naturalistic movie viewing.

Aligned to the Nature Communications submission of **9 September 2026**.

## Overview

- Resting-state spectral parameterization and a data-informed posterior exponent readout
- Rest-to-movie modulation of the condition-averaged posterior exponent
- Film-level Aperiodic-ISC during *Partly Cloudy* (TD-template; `min_overlap=50`)
- Classic synchrony controls (broadband envelope ISC; within-group alpha PLV)
- Reconstructed-template permutation inference (equal-tail two-sided *p* = 0.00020)
- Estimator-matched rest–movie sensitivity (Table S12)
- External HBN *The Present* Aperiodic-ISC (directionally consistent, not replication)

Raw EEG is not included.

## Reproduce manuscript figures from shipped CSVs

From the repository root:

```bash
python scripts/figures/main/build_main_figures.py
python scripts/figures/supplementary/build_si_figures.py
```

Outputs go to `outputs/figures/main/` and `outputs/figures/supplementary/`.
Palette: ASD `#B22222`, TD `#000080`.

Source tables live in `figure_source_data/` (mirrored under `data/figure_source/`).

## Analysis scripts (need local EEG / trajectories)

Run from the **repository root** so `config/` and `src/` resolve.

```bash
# Film-level Partly Cloudy Aperiodic-ISC
python scripts/movie/22_calc_aperiodic_isc.py --min_overlap_points 50

# Classic synchrony controls
python scripts/movie/24_sync_control_classic_isc.py --min_overlap_points 50

# Covariate-adjusted ISC models
python scripts/movie/26b_partly_cloudy_isc_models.py

# TD-template permutation (slow; B=10000) then equal-tail recalibration
python scripts/movie/27_td_template_permutation.py
python scripts/movie/27b_recalibrate_permutation_p.py

# Rest–movie mean-level analysis + matched-estimator sensitivity
python scripts/movie/25_rest_movie_state_analysis.py
python scripts/movie/25d_rest_movie_matched_estimator.py

# HBN The Present
python scripts/hbn/hbn_isc_calc.py
```

The permutation **null CSV is shipped**, so `27b_recalibrate_permutation_p.py` can be run without EEG. Re-running `27_td_template_permutation.py` requires `specparam_exponent_timeseries_global.csv`.

Primary permutation *p* values in the manuscript are:

- one-sided TD > ASD: 9.999e-05
- equal-tail two-sided: 0.00020

Do not use `|Tb| ≥ |Tobs|` as the primary two-sided *p* (the reconstructed-template null is shifted).

## Repository layout

| Path | Description |
|------|-------------|
| `src/` | Shared analysis code |
| `config/` | YAML configs (relative paths; placeholders for raw EEG / HBN BIDS) |
| `scripts/resting/` | Resting-state pipeline |
| `scripts/movie/` | Partly Cloudy Aperiodic-ISC, synchrony, permutation, rest–movie |
| `scripts/hbn/` | HBN The Present movie analyses |
| `scripts/figures/main/` | Manuscript Figures 1–5 |
| `scripts/figures/supplementary/` | Supplementary Figures S1–S5 |
| `scripts/figures/_archived_20260812/` | Pre-film-level plotting scripts (not manuscript-facing) |
| `figure_source_data/` | Figure / table source CSVs |
| `data/supplementary_source/` | SI source tables, permutation null, matched-estimator headlines |

## Notes

Nested scripts locate the repository by walking upward until `src/config.py` is found. Run them from any working directory.

HBN resting-state scripts remain for provenance. They are not part of the submitted manuscript-facing external analysis.

Event-segment Aperiodic-ISC (`scripts/movie/23_calc_segment_isc.py`) is retained for provenance and is not the primary naturalistic contrast.

See `MANUSCRIPT_ALIGNMENT_20260909.md` for remaining script-vs-text notes.

## License

Code is released under the MIT license. Derived tables are provided for research reuse with appropriate citation.
