# asd-eeg-aperiodic

Code and de-identified derived data for analyses of posterior aperiodic EEG activity in autistic and typically developing children during resting-state and naturalistic movie viewing.

## Overview

This repository contains the analysis scripts, configuration files, and derived source data used to generate the manuscript figures and statistical tables. Raw EEG recordings are not included.

The analyses cover:

- resting-state EEG spectral parameterization
- posterior-region aperiodic exponent analyses
- naturalistic movie Aperiodic-ISC analyses
- rest-movie coupling analyses
- external Healthy Brain Network The Present movie convergence analyses

## Repository Structure

| Path | Description |
|------|-------------|
| `src/` | Shared analysis code |
| `config/` | YAML configuration files with relative paths |
| `scripts/resting/` | Resting-state EEG analysis scripts |
| `scripts/movie/` | Movie-viewing Aperiodic-ISC analysis scripts |
| `scripts/hbn/` | Healthy Brain Network The Present analysis scripts |
| `scripts/figures/` | Figure and source-data export scripts |
| `data/figure_source/` | Source data for main figures |
| `data/supplementary_source/` | Source data for supplementary figures and tables |
| `data/tables/` | Statistical summary tables |
| `figure_source_data/` | Legacy source-data mirror used by older plotting scripts |

## Installation

```bash
pip install -r requirements.txt
```

## Reproducing Figures

Figures can be regenerated from the shipped derived CSV files:

```bash
python scripts/figures/main/make_figure1_resting_discovery_refined.py
python scripts/figures/supplementary/plot_supp_s8_rest_movie_coupling.py
```

Generated files are written to `outputs/figures/`.

## Data Access

This repository includes de-identified derived data only. It does not include:

- raw EEG recordings
- item-level clinical instrument data
- raw Healthy Brain Network BIDS data

Healthy Brain Network data should be obtained directly from the HBN data portal under the applicable data-use agreement.

## Notes

Configuration files use paths relative to the repository root. Local raw-data paths should be set on the user's machine and should not be committed.

Earlier exploratory HBN resting-state scripts are retained for provenance, but the current external-validation analysis uses the HBN The Present movie cohort.

## License

Code is released under the MIT license. Derived tables are provided for research reuse with appropriate citation.
