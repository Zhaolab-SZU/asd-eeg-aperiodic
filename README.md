# asd-eeg-aperiodic

Analysis code and **de-identified derived data** for posterior aperiodic EEG dynamics in autistic and typically developing children (resting-state + naturalistic movie viewing).

**Repository:** https://github.com/Zhaolab-SZU/asd-eeg-aperiodic

This bundle supports the manuscript prepared for *Nature Human Behaviour*. A DOI-backed archive (Zenodo/OSF) will be linked here before publication.

## What is included

| Path | Contents |
|------|----------|
| `scripts/resting/` | Resting EEG → PSD → specparam → ROI / channel models |
| `scripts/movie/` | Partly Cloudy time-resolved aperiodic + Aperiodic-ISC + synchrony controls |
| `scripts/hbn/` | Healthy Brain Network external analyses (code only; obtain HBN under DUA) |
| `scripts/figures/` | Figure source export utilities + current main/supplementary plotting scripts |
| `src/` | Shared analysis library |
| `config/` | Relative-path YAML configs (see `config/README_PATHS.md`) |
| `data/figure_source/` | De-identified CSVs for main Figures 1–5 |
| `data/supplementary_source/` | De-identified CSVs for supplementary figures/tables |
| `data/tables/` | Additional statistical summary tables (see gap list there) |
| `figure_source_data/` | Legacy mirror of figure source tables (kept for older scripts) |

## What is **not** included

- Raw EEG recordings
- Item-level clinical instruments beyond summary ADOS totals used in analyses
- Healthy Brain Network raw BIDS data (obtain separately under the HBN data-use agreement)

## Quick start (regenerate figures from shipped CSVs)

```bash
pip install -r requirements.txt
python scripts/figures/main/make_figure1_resting_discovery_refined.py
python scripts/figures/supplementary/plot_supp_s8_rest_movie_coupling.py
```

Outputs are written under `outputs/figures/` (created at runtime; gitignored).

## Full pipeline

Resting → movie → HBN scripts expect local raw/BIDS roots configured in `config/*.yaml`.  
Replace placeholders locally; **do not commit absolute machine paths**.  
See `PIPELINE_MAP.md` for ideal script names ↔ development filenames.

## Data / code availability (manuscript wording)

De-identified participant-level derived data and analysis scripts supporting the main statistical analyses and figures are provided in this repository and will be archived with a DOI before publication. Raw EEG and item-level clinical data are not publicly available; restricted access may be requested from the corresponding author subject to ethics approval and a data-use agreement.

## Statistical summary tables (S4a / S4b / HBN FSIQ)

Shipped under `data/tables/` and `statistical_tables/`, with recompute scripts:

- `scripts/resting/10c_posterior_roi_nested_split_validation.py` (S4a)
- `scripts/resting/10d_supplementary_frontal_comparison.py` (S4b)
- `scripts/hbn/hbn_isc_fsiq_adjusted.py`
- `scripts/figures/export_statistical_summary_tables.py`

HBN subject-level CSVs use pseudonymous IDs (`HBN_0001` …); original NDAR identifiers are not included in this public bundle.

## License

Code: MIT (`LICENSE`).  
Derived tables: intended for open research use with the manuscript; prefer CC-BY 4.0 if depositing separately on Zenodo.
