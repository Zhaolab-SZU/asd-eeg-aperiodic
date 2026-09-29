# asd-eeg-aperiodic

Code and de-identified participant-level derived data for posterior aperiodic EEG dynamics in autistic and typically developing children during resting state and naturalistic movie viewing.

This repository is aligned to the manuscript submitted to **Nature Mental Health on 28 September 2026**:

> *Aperiodic EEG dynamics reveal reduced naturalistic temporal alignment in childhood autism*

The release corresponding to the submitted manuscript is **`v2.0-nmh-submitted`**. Until that release is tagged, use the `codex/nmh-submitted-20260928` branch for the manuscript-aligned snapshot.

## Reproduce the submitted manuscript

### 1. Figures from shipped source data (no EEG required)

```bash
python -m pip install -r requirements-ci.txt matplotlib Pillow
python scripts/figures/main/build_main_figures.py
python scripts/figures/supplementary/build_si_figures.py
python -m pytest
```

Outputs are written to `outputs/figures/main/` and `outputs/figures/supplementary/`.
The submitted main-figure order is:

1. resting posterior aperiodic activity;
2. spatial robustness;
3. naturalistic Aperiodic-ISC in Partly Cloudy and HBN The Present;
4. age-related resting posterior exponent analyses;
5. resting posterior exponent–ADOS associations.

### 2. Locked manuscript statistics from shipped derived tables

| Submitted result | Public input | Manuscript-facing inference |
|---|---|---|
| Partly Cloudy Aperiodic-ISC (58 ASD, 78 TD) | `figure_source_data/fig5_partly_cloudy_aperiodic_isc_*.csv` | Welch *p*=1.95×10⁻⁷; reconstructed-template equal-tail permutation *p*=0.00020 |
| HBN The Present Aperiodic-ISC (119 ASD, 119 TD) | `figure_source_data/fig5_the_present_aperiodic_isc*.csv` | Welch *p*=0.0099; template-corrected permutation *p*=0.032; strict cross-fit *p*=0.018 |
| Primary-cohort Aperiodic-ISC Group × Age | `data/manuscript_tables/table_s18_group_by_age_models.csv` | nominal HC3 *p*=0.0495; reconstructed-template interval spans zero |
| HBN Aperiodic-ISC Group × Age | `data/manuscript_tables/table_s18_group_by_age_models.csv` | no evidence for interaction across prespecified checks |
| Film-level Aperiodic-ISC–ADOS | `data/manuscript_tables/table_s11_partly_cloudy_aperiodic_isc.csv` | no robust association |

The submitted Supplementary Tables are exported verbatim to `data/manuscript_tables/`. They are the numerical ground truth for the public release. See `SUBMITTED_MANUSCRIPT_ALIGNMENT_20260928.md` for the manuscript-to-code map and outstanding reproducibility items.

### 3. Full recomputation (restricted local EEG or public HBN BIDS required)

```bash
python -m pip install -r requirements.txt

# Primary Partly Cloudy pipeline
python scripts/movie/22_calc_aperiodic_isc.py --min_overlap_points 50
python scripts/movie/24_sync_control_classic_isc.py --min_overlap_points 50
python scripts/movie/26b_partly_cloudy_isc_models.py
python scripts/movie/27_td_template_permutation.py
python scripts/movie/27b_recalibrate_permutation_p.py

# HBN The Present base pipeline
python scripts/hbn/hbn_isc_calc.py
```

The v3–v5 manuscript analyses require additional scripts and machine-readable outputs from the analysis workspace. Their exact commands, seeds, file manifest, privacy constraints and acceptance tests are specified in `analysis_requests/ANALYSIS_WAREHOUSE_COMPLETION_v6.md`. The public repository must not be described as fully end-to-end reproducible until that manifest is satisfied.

## Public versus restricted material

| Layer | Publicly shipped | Restricted / obtained separately |
|---|---|---|
| Figure source data and submitted summary tables | Yes | — |
| De-identified participant-level derived measures | Yes, where permitted | — |
| Primary and sensitivity analysis code | Yes or explicitly tracked in the v6 completion manifest | — |
| Raw primary-cohort EEG and item-level clinical records | No | Institutional governance and consent restrictions apply |
| HBN raw EEG | No local redistribution | Obtain through the HBN EEG BIDS/NEMAR resources under their terms |

No raw EEG, direct identifiers, dates of birth or item-level clinical records belong in this repository.

## Locked software environment

The submitted analyses used Python 3.12.2, MNE 1.10.1, specparam 2.0.0rc6, mne-icalabel 0.8.1, NumPy 2.3.3, SciPy 1.16.1, pandas 2.3.3, statsmodels 0.14.6, scikit-learn 1.7.2, matplotlib 3.10.7, seaborn 0.13.2, pingouin 0.6.1, mne-connectivity 0.8.1 and autoreject 0.4.4. See `requirements.txt` and `environment.yml`.

## Fisher *z* convention

Subject-level Aperiodic-ISC is `z = atanh(r)` per subject, and group inference uses the mean of subject-level *z*. Auxiliary summaries distinguish the mean of Fisher-*z* values from `z(mean r)`.

## License and citation

MIT (`LICENSE`). Citation metadata are provided in `CITATION.cff`. An archival DOI will be added to both the manuscript and repository metadata when deposited.
