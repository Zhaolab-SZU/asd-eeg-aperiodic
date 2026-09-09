# asd-eeg-aperiodic

Code and de-identified derived tables for posterior aperiodic EEG dynamics in autistic and typically developing children (resting-state and naturalistic movie viewing).

Aligned to the Nature Communications submission of **9 September 2026**.
Tag this snapshot as **`v1.0-submitted`**.

## Reproduce the manuscript

Three layers. Stop at the first layer that answers your question.

### 1. Figures from shipped CSVs (no EEG)

```bash
python -m pip install -r requirements-ci.txt matplotlib Pillow
python scripts/figures/main/build_main_figures.py
python scripts/figures/supplementary/build_si_figures.py
python -m pytest
```

Outputs: `outputs/figures/main/` and `outputs/figures/supplementary/`.
Palette: ASD `#B22222`, TD `#000080`.

### 2. Manuscript statistics from shipped derived tables (no EEG)

| Result | Script | Required input (shipped) | Output |
|--------|--------|--------------------------|--------|
| Fig. 5B Partly Cloudy Aperiodic-ISC (n=58/78, Welch *t*) | inspect CSV | `figure_source_data/fig5_partly_cloudy_aperiodic_isc_*.csv` | same CSV |
| Permutation *p* (one-sided 9.999e-05; equal-tail 0.00020) | `scripts/movie/27b_recalibrate_permutation_p.py` | `data/supplementary_source/td_template_permutation/permutation_null.csv` | `…/p_value_recalibration/` |
| Table S12 matched-estimator headline | inspect CSV | `data/supplementary_source/rest_movie_matched_estimator/matched_estimator_headline.csv` | F=4.52, *p*=0.035 |
| HBN The Present ISC (Fig. 5B / S5) | inspect CSV | `figure_source_data/fig5_the_present_aperiodic_isc*.csv` | same CSV |

`27_td_template_permutation.py` is the permutation **engine** (rebuilds the TD template under label shuffles; needs exponent trajectories). `27b_recalibrate_permutation_p.py` is the manuscript-facing **p-value implementation** applied to the saved null. Do not treat `27_`'s column `p_emp_two_sided` (`|Tb| ≥ |Tobs|`) as the paper's two-sided *p*.

### 3. Full recompute (local EEG / HBN BIDS; not shipped)

```bash
python -m pip install -r requirements.txt   # or: conda env create -f environment.yml
# Film-level Partly Cloudy Aperiodic-ISC
python scripts/movie/22_calc_aperiodic_isc.py --min_overlap_points 50
python scripts/movie/24_sync_control_classic_isc.py --min_overlap_points 50
python scripts/movie/26b_partly_cloudy_isc_models.py
python scripts/movie/27_td_template_permutation.py   # slow; B=10000
python scripts/movie/27b_recalibrate_permutation_p.py
python scripts/movie/25_rest_movie_state_analysis.py
python scripts/movie/25d_rest_movie_matched_estimator.py
python scripts/hbn/hbn_isc_calc.py
```

Primary permutation *p* values: one-sided TD > ASD = 9.999e-05; equal-tail two-sided = 0.00020.

## What you can and cannot rerun

| Layer | Public files enough? | Needs restricted raw EEG / HBN BIDS |
|-------|----------------------|-------------------------------------|
| Build Fig. 1–5 / S1–S5 | Yes | No |
| Recompute permutation *p* from the shipped null (`27b`) | Yes | No |
| Re-run 10,000 reconstructed-template permutations (`27_`) | No (needs `specparam_exponent_timeseries_global.csv`) | Trajectories from movie EEG |
| Matched-estimator rest sliding fits (`25d`) | Headline CSV yes; full refit no | Resting FIF |
| specparam / ISC from scratch | No | Raw EEG / HBN under DUA |

Nested scripts walk upward until `src/config.py` is found.

## Manuscript-facing movie / HBN scripts

See `scripts/movie/README.md` and `scripts/hbn/README.md`. Event-segment ISC, older rest–movie coupling, and HBN resting pipelines live in `scripts/legacy/` and are not the submitted primary path.

## Dependencies

Pinned in `requirements.txt` (analysis, **specparam==2.0.0**) and `requirements-ci.txt` (tests). `environment.yml` installs the analysis pin on Python 3.11. Do not upgrade specparam without re-checking `src/specparam_utils.py`.

## Fisher *z*

Subject-level Aperiodic-ISC is `z = atanh(r)` per subject, then group tests use **mean of z**. Auxiliary pairwise / time-resolved summaries now store `mean_pairwise_z` / `mean_loo_z` as mean of *z*; `z_of_mean_*_r` is `z(mean r)` and is not used for inference.

## License and citation

MIT (`LICENSE`). Cite via `CITATION.cff` / the submitted article *Aperiodic EEG dynamics reveal reduced naturalistic temporal alignment in childhood autism*.
