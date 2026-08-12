# Statistical summary tables

Place de-identified summary CSVs that underpin manuscript/SI tables here.

Do not place raw EEG or item-level clinical raw responses in this folder.

## Contents

| File | Manuscript / SI target | Source script |
|------|------------------------|---------------|
| `table1_participant_characteristics.csv` | Main Table 1 participant characteristics | Current manuscript table source, locked to `Manuscript_20260812(q).docx` |
| `table_s4a_nested_split_summary_repeats.csv` | Supplementary Table S4a (primary: 200× 70/30) | `scripts/28_posterior_roi_nested_split_validation.py` (+ `src/posterior_roi_nested_split.py`) |
| `table_s4a_nested_split_summary_kfold.csv` | S4a (5-fold nested) | same |
| `table_s4a_nested_split_single50.csv` | S4a (single 50/50 split) | same |
| `table_s4a_channel_selection_frequency_repeats.csv` | S4a channel selection rates | same |
| `table_s4b_frontal_group_ols.csv` | Supplementary Table S4b (region group OLS) | `scripts/29_supplementary_frontal_comparison.py` (+ `src/frontal_comparison_analysis.py`) |
| `table_s4b_frontal_group_age_interaction.csv` | S4b (group × age) | same |
| `table_s4b_frontal_ados_partial.csv` | S4b (ADOS partial Pearson) | same |
| `table_s4b_frontal_vs_posterior_interaction.csv` | S4b (group × region key terms) | same |
| `table_s4b_movie_isc_effects.csv` | S4b exploratory movie ISC (group summaries only) | same |
| `hbn_fsiq_adjusted_isc.csv` | HBN residual-FSIQ-adjusted Aperiodic-ISC | `scripts/hbn/hbn_isc_fsiq_adjusted.py` |

## Re-export / recompute

Development repository:

```bash
python scripts/28_posterior_roi_nested_split_validation.py --n-repeats 200 --n-folds 5
python scripts/29_supplementary_frontal_comparison.py
python scripts/hbn/hbn_isc_fsiq_adjusted.py
python scripts/export_statistical_summary_tables.py
```

Release-bundle equivalents (under `github_release/`):

```bash
python scripts/resting/10c_posterior_roi_nested_split_validation.py --n-repeats 200 --n-folds 5
python scripts/resting/10d_supplementary_frontal_comparison.py
python scripts/hbn/hbn_isc_fsiq_adjusted.py
python scripts/figures/export_statistical_summary_tables.py
```

Library modules: `src/posterior_roi_nested_split.py`, `src/frontal_comparison_analysis.py`.
