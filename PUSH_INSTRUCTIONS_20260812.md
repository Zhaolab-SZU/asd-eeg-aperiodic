# Push instructions - 2026-08-12

This staging copy is aligned to the current manuscript and supplementary files.

## Current frontal-comparison update

Updated from `frontal_comparison_rerun_bundle_20260812`:

- `src/frontal_comparison_analysis.py`
- `data/tables/table_s4b_frontal_group_ols.csv`
- `data/tables/table_s4b_frontal_group_age_interaction.csv`
- `data/tables/table_s4b_frontal_vs_posterior_interaction.csv`
- matching mirrors under `statistical_tables/`

Current values:

- Frontal group effect: beta = 0.114, p = 0.013.
- Posterior group effect: beta = 0.133, p = 1.3e-4.
- Group x region: beta approximately -0.002, p = 0.95.
- Frontal group x age: beta = 0.0020, p = 0.35.
- Posterior group x age in S4b: beta = 0.0052, p = 7.1e-4.

The main developmental result remains locked to the primary development table (`s4_development_interactions.csv`; beta = 0.0048) unless the full development source table is explicitly re-locked.

## Movie Aperiodic-ISC covariate sensitivity

Added as Supplementary Table S11 covariate-sensitivity support:

- `scripts/movie/26_movie_isc_covariate_sensitivity.py`
- `data/tables/movie_isc_covariate_sensitivity.csv`
- matching mirror under `statistical_tables/`

Interpretation to preserve in manuscript/SI: demographic-adjusted models retained TD > ASD direction across mentalizing, pain-related and neutral segments, with all three FDR-significant. Additional valid-window and mean movie R2 sensitivity models retained the same direction, with strongest support for mentalizing and neutral and weaker support for pain-related segments.

## Movie segment valid-window threshold

Clarified the segment-level Aperiodic-ISC inclusion rule in the manuscript/SI:

- Main movie ISC path uses `min_overlap_points = 10` for valid overlapping participant-template time points within a segment.
- The subject-level movie QC threshold remains >=50 usable artifact-free 2-s epochs across the movie; it is not a per-segment threshold.
- In the present TD-template cohort, no participants were excluded from segment-level analyses based on the `min_overlap_points` threshold.

## Push from another device

```bash
git status
git diff --stat
python scripts/figures/validate_supplementary_source_data.py
git add .
git commit -m "Align public code and source data with current manuscript"
git push origin main
```
