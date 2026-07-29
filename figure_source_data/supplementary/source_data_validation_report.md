# Supplementary source data validation report

Directory: `figure_source_data\supplementary`

## 1. File existence

- `s1_cohort_flow.csv`: **PASS** (14 rows)
- `s2_loocv_electrode_survival.csv`: **PASS** (4 rows)
- `s2_loocv_criteria_summary.csv`: **PASS** (7 rows)
- `s3_sensitivity_models.csv`: **PASS** (7 rows)
- `s3_fixed_knee_subjects.csv`: **PASS** (138 rows)
- `s3_knee_qc.csv`: **PASS** (3 rows)
- `s4_development_predictions.csv`: **PASS** (480 rows)
- `s4_development_interactions.csv`: **PASS** (7 rows)
- `s4_development_diagnostics.csv`: **PASS** (280 rows)
- `s5_iaf_subjects.csv`: **PASS** (138 rows)
- `s5_iaf_models.csv`: **PASS** (5 rows)
- `s6_isc_timecourse.csv`: **PASS** (1330 rows)
- `s6_td_template_segment_summary.csv`: **PASS** (6 rows)
- `s6_within_group_segment_summary.csv`: **PASS** (6 rows)
- `s6_event_boundaries.csv`: **PASS** (3 rows)
- `s7_synchrony_controls.csv`: **PASS** (24 rows)
- `s7_envelope_adjusted.csv`: **PASS** (3 rows)
- `s7_gaze_sensitivity.csv`: **PASS** (6 rows)
- `s8_hbn_movie_subjects.csv`: **PASS** (476 rows)
- `s8_hbn_movie_summary.csv`: **PASS** (4 rows)
- `s8_hbn_resting_models.csv`: **PASS** (8 rows)
- `s8_hbn_resting_subjects.csv`: **PASS** (224 rows)
- `s9_coupling_subjects.csv`: **PASS** (136 rows)
- `s9_coupling_models.csv`: **PASS** (2 rows)
- `s9_coupling_bootstrap_summary.csv`: **PASS** (6 rows)
- `s9_coupling_bootstrap.csv`: **PASS** (12000 rows)
- `source_data_manifest.csv`: **PASS** (26 rows)

## 2. Sample size checks (vs supp_table_s1)

- Registration/effective resting sample: **PASS** (file 168/80/88 vs expected 168/80/88)
- Primary resting-state spectral cohort: **PASS** (file 138/61/77 vs expected 138/61/77)
- Posterior resting cohort (primary posterior exponent): **PASS** (file 138/61/77 vs expected 138/61/77)
- ADOS complete-case subset: **PASS** (file 60/60/0 vs expected 60/60/0)
- Movie spectral-QC / ISC cohort: **PASS** (file 136/58/78 vs expected 136/58/78)
- Rest–movie paired cohort: **PASS** (file 104/46/58 vs expected 104/46/58)
- Dual-paradigm post-QC matched cohort: **PASS** (file 68/34/34 vs expected 68/34/34)
- HBN matched cohort: **PASS** (file 238/119/119 vs expected 238/119/119)
- HBN eyes-open matched subset: **PASS** (file 224/112/112 vs expected 224/112/112)

## 3. Key result consistency

- LOOCV all-four FDR: **PASS** (125/138, expected 125/138)
- Strict-QC posterior β: **PASS** (β=0.139, n=90)
- IQ-balanced posterior β: **PASS** (β=0.123, n=76)

## 4. Gap-fill checks

- s7 envelope-adjusted pain p: **PASS** (p=5.40e-05)
- s7 Alpha PLV pain p≈0.97: **PASS**
- s8 HBN pseudonymous subjects: **PASS** (n=476)
- s9 bootstrap iterations: **PASS** (n=12000)
- s4 sensitivity cohort interactions: **PASS**

## Overall verdict: **PASS**
