# Supplementary source data validation report

Directory: `figure_source_data/supplementary`

## 1. File existence

- `s1_cohort_flow.csv`: **PASS** (9 rows)
- `s1_participant_characteristics.csv`: **PASS** (61 rows)
- `s2_loocv_electrode_survival.csv`: **PASS** (4 rows)
- `s2_loocv_criteria_summary.csv`: **PASS** (7 rows)
- `s3_sensitivity_models.csv`: **PASS** (7 rows)
- `s3_fixed_knee_subjects.csv`: **PASS** (138 rows)
- `s3_knee_qc.csv`: **PASS** (3 rows)
- `s4_development_predictions.csv`: **PASS** (480 rows)
- `s4_development_predictions_with_ci.csv`: **PASS** (480 rows)
- `s4_development_interactions.csv`: **PASS** (7 rows)
- `s4_development_diagnostics.csv`: **PASS** (280 rows)
- `s5_iaf_subjects.csv`: **PASS** (138 rows)
- `s5_iaf_models.csv`: **PASS** (5 rows)
- `s6_isc_timecourse.csv`: **PASS** (1330 rows)
- `s6_td_template_segment_summary.csv`: **PASS** (6 rows)
- `s6_within_group_segment_summary.csv`: **PASS** (6 rows)
- `s6_event_boundaries.csv`: **PASS** (3 rows)
- `s6_isc_subject_level.csv`: **PASS** (816 rows)
- `s6_isc_group_ci.csv`: **PASS** (12 rows)
- `s7_synchrony_controls.csv`: **PASS** (24 rows)
- `s7_envelope_adjusted.csv`: **PASS** (3 rows)
- `s7_gaze_sensitivity.csv`: **PASS** (6 rows)
- `s8_hbn_movie_subjects.csv`: **PASS** (476 rows)
- `s8_hbn_movie_summary.csv`: **PASS** (4 rows)
- `s9_coupling_subjects.csv`: **PASS** (136 rows)
- `s9_coupling_models.csv`: **PASS** (2 rows)
- `s9_coupling_bootstrap_summary.csv`: **PASS** (6 rows)
- `s9_coupling_bootstrap.csv`: **PASS** (12000 rows)
- `source_data_manifest.csv`: **PASS** (25 rows)

## 2. Sample size checks (vs supp_table_s1)

- Registration/effective resting sample: **PASS** (file 168/80/88 vs expected 168/80/88)
- Primary resting-state spectral cohort: **PASS** (file 138/61/77 vs expected 138/61/77)
- Paired rest-to-movie exponent: **PASS** (file 136/61/75 vs expected 136/61/75)
- Movie Aperiodic-ISC cohort: **PASS** (file 136/58/78 vs expected 136/58/78)
- Resting + movie matched: **PASS** (file 92/46/46 vs expected 92/46/46)
- IQ-balanced subset: **PASS** (file 76/38/38 vs expected 76/38/38)
- Strict specparam-QC: **PASS** (file 90/44/46 vs expected 90/44/46)
- HBN The Present matched cohort: **PASS** (file 238/119/119 vs expected 238/119/119)
- The Present movie Aperiodic-ISC: **PASS** (file 238/119/119 vs expected 238/119/119)

## 3. Participant-characteristics checks

- Resting 1:1 matched cohort Sex M/F: **PASS** (file ASD=51/4, TD=43/12, p=nan; expected ASD=51/4, TD=43/12)
- Movie Aperiodic-ISC cohort Sex M/F: **PASS** (file ASD=53/5, TD=58/20, p=0.013; expected ASD=53/5, TD=58/20, p=0.013)
- Resting + movie matched cohort Sex M/F: **PASS** (file ASD=39/7, TD=39/7, p=nan; expected ASD=39/7, TD=39/7)
- Dual-paradigm post-QC matched cohort Sex M/F: **PASS** (file ASD=31/3, TD=26/8, p=nan; expected ASD=31/3, TD=26/8)
- HBN The Present matched cohort Sex M/F: **PASS** (file ASD=18/101, TD=18/101, p=nan; expected ASD=18/101, TD=18/101)

## 4. QC config checks

- Primary movie min usable epochs: **PASS** (file 50; expected 50)
- Primary movie invalid-channel ratio: **PASS** (file 0.3; expected 0.3)
- HBN The Present min usable epochs: **PASS** (file 40; expected 40)
- HBN matched-cohort min usable epochs: **PASS** (file 40; expected 40)
- HBN The Present invalid-channel ratio: **PASS** (file 0.2; expected 0.2)
- HBN The Present min R2: **PASS** (file 0.9; expected 0.9)
- HBN The Present fit-error top percentile: **PASS** (file 5.0; expected 5.0)

## 5. Key result consistency

- LOOCV all-four FDR: **PASS** (125/138, expected 125/138)
- Strict-QC posterior β: **PASS** (β=0.139, n=90)
- IQ-balanced posterior β: **PASS** (β=0.123, n=76)

## 6. Gap-fill checks

- s7 envelope-adjusted pain p: **PASS** (p=5.40e-05)
- s7 Alpha PLV pain p≈0.97: **PASS**
- s8 HBN pseudonymous subjects: **PASS** (n=476)
- s9 bootstrap iterations: **PASS** (n=12000)
- s4 sensitivity cohort interactions: **PASS**
- s4 predictions with CI grid complete: **PASS**
- s4 predictions CI bounds valid: **PASS**
- s4 predictions finite values: **PASS**
- s4 predictions mean CI (not pred interval): **PASS**
- s4 CI file vs reference predictions: **PASS** (max |diff|=1.998e-15)
- s6 subject-level row count (816) and n subjects (136): **PASS**
- s6 subject means vs summary tables: **PASS**
- s6 group CI bounds valid: **PASS**

## Overall verdict: **PASS**
