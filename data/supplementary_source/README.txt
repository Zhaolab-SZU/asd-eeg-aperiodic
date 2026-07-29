Supplementary Figures S1-S9 source data bundle
Generated: 2026-07-09 (gap-fill refresh)

Contents (28 files)
- s1_cohort_flow.csv
- s2_loocv_electrode_survival.csv, s2_loocv_criteria_summary.csv
- s3_sensitivity_models.csv, s3_fixed_knee_subjects.csv, s3_knee_qc.csv
- s4_development_predictions.csv, s4_development_interactions.csv, s4_development_diagnostics.csv
- s5_iaf_subjects.csv, s5_iaf_models.csv
- s6_isc_timecourse.csv, s6_td_template_segment_summary.csv, s6_within_group_segment_summary.csv, s6_event_boundaries.csv
- s7_synchrony_controls.csv, s7_envelope_adjusted.csv, s7_gaze_sensitivity.csv
- s8_hbn_movie_summary.csv, s8_hbn_movie_subjects.csv, s8_hbn_resting_models.csv, s8_hbn_resting_subjects.csv
- s9_coupling_subjects.csv, s9_coupling_models.csv, s9_coupling_bootstrap_summary.csv, s9_coupling_bootstrap.csv
- source_data_manifest.csv
- source_data_validation_report.md

Previously missing items now included
- s7_envelope_adjusted.csv (from derivatives/derivatives_task_movie/stats/classic_isc/)
- s7 Alpha PLV ISC rows in s7_synchrony_controls.csv
- s8_hbn_movie_subjects.csv (pseudonymous HBN_#### matched ISC)
- s9_coupling_bootstrap.csv (4000 iterations × 3 segments)
- S4 IQ-balanced / Strict-QC age interactions in s4_development_interactions.csv

Regenerate
  python scripts/recompute_s4_age_interaction_sensitivity.py
  python scripts/export_supplementary_source_data.py
  python scripts/validate_supplementary_source_data.py

Validation reference: supplementary_20260704.docx Tables/Results
