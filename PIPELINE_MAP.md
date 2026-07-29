# Pipeline map (ideal release name → original repository path)

| Bundle path | Original path | Note |
|-------------|---------------|------|
| `scripts/resting/00_check_environment.py` | `scripts/00_check_environment.py` | Environment / dependency check |
| `scripts/resting/01_prepare_participants.py` | `scripts/01_prepare_participants.py` | Participant table preparation |
| `scripts/resting/02_preprocess_eeg.py` | `scripts/02_preprocess_eeg.py` | Resting EEG preprocess (import, filter, bad-channel, ICA, epoch). Maps to idealized 00_raw_import / 01_bad_channel / 02_ica / 03_epoch steps. |
| `scripts/resting/03_compute_psd.py` | `scripts/03_compute_psd.py` | Welch PSD |
| `scripts/resting/04_run_specparam_fixed.py` | `scripts/04_run_specparam.py` | Primary fixed-mode specparam |
| `scripts/resting/05_specparam_qc.py` | `scripts/05_specparam_qc.py` | specparam fit QC |
| `scripts/resting/05b_run_specparam_knee.py` | `scripts/93_posterior_knee_mode_sensitivity.py` | Knee-mode sensitivity (idealized 05_run_specparam_knee) |
| `scripts/resting/06_extract_global_posterior_exponent.py` | `scripts/06_compute_roi_metrics.py` | Global / ROI / posterior exponent extraction |
| `scripts/resting/07_demographic_and_qc_stats.py` | `scripts/07_demographic_and_qc_stats.py` | Demographics + QC stats |
| `scripts/resting/07b_table1_main_cohort.py` | `scripts/07b_table1_main_cohort.py` | Table 1 cohort characteristics |
| `scripts/resting/08_model_global_spectral.py` | `scripts/08_main_group_analysis.py` | Global exponent / offset group models |
| `scripts/resting/09_roi_mixed_model.py` | `scripts/09_roi_mixed_model.py` | ROI mixed models |
| `scripts/resting/10_channel_fdr_analysis.py` | `scripts/10_channel_level_analysis.py` | Channel-level effects + FDR |
| `scripts/resting/10b_loocv_fdr_survival.py` | `scripts/27_posterior_roi_loocv_fdr.py` | Leave-one-subject-out FDR survival (Supp Fig S1) |
| `scripts/resting/11_clinical_correlation.py` | `scripts/11_clinical_correlation.py` | ADOS / clinical associations |
| `scripts/resting/12_iaf_control_model.py` | `scripts/12_periodic_peak_analysis.py` | Individual alpha frequency / periodic controls |
| `scripts/resting/13_traditional_band_power.py` | `scripts/13_traditional_band_power.py` | Traditional band-power controls |
| `scripts/resting/14_sensitivity_analysis.py` | `scripts/14_sensitivity_analysis.py` | Primary sensitivity suite (IQ-balance, covariates, etc.) |
| `scripts/resting/14b_qc_sensitivity_followup.py` | `scripts/17_qc_and_sensitivity_followup.py` | QC follow-up sensitivities |
| `scripts/resting/14c_iclabel_artifact_sensitivity.py` | `scripts/23_iclabel_artifact_sensitivity.py` | ICLabel artifact sensitivity |
| `scripts/resting/14d_posterior_roi_sensitivity.py` | `scripts/95_posterior_roi_sensitivity.py` | Posterior ROI definition sensitivity |
| `scripts/resting/15_match_iq_and_postqc_cohorts.py` | `scripts/69_build_postqc_matched_cohorts.py` | IQ / post-QC matched cohort construction |
| `scripts/resting/15b_rematch_after_qc.py` | `scripts/66_rematch_after_qc.py` | Rematch after QC exclusions |
| `scripts/resting/16_generate_report_tables.py` | `scripts/16_generate_report_tables.py` | Manuscript / report tables |
| `scripts/resting/17_development_interaction.py` | `scripts/19_development_and_reliability_extension.py` | Age × group developmental analyses |
| `scripts/resting/17b_normative_td_reference.py` | `scripts/90_normative_exponent_analysis.py` | TD-reference normative deviation |
| `scripts/resting/17c_spectral_maturation.py` | `scripts/91_spectral_maturation_joint_model.py` | Spectral maturation joint model |
| `scripts/resting/17d_nonlinear_age_spline.py` | `scripts/92_nonlinear_age_sensitivity.py` | Nonlinear age / spline sensitivity |
| `scripts/movie/20_movie_prepare_participants.py` | `scripts/31_prepare_movie_participants.py` | Movie participant preparation |
| `scripts/movie/20b_align_exponent_movie_labels.py` | `scripts/31_align_exponent_with_movie_labels.py` | Align resting exponents with movie labels |
| `scripts/movie/21_time_resolved_specparam_isc.py` | `scripts/97_posterior_movie_specparam_isc.py` | Time-resolved posterior specparam + ISC pipeline |
| `scripts/movie/22_calc_aperiodic_isc.py` | `scripts/68_compute_aperiodic_isc.py` | TD-template and within-group Aperiodic-ISC |
| `scripts/movie/23_calc_segment_isc.py` | `scripts/33_compute_segment_isc.py` | Segment-level ISC helpers |
| `scripts/movie/24_sync_control_classic_isc.py` | `scripts/69_compute_classic_isc_controls.py` | Envelope ISC / alpha PLV controls |
| `scripts/movie/24b_envelope_partial_isc.py` | `scripts/93_aperiodic_envelope_partial_isc.py` | Envelope-adjusted Aperiodic-ISC (Supp Fig S6) |
| `scripts/movie/25_rest_movie_state_analysis.py` | `scripts/35_rest_movie_posterior_state_analysis.py` | Rest–movie posterior exponent modulation |
| `scripts/movie/25b_rest_movie_coupling.py` | `scripts/34_resting_to_movie_coupling.py` | Resting-to-movie coupling models |
| `scripts/movie/25c_rest_movie_coupling_overall_isc.py` | `scripts/71_resting_to_movie_coupling_overall_isc.py` | Coupling with overall ISC |
| `scripts/hbn/100_hbn_inventory.py` | `scripts/100_hbn_inventory.py` | HBN file inventory |
| `scripts/hbn/hbn_preprocess_resting.py` | `scripts/101_hbn_preprocess_resting.py` | HBN resting preprocess |
| `scripts/hbn/hbn_preprocess_present.py` | `scripts/101_hbn_preprocess_thepresent.py` | HBN The Present movie preprocess |
| `scripts/hbn/hbn_specparam_roi.py` | `scripts/102_hbn_specparam_roi.py` | HBN ROI specparam |
| `scripts/hbn/hbn_external_validation.py` | `scripts/103_hbn_external_validation.py` | HBN external validation entry |
| `scripts/hbn/hbn_isc_calc.py` | `scripts/113_hbn_thepresent_aperiodic_isc.py` | HBN The Present Aperiodic-ISC |
| `scripts/hbn/hbn_isc_matched.py` | `scripts/114_hbn_thepresent_isc_matched.py` | Age/IQ/sex-matched HBN ISC |
| `scripts/hbn/hbn_isc_matched_global.py` | `scripts/115_hbn_thepresent_isc_matched_global.py` | Matched HBN ISC global variant |
| `scripts/hbn/hbn_rest_eyes_open_matched.py` | `scripts/143_hbn_eo_matched_external_validation.py` | HBN eyes-open matched resting convergence |
| `scripts/hbn/hbn_eo_posterior_exponent.py` | `scripts/hbn_eo_posterior_exponent.py` | HBN EO posterior exponent helper |
| `scripts/figures/prepare_figure_source_data.py` | `scripts/prepare_figure_source_data.py` | Assemble main-figure source CSVs |
| `scripts/figures/export_supplementary_source_data.py` | `scripts/export_supplementary_source_data.py` | Export supplementary source tables |
| `scripts/figures/export_s4_development_predictions_with_ci.py` | `scripts/export_s4_development_predictions_with_ci.py` | S4 development CI trajectories |
| `scripts/figures/export_s6_isc_subject_level.py` | `scripts/export_s6_isc_subject_level.py` | S6 subject-level ISC |
| `scripts/figures/export_s7_synchrony_subject_level.py` | `scripts/export_s7_synchrony_subject_level.py` | S7 synchrony subject-level |
| `scripts/figures/export_fig5_rest_movie_exponent_subjects.py` | `scripts/export_fig5_rest_movie_exponent_subjects.py` | Fig.5A rest–movie subject table |
| `scripts/figures/validate_supplementary_source_data.py` | `scripts/validate_supplementary_source_data.py` | Validate supplementary source bundle |
| `scripts/figures/recompute_s4_age_interaction_sensitivity.py` | `scripts/recompute_s4_age_interaction_sensitivity.py` | Recompute S4 age×group sensitivities |
| `scripts/figures/plot_supplementary_figures.py` | `scripts/plot_supplementary_figures.py` | Plot Supp Figs S1/S4–S7 |
| `scripts/figures/plot_fig2a_channelwise_topomap.py` | `scripts/plot_fig2a_channelwise_topomap.py` | Fig.2A topomap |
| `scripts/figures/plot_posterior_robustness_forest.py` | `scripts/plot_posterior_robustness_forest.py` | Fig.2B robustness forest |
| `scripts/figures/plot_ados_posterior_partialcorr_main.py` | `scripts/64_plot_ados_posterior_partialcorr_main.py` | Fig.4 ADOS partial-correlation panels |
| `src/config.py` | `src/config.py` | library module |
| `src/io_utils.py` | `src/io_utils.py` | library module |
| `src/eeg_preprocessing.py` | `src/eeg_preprocessing.py` | library module |
| `src/psd_utils.py` | `src/psd_utils.py` | library module |
| `src/specparam_utils.py` | `src/specparam_utils.py` | library module |
| `src/roi_utils.py` | `src/roi_utils.py` | library module |
| `src/qc_utils.py` | `src/qc_utils.py` | library module |
| `src/stats_utils.py` | `src/stats_utils.py` | library module |
| `src/paper_figures.py` | `src/paper_figures.py` | library module |
| `src/plotting_utils.py` | `src/plotting_utils.py` | library module |
| `src/submission_style.py` | `src/submission_style.py` | library module |
| `src/iclabel_sensitivity.py` | `src/iclabel_sensitivity.py` | library module |
| `src/extension_analysis.py` | `src/extension_analysis.py` | library module |
| `src/normative_analysis.py` | `src/normative_analysis.py` | library module |
| `src/spectral_maturation_analysis.py` | `src/spectral_maturation_analysis.py` | library module |
| `src/nonlinear_age_sensitivity.py` | `src/nonlinear_age_sensitivity.py` | library module |
| `src/posterior_knee_sensitivity.py` | `src/posterior_knee_sensitivity.py` | library module |
| `src/posterior_roi_loocv.py` | `src/posterior_roi_loocv.py` | library module |
| `src/aperiodic_isc.py` | `src/aperiodic_isc.py` | library module |
| `src/aperiodic_isc_analysis.py` | `src/aperiodic_isc_analysis.py` | library module |
| `src/classic_isc_analysis.py` | `src/classic_isc_analysis.py` | library module |
| `src/movie_isc_partial_analysis.py` | `src/movie_isc_partial_analysis.py` | library module |
| `src/coordination_feature_analysis.py` | `src/coordination_feature_analysis.py` | library module |
| `src/hbn_external.py` | `src/hbn_external.py` | library module |
| `src/hbn_aperiodic_isc.py` | `src/hbn_aperiodic_isc.py` | library module |
| `src/hbn_eo_exponent.py` | `src/hbn_eo_exponent.py` | library module |
| `src/hbn_main_matched_cohort.py` | `src/hbn_main_matched_cohort.py` | library module |
| `src/hbn_confirmatory_replication.py` | `src/hbn_confirmatory_replication.py` | library module |
| `src/hbn_strict_replication.py` | `src/hbn_strict_replication.py` | library module |
| `src/hbn_spatial_stats.py` | `src/hbn_spatial_stats.py` | library module |
| `src/artifact_defense_analysis.py` | `src/artifact_defense_analysis.py` | library module |
| `src/robustness_utils.py` | `src/robustness_utils.py` | library module |
| `src/roi_effect_sizes.py` | `src/roi_effect_sizes.py` | library module |
| `src/spatial_topography_analysis.py` | `src/spatial_topography_analysis.py` | library module |
| `src/clinical_matched_analysis.py` | `src/clinical_matched_analysis.py` | library module |
| `src/clinical_age_interaction_analysis.py` | `src/clinical_age_interaction_analysis.py` | library module |
| `src/__init__.py` | `src/__init__.py` | library module |
| `src/bp_figures/` | `src/bp_figures/` | figure helpers |
| `src/spectral_pipeline.py` | `(alias)` | ideal-name re-export |
| `src/stats_models.py` | `(alias)` | ideal-name re-export |
| `src/isc_calculator.py` | `(alias)` | ideal-name re-export |
| `src/artifact_ica.py` | `(alias)` | ideal-name re-export |
| `src/matching_tools.py` | `(alias)` | ideal-name re-export |
| `config/config.yaml` | `config/config.yaml` | config |
| `config/roi_channels.yaml` | `config/roi_channels.yaml` | config |
| `config/config_task_movie.yaml` | `config/config_task_movie.yaml` | config |
| `config/config_task_movie_matched_postqc.yaml` | `config/config_task_movie_matched_postqc.yaml` | config |
| `config/config_task_movie_both_postqc.yaml` | `config/config_task_movie_both_postqc.yaml` | config |
| `config/config_resting_matched.yaml` | `config/config_resting_matched.yaml` | config |
| `config/config_resting_matched_postqc.yaml` | `config/config_resting_matched_postqc.yaml` | config |
| `config/config_hbn.yaml` | `config/config_hbn_thepresent.yaml` | config |
| `config/config_hbn_thepresent.yaml` | `config/config_hbn_thepresent.yaml` | config |
| `config/config_hbn_resting.yaml` | `config/config_hbn_resting.yaml` | config |
| `config/config_hbn_external.yaml` | `config/config_hbn_external.yaml` | config |
| `config/roi_hbn129.yaml` | `config/roi_channels_hbn129.yaml` | config |
| `config/roi_channels_hbn129.yaml` | `config/roi_channels_hbn129.yaml` | config |
| `figure_source_data/` | `figure_source_data/` | figure source CSVs |
| `figure_source_data/outputs_figure_source_data/` | `outputs/figure_source_data/` | main Fig.1–5 authoritative CSVs |

## Excluded from this bundle

- Raw EEG / BIDS under `data/`
- Large `derivatives/` intermediates (`.fif`, etc.)
- Jansen–Rit mechanistic branch (`96–146`, `src/jr_*`)
- Gaze sensitivity scripts (removed from manuscript)
- ML / classifier exploration scripts
- Local tools (`.tools/`), Word backups, notebooks

| `scripts/resting/10c_posterior_roi_nested_split_validation.py` | `scripts/28_posterior_roi_nested_split_validation.py` | Nested split-sample ROI validation (Supp Table S4a) |
| `scripts/resting/10d_supplementary_frontal_comparison.py` | `scripts/29_supplementary_frontal_comparison.py` | Frontal comparison analyses (Supp Table S4b) |
| `scripts/hbn/hbn_isc_fsiq_adjusted.py` | `scripts/hbn/hbn_isc_fsiq_adjusted.py` | HBN FSIQ-adjusted Aperiodic-ISC OLS |
| `scripts/figures/export_statistical_summary_tables.py` | `scripts/export_statistical_summary_tables.py` | Export S4a/S4b/HBN FSIQ summary CSVs |
| `src/posterior_roi_nested_split.py` | `src/posterior_roi_nested_split.py` | library module |
| `src/frontal_comparison_analysis.py` | `src/frontal_comparison_analysis.py` | library module |

