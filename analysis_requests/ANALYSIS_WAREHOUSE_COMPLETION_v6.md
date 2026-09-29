# ANALYSIS_PLAN_v6 — NMH submitted-repository completion

## 1. Objective

Complete the public GitHub repository so that every analysis claimed in the Nature Mental Health submission of 28 September 2026 can be audited from de-identified derived inputs and rerun from the appropriate restricted/public raw inputs.

This is a release-engineering task, not a new hypothesis-generating analysis. The submitted manuscript and data/manuscript_tables are the numerical locks. Do not change the analysis population, model specification, correction family, seed or interpretation to obtain cleaner results.

The analysis workspace must return one consolidated report:

    outputs/reports/ANALYSIS_PLAN_v6_full_results_report.md

The report must describe every copied/generated file, command, input hash, output hash, numerical comparison with the submitted lock, privacy audit and any unresolved discrepancy.

## 2. Non-negotiable privacy constraints

Public deliverables may contain pseudonymous study IDs and de-identified derived variables only. Do not export raw EEG, dates of birth, acquisition dates, free text, item-level clinical responses, local usernames, drive letters, absolute paths or re-identification keys. Replace local paths in metadata with repository-relative paths.

Before delivery, scan all text-like files for Windows drive prefixes, /Users/, user home paths, email addresses other than the corresponding-author address already present in CITATION.cff, and known direct-identifier columns.

## 3. Locked environment

Use Python 3.12.2 and the exact versions in requirements.txt. Record the actual interpreter and package freeze in:

    outputs/reports/environment_freeze_v6.txt

If the analysis workspace cannot recreate a package version, report the conflict; do not silently substitute a different environment.

## 4. Required analysis deliveries

### 4.1 v1 film-level Aperiodic-ISC–ADOS

Required script:

    scripts/movie/film_level_isc_ados_exploratory.py

Required public outputs:

    outputs/tables/film_level_isc_ados/
    outputs/figures/supplementary/film_level_isc_ados_total.png
    outputs/figures/supplementary/film_level_isc_ados_total.pdf
    outputs/figures/supplementary/film_level_isc_ados_total.svg
    outputs/reports/ANALYSIS_PLAN_v1_full_results_report.md

Required locks:

- movie spectral-QC ASD n=58;
- ADOS-total complete-case n=57;
- partial r=0.022 with 95% CI approximately [−0.24, 0.28];
- standardized beta=0.022 with HC3 95% CI approximately [−0.32, 0.36];
- raw p=0.90 and FDR q=0.90;
- reconstructed-template bootstrap 95% interval approximately [−0.32, 0.34], B=5,000;
- classification no_robust_evidence.

The output bundle must include the joined analysis frame, missingness audit, model table, multiplicity table, bootstrap summary and bootstrap null/distribution needed to reproduce the interval.

### 4.2 v2 temporal mechanism checks

Required script (use this public name even if the internal script has another name):

    scripts/movie/aperiodic_isc_mechanism_v2.py

Required public outputs:

    outputs/tables/aperiodic_isc_mechanism_v2/
    outputs/reports/ANALYSIS_PLAN_v2_full_results_report.md

The directory must include, at minimum:

- temporal-shift null summary and the de-identified null draws;
- lagged ISC subject metrics;
- lagged_isc_fdr_results.csv with permutation-based FDR as the inferential column and conventional FDR clearly labelled audit-only;
- trajectory_structure_participant_metrics.csv;
- staged model coefficients M0–M4 with raw and standardized coefficients;
- variable-level VIF table;
- an analysis metadata JSON containing seed 20260924, cohort counts and input SHA-256 hashes.

Required locks:

- ASD mean z=0.0582816; TD mean z=0.1133553; TD−ASD=0.0550737;
- temporal-shift equal-tail group-difference p=0.00019998;
- zero-lag Cohen d=1.002 and max-lag Cohen d=0.725, approximately 27.7% attenuation;
- max-lag permutation FDR q=0.00119988;
- M4 raw group beta=0.0545347, 95% CI [0.0305789, 0.0784904], p=8.12701×10⁻6;
- classification stimulus_locked_partial_timing_variability_with_residual_alignment_robust_to_controls.

### 4.3 v3 primary-cohort Group × Age

Required script:

    scripts/movie/film_level_isc_age_v3.py

Use seed 20260925 and B=5,000 reconstructed-template bootstrap iterations.

Required public outputs:

    outputs/tables/film_level_isc_age_v3/
    outputs/figures/supplementary/film_level_isc_age_v3.png/.pdf/.svg
    outputs/figures/supplementary/film_level_isc_age_v3_forest.png/.pdf/.svg
    outputs/reports/ANALYSIS_PLAN_v3_full_results_report.md

Required tables include age_descriptives.csv, analysis_metadata.json, primary_model.csv, simple_slopes_and_contrasts.csv, staged_models.csv, vif.csv, robustness.csv, bootstrap_summary.json and either bootstrap draws or a deterministic compact representation sufficient to recompute the interval.

Verify these input SHA-256 values before rerunning:

- ISC subjects: c0c106e0e7b1d3a81de4f1fedc584d2334f7aace5eeb22342144b36f62c10a80
- ISC stats: 3c52489752cffb6a092f687e6787f234f07f4abeeea8f086d14ce1cefa2bc00f
- analysis frame: 3940092d333fcf947b46bbf6057c41b7658e01eacbab87abc3a42fdf14b4b8a8
- trajectory metrics: 3b971711558824914f10e374d189bf08a7e4c3751382739bb45d9d1043fe35b2
- locked participant list: c6f80deee2a206c5335ab057cd0681c07887a487c286b9a8cbe675483e806f65
- concatenated trajectory input: 4521f6516c1f9bcbb0ac5aea3cf170fff6bab3f4de1c4d6098259c9de6131fca

Required locks:

- n=136 (58 ASD, 78 TD);
- interaction=−0.0121026 Fisher-z/year, HC3 95% CI [−0.0241767, −0.0000284], p=0.0494594;
- ASD slope=0.0102884/year; TD slope=−0.0018142/year;
- bootstrap median interaction approximately −0.0105/year with 95% interval [−0.0233, 0.0019];
- classification sensitivity_dependent_age_slope_difference.

### 4.4 v4 HBN template-dependence correction

Required script:

    scripts/hbn/hbn_template_dependency_v4.py

Required command:

    python scripts/hbn/hbn_template_dependency_v4.py --config config/config_hbn_thepresent.yaml --seed 20260926 --n-permutations 10000 --n-folds 5 --n-fold-repeats 100 --skip-reverse-template

If the existing script lacks --skip-reverse-template, add it. The public release and submitted narrative require the reconstructed TD-template and strict cross-fit analyses; reverse-template results are not required and must not be reintroduced into manuscript-facing summaries.

Required public outputs:

    outputs/tables/hbn_template_dependency_v4/
    outputs/figures/supplementary/hbn_template_dependency_v4.png/.pdf/.svg
    outputs/reports/ANALYSIS_PLAN_v4_full_results_report.md

Include observed models, reconstructed-template permutation summary/null, frozen pair-level fold assignment, cross-fit subject values, repeat-level estimates, epoch sensitivity, input hashes and leakage assertions.

Required locks:

- 119 ASD–TD matched pairs;
- sliding-window conventional Welch p=0.0099;
- FSIQ-adjusted TD−ASD beta=0.0311061;
- reconstructed-template equal-tail p=0.0317968 (reported 0.032);
- strict cross-fit beta=0.0330342 and p=0.0177982 (reported 0.018);
- epoch template-corrected p approximately 0.060;
- classification template_corrected_hbn_support.

### 4.5 v5 HBN Group × Age

Required script:

    scripts/hbn/hbn_group_age_v5.py

Required command:

    python scripts/hbn/hbn_group_age_v5.py --config config/config_hbn_thepresent.yaml --seed 20260927 --n-permutations 10000 --n-folds 5 --n-fold-repeats 100

Required public outputs:

    outputs/tables/hbn_group_age_v5/
    outputs/figures/supplementary/hbn_group_age_v5.png/.pdf/.svg
    outputs/figures/supplementary/hbn_group_age_v5_forest.png/.pdf/.svg
    outputs/reports/ANALYSIS_PLAN_v5_full_results_report.md

Required locks:

- n=238, 119 per group and 119 matched pairs;
- Group × Age=−0.0019984 Fisher-z/year, HC3 95% CI [−0.0235861, 0.0195893], p=0.856025;
- pair-cluster p=0.849100;
- reconstructed-template equal-tail p=0.819718;
- cross-fit interaction=−0.00169337, p=0.865513;
- cross-cohort common-support p=0.590141;
- robust-M p=0.626688;
- 2-s epoch interaction=0.00180461, template p=0.915308;
- classification directional_but_inconclusive_hbn_age_evidence.

Reverse-template analyses must not be run or discussed in v5.

## 5. Repository-relative packaging

Return a single directory tree rooted at asd-eeg-aperiodic/ containing only the required scripts, de-identified inputs permitted for release, outputs and reports. Preserve the public paths specified above. Do not return a ZIP with an additional nested repository level.

Every analysis script must:

1. locate the repository root without a machine-specific path;
2. accept the documented seed/configuration arguments;
3. create its output directories;
4. record package versions, input hashes, cohort flow and missingness;
5. fail loudly on cohort/count/hash drift;
6. use Fisher-z subject values consistently;
7. write CSV/JSON before drawing figures;
8. produce no analysis result that exists only in console output.

## 6. Full v6 report requirements

The consolidated report must contain:

1. executive classification for v1–v5;
2. exact commands executed and exit status;
3. environment/version table;
4. input/output SHA-256 manifest;
5. cohort and missingness audit;
6. one comparison table with submitted value, recomputed value, absolute difference and pass/fail;
7. list of public files copied/generated;
8. privacy scan results;
9. test results;
10. unresolved issues and whether they block the v2.0-nmh-submitted tag.

Also write:

    outputs/reports/release_manifest_v6.csv
    outputs/reports/submitted_number_comparison_v6.csv
    outputs/reports/privacy_scan_v6.txt

## 7. Acceptance tests

From a clean clone, the following must pass:

    python -m pytest
    python scripts/figures/main/build_main_figures.py
    python scripts/figures/supplementary/build_si_figures.py

Add number-lock tests for the v1–v5 values above. Tests may use compact shipped null distributions and de-identified derived tables, but must not require raw primary-cohort EEG.

Tolerance policy:

- exact equality for n, seed, folds and iteration counts;
- absolute tolerance 5×10⁻4 for manuscript-rounded coefficients/CIs;
- absolute tolerance 5×10⁻5 for p values below 0.01 unless the stored report gives more precise values;
- classifications and direction must match exactly.

## 8. Final decision rule

Return release_ready only if all mandatory files exist, all locks pass, privacy scanning is clean and the clean-clone commands succeed. Otherwise return release_blocked and enumerate the missing or discordant items. Do not label the repository fully reproducible merely because the submitted CSV tables are present.
