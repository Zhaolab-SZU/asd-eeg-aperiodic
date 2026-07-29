# S6 Aperiodic-ISC subject-level export validation

Subject file: `figure_source_data\supplementary\s6_isc_subject_level.csv`
Group CI file: `figure_source_data\supplementary\s6_isc_group_ci.csv`

## Upstream sources

- TD-template: `derivatives\derivatives_task_movie\stats\aperiodic_isc\aperiodic_isc_td_template_subject_values.csv` (isc_z scale)
- Within-group LOO: `derivatives\derivatives_task_movie\stats\aperiodic_isc\aperiodic_isc_within_group_subject_values.csv` (isc_z scale)
- Summary reference: `s6_td_template_segment_summary.csv`, `s6_within_group_segment_summary.csv`
- Group tests: `outputs\tables\followup_exploration\isc_mechanism_group_tests.csv`

## Cohort

- Subject-level rows: 816 (expected 816 = 136 subjects × 3 segments × 2 definitions)
- Unique subjects: 136

## Mean reproduction vs summary tables

All segment × group × isc_definition means **match** summary `mean_isc` (tolerance 1e-12).

## Group 95% CI method

- mean_isc: subject-level mean of Fisher-z ISC (`isc_z`)
- se: SD / sqrt(n) with ddof=1
- ci_low/ci_high: mean ± t_(0.975, n−1) × SE (within-group mean CI)
- comparison_p / fdr_p: from `isc_mechanism_group_tests.csv` (Welch t-test on isc_z)

## Overall verdict: **PASS**