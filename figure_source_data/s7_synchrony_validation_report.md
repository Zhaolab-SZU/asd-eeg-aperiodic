# S7 synchrony-control export validation

Subject file: `figure_source_data\s7_synchrony_subject_level.csv`
Group effects file: `figure_source_data\s7_synchrony_group_effects.csv`
Reference table: `figure_source_data\supplementary\s7_synchrony_controls.csv` (within_group_loo rows only)

## Upstream sources

- Aperiodic within-group LOO: `derivatives\derivatives_task_movie\stats\aperiodic_isc\aperiodic_isc_within_group_subject_values.csv` (`isc_z`)
- Envelope within-group LOO: `derivatives\derivatives_task_movie\stats\classic_isc\envelope_within_group_subject_values.csv` (`isc_z`)
- Alpha PLV within-group LOO: `derivatives\derivatives_task_movie\stats\classic_isc\alpha_plv_within_group_subject_values.csv` (`isc_r`)
- Mechanism group tests: `outputs\tables\followup_exploration\isc_mechanism_group_tests.csv`

## Cohort and scale

- Subject-level rows: 1224 (expected 1224 = 136 subjects × 3 segments × 3 metrics)
- Unique subjects: 136 (expected 136)
- All metrics use `isc_definition = within_group_loo`
- Aperiodic-ISC & Envelope ISC on Fisher-z (`isc_z`); Alpha PLV ISC on PLV correlation (`isc_r`)

## 1. Group means and TD−ASD effects vs legacy `s7_synchrony_controls.csv`

All within_group_loo means, group effects, SE (where present), p, and fdr_p **match** legacy table (tol 1e-12).

## 2. Sample sizes

- Per segment × metric: n_asd = 58, n_td = 78 (movie synchrony-control cohort, n = 136)
- Legacy group-level table stored n = 136 on each row; new tables report group-specific counts.

## 3. Alpha PLV SE and 95% CI (Welch TD−ASD difference)

| segment | group_effect | se | ci_low | ci_high | p |
|---------|--------------|----|--------|---------|---|
| mentalizing | 0.0118208066207 | 0.0116207395826 | -0.0111807812783 | 0.0348223945196 | 0.311038886222 |
| neutral | 0.00428941629167 | 0.00533936877681 | -0.00627520648964 | 0.014854039073 | 0.423254647867 |
| pain | 0.000409526126171 | 0.0107239146027 | -0.0208064531997 | 0.021625505452 | 0.969596274073 |

- Method: Welch unequal-variance t-test on subject-level `isc_r`; SE = sqrt(s²_ASD/n_ASD + s²_TD/n_TD);
  95% CI = group_effect ± t_(0.975, df_Welch–Satterthwaite) × SE.

## 4. FDR correction family

- **Aperiodic-ISC** (`within_group_*`) and **Envelope ISC** (`envelope_*`):
  `fdr_p` taken from `isc_mechanism_group_tests.csv`, Benjamini–Hochberg across **12** mechanism
  comparisons (td_template, within_group, template_gap, envelope × mental/pain/neutral).
- **Alpha PLV ISC**: not included in that 12-test family; `fdr_p` left empty (NaN), matching legacy
  `s7_synchrony_controls.csv`. Raw two-sided Welch p-values reported.

## 5. Envelope-adjusted analysis (`s7_envelope_adjusted.csv`)

- Validated file: `figure_source_data\supplementary\s7_envelope_adjusted.csv`
- Upstream: `derivatives\derivatives_task_movie\stats\classic_isc\aperiodic_envelope_partial_analysis.csv`

Required fields present; all numeric values **match** upstream (tol 1e-12).

Fields checked: event_type, pearson_r, shared_variance_pct, envelope_adjusted_group_beta_z,
envelope_adjusted_group_se, envelope_adjusted_group_p, envelope_adjusted_group_fdr_p,
partial_cohen_d, effect_retained_pct.

## Overall verdict: **PASS**