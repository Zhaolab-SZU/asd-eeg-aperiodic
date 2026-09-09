# Manuscript Alignment Audit - 2026-08-12

## Scope

This audit aligns the public staging repository with the 2026-08-12(q) manuscript and supplementary files. The current manuscript-facing HBN analysis is restricted to the external HBN The Present movie Aperiodic-ISC cohort. HBN resting-state convergence analyses are no longer presented in the manuscript or supplementary source-data manifest.

## Changes Applied

- Updated Supplementary Figure S1 plotting script to the current stacked layout:
  - Panel A: primary cohort flow.
  - Panel B: HBN The Present matched cohort -> The Present movie Aperiodic-ISC.
  - Removed eyes-open and eyes-closed HBN resting nodes from the generated S1 figure.
- Updated `s1_cohort_flow.csv` in both source-data locations to match the current Figure S1 nodes and sample sizes.
- Updated supplementary source-data export and validation scripts:
  - S8 now exports and validates HBN movie source data only.
  - HBN resting `s8_hbn_resting_*` files are no longer expected by the current manifest or validation script.
  - LOOCV "at least three" posterior-electrode survival checks are unchanged because they refer to spatial robustness, not posterior ROI inclusion QC.
- Updated source-data manifests and README files to remove manuscript-facing HBN resting entries.
- Added manuscript-facing participant-characteristics source tables:
  - `data/tables/table1_participant_characteristics.csv`
  - `data/supplementary_source/s1_participant_characteristics.csv`
  - `figure_source_data/supplementary/s1_participant_characteristics.csv`
- Locked the current sex counts from the manuscript:
  - Movie Aperiodic-ISC: ASD 53/5, TD 58/20, p = 0.013.
  - Resting 1:1 matched: ASD 51/4, TD 43/12.
  - Resting + movie matched: ASD 39/7, TD 39/7.
  - Dual-paradigm post-QC matched: ASD 31/3, TD 26/8.
  - HBN The Present matched: ASD 18/101, TD 18/101.
- Updated HBN The Present config to explicitly encode the current manuscript QC:
  - At least 40 usable artefact-free 2-s movie epochs.
  - No more than 20% invalid scalp channels.
  - Specparam R2 >= 0.90 and top 5% fit-error exclusion.
  - Homologous posterior ROI inclusion remains at least 50% (>=2/4 electrodes) via `roi_channels_hbn129.yaml`.
- Updated `PIPELINE_MAP.md`, `config/README_PATHS.md`, and `README.md` so the public manuscript-facing path points to HBN The Present movie analyses.
- Marked the old HBN supplementary convergence plot as an archived movie-only visual check and removed its dependency on HBN resting source tables.

## Validation

- `python3 scripts/figures/validate_supplementary_source_data.py`
  - PASS.
  - Current report: `figure_source_data/supplementary/source_data_validation_report.md`.
  - Now checks corrected sex counts and primary-vs-HBN movie QC thresholds.
- `python3 scripts/figures/supplementary/draw_s1_cohort_flow_pil.py`
  - Wrote:
    - `outputs/figures/supplementary/Supplementary_Figure_S1_cohort_flow.png`
    - `outputs/figures/supplementary/Supplementary_Figure_S1_cohort_flow.pdf`
- Text search over manuscript-facing figure/data/manifest paths found no remaining references to:
  - `s8_hbn_resting`
  - `HBN eyes-open matched subset`
  - `Eyes-open resting subset`
  - `Eyes-closed resting subset`
  - `config_hbn_resting`

## Remaining Provenance-Only Files

Some old HBN resting scripts/configs remain in the repository for provenance and because this audit avoided destructive file removal:

- `config/config_hbn_resting.yaml`
- `config/config_hbn_external.yaml`
- `scripts/hbn/hbn_preprocess_resting.py`
- `scripts/hbn/hbn_rest_eyes_open_matched.py`
- `scripts/hbn/hbn_eo_posterior_exponent.py`
- `src/hbn_eo_exponent.py`
- `src/hbn_confirmatory_replication.py`
- `src/hbn_strict_replication.py`
- resting HBN CSV artifacts under `figure_source_data/`

These files are not part of the current manuscript-facing source-data manifest or the current public pipeline map.

## Current Verdict

The manuscript-facing code/data path is aligned with the 2026-08-12(q)稿件: HBN resting-state analyses have been removed from the active source-data and figure pipeline, HBN The Present movie convergence remains available, participant-characteristics source data match the current tables, and posterior ROI inclusion configs remain at `min_valid_channel_ratio: 0.5`.

Before a GitHub release, consider either physically deleting the provenance-only HBN resting files in a separate cleanup commit or moving them under an explicit `archive/` directory after author approval.
