# Nature Mental Health submission alignment

Locked source: the manuscript and Supplementary Information submitted on 28 September 2026. This document records how the public repository maps to that submitted scientific record.

## Scientific narrative

The submitted main line is:

1. a clinically contextualized resting posterior aperiodic phenotype;
2. reduced TD-referenced temporal coordination during Partly Cloudy;
3. persistence after broadband-envelope, timing and trajectory-structure checks;
4. directionally consistent, template-corrected support in HBN The Present;
5. cautious exploratory age analyses and no robust film-level Aperiodic-ISC–ADOS association.

Rest-to-movie mean-exponent comparisons and reverse-template interpretation are not principal submitted claims and must not be reintroduced into the README, main-figure order or release summary.

## Submitted figure map

| Item | Public builder | Source-data status |
|---|---|---|
| Figure 1 | scripts/figures/main/_make_figure1.py | shipped |
| Figure 2 | scripts/figures/main/_make_figure2.py | shipped |
| Figure 3 | scripts/figures/main/_make_figure3.py | shipped; Partly Cloudy and HBN |
| Figure 4 | scripts/figures/main/_make_figure4.py | shipped |
| Figure 5 | scripts/figures/main/_make_figure5.py | shipped |
| Supplementary Figures S1–S5 | scripts/figures/supplementary/ | shipped |
| Supplementary Tables S1–S18 | data/manuscript_tables/ | exported from the submitted SI |

## Reproducibility status

| Analysis family | Submitted numerical lock | Public end-to-end script |
|---|---|---|
| Resting primary/robustness/clinical | yes | present |
| Partly Cloudy primary ISC and reconstructed permutation | yes | present |
| Film-level ISC–ADOS (v1) | yes | pending analysis-workspace delivery |
| Temporal-shift, lag and trajectory structure (v2) | yes | pending analysis-workspace delivery |
| Primary-cohort Group × Age (v3) | yes | pending analysis-workspace delivery |
| HBN template correction and strict cross-fit (v4) | yes | pending analysis-workspace delivery |
| HBN Group × Age (v5) | yes | pending analysis-workspace delivery |

The exact completion contract is analysis_requests/ANALYSIS_WAREHOUSE_COMPLETION_v6.md. Until it passes, the repository supports figure regeneration and numerical audit but not every submitted sensitivity analysis from raw/derived trajectories.

## Numerical locks added in this alignment

- Partly Cloudy: ASD n=58, TD n=78; mean difference TD−ASD approximately 0.055 Fisher-z; Welch p approximately 1.95×10⁻7; reconstructed-template equal-tail p=0.00020.
- HBN: 119 matched ASD–TD pairs; reconstructed-template p=0.032; strict cross-fit p=0.018.
- Primary-cohort Group × Age: −0.0121 Fisher-z/year, HC3 p=0.0495, but the reconstructed-template bootstrap interval spans zero.
- HBN Group × Age: −0.0020 Fisher-z/year, HC3 p=0.856; reconstructed-template p=0.820.
- Film-level ISC–ADOS total: partial r=0.022, p=q=0.90, n=57.

These values must be checked against data/manuscript_tables and analysis output files before a release tag is created.

## Release gate

Do not create v2.0-nmh-submitted until:

1. the v6 analysis-workspace manifest is complete;
2. all number-lock tests pass;
3. all main and supplementary figures rebuild without manual editing;
4. the repository contains no direct identifiers, raw primary-cohort EEG, item-level clinical data or machine-specific absolute paths;
5. an archival DOI is added if available.
