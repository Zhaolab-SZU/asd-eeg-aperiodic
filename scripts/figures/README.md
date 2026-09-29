# Manuscript figure builders

These Python/matplotlib builders are aligned to the manuscript submitted to Nature Mental Health on 28 September 2026.

## Main figures

```bash
python scripts/figures/main/build_main_figures.py
```

| Figure | Builder | Submitted content |
|---|---|---|
| 1 | `_make_figure1.py` | Resting posterior aperiodic activity |
| 2 | `_make_figure2.py` | Spatial robustness |
| 3 | `_make_figure3.py` | Partly Cloudy and HBN The Present Aperiodic-ISC |
| 4 | `_make_figure4.py` | Age-related resting posterior exponent analyses |
| 5 | `_make_figure5.py` | Resting posterior exponent–ADOS associations |

## Supplementary figures

```bash
python scripts/figures/supplementary/build_si_figures.py
```

| Figure | Submitted content |
|---|---|
| S1 | Cohort flow |
| S2 | Developmental sensitivity |
| S3 | Individual alpha frequency control |
| S4 | Synchrony controls |
| S5 | HBN The Present sliding-window and non-overlapping 2-s parameterization check |

Each builder reads shipped CSV source data and writes PNG, PDF and SVG files under `outputs/figures/`. Text remains editable in SVG/PDF. The HBN inferential corrections reported in the legend and Supplementary Tables are supplied by the v4 analysis; the plotted subject values are the de-identified source data shipped here.
