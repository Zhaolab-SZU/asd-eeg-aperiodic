# Figure scripts

Manuscript-facing builders:

```bash
python scripts/figures/main/build_main_figures.py
python scripts/figures/supplementary/build_si_figures.py
```

Palette: ASD `#B22222`, TD `#000080`. Outputs: `outputs/figures/main/` and `outputs/figures/supplementary/`.

CSV search order is `figure_source_data/` → `data/figure_source/` → `figure_source_data/outputs_figure_source_data/`. Supplementary tables also fall back to `data/supplementary_source/`.

`_archived_20260812/` holds the 12 August 2026 plotters (segment violins; SI numbering S1–S8). Do not use them for the submitted manuscript.
