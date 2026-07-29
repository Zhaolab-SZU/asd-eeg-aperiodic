# Fig. 2A topomap rebuild package (plot-only)

Self-contained export for Codex to tweak **layout** without changing channel statistics or spatial interpolation pattern.

## Quick start

```bash
cd figure_source_data/fig2_topomap_rebuild
python plot_fig2a_topomap.py
```

Outputs land in `output/panel_A_channelwise_effect_topomap.{png,tiff,pdf,svg}`.

Dependencies: `numpy`, `pandas`, `matplotlib`, `mne`.

## Files

| Path | Role |
|------|------|
| `plot_fig2a_topomap.py` | **Primary rebuild script** (plot-only; reads `./data/`) |
| `original_plot_fig2a_channelwise_topomap.py` | Copy of repo `scripts/plot_fig2a_channelwise_topomap.py` (reference) |
| `plot_config.json` | Frozen display constants (vlim, montage, sphere, colormap name) |
| `data/channel_group_effects.csv` | **64-channel β / SE / p / q / FDR / posterior flags** (do not recompute) |
| `data/channel_mne_topomap_coords.csv` | MNE `_find_topomap_coords` x/y per channel (+ merged stats for QA) |
| `data/posterior_cluster_callouts.csv` | E33/E36/E37/E38 FDR stats + external label anchor positions |
| `data/colormap_reference_tmap_cbar_stops.csv` | 11-stop hex palette (`reference_tmap_cbar`) |
| `data/channel_group_effects_upstream_fig2_channel_level.csv` | Upstream copy incl. alternate x/y columns |
| `data/topomap_channel_effects_mne_layout_reference.csv` | Earlier combined layout+effect table |
| `reference/panel_A_channelwise_effect_topomap_report.json` | Last successful run metadata from main pipeline |

## Colour scale & colormap (current published figure)

- **β color scale**: fixed symmetric **`vlim = ±0.15`** (CLI default `--vlim 0.15`).
  - Not auto-scaled in the published panel; auto mode exists in the original repo script only when `--vlim` is omitted and uses 88th-percentile |β| clipped to [0.12, 0.16].
- **Colormap name**: `reference_tmap_cbar` (custom `LinearSegmentedColormap`, **not** matplotlib built-in `RdBu_r`).
- **Stops**: see `data/colormap_reference_tmap_cbar_stops.csv` or `REFERENCE_CMAP_STOPS` in the script.
- **Colorbar label**: `β (TD − ASD)`; ticks at `[-vlim, -vlim/2, 0, vlim/2, vlim]`.

## Interpolation & coordinates

- Montage: `GSN-HydroCel-64_1.0` (EGI 64), channels `E1`–`E64`.
- MNE `plot_topomap` with `sphere=0.08`, `image_interp='cubic'`, `extrapolate='head'`, `res=256`.
- Electrode 2D positions from `mne.channels.layout._find_topomap_coords` (exported in `channel_mne_topomap_coords.csv`).
- Head-circle clip radius = 0.08; in-head sensors only shown as black dots.
- Posterior FDR channels (E33/E36/E37/E38): larger markers + straight leader lines to external labels (`posterior_cluster_callouts.csv`).

## Statistics (frozen — do not refit)

Model (reference only):

```
aperiodic_exponent ~ C(group) + age_months + C(sex) + IQ_total + usable_epochs
```

- Group coding: **TD − ASD** (positive β → TD exponent higher).
- FDR: BH across 64 channels, α = 0.05.
- FDR-significant: **E33, E36, E37, E38** only.

## Safe layout tweaks for Codex

- `POSTERIOR_CALLOUT` label positions, font sizes, leader line width.
- Figure size, margins, colorbar fraction/pad/shrink.
- Panel label position (`ax.text` for `"A"`).
- DPI / export formats.

## Do NOT change (preserves spatial pattern)

- `channel_group_effects.csv` β values.
- `TOPO_SPHERE`, montage name, MNE interpolation settings unless intentionally re-validating against reference PNG.
- `vlim` away from 0.15 without explicit request (alters colour saturation).

## Validation

Compare rebuild output to `reference/panel_A_channelwise_effect_topomap_report.json`:
- `vlim`: 0.15
- `fdr_channels`: E33, E36, E37, E38
- Posterior β values in report `posterior_stats`.
