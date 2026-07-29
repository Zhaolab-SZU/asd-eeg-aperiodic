# Path configuration for the public release bundle

All YAML configs under `config/` use **paths relative to the repository root**
(the folder that contains `config/`, `src/`, and `scripts/`).

Raw EEG and HBN BIDS data are **not** included in this bundle.

## Placeholder convention

| Key | Meaning | What you should do |
|-----|---------|-------------------|
| `raw_data` | Local resting / movie EEG root | Create `data/raw/` or set to `/PATH/TO/YOUR/RAW_EEG` **locally** (do not commit absolute paths) |
| `participants_file` | Participant table | Place de-identified CSVs under `data/participants/` |
| `derivatives_root` | Intermediate products | Defaults under `derivatives*` (created on run) |
| `outputs_root` | Tables / figures | Defaults under `outputs*` |
| `bids_root` | HBN BIDS download root | Obtain HBN under DUA; point here locally |
| `manifest_file` | HBN scan / inclusion manifest | Generated after HBN inventory scripts |

Example local override (keep only on your machine):

```yaml
paths:
  raw_data: "/PATH/TO/YOUR/RAW_EEG"
  bids_root: "/PATH/TO/HBN_BIDS"
```

`src/config.py` resolves any **relative** path against the repository root at runtime.
Do not commit machine-specific absolute paths.

## Config files

| File | Role |
|------|------|
| `config.yaml` | Primary resting-state analysis |
| `config_resting_matched*.yaml` | Matched / post-QC resting cohorts |
| `config_task_movie*.yaml` | Partly Cloudy movie task |
| `config_hbn.yaml` | Alias of The Present movie config |
| `config_hbn_thepresent.yaml` | HBN The Present ISC |
| `config_hbn_resting.yaml` | HBN resting |
| `config_hbn_external.yaml` | HBN external replication suite |
| `roi_channels.yaml` | Main-study 64-ch ROIs |
| `roi_hbn129.yaml` | Alias of HBN 129-ch ROI map |
