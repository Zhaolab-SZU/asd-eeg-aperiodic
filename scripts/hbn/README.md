# HBN scripts

The manuscript-facing external analysis is HBN The Present movie Aperiodic-ISC. It provides cross-stimulus support, not a strict replication.

| Script | Role | Release status |
|---|---|---|
| `100_hbn_inventory.py` | File inventory | present |
| `hbn_preprocess_present.py` | The Present preprocessing | present |
| `hbn_specparam_roi.py` | ROI specparam | present |
| `hbn_isc_calc.py` | Base The Present Aperiodic-ISC | present |
| `hbn_isc_matched.py` | Age/IQ/sex matching | present |
| `hbn_isc_matched_global.py` | Matched global variant | present |
| `hbn_isc_fsiq_adjusted.py` | FSIQ-adjusted OLS | present |
| `hbn_template_dependency_v4.py` | Reconstructed-template permutation and strict cross-fit | pending v6 transfer |
| `hbn_group_age_v5.py` | Exploratory HBN Group × Age | pending v6 transfer |

The submitted locks are reconstructed-template p=0.032 and strict cross-fit p=0.018 for the primary HBN contrast. Reverse-template interpretation was removed from the submitted narrative and must not be presented as a principal analysis.

HBN resting-state pipelines are archived under `scripts/legacy/hbn/`. See `analysis_requests/ANALYSIS_WAREHOUSE_COMPLETION_v6.md` for the exact missing-file contract.
