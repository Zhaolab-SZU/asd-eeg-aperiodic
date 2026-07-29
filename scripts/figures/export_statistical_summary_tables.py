# Ideal release name: export_statistical_summary_tables.py
# Original path: scripts/export_statistical_summary_tables.py
# Note: Package SI summary CSVs into statistical_tables/
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
Export de-identified summary CSVs for manuscript/SI tables into
statistical_tables/ (and a mirror under outputs/tables/statistical_summary/).

Sources (already computed; this script only packages):
  - Supplementary Table S4a: nested / split-sample ROI validation
  - Supplementary Table S4b: frontal vs posterior comparison
  - HBN FSIQ-adjusted posterior Aperiodic-ISC OLS
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

# scripts/figures/ → github_release root
RELEASE_ROOT = Path(__file__).resolve().parents[2]
_ANALYSIS = RELEASE_ROOT.parent
PROJECT_ROOT = (
    _ANALYSIS
    if (_ANALYSIS / "outputs" / "tables").exists()
    else RELEASE_ROOT
)

OUT_RELEASE = RELEASE_ROOT / "statistical_tables"
OUT_MIRROR = PROJECT_ROOT / "outputs" / "tables" / "statistical_summary"

ROBUST = PROJECT_ROOT / "outputs" / "tables" / "robustness"
FRONTAL = PROJECT_ROOT / "outputs" / "tables" / "supplementary_frontal"
FSIQ = PROJECT_ROOT / "outputs" / "tables" / "hbn_fsiq_adjusted"


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Re-run the corresponding analysis script first."
        )
    return path


def _copy(src: Path, dst_name: str) -> Path:
    dst = OUT_RELEASE / dst_name
    shutil.copy2(src, dst)
    OUT_MIRROR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, OUT_MIRROR / dst_name)
    return dst


def _export_hbn_fsiq() -> Path:
    src = _require(FSIQ / "hbn_isc_fsiq_adjusted_ols.csv")
    df = pd.read_csv(src)
    out = df[
        [
            "analysis",
            "n",
            "n_asd",
            "n_td",
            "beta_td_minus_asd",
            "se",
            "ci_low",
            "ci_high",
            "p",
            "unadj_delta",
            "unadj_p",
            "mean_iq_asd",
            "mean_iq_td",
        ]
    ].copy()
    out.insert(0, "cohort", "matched_HBN")
    out.insert(1, "roi", "posterior")
    out.insert(2, "model", "isc_z ~ C(group) + IQ_total")
    out["note"] = "Residual-FSIQ-adjusted sensitivity; beta = TD - ASD"
    path = OUT_RELEASE / "hbn_fsiq_adjusted_isc.csv"
    out.to_csv(path, index=False)
    out.to_csv(OUT_MIRROR / "hbn_fsiq_adjusted_isc.csv", index=False)
    return path


def _export_s4b_interaction_key() -> Path:
    terms = pd.read_csv(_require(FRONTAL / "frontal_posterior_region_mixed_terms.csv"))
    meta = pd.read_csv(_require(FRONTAL / "frontal_posterior_region_mixed_meta.csv"))
    key = terms[
        terms["term"].isin(
            [
                "C(group)[T.TD]",
                "C(region)[T.frontal]",
                "C(group)[T.TD]:C(region)[T.frontal]",
            ]
        )
    ].copy()
    key["method"] = meta["method"].iloc[0]
    key["formula"] = meta["formula"].iloc[0]
    key["n_subjects"] = int(meta["n_subjects"].iloc[0])
    key["used_mixedlm"] = bool(meta["used_mixedlm"].iloc[0])
    path = OUT_RELEASE / "table_s4b_frontal_vs_posterior_interaction.csv"
    key.to_csv(path, index=False)
    key.to_csv(OUT_MIRROR / "table_s4b_frontal_vs_posterior_interaction.csv", index=False)
    return path


def _write_readme() -> Path:
    text = """# Statistical summary tables

Place de-identified summary CSVs that underpin manuscript/SI tables here.

Do not place raw EEG or item-level clinical raw responses in this folder.

## Contents

| File | Manuscript / SI target | Source script |
|------|------------------------|---------------|
| `table_s4a_nested_split_summary_repeats.csv` | Supplementary Table S4a (primary: 200× 70/30) | `scripts/28_posterior_roi_nested_split_validation.py` (+ `src/posterior_roi_nested_split.py`) |
| `table_s4a_nested_split_summary_kfold.csv` | S4a (5-fold nested) | same |
| `table_s4a_nested_split_single50.csv` | S4a (single 50/50 split) | same |
| `table_s4a_channel_selection_frequency_repeats.csv` | S4a channel selection rates | same |
| `table_s4b_frontal_group_ols.csv` | Supplementary Table S4b (region group OLS) | `scripts/29_supplementary_frontal_comparison.py` (+ `src/frontal_comparison_analysis.py`) |
| `table_s4b_frontal_group_age_interaction.csv` | S4b (group × age) | same |
| `table_s4b_frontal_ados_partial.csv` | S4b (ADOS partial Pearson) | same |
| `table_s4b_frontal_vs_posterior_interaction.csv` | S4b (group × region key terms) | same |
| `table_s4b_movie_isc_effects.csv` | S4b exploratory movie ISC (group summaries only) | same |
| `hbn_fsiq_adjusted_isc.csv` | HBN residual-FSIQ-adjusted Aperiodic-ISC | `scripts/hbn/hbn_isc_fsiq_adjusted.py` |

## Re-export / recompute

Development repository:

```bash
python scripts/28_posterior_roi_nested_split_validation.py --n-repeats 200 --n-folds 5
python scripts/29_supplementary_frontal_comparison.py
python scripts/hbn/hbn_isc_fsiq_adjusted.py
python scripts/export_statistical_summary_tables.py
```

Release-bundle equivalents (under `github_release/`):

```bash
python scripts/resting/10c_posterior_roi_nested_split_validation.py --n-repeats 200 --n-folds 5
python scripts/resting/10d_supplementary_frontal_comparison.py
python scripts/hbn/hbn_isc_fsiq_adjusted.py
python scripts/figures/export_statistical_summary_tables.py
```

Library modules: `src/posterior_roi_nested_split.py`, `src/frontal_comparison_analysis.py`.
"""
    path = OUT_RELEASE / "README.md"
    path.write_text(text, encoding="utf-8")
    OUT_MIRROR.mkdir(parents=True, exist_ok=True)
    (OUT_MIRROR / "README.md").write_text(text, encoding="utf-8")
    return path


def main() -> None:
    OUT_RELEASE.mkdir(parents=True, exist_ok=True)
    OUT_MIRROR.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    written.append(
        _copy(
            _require(ROBUST / "posterior_roi_nested_split_summary_repeats.csv"),
            "table_s4a_nested_split_summary_repeats.csv",
        )
    )
    written.append(
        _copy(
            _require(ROBUST / "posterior_roi_nested_split_summary_kfold.csv"),
            "table_s4a_nested_split_summary_kfold.csv",
        )
    )
    written.append(
        _copy(
            _require(ROBUST / "posterior_roi_nested_split_single50.csv"),
            "table_s4a_nested_split_single50.csv",
        )
    )
    written.append(
        _copy(
            _require(ROBUST / "posterior_roi_nested_split_selection_freq_repeats.csv"),
            "table_s4a_channel_selection_frequency_repeats.csv",
        )
    )

    written.append(
        _copy(
            _require(FRONTAL / "frontal_posterior_group_ols.csv"),
            "table_s4b_frontal_group_ols.csv",
        )
    )
    written.append(
        _copy(
            _require(FRONTAL / "frontal_posterior_group_age_interaction.csv"),
            "table_s4b_frontal_group_age_interaction.csv",
        )
    )
    written.append(
        _copy(
            _require(FRONTAL / "frontal_posterior_ados_partial.csv"),
            "table_s4b_frontal_ados_partial.csv",
        )
    )
    written.append(_export_s4b_interaction_key())
    written.append(
        _copy(
            _require(FRONTAL / "frontal_posterior_movie_isc_effects.csv"),
            "table_s4b_movie_isc_effects.csv",
        )
    )

    written.append(_export_hbn_fsiq())
    written.append(_write_readme())

    print(f"Wrote {len(written)} files to {OUT_RELEASE}")
    for p in written:
        print(f"  - {p.name}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
