# Ideal release name: 10d_supplementary_frontal_comparison.py
# Original path: scripts/29_supplementary_frontal_comparison.py
# Note: Frontal vs posterior comparison (Supp Table S4b)
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
29_supplementary_frontal_comparison.py
--------------------------------------
Supplementary frontal comparison analysis:

1) Frontal (and posterior) group OLS with identical covariates
2) Mixed model: exponent ~ group × region(frontal/posterior) + covariates + (1|subject)
3) Developmental (group×age) and ASD ADOS partial correlations for frontal vs posterior
4) Exploratory movie TD-template Aperiodic-ISC for frontal vs posterior (FDR across tests)

Outputs: outputs/tables/supplementary_frontal/ and reports/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _project_root() -> Path:
    """Resolve analysis repo when this copy lives under github_release/scripts/resting/."""
    here = Path(__file__).resolve()
    bundle = here.parents[2]
    parent = bundle.parent
    for cand in (parent, bundle):
        if (cand / "src" / "frontal_comparison_analysis.py").exists() and (
            (cand / "config").exists()
            or (cand / "figure_source_data" / "roi_subject_wide_primary.csv").exists()
        ):
            return cand
    return bundle


PROJECT_ROOT = _project_root()
sys.path.insert(0, str(PROJECT_ROOT))
_BUNDLE = Path(__file__).resolve().parents[2]
if str(_BUNDLE) not in sys.path:
    sys.path.insert(0, str(_BUNDLE))

from src.config import load_config, setup_logging  # noqa: E402
import src.frontal_comparison_analysis as fca  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supplementary frontal comparison analysis")
    p.add_argument("--config", type=str, default=None)
    p.add_argument(
        "--skip-movie-isc",
        action="store_true",
        help="Skip exploratory movie ISC (resting analyses only)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    log = setup_logging(cfg, name="supp_frontal_comparison")

    if args.skip_movie_isc:

        def _skip(*_a, **_k):
            raise RuntimeError("skipped by --skip-movie-isc")

        fca.run_movie_frontal_isc = _skip  # type: ignore

    paths = fca.run_frontal_comparison_pipeline(
        PROJECT_ROOT,
        Path(cfg["paths"]["outputs_root"]),
    )
    for name, path in paths.items():
        log.info("%s -> %s", name, path)
    log.info("Done.")


if __name__ == "__main__":
    main()
