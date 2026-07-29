# Ideal release name: 24b_envelope_partial_isc.py
# Original path: scripts/93_aperiodic_envelope_partial_isc.py
# Note: Envelope-adjusted Aperiodic-ISC (Supp Fig S6)
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
93_aperiodic_envelope_partial_isc.py
------------------------------------
Quantify partial overlap between within-group Aperiodic-ISC and broadband
envelope ISC. Tests whether Aperiodic-ISC group differences remain after
controlling envelope ISC (ANCOVA on Fisher z).

Input:
  derivatives/derivatives_task_movie/stats/aperiodic_isc/aperiodic_isc_within_group_subject_values.csv
  derivatives/derivatives_task_movie/stats/classic_isc/envelope_within_group_subject_values.csv

Output:
  derivatives/derivatives_task_movie/stats/classic_isc/aperiodic_envelope_partial_analysis.csv
  derivatives/derivatives_task_movie/stats/classic_isc/aperiodic_envelope_partial_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import save_csv  # noqa: E402
from src.movie_isc_partial_analysis import (  # noqa: E402
    build_manuscript_summary_table,
    load_merged_within_group_isc,
    run_aperiodic_envelope_partial_analysis,
)


def main() -> None:
    cfg = load_config()
    log = setup_logging(cfg, name="aperiodic_envelope_partial_isc")

    stats_root = Path(cfg["paths"]["derivatives_root"]) / "derivatives_task_movie" / "stats"
    ap_path = stats_root / "aperiodic_isc" / "aperiodic_isc_within_group_subject_values.csv"
    env_path = stats_root / "classic_isc" / "envelope_within_group_subject_values.csv"
    out_dir = stats_root / "classic_isc"

    if not ap_path.exists():
        raise FileNotFoundError(ap_path)
    if not env_path.exists():
        raise FileNotFoundError(env_path)

    merged = load_merged_within_group_isc(ap_path, env_path)
    partial = run_aperiodic_envelope_partial_analysis(merged)
    summary = build_manuscript_summary_table(partial)

    save_csv(partial, out_dir / "aperiodic_envelope_partial_analysis.csv")
    save_csv(summary, out_dir / "aperiodic_envelope_partial_summary.csv")

    log.info("Saved partial overlap analysis (%d event types)", len(partial))
    for _, row in partial.iterrows():
        log.info(
            "%s: r=%.3f (shared var %.1f%%), raw p=%.2e, envelope-adjusted p=%.2e, FDR p=%.2e",
            row["event_type"],
            row["pearson_r"],
            row["shared_variance_pct"],
            row["raw_p"],
            row["ancova_group_p"],
            row["ancova_group_fdr_p"],
        )


if __name__ == "__main__":
    main()
