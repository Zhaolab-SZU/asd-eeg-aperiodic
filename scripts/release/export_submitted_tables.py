#!/usr/bin/env python3
"""Export the locked NMH Supplementary Information tables to CSV.

The submitted DOCX is an input supplied by the authors and is intentionally not
committed. This exporter makes the public, machine-readable table snapshot from
that source without modifying values or interpreting statistical fields.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from docx import Document


TABLE_STEMS = [
    "table_s01a_participant_characteristics",
    "table_s01b_analytic_cohort_summary",
    "table_s02",
    "table_s03",
    "table_s04",
    "table_s05",
    "table_s06",
    "table_s07",
    "table_s08",
    "table_s09",
    "table_s10",
    "table_s11_partly_cloudy_aperiodic_isc",
    "table_s12_state_modulation",
    "table_s13_hbn_template_dependency",
    "table_s14_method_comparison",
    "table_s15_temporal_shift",
    "table_s16_lag_analysis",
    "table_s17a_trajectory_metrics",
    "table_s17b_staged_models",
    "table_s18_group_by_age_models",
]


def clean_cell(value: str) -> str:
    """Collapse Word line breaks and non-breaking spaces without changing text."""
    value = value.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", value).strip()


def unique_headers(values: list[str]) -> list[str]:
    """Create stable CSV headers when Word contains repeated/blank headings."""
    result: list[str] = []
    counts: dict[str, int] = {}
    for index, value in enumerate(values, start=1):
        base = value or f"column_{index}"
        counts[base] = counts.get(base, 0) + 1
        result.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return result


def export_tables(docx_path: Path, output_dir: Path) -> None:
    document = Document(docx_path)
    if len(document.tables) != len(TABLE_STEMS):
        raise RuntimeError(
            f"Expected {len(TABLE_STEMS)} submitted tables, found {len(document.tables)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (table, stem) in enumerate(zip(document.tables, TABLE_STEMS)):
        rows = [[clean_cell(cell.text) for cell in row.cells] for row in table.rows]
        headers = unique_headers(rows[0])
        target = output_dir / f"{stem}.csv"
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(headers)
            writer.writerows(rows[1:])
        print(f"[{index:02d}] {target} ({len(rows) - 1} data rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("supplementary_docx", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/manuscript_tables"),
    )
    args = parser.parse_args()
    export_tables(args.supplementary_docx, args.output_dir)


if __name__ == "__main__":
    main()
