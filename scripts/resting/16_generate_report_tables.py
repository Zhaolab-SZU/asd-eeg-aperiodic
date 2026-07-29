# Ideal release name: 16_generate_report_tables.py
# Original path: scripts/16_generate_report_tables.py
# Note: Manuscript / report tables
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
16_generate_report_tables.py
----------------------------
汇总各分析结果为报告用总表。

输入: derivatives/stats/, outputs/tables/
输出: outputs/reports/analysis_summary.md
      outputs/tables/all_results_combined.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.io_utils import save_csv  # noqa: E402


def main() -> None:
    cfg = load_config()
    log = setup_logging(cfg, name="generate_report")

    deriv = Path(cfg["paths"]["derivatives_root"])
    stats_dir = deriv / "stats"
    tables_dir = Path(cfg["paths"]["outputs_root"]) / "tables"
    reports_dir = Path(cfg["paths"]["outputs_root"]) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    stat_files = list(stats_dir.glob("*.csv")) if stats_dir.exists() else []
    table_files = list(tables_dir.glob("*.csv")) if tables_dir.exists() else []

    combined = []
    lines = ["# 分析报告摘要\n", f"统计结果文件数: {len(stat_files)}\n", "## 主要结果文件\n"]

    for f in sorted(stat_files):
        lines.append(f"- `{f.name}`")
        try:
            df = pd.read_csv(f)
            df["source_file"] = f.name
            combined.append(df)
        except Exception as exc:
            log.warning("无法读取 %s: %s", f, exc)

    for f in sorted(table_files):
        lines.append(f"- tables/{f.name}")

    if combined:
        all_df = pd.concat(combined, ignore_index=True, sort=False)
        save_csv(all_df, tables_dir / "all_results_combined.csv")

    # 主分析显著性摘要
    main_path = stats_dir / "main_group_analysis.csv"
    if main_path.exists():
        main_df = pd.read_csv(main_path)
        group_rows = main_df[main_df["term"].str.contains("group", case=False, na=False)]
        lines.append("\n## 主分析 group 效应\n")
        for _, r in group_rows.iterrows():
            lines.append(
                f"- {r['outcome']}: coef={r['coef']:.4f}, p={r['pvalue']:.4f} (n={r['n_obs']})"
            )

    report_path = reports_dir / "analysis_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("报告已生成: %s", report_path)


if __name__ == "__main__":
    main()
