# Ideal release name: 17_development_interaction.py
# Original path: scripts/19_development_and_reliability_extension.py
# Note: Age × group developmental analyses
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
19_development_and_reliability_extension.py
--------------------------------------------
扩展分析：年龄 group×age 交互 + split-half 信度。

用法（项目根目录）:
  python scripts/19_development_and_reliability_extension.py
  python scripts/19_development_and_reliability_extension.py --max-subjects 5
  python scripts/19_development_and_reliability_extension.py --skip-split-half
  python scripts/19_development_and_reliability_extension.py --n-jobs 4 --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging  # noqa: E402
from src.extension_analysis import (  # noqa: E402
    compute_split_half_reliability,
    fit_interaction_models,
    load_main_qc_cohort,
    plot_development_interaction,
    plot_split_half_reliability,
    run_split_half_batch,
    write_extension_report,
)
from src.io_utils import save_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Development interaction + split-half reliability")
    p.add_argument("--max-subjects", type=int, default=None, help="试跑：限制 split-half 被试数")
    p.add_argument("--overwrite", action="store_true", help="覆盖已有输出")
    p.add_argument("--skip-split-half", action="store_true", help="仅运行年龄交互部分")
    p.add_argument("--skip-development", action="store_true", help="仅运行 split-half")
    p.add_argument("--n-jobs", type=int, default=1, help="split-half 并行数（joblib threads）")
    return p.parse_args()


def _exists_ok(path: Path, overwrite: bool) -> bool:
    return path.exists() and not overwrite


def main() -> None:
    args = parse_args()
    cfg = load_config()
    log = setup_logging(cfg, name="extension_dev_reliability")
    deriv = Path(cfg["paths"]["derivatives_root"])
    outputs = Path(cfg["paths"]["outputs_root"])

    tables_dir = outputs / "tables" / "extension"
    fig_dir = outputs / "figures" / "extension"
    reports_dir = outputs / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_main_qc_cohort(cfg, deriv)
    log.info("主分析 QC 队列: N=%d (ASD=%d, TD=%d)",
             len(cohort), (cohort["group"] == "ASD").sum(), (cohort["group"] == "TD").sum())

    interaction_path = tables_dir / "development_interaction_models.csv"
    interaction_df = None

    if not args.skip_development:
        log.info("=== 年龄交互模型 ===")
        if _exists_ok(interaction_path, args.overwrite):
            log.info("跳过拟合（已存在）: %s", interaction_path)
            import pandas as pd
            interaction_df = pd.read_csv(interaction_path)
        else:
            interaction_df = fit_interaction_models(cohort)
            save_csv(interaction_df, interaction_path)
            log.info("已保存: %s", interaction_path)

        for outcome, stem in [
            ("global_exponent", "fig_development_interaction_exponent"),
            ("global_offset", "fig_development_interaction_offset"),
        ]:
            fig_base = fig_dir / stem
            if _exists_ok(fig_base.with_suffix(".png"), args.overwrite):
                log.info("跳过作图（已存在）: %s", fig_base)
            else:
                plot_development_interaction(cohort, outcome, fig_base)
                log.info("已保存图: %s", fig_base)

    reliability_df = None
    subject_metrics_path = tables_dir / "split_half_subject_metrics.csv"

    if not args.skip_split_half:
        log.info("=== Split-half reliability ===")
        epochs_dir = deriv / "epochs"

        if _exists_ok(subject_metrics_path, args.overwrite):
            log.info("读取已有 split-half 指标: %s", subject_metrics_path)
            import pandas as pd
            subject_df = pd.read_csv(subject_metrics_path)
        else:
            subject_df = run_split_half_batch(
                cohort, epochs_dir, cfg,
                max_subjects=args.max_subjects,
                n_jobs=args.n_jobs,
            )
            if subject_df.empty:
                log.error(
                    "split-half 无有效被试。请确认 derivatives/epochs/{subject_id}_epochs.fif "
                    "或 {subject_id}-epo.fif 存在（需先运行 02_preprocess_eeg.py）。"
                )
            save_csv(subject_df, subject_metrics_path)
            log.info("已保存: %s (%d 被试)", subject_metrics_path, len(subject_df))

        if not subject_df.empty:
            reliability_path = tables_dir / "split_half_reliability.csv"
            reliability_df = compute_split_half_reliability(subject_df)
            save_csv(reliability_df, reliability_path)
            log.info("已保存: %s", reliability_path)

            fig_sh = fig_dir / "fig_split_half_reliability"
            if _exists_ok(fig_sh.with_suffix(".png"), args.overwrite):
                log.info("跳过作图（已存在）: %s", fig_sh)
            else:
                plot_split_half_reliability(subject_df, reliability_df, fig_sh)
                log.info("已保存图: %s", fig_sh)

    report_path = reports_dir / "extension_development_reliability.md"
    if interaction_df is None:
        import pandas as pd
        interaction_df = pd.read_csv(interaction_path) if interaction_path.exists() else pd.DataFrame()
    if reliability_df is None and (tables_dir / "split_half_reliability.csv").exists():
        import pandas as pd
        reliability_df = pd.read_csv(tables_dir / "split_half_reliability.csv")

    if not interaction_df.empty:
        if reliability_df is None or reliability_df.empty:
            import pandas as pd
            reliability_df = pd.DataFrame({
                "metric": ["global_exponent", "global_offset", "alpha_pw"],
                "n_subjects": [0, 0, 0],
                "spearman_rho": [float("nan")] * 3,
                "spearman_p": [float("nan")] * 3,
                "spearman_brown_spearman": [float("nan")] * 3,
            })
            log.warning("信度结果缺失，报告仅含年龄交互部分")
        write_extension_report(report_path, interaction_df, reliability_df, len(cohort))
        log.info("已保存报告: %s", report_path)
    else:
        log.warning("报告未生成（缺少交互模型结果）")

    print("\n" + "=" * 60)
    print("扩展分析输出")
    print("=" * 60)
    for label, paths in [
        ("Tables", list(tables_dir.glob("*.csv"))),
        ("Figures", sorted({p.with_suffix("") for p in fig_dir.glob("*.png")})),
        ("Report", [report_path] if report_path.exists() else []),
    ]:
        print(f"\n{label}:")
        for p in paths:
            print(f"  {p.resolve()}")


if __name__ == "__main__":
    main()
