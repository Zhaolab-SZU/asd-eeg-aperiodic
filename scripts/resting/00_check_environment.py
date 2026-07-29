# Ideal release name: 00_check_environment.py
# Original path: scripts/00_check_environment.py
# Note: Environment / dependency check
# This file is a copy for the public github_release/ bundle.

#!/usr/bin/env python
"""
00_check_environment.py
-----------------------
检查 Python 环境与核心依赖是否可用。

输入: 无
输出: 控制台日志；derivatives/logs/environment_check.txt
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_project_root, load_config, setup_logging  # noqa: E402

REQUIRED_PACKAGES = [
    "mne",
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "statsmodels",
    "sklearn",
    "joblib",
    "yaml",
]


def check_package(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name if name != "yaml" else "yaml")
        ver = getattr(mod, "__version__", "unknown")
        return True, ver
    except ImportError as exc:
        return False, str(exc)


def main() -> None:
    log = setup_logging(name="env_check")
    root = get_project_root()
    log.info("项目根目录: %s", root)

    try:
        cfg = load_config()
        log.info("配置文件加载成功: %s", root / "config" / "config.yaml")
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    # specparam
    try:
        import specparam  # noqa: F401
        log.info("specparam: OK (%s)", specparam.__version__)
    except ImportError:
        try:
            import fooof  # noqa: F401
            log.info("fooof (legacy): OK")
        except ImportError:
            log.error("未安装 specparam / fooof，请运行: pip install specparam")
            sys.exit(1)

    failed = []
    lines = [f"Python {sys.version}", f"Project root: {root}", ""]
    for pkg in REQUIRED_PACKAGES:
        ok, info = check_package(pkg)
        status = f"OK ({info})" if ok else f"FAIL ({info})"
        lines.append(f"{pkg}: {status}")
        log.info("%-15s %s", pkg, status)
        if not ok:
            failed.append(pkg)

    log_dir = Path(cfg["paths"]["logs_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out_file = log_dir / "environment_check.txt"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    log.info("环境检查报告已保存: %s", out_file)

    if failed:
        log.error("缺少依赖: %s\n请运行: pip install -r requirements.txt", failed)
        sys.exit(1)
    log.info("环境检查通过。")


if __name__ == "__main__":
    main()
