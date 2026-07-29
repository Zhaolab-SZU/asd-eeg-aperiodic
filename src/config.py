"""加载与管理项目配置。

Public-release note
-------------------
YAML ``paths.*`` entries are **relative to the repository root** (the parent of
``src/`` / ``config/``). Raw EEG and HBN BIDS are not shipped; set local
absolute paths only on your machine and do not commit them. See
``config/README_PATHS.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Repository root: parent of src/ (works for github_release/ and the main repo)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_project_root() -> Path:
    """返回项目根目录。"""
    return PROJECT_ROOT


def load_yaml(path: Path) -> dict[str, Any]:
    """加载 YAML 配置文件。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """加载主配置 config.yaml。"""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    cfg = load_yaml(config_path)
    # 将相对路径解析为绝对路径
    paths = cfg.get("paths", {})
    for key in ("raw_data", "participants_file", "derivatives_root", "outputs_root", "logs_dir",
                "bids_root", "manifest_file"):
        if key in paths and paths[key]:
            p = Path(paths[key])
            if not p.is_absolute():
                paths[key] = str((PROJECT_ROOT / p).resolve())
    cfg["paths"] = paths
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def load_roi_config(roi_path: Path | None = None) -> dict[str, Any]:
    """加载 ROI 电极配置。"""
    if roi_path is None:
        roi_path = PROJECT_ROOT / "config" / "roi_channels.yaml"
    return load_yaml(roi_path)


def resolve_path(cfg: dict[str, Any], *parts: str) -> Path:
    """基于 derivatives_root 或项目根拼接路径。"""
    root = Path(cfg.get("paths", {}).get("derivatives_root", PROJECT_ROOT / "derivatives"))
    return Path(root).joinpath(*parts)


def setup_logging(cfg: dict[str, Any] | None = None, name: str = "asd_eeg") -> logging.Logger:
    """配置日志：控制台 + 文件。"""
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)

    log_dir = PROJECT_ROOT / "derivatives" / "logs"
    if cfg:
        log_dir = Path(cfg.get("paths", {}).get("logs_dir", log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log
