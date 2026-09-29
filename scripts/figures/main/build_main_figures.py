"""Build manuscript Figures 1–5.

Usage (from repository root):
  python scripts/figures/main/build_main_figures.py
  python scripts/figures/main/build_main_figures.py 1 3 5
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "src" / "config.py").is_file():
    if ROOT.parent == ROOT:
        raise RuntimeError("Cannot locate repository root (missing src/config.py)")
    ROOT = ROOT.parent

SCRIPTS = ROOT / "scripts" / "figures" / "main"
sys.path.insert(0, str(SCRIPTS))


def main(which=None):
    from _make_figure1 import make_figure1
    from _make_figure2 import make_figure2
    from _make_figure3 import make_figure3
    from _make_figure4 import make_figure4
    from _make_figure5 import make_figure5

    builders = {
        1: make_figure1,
        2: make_figure2,
        3: make_figure3,
        4: make_figure4,
        5: make_figure5,
    }
    targets = which or sorted(builders)
    for number in targets:
        builders[number]()
    print("main figures →", ROOT / "outputs" / "figures" / "main")


if __name__ == "__main__":
    nums = [int(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else None
    main(nums)
