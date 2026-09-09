"""Build Supplementary Figures S1–S5.

Usage (from repository root):
  python scripts/figures/supplementary/build_si_figures.py
  python scripts/figures/supplementary/build_si_figures.py 1 4 5
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "src" / "config.py").is_file():
    if ROOT.parent == ROOT:
        raise RuntimeError("Cannot locate repository root (missing src/config.py)")
    ROOT = ROOT.parent

SCRIPTS = ROOT / "scripts" / "figures" / "supplementary"
sys.path.insert(0, str(SCRIPTS))


def main(which=None):
    from _make_s1 import make_s1
    from _make_s2 import make_s2
    from _make_s3 import make_s3
    from _make_s4 import make_s4
    from _make_s5 import make_s5

    builders = {
        1: make_s1,
        2: make_s2,
        3: make_s3,
        4: make_s4,
        5: make_s5,
    }
    targets = which or sorted(builders)
    for n in targets:
        builders[n]()
    print("SI figures →", ROOT / "outputs" / "figures" / "supplementary")


if __name__ == "__main__":
    nums = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else None
    main(nums)
