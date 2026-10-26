"""Shared constants of the figure scripts: paths, method labels, skip helper."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULT = ROOT / "result"
OUT_DIR = ROOT / "paper_assets" / "figures" / "generated"


def require(*paths: Path) -> bool:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        print("[SKIP] missing input(s): " + ", ".join(str(p) for p in missing))
        return False
    return True
