"""Test path setup: the package is importable without installation."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

for entry in (str(PACKAGE_ROOT / "src"), str(PACKAGE_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)
