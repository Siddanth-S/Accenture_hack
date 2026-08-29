"""Test configuration.

Puts `src/` on the path so tests import the package as `preflight.*`, exactly
the way `eval/run_scenarios.py` and the proxy do. No install step, no editable
package — the suite runs with a bare `pytest` from the repo root.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
