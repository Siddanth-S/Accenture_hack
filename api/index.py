"""Vercel serverless entrypoint for Preflight.

Vercel's @vercel/python runtime serves the ASGI `app` exported here. We put
`src/` on the path and force the SQLite ledger into /tmp (Vercel's only writable
directory) before importing the app.

NOTE: Vercel is serverless/stateless. The in-memory session accumulator and turn
store do NOT persist across cold invocations, so the multi-turn session features
(Live trajectory, session risk building across turns) may reset. The Showdown,
Governance, external benchmark (cached), landing and single-response scoring all
work fine. For the full stateful experience, set PREFLIGHT_REDIS_URL to a hosted
Redis, or deploy on an always-on host (Render/Railway).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The bundle filesystem is read-only except /tmp — point the ledger there.
os.environ.setdefault("PREFLIGHT_DB", "/tmp/preflight.db")

from preflight.proxy import app  # noqa: E402  (import after path/env setup)
