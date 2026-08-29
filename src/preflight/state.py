"""Shared turn store between proxy and console.

Holds prompt + response text for each session turn — the raw bodies the console
renders in the transcript view. This is deliberately NOT the ledger: invariant 7
says the tamper-evident ledger never keeps raw prompt bodies, so this ephemeral
UI store is where they live instead, under a TTL, separate from the durable
audit trail.

Like the session accumulator (see [[session.py]]) the store is pluggable:

  - in-memory by default, so the prototype runs with nothing installed;
  - Redis when PREFLIGHT_REDIS_URL is set, so transcripts survive a restart and
    are shared across workers — the same fix, and the same env var, as the
    session store. Without it, two uvicorn workers behind a load balancer show
    different halves of a conversation depending on who served each turn, and a
    restart blanks the console mid-demo.

The module-level `record_turn` / `get_turns` / `clear_turns` functions are kept
as the public surface so proxy.py and console.py do not care which backend is
live. Values are plain JSON-serialisable dicts, so the Redis backend uses JSON
rather than pickle — no trusted-unpickle surface for data that is only ever read
back to render a page.
"""
from __future__ import annotations

import json
import os
import sys
import time

# Match the session store's dormancy window: a transcript with no turns for this
# long is evicted. Redis enforces it itself; the memory backend sweeps on read.
TURN_TTL_S = 60 * 60 * 4


class _MemoryTurnBackend:
    """Process-local dict. Lost on restart, not shared across workers — the
    limitation the Redis backend exists to remove."""

    kind = "memory"

    def __init__(self) -> None:
        self._d: dict[str, list[dict]] = {}
        self._seen: dict[str, float] = {}

    def get(self, sid: str) -> list[dict]:
        self._evict()
        return list(self._d.get(sid, []))

    def append(self, sid: str, turn: dict) -> None:
        self._d.setdefault(sid, []).append(turn)
        self._d[sid].sort(key=lambda t: t["turn"])
        self._seen[sid] = time.time()

    def clear(self, sid: str) -> None:
        self._d.pop(sid, None)
        self._seen.pop(sid, None)

    def _evict(self) -> None:
        now = time.time()
        for k in [k for k, seen in self._seen.items()
                  if now - seen > TURN_TTL_S]:
            self._d.pop(k, None)
            self._seen.pop(k, None)


class _RedisTurnBackend:
    """Redis-backed store — survives restarts, shared across workers.

    One JSON list per session under a TTL key. Read-modify-write on append: the
    transcript is short (a handful of turns) and only the proxy writes it, so the
    race window is negligible and not worth a Lua script for a demo tool.
    """

    kind = "redis"
    _PREFIX = "preflight:turns:"

    def __init__(self, client) -> None:
        self._r = client

    def _key(self, sid: str) -> str:
        return f"{self._PREFIX}{sid}"

    def get(self, sid: str) -> list[dict]:
        raw = self._r.get(self._key(sid))
        return json.loads(raw) if raw else []

    def append(self, sid: str, turn: dict) -> None:
        turns = self.get(sid)
        turns.append(turn)
        turns.sort(key=lambda t: t["turn"])
        self._r.set(self._key(sid), json.dumps(turns), ex=TURN_TTL_S)

    def clear(self, sid: str) -> None:
        self._r.delete(self._key(sid))


def _make_backend():
    """Redis if PREFLIGHT_REDIS_URL is set and reachable, else in-memory.

    A configured-but-unreachable Redis degrades to in-memory with a loud warning
    rather than crashing — the same explicit-degradation contract the session
    store and detectors follow.
    """
    url = os.getenv("PREFLIGHT_REDIS_URL")
    if not url:
        return _MemoryTurnBackend()
    try:
        import redis  # optional dependency; only imported when configured
        client = redis.Redis.from_url(url, socket_connect_timeout=1)
        client.ping()
        return _RedisTurnBackend(client)
    except Exception as exc:  # ImportError or connection failure
        print(
            f"[preflight] PREFLIGHT_REDIS_URL set but Redis is unavailable "
            f"({exc!r}); falling back to in-memory turn store. Transcripts will "
            f"NOT survive restart or span workers.",
            file=sys.stderr,
        )
        return _MemoryTurnBackend()


class TurnStore:
    """Turn store with a pluggable backend. See module docstring."""

    def __init__(self, backend=None) -> None:
        self._backend = backend if backend is not None else _make_backend()

    @property
    def kind(self) -> str:
        return getattr(self._backend, "kind", "custom")

    def record(self, session_id: str, turn: int, prompt: str, response: str,
               model_requested: str, model_routed: str,
               cost_saved_usd: float) -> None:
        self._backend.append(session_id, {
            "turn":            turn,
            "prompt":          prompt,
            "response":        response,
            "model_requested": model_requested,
            "model_routed":    model_routed,
            "cost_saved_usd":  round(cost_saved_usd, 6),
        })

    def get(self, session_id: str) -> list[dict]:
        return self._backend.get(session_id)

    def clear(self, session_id: str) -> None:
        self._backend.clear(session_id)


# Module-level singleton + thin functions: the public surface the rest of the
# app imports. Keeping these as functions means proxy.py / console.py never touch
# the backend directly.
_STORE = TurnStore()


def store_kind() -> str:
    return _STORE.kind


def record_turn(session_id: str, turn: int, prompt: str, response: str,
                model_requested: str, model_routed: str,
                cost_saved_usd: float) -> None:
    _STORE.record(session_id, turn, prompt, response,
                  model_requested, model_routed, cost_saved_usd)


def get_turns(session_id: str) -> list[dict]:
    return _STORE.get(session_id)


def clear_turns(session_id: str) -> None:
    _STORE.clear(session_id)
