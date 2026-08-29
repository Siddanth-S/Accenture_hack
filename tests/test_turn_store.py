"""Turn store tests — pluggable backend, same public surface.

Covers the in-memory default, the module-level functions the app imports, the
JSON Redis backend against a fake client, and the honest degrade-to-memory when
a configured Redis is unreachable.
"""
from __future__ import annotations

import json

from preflight import state
from preflight.state import (
    TurnStore,
    _MemoryTurnBackend,
    _RedisTurnBackend,
    _make_backend,
)


def _record(store: TurnStore, sid: str, turn: int, prompt="p", response="r"):
    store.record(sid, turn, prompt, response,
                 model_requested="claude-sonnet-5", model_routed="gpt-4o-mini",
                 cost_saved_usd=0.01)


# -- in-memory backend -----------------------------------------------------

def test_memory_record_and_get():
    store = TurnStore(_MemoryTurnBackend())
    _record(store, "s1", 1, prompt="hello")
    _record(store, "s1", 2, prompt="again")
    turns = store.get("s1")
    assert [t["turn"] for t in turns] == [1, 2]
    assert turns[0]["prompt"] == "hello"
    assert store.kind == "memory"


def test_memory_orders_by_turn():
    store = TurnStore(_MemoryTurnBackend())
    _record(store, "s1", 3)
    _record(store, "s1", 1)
    _record(store, "s1", 2)
    assert [t["turn"] for t in store.get("s1")] == [1, 2, 3]


def test_memory_get_returns_copy():
    store = TurnStore(_MemoryTurnBackend())
    _record(store, "s1", 1)
    store.get("s1").append({"turn": 99})
    assert [t["turn"] for t in store.get("s1")] == [1]  # not mutated


def test_memory_clear():
    store = TurnStore(_MemoryTurnBackend())
    _record(store, "s1", 1)
    store.clear("s1")
    assert store.get("s1") == []


# -- module-level functions (default singleton) ----------------------------

def test_module_functions_roundtrip():
    sid = "module-test-session"
    state.clear_turns(sid)
    state.record_turn(sid, 1, "prompt", "response", "req", "routed", 0.02)
    turns = state.get_turns(sid)
    assert len(turns) == 1
    assert turns[0]["response"] == "response"
    assert turns[0]["cost_saved_usd"] == 0.02
    assert state.store_kind() in {"memory", "redis"}
    state.clear_turns(sid)
    assert state.get_turns(sid) == []


# -- Redis backend (fake client) -------------------------------------------

class _FakeRedis:
    """Minimal get/set/delete over a dict, storing the same JSON strings a real
    client would. Proves the backend serialises correctly without a live Redis."""

    def __init__(self):
        self.d: dict[str, str] = {}

    def get(self, k):
        return self.d.get(k)

    def set(self, k, v, ex=None):
        self.d[k] = v

    def delete(self, k):
        self.d.pop(k, None)


def test_redis_backend_serialises_json():
    fake = _FakeRedis()
    store = TurnStore(_RedisTurnBackend(fake))
    _record(store, "s1", 2, prompt="second")
    _record(store, "s1", 1, prompt="first")
    # Stored as a JSON string under the namespaced key.
    raw = fake.d["preflight:turns:s1"]
    assert isinstance(raw, str)
    assert [t["turn"] for t in json.loads(raw)] == [1, 2]
    # And reads back sorted through the public API.
    assert [t["prompt"] for t in store.get("s1")] == ["first", "second"]
    assert store.kind == "redis"


def test_redis_backend_clear():
    fake = _FakeRedis()
    store = TurnStore(_RedisTurnBackend(fake))
    _record(store, "s1", 1)
    store.clear("s1")
    assert store.get("s1") == []
    assert "preflight:turns:s1" not in fake.d


# -- degrade-to-memory when Redis is configured but unreachable ------------

def test_unreachable_redis_falls_back_to_memory(monkeypatch, capsys):
    # A bogus URL that cannot connect within the 1s timeout.
    monkeypatch.setenv("PREFLIGHT_REDIS_URL", "redis://127.0.0.1:6390/0")
    backend = _make_backend()
    assert backend.kind == "memory"
    # And it says so loudly rather than pretending.
    assert "falling back to in-memory turn store" in capsys.readouterr().err


def test_no_redis_url_uses_memory(monkeypatch):
    monkeypatch.delenv("PREFLIGHT_REDIS_URL", raising=False)
    assert _make_backend().kind == "memory"
