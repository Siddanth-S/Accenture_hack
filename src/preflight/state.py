"""Shared in-process state between proxy and console.

Holds the in-memory turn store — prompt + response text for each session turn.
Not persisted to the ledger (invariant 7: ledger never keeps raw prompt bodies).
Lost on restart; repopulate with POST /replay.
"""
from __future__ import annotations

# session_id -> list of turn dicts, ordered by turn number
turn_store: dict[str, list[dict]] = {}


def record_turn(session_id: str, turn: int, prompt: str, response: str,
                model_requested: str, model_routed: str, cost_saved_usd: float) -> None:
    if session_id not in turn_store:
        turn_store[session_id] = []
    turn_store[session_id].append({
        "turn":            turn,
        "prompt":          prompt,
        "response":        response,
        "model_requested": model_requested,
        "model_routed":    model_routed,
        "cost_saved_usd":  round(cost_saved_usd, 6),
    })
    turn_store[session_id].sort(key=lambda t: t["turn"])


def get_turns(session_id: str) -> list[dict]:
    return turn_store.get(session_id, [])
