"""OpenAI-compatible proxy with Preflight oversight.

Integration is a one-line change in any OpenAI SDK client:

    client = OpenAI(base_url="http://localhost:8000", api_key="any")

Every request passes through the staged detection cascade. The decision
is written to the ledger and returned in response headers so callers can
observe what was intercepted without changing their parsing code.

Run:
    uvicorn preflight.proxy:app --port 8000 --reload

Config (env vars or .env file):
    UPSTREAM_BASE_URL   upstream endpoint  (default: https://openrouter.ai/api/v1)
    UPSTREAM_API_KEY    API key            (or API_KEY from .env)
    PREFLIGHT_POLICY    path to policy YAML
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Load .env before anything reads env vars
# ---------------------------------------------------------------------------
_env_file = ROOT / ".env"
if _env_file.exists():
    for _raw in _env_file.read_text().splitlines():
        _raw = _raw.strip()
        if _raw and not _raw.startswith("#") and "=" in _raw:
            _k, _v = _raw.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from .console import router as console_router  # noqa: E402
from .engine import Engine, InferenceResult    # noqa: E402
from .ledger import Ledger                     # noqa: E402
from .policy import load_policy                # noqa: E402
from .schemas import Action, RequestContext    # noqa: E402
from .session import SessionStore              # noqa: E402
from . import state                            # noqa: E402

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_policy_path = os.getenv("PREFLIGHT_POLICY", str(ROOT / "policies" / "default.yaml"))
_policy   = load_policy(_policy_path)
_sessions = SessionStore()
_engine   = Engine(_policy, _sessions)
_ledger   = Ledger(ROOT / "data" / "preflight.db")

UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "https://openrouter.ai/api/v1")
UPSTREAM_API_KEY  = os.getenv("UPSTREAM_API_KEY") or os.getenv("API_KEY", "")

# Map internal model names to OpenRouter namespaced names.
_OPENROUTER_MODELS: dict[str, str] = {
    "claude-opus-5":      "anthropic/claude-opus-4",
    "claude-sonnet-5":    "anthropic/claude-sonnet-4-5",
    "claude-haiku-4-5":   "anthropic/claude-haiku-4-5",
    "gpt-4o":             "openai/gpt-4o",
    "gpt-4o-mini":        "openai/gpt-4o-mini",
    "local/qwen2.5-1.5b": "openai/gpt-4o-mini",   # local → cheapest cloud fallback
}


def _upstream_model(internal: str) -> str:
    if "openrouter" in UPSTREAM_BASE_URL:
        return _OPENROUTER_MODELS.get(internal, internal)
    return internal


app = FastAPI(
    title="Preflight LLM Gateway",
    description="Model-agnostic LLM oversight proxy.",
    version="0.1.0",
)
app.include_router(console_router)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decision_headers(decision) -> dict[str, str]:
    sr = decision.session_risk
    return {
        "x-preflight-action":        decision.action.value,
        "x-preflight-latency-ms":    str(round(decision.latency_ms, 1)),
        "x-preflight-reason":        decision.reason[:200],
        "x-preflight-model-routed":  decision.routed_model,
        "x-preflight-eg":            str(round(sr.escalation_gradient, 3)),
        "x-preflight-contamination": str(round(sr.contamination, 3)),
        "x-preflight-retry-burn":    str(round(sr.retry_burn, 3)),
        "x-preflight-cost-creep":    str(round(sr.cost_creep, 3)),
    }


def _block(status: int, message: str, headers: dict[str, str]) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "preflight_block",
                           "code": "content_policy_violation"}},
        headers=headers,
    )


def _stub_completion(model: str, text: str, tokens_in: int, tokens_out: int) -> dict:
    return {
        "id": f"preflight-stub-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out,
                  "total_tokens": tokens_in + tokens_out},
    }

# ---------------------------------------------------------------------------
# Main proxy endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body      = await request.json()
    messages  = body.get("messages", [])
    prompt    = " ".join(m["content"] for m in messages if m.get("role") == "user") or ""

    session_id = request.headers.get("x-session-id", str(uuid.uuid4()))
    use_case   = request.headers.get("x-use-case", "default")
    turn       = int(request.headers.get("x-turn", "1"))

    ctx = RequestContext(
        session_id=session_id, use_case=use_case, turn=turn,
        requested_model=body.get("model", "gpt-4o-mini"), sources=[],
    )

    # Stage 0 ----------------------------------------------------------------
    pre_findings, pre_faults, rd = _engine.preflight(ctx, prompt)
    ctx.routed_model = rd.routed

    if pre_faults:
        return _block(400, f"Request blocked at preflight: {', '.join(pre_faults)}", {
            "x-preflight-action": "block",
            "x-preflight-stage":  "0",
            "x-preflight-faults": "; ".join(pre_faults),
        })

    # Forward to upstream ----------------------------------------------------
    if not UPSTREAM_API_KEY:
        response_text = (
            "Preflight demo mode — no API key set. "
            "Request passed Stage 0. Set UPSTREAM_API_KEY to proxy to a real model."
        )
        tokens_in     = max(1, len(prompt.split()) * 4 // 3)
        tokens_out    = max(1, len(response_text.split()) * 4 // 3)
        upstream_data = _stub_completion(body.get("model", "demo"),
                                         response_text, tokens_in, tokens_out)
    else:
        upstream_model = _upstream_model(rd.routed)
        upstream_body  = {**body, "model": upstream_model, "stream": False}
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{UPSTREAM_BASE_URL}/v1/chat/completions",
                    json=upstream_body,
                    headers={"Authorization": f"Bearer {UPSTREAM_API_KEY}",
                             "Content-Type": "application/json"},
                )
                resp.raise_for_status()
                upstream_data = resp.json()
        except httpx.HTTPStatusError as exc:
            return JSONResponse(status_code=exc.response.status_code,
                                content={"error": {"message": str(exc), "type": "upstream_error"}})
        except httpx.RequestError as exc:
            return JSONResponse(status_code=502,
                                content={"error": {"message": f"Upstream unreachable: {exc}",
                                                   "type": "upstream_error"}})

        response_text = upstream_data["choices"][0]["message"]["content"]
        usage         = upstream_data.get("usage", {})
        tokens_in     = usage.get("prompt_tokens",    max(1, len(prompt.split()) * 4 // 3))
        tokens_out    = usage.get("completion_tokens", max(1, len(response_text.split()) * 4 // 3))

    # Stages 1-3 -------------------------------------------------------------
    result   = InferenceResult(text=response_text, tokens_in=tokens_in, tokens_out=tokens_out)
    decision = await _engine.evaluate_response(ctx, prompt, result)
    _ledger.append(decision, _policy.version_hash)

    # Store prompt + response for the session detail view
    state.record_turn(session_id, turn, prompt, response_text,
                      rd.requested, rd.routed, rd.saved_usd)

    headers = _decision_headers(decision)
    if decision.action == Action.BLOCK:
        return _block(400, f"Response blocked: {decision.reason}", headers)
    if decision.action == Action.REDACT:
        upstream_data["choices"][0]["message"]["content"] = decision.response_preview
    return JSONResponse(content=upstream_data, headers=headers)

# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

_SCENARIOS_PATH = ROOT / "data" / "sessions" / "preflight-sessions-v1.jsonl"


def _load_scenarios() -> list[dict]:
    return [json.loads(l) for l in _SCENARIOS_PATH.read_text().splitlines() if l.strip()]


async def _run_scenario(scen: dict) -> dict:
    sid = f"replay-{scen['session_id']}"
    _sessions.reset(sid)
    state.turn_store.pop(sid, None)
    turns_out = []

    for t in scen["turns"]:
        ctx = RequestContext(
            session_id=sid, use_case=scen.get("use_case", "default"),
            turn=t["turn"], requested_model="claude-sonnet-5",
            sources=t.get("sources", []), stateless_mode=False,
        )
        result = InferenceResult(
            text=t["response"],
            tokens_in=max(1, len(t["prompt"].split()) * 4 // 3),
            tokens_out=max(1, len(t["response"].split()) * 4 // 3),
        )
        decision = await _engine.evaluate_response(ctx, t["prompt"], result)
        _ledger.append(decision, _policy.version_hash)
        state.record_turn(sid, t["turn"], t["prompt"], t["response"],
                          ctx.requested_model, decision.routed_model, 0.0)
        turns_out.append({
            "turn":       t["turn"],
            "action":     decision.action.value,
            "latency_ms": round(decision.latency_ms, 2),
            "reason":     decision.reason,
        })

    return {"scenario": scen["name"], "label": scen["label"],
            "session_id": sid, "turns": turns_out}


@app.get("/replay")
async def list_scenarios():
    return {"scenarios": [
        {"name": s["name"], "label": s["label"],
         "turns": len(s["turns"]), "unsafe_from_turn": s["unsafe_from_turn"],
         "note": s.get("note", "")}
        for s in _load_scenarios()
    ]}


@app.post("/replay")
async def replay_all():
    results = [await _run_scenario(s) for s in _load_scenarios()]
    return {"replayed": len(results), "results": results}


@app.post("/replay/{scenario_slug}")
async def replay_one(scenario_slug: str):
    needle    = scenario_slug.lower().replace("-", " ").replace("_", " ")
    scenarios = _load_scenarios()
    match     = next((s for s in scenarios if needle in s["name"].lower()), None)
    if not match:
        raise HTTPException(status_code=404, detail={
            "error": f"No scenario matching '{scenario_slug}'",
            "available": [s["name"] for s in scenarios],
        })
    return await _run_scenario(match)

# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok", "policy": _policy.id,
        "policy_version": _policy.version_hash,
        "jurisdiction": _policy.jurisdiction,
        "upstream": UPSTREAM_BASE_URL,
        "demo_mode": not bool(UPSTREAM_API_KEY),
    }


@app.get("/v1/decisions")
async def recent_decisions(limit: int = 20):
    return {"decisions": _ledger.recent(limit)}


@app.get("/v1/decisions/queue")
async def review_queue(limit: int = 25):
    return {"queue": _ledger.queue(limit)}


@app.get("/v1/sessions/{session_id}/turns")
async def session_turns(session_id: str):
    return {"session_id": session_id, "turns": state.get_turns(session_id)}
