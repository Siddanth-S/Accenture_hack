"""HTMX console — three views: Live / Review Queue / Governance."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, Form
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .ledger import Ledger
from .policy import load_policy
from . import state

ROOT    = Path(__file__).resolve().parents[2]
_policy = load_policy(os.getenv("PREFLIGHT_POLICY", str(ROOT / "policies" / "default.yaml")))
_ledger = Ledger(ROOT / "data" / "preflight.db")

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

router = APIRouter()

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_SEVERITY = {"pass": 0, "pass_with_citations": 0,
             "redact": 1, "regenerate": 1, "escalate": 2, "block": 2}

_STRIP_BORDER = {0: "#3D5E3D", 1: "#7A5E1A", 2: "#7A1A1A"}


def _sessions(decisions: list[dict]) -> list[dict]:
    """Group decisions by session, most-recently-active first."""
    by_sid: dict[str, dict] = {}
    for d in reversed(decisions):      # oldest first so turns append in order
        sid = d["session_id"]
        if sid not in by_sid:
            by_sid[sid] = dict(session_id=sid, use_case=d["use_case"],
                               turns=[], last_ts=d["ts"], worst_sev=0)
        s = by_sid[sid]
        s["turns"].append(dict(
            turn=d["turn"],
            action=d["action"],
            latency_ms=round(d["latency_ms"], 1),
            reason=d["reason"],
        ))
        s["last_ts"] = d["ts"]
        sev = _SEVERITY.get(d["action"], 0)
        if sev > s["worst_sev"]:
            s["worst_sev"] = sev
    for s in by_sid.values():
        s["strip_border"] = _STRIP_BORDER[s["worst_sev"]]
    return sorted(by_sid.values(), key=lambda s: s["last_ts"], reverse=True)


def _ctx(view: str, extra: dict) -> dict:
    return {"policy_id": _policy.id, "jurisdiction": _policy.jurisdiction,
            "active_view": view, **extra}


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return await live(request)


@router.get("/console/live", response_class=HTMLResponse)
async def live(request: Request):
    decisions = _ledger.recent(300)
    sessions  = _sessions(decisions)
    stats     = _ledger.stats()
    return TEMPLATES.TemplateResponse(request, "live.html", {
        "sessions": sessions[:25],
        "stats": stats,
        **_ctx("live", {}),
    })


@router.get("/console/partials/strips", response_class=HTMLResponse)
async def strips_partial(request: Request):
    decisions = _ledger.recent(300)
    sessions  = _sessions(decisions)
    return TEMPLATES.TemplateResponse(request, "partials/strips.html", {
        "sessions": sessions[:25],
    })


@router.get("/console/partials/stats", response_class=HTMLResponse)
async def stats_partial(request: Request):
    return TEMPLATES.TemplateResponse(request, "partials/stats.html", {
        "stats": _ledger.stats(),
    })


@router.get("/console/queue", response_class=HTMLResponse)
async def queue_view(request: Request):
    return TEMPLATES.TemplateResponse(request, "queue.html", {
        "items": _ledger.queue(50),
        **_ctx("queue", {}),
    })


@router.get("/console/session/{session_id}", response_class=HTMLResponse)
async def session_detail(request: Request, session_id: str):
    rows     = _ledger.by_session(session_id)
    turns_kv = {t["turn"]: t for t in state.get_turns(session_id)}

    turns = []
    for row in rows:
        payload = json.loads(row["payload"])
        t_store = turns_kv.get(row["turn"], {})
        turns.append({
            "turn":             row["turn"],
            "action":           row["action"],
            "stakes":           row["stakes"],
            "latency_ms":       round(row["latency_ms"], 1),
            "cost_usd":         payload.get("cost_usd", 0),
            "reason":           row["reason"],
            "triggered_by":     payload.get("triggered_by", []),
            "degraded":         payload.get("degraded", []),
            "model_requested":  t_store.get("model_requested", payload.get("requested_model", "")),
            "model_routed":     t_store.get("model_routed",    payload.get("routed_model", "")),
            "cost_saved_usd":   t_store.get("cost_saved_usd", 0),
            "prompt":           t_store.get("prompt", ""),
            "response":         t_store.get("response", payload.get("response_preview", "")),
            "risk":             payload.get("risk", {}),
            "session_risk":     payload.get("session_risk", {}),
            "stage_latency_ms": payload.get("stage_latency_ms", {}),
            "findings":         payload.get("findings", []),
            "claims":           payload.get("claims", []),
        })

    # Infer scenario name from session_id for replay sessions
    scenario_name = session_id
    if session_id.startswith("replay-"):
        scenario_name = session_id[7:].replace("-", " ").title()

    return TEMPLATES.TemplateResponse(request, "session.html", {
        "session_id":    session_id,
        "scenario_name": scenario_name,
        "turns":         turns,
        "has_prompts":   any(t["prompt"] for t in turns),
        **_ctx("live", {}),
    })


@router.get("/console/review-form/{seq}", response_class=HTMLResponse)
async def review_form(request: Request, seq: int):
    html = f"""
<form hx-post="/console/review/{seq}"
      hx-target="#review-{seq}"
      hx-swap="outerHTML"
      style="display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap; margin-top:0.35rem;">
  <select name="to_action"
          style="background:var(--panel); border:1px solid var(--border); color:#C8CDD8;
                 font-size:0.7rem; padding:0.25rem 0.4rem; border-radius:3px;">
    <option value="pass">pass — approve</option>
    <option value="escalate">escalate — keep</option>
    <option value="block">block — harden</option>
  </select>
  <input name="rationale" placeholder="rationale (required)" required
         style="flex:1; min-width:160px; background:var(--panel); border:1px solid var(--border);
                color:#C8CDD8; font-size:0.7rem; padding:0.25rem 0.5rem; border-radius:3px;"/>
  <input name="reviewer" placeholder="reviewer id"
         style="width:130px; background:var(--panel); border:1px solid var(--border);
                color:#C8CDD8; font-size:0.7rem; padding:0.25rem 0.5rem; border-radius:3px;"/>
  <button type="submit"
          style="background:var(--cleared); color:#0D1210; font-size:0.68rem; font-weight:700;
                 border:none; padding:0.28rem 0.7rem; border-radius:3px; cursor:pointer;
                 font-family:inherit; letter-spacing:0.04em;">
    Submit
  </button>
</form>"""
    return HTMLResponse(html)


@router.post("/console/review/{seq}", response_class=HTMLResponse)
async def submit_review(
    request: Request,
    seq: int,
    to_action: str = Form(...),
    rationale: str = Form(...),
    reviewer: str = Form(default="reviewer"),
):
    row = _ledger._conn.execute(
        "SELECT action FROM ledger WHERE seq=?", (seq,)
    ).fetchone()
    from_action = row["action"] if row else "unknown"
    _ledger.record_override(seq, reviewer or "reviewer", from_action, to_action, rationale)
    color = "var(--cleared)" if to_action == "pass" else "var(--caution)"
    html = (
        f'<span style="font-size:0.68rem; color:{color}; font-weight:600;">'
        f'✓ {to_action} · {reviewer or "reviewer"}</span>'
    )
    return HTMLResponse(html)


@router.get("/console/chat", response_class=HTMLResponse)
async def chat_view(request: Request):
    return TEMPLATES.TemplateResponse(request, "chat.html", {
        **_ctx("chat", {}),
    })


@router.get("/console/governance", response_class=HTMLResponse)
async def governance(request: Request):
    stats = _ledger.stats()
    lats  = _ledger.latencies()

    buckets = [0] * 10
    for lat in lats:
        buckets[min(9, int(lat))] += 1

    return TEMPLATES.TemplateResponse(request, "governance.html", {
        "stats": stats,
        "actions_json": json.dumps([
            {"name": k, "value": v}
            for k, v in stats.get("actions", {}).items()
        ]),
        "lat_buckets_json": json.dumps(buckets),
        "lat_labels_json": json.dumps(
            ["0–1", "1–2", "2–3", "3–4", "4–5",
             "5–6", "6–7", "7–8", "8–9", "9ms+"]
        ),
        **_ctx("governance", {}),
    })
