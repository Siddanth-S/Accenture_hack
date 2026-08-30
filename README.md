# Preflight — ControlPlane Checker

**Accenture Innovation Challenge 2026 · Round 2 · Problem Track 1**
Team Preflight — NITK Surathkal

---

## The thesis

AI failure is not the problem. The delay in noticing it is.

Enterprises discover AI failures through a weekly eval, a month-end cost
review, or a customer complaint — days after a user already acted on the
answer. Guardrail libraries filter one response at a time. Observability
platforms draw graphs after the fact. Neither one decides, and neither
carries session state.

That last gap is the one that matters:

> **A single response is not the unit of failure. A session is.**

Retry loops, creeping cost, claim contamination and gradual jailbreak
escalation do not exist inside any single response. A stateless checker
cannot see them in principle — not merely in practice.

Preflight is a model-agnostic sidecar that wraps the inference call rather
than following it, scores every interaction on **performance, cost and
responsibility**, and carries risk forward across the whole conversation.

---

## Results

Run `python -m eval.run_scenarios` to reproduce:

```
                               stateless   session-aware
  ------------------------------------------------------
  unsafe caught                   4/6             6/6
  detection rate                    67%           100%
  false positives                 0/1             0/1

  2 turn(s) intervened on ONLY because of session state
```

Scoring is **timely** detection — intervened at or before the turn the
scenario becomes unsafe. A checker that fires on turn 9 of a breach that
began on turn 4 has prevented nothing, so "caught somewhere in the session"
would be the wrong metric.

The stateless column is not a strawman. Every competing guardrail library is
stateless by construction.

---

## Quickstart

```bash
pip install -r requirements.txt
python -m eval.run_scenarios          # stateless vs session-aware detection
python -m eval.calibrate              # conformal threshold selection (the proof)
```

No model downloads, no GPU, no API keys. Detectors degrade to documented
lexical fallbacks and say so in the output.

`eval.calibrate` scores a labelled corpus through the real groundedness
pipeline and reports the threshold the conformal guarantee selects. At a small
sample it *refuses* the 5% budget and says so — the number is derived, never
hand-typed. `--write` patches the policy; `--alpha` sets the risk budget.

---

## Architecture — the staged cascade

| Stage | Budget | What runs | Blocking? |
|---|---|---|---|
| 0 · preflight | 1–5 ms | PII, injection, routing, budget | yes, pre-token |
| 1 · streaming | ~0 ms | claim extraction, canary, output PII | during generation |
| 2 · verification | 30–150 ms | NLI entailment per claim | gated by stakes |
| 3 · deep | 300 ms–2 s | semantic entropy, counterfactual bias | tail only |

**On the 80 ms figure from our Round 1 deck:** it was wrong, and we are
correcting it rather than defending it. You cannot run NLI entailment on
every claim plus a counterfactual probe in 80 ms — a counterfactual probe
alone needs a second generation.

The defensible claim is stage 1. Work done *while tokens are still
streaming* costs no wall-clock time on the inference path, because the model
is the bottleneck, not us. That is what "sidecar, not gatekeeper" actually
buys. Roughly 70% of traffic terminates at stage 0–1; the expensive tail runs
where the stakes justify it.

And it is *real*, not just described: `POST /v1/chat/completions` with
`stream: true` streams tokens straight through while the deterministic Stage-1
checks (canary leak, response PII) run on the growing buffer and **cut the
stream the instant a provable fault fires**. When the answer completes, the
full engine runs, the decision is sealed into the ledger, and a trailing
`preflight` SSE event returns the verdict in-band — so a caller learns what was
decided without a second request.

---

## What makes this different

**Session risk accumulator** (`session.py`) — four trajectory signals that
are invisible to any single-response checker:
*escalation gradient* (monotone walk toward a policy boundary),
*contamination* (turn 6 reasoning from an ungrounded claim asserted at turn 2),
*retry burn*, *cost creep*. The store is pluggable: in-memory for the
prototype, **Redis** when `PREFLIGHT_REDIS_URL` is set — so the accumulator
survives restarts and is shared across workers instead of silently fragmenting
per worker behind a load balancer. A configured-but-unreachable Redis degrades
to in-memory with a loud warning; `/health` reports which backend is live.

**Paraphrase-robust injection detection** (`detectors/core.py`) — fixed-string
signatures only catch their exact wording ("ignore all previous instructions"),
so a swap to "kindly disregard the earlier directives" slipped straight
through. Detection now also matches *intent* — a manipulation verb near a
control-surface object, with synonyms on both sides — catching paraphrases with
no model. Heuristic matches escalate rather than hard-block (only a 0.95 exact
signature blocks), and a trained classifier can be plugged in via
`set_injection_classifier()`.

**Claim-level verdicts** (`claims.py`) — atomic claims with character offsets,
so we can redact or regenerate one clause instead of blocking a whole answer.
`UNVERIFIABLE` is a first-class verdict: the brief notes there is often no
real-time ground truth, and a checker that forces every claim into
supported/unsupported is manufacturing confidence it does not have.

**Conformal risk control** (`calibration.py`, run via `python -m eval.calibrate`)
— thresholds are derived from an operator-chosen risk budget, not hand-tuned.
The claim we can defend: *"the probability an unsafe response passes is bounded
at alpha, with 1−delta confidence, distribution-free."* The knob a compliance
officer turns is "what miss rate can we accept", not "what should the
groundedness threshold be". The calibrator is honest about its own sample size:
below the data needed for a 5% budget it refuses to emit a threshold rather
than shipping an undefended number.

**Cost as a gate, not an invoice** (`router.py`) — routing happens before a
token exists. High-stakes traffic is never downgraded, and we never route
*up*: silently spending more of someone's money is not a safety feature.

**Blocking only on deterministic faults** (`policy.py`) — a matched PII
pattern with a valid checksum, a fired canary, a validated injection
signature. Probabilistic detectors never hard-block, because 0.82 is a reason
to regenerate, not a reason to be certain. Blocking on a model's guess is how
you train users to route around the guardrail.

**Tamper-evident ledger** (`ledger.py`) — each record carries the SHA-256 of
its predecessor. "We log decisions" becomes "we can prove our logs are
intact", and `verify_chain()` names the exact sequence number of any break.

---

## Limitations

Stated deliberately — a checker that hides its own failure modes is the thing
we are arguing against.

- Groundedness runs on lexical + numeric overlap **by default**, with
  confidence capped at 0.65 so a fallback verdict can never present as
  model-grade evidence. Setting `PREFLIGHT_NLI_MODEL` loads a real 3-way NLI
  cross-encoder (`nli.py`) and groundedness becomes true entailment — able to
  separate `CONTRADICTED` (a source says the opposite) from `UNSUPPORTED`
  (the source is silent), which the lexical path cannot. On the scenario set,
  turning it on lifts even the *stateless* column to 6/6. Left opt-in so the
  default run needs no model download; the engine reports which mode is live
  and `/health` exposes `groundedness: entailment|lexical`.
- Retry detection uses 4-character prefix stemming, which will occasionally
  collide unrelated words. Embedding cosine is the production path.
- The conformal guarantee assumes calibration and production data are
  exchangeable. Under distribution shift it degrades — which is why drift
  monitoring must feed recalibration.
- The calibration corpus (`data/calibration/`) is *synthetic* — templated
  across domains for coverage and reproducibility, not sampled from production
  traffic. The groundedness scores it produces are real detector outputs; the
  inputs are generated. It exists to make the conformal procedure runnable and
  the guarantee's sample-size honesty visible, not to stand in for real data.
- Counterfactual bias probing requires a second generation. It is sampled,
  not run per-request, and the cost is real.
- `preflight-sessions-v1` is 7 hand-built sessions. Enough to demonstrate the
  mechanism, not enough to make statistical claims from.

## Operating it

| Surface | What it does |
|---|---|
| `GET /health` | live config — demo/live mode, session + turn store backend, auth, streaming |
| `GET /metrics` | Prometheus exposition: decisions by verdict, flag rate, p50/p95/p99 engine latency, cost + cost-avoided. *We argue observability tools only draw graphs after the fact — Preflight is observable too, it just also decides.* Counters derive from the sealed ledger, so they cannot drift from what was recorded. |
| Governance → **Tamper / Verify** | press one button to silently rewrite a historical record, another to re-walk the chain and watch it name the exact broken `seq`. The tamper-evidence claim, made pressable. |

Optional hardening, all off by default so the demo needs no configuration:

- `PREFLIGHT_API_KEY` — gate the inference endpoints behind a bearer token.
- `PREFLIGHT_RATE_LIMIT_PER_MIN` — per-session sliding-window limit (default 60).
- `PREFLIGHT_REDIS_URL` — persist the session accumulator + turn store across
  cold starts and workers; without it, both degrade to in-memory with a warning
  (which is why the *deployed* multi-turn demo wants Redis or an always-on host,
  not serverless).

Honesty carries through to routing: if the model that actually answered differs
from the one Preflight routed to — e.g. an upstream quota fallback — the
response reports `served_model` and a `routing_fallback` flag rather than
pretending the routed model replied.

## Repo map

```
src/preflight/
  schemas.py       risk vector, claims, decisions
  session.py       the session risk accumulator
  claims.py        extraction + groundedness verification
  detectors/core.py PII, injection, entropy, bias, capability negotiation
  policy.py        declarative YAML -> action ladder
  calibration.py   conformal risk control
  router.py        cost-aware routing + waste ledger
  engine.py        the staged cascade
policies/          per-jurisdiction policy, hot-swappable
data/sessions/     preflight-sessions-v1 adversarial multi-turn set
eval/              scenario runner
```
