"""Pre-flight cost control.

Most guardrail products treat cost as an observability problem — you find out
at month end. Preflight treats it as a gate: the routing decision happens
BEFORE a token exists, so the cheapest sufficient model is chosen rather than
audited afterwards.

The waste ledger is the part a CFO cares about. Three kinds of spend that
never appear as a line item anywhere:

  downgrade savings - a frontier model doing a lookup a small model handles
  retrieval waste   - retrieved chunks that no claim in the answer used
  retry burn        - the same question answered three times

None of these are visible per-response. All of them are visible per-session,
which is the same argument as the risk accumulator, applied to money.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# USD per 1K tokens (in, out). Illustrative — real prices go in config.
MODEL_COSTS: dict[str, tuple[float, float]] = {
    "claude-opus-5":       (0.015, 0.075),
    "claude-sonnet-5":     (0.003, 0.015),
    "claude-haiku-4-5":    (0.0008, 0.004),
    "gpt-4o":              (0.005, 0.015),
    "gpt-4o-mini":         (0.00015, 0.0006),
    "local/qwen2.5-1.5b":  (0.0, 0.0),
}

# Ordered cheapest -> most capable within a tier ladder.
LADDER = [
    "local/qwen2.5-1.5b",
    "gpt-4o-mini",
    "claude-haiku-4-5",
    "claude-sonnet-5",
    "gpt-4o",
    "claude-opus-5",
]


@dataclass
class RouteDecision:
    requested: str
    routed: str
    complexity: str
    reason: str
    est_cost_usd: float
    est_cost_if_requested: float

    @property
    def saved_usd(self) -> float:
        return round(max(0.0, self.est_cost_if_requested - self.est_cost_usd), 6)


_COMPLEX_SIGNALS = [
    (r"\b(analyse|analyze|compare|evaluate|assess|synthesi[sz]e)\b", 2),
    (r"\b(why|explain|reasoning|justify|implications?)\b", 1),
    (r"\b(step[-\s]by[-\s]step|walk me through|derive|prove)\b", 2),
    (r"\b(code|implement|refactor|debug|algorithm)\b", 2),
    (r"\b(legal|regulatory|compliance|contract|clinical|diagnos)\b", 2),
    (r"\b(draft|write|compose)\b", 1),
]
_SIMPLE_SIGNALS = [
    (r"^\s*(what is|who is|when is|where is|define)\b", -2),
    (r"\b(status|balance|hours|address|phone|email)\b", -1),
    (r"^\s*(hi|hello|thanks|thank you|ok|okay)\b", -3),
]


def classify_complexity(prompt: str) -> tuple[str, int]:
    """Cheap deterministic complexity classification.

    Deliberately rules-based, not a model call. Spending an LLM call to
    decide which LLM to call is how routing layers eat their own savings —
    a mistake worth naming out loud, because it is a common one.
    """
    score = 0
    for pat, w in _COMPLEX_SIGNALS:
        if re.search(pat, prompt, re.IGNORECASE):
            score += w
    for pat, w in _SIMPLE_SIGNALS:
        if re.search(pat, prompt, re.IGNORECASE):
            score += w

    words = len(prompt.split())
    if words > 120:
        score += 2
    elif words > 50:
        score += 1
    elif words < 12:
        score -= 1

    if score <= -1:
        return "trivial", score
    if score <= 1:
        return "simple", score
    if score <= 4:
        return "moderate", score
    return "complex", score


TIER_FLOOR = {
    "trivial":  "gpt-4o-mini",
    "simple":   "gpt-4o-mini",
    "moderate": "claude-haiku-4-5",
    "complex":  "claude-sonnet-5",
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    cin, cout = MODEL_COSTS.get(model, (0.003, 0.015))
    return round((tokens_in / 1000) * cin + (tokens_out / 1000) * cout, 6)


def route(
    prompt: str,
    requested_model: str,
    *,
    stakes: str = "medium",
    allowed: list[str] | None = None,
    est_tokens_out: int = 400,
) -> RouteDecision:
    """Choose the cheapest sufficient model.

    Two guardrails on the guardrail:

      - High-stakes traffic is never downgraded. Saving $0.004 on a loan
        decision is not a trade any enterprise wants made automatically.
      - We never ROUTE UP. If the caller asked for a cheap model, that is
        their call; silently spending more of someone's money is not a
        safety feature.
    """
    complexity, score = classify_complexity(prompt)
    tokens_in = max(1, len(prompt.split()) * 4 // 3)
    baseline = estimate_cost(requested_model, tokens_in, est_tokens_out)

    if stakes in ("high", "critical"):
        return RouteDecision(
            requested=requested_model, routed=requested_model,
            complexity=complexity,
            reason=f"stakes={stakes}: routing disabled, capability over cost",
            est_cost_usd=baseline, est_cost_if_requested=baseline,
        )

    floor = TIER_FLOOR[complexity]
    candidates = [m for m in LADDER if m in (allowed or LADDER)]
    if not candidates:
        candidates = LADDER

    try:
        floor_idx = candidates.index(floor)
    except ValueError:
        floor_idx = 0
    try:
        req_idx = candidates.index(requested_model)
    except ValueError:
        req_idx = len(candidates) - 1

    chosen_idx = min(floor_idx, req_idx)  # never route up
    routed = candidates[chosen_idx]
    cost = estimate_cost(routed, tokens_in, est_tokens_out)

    if routed == requested_model:
        reason = f"complexity={complexity} (score {score}): requested model retained"
    else:
        reason = (
            f"complexity={complexity} (score {score}): "
            f"{requested_model} -> {routed}"
        )

    return RouteDecision(
        requested=requested_model, routed=routed, complexity=complexity,
        reason=reason, est_cost_usd=cost, est_cost_if_requested=baseline,
    )


def retrieval_waste(sources: list[dict], used_source_ids: set[str]) -> dict:
    """Retrieved context that no claim in the answer actually used.

    Measurable because we already attribute every claim to a source during
    verification — the attribution is free, it falls out of groundedness
    checking. Chunks with zero attributions were paid for and read by nobody.
    """
    if not sources:
        return {"chunks": 0, "unused": 0, "wasted_tokens": 0, "waste_ratio": 0.0}
    unused = [s for s in sources if s.get("id") not in used_source_ids]
    wasted_tokens = sum(len(s.get("text", "").split()) * 4 // 3 for s in unused)
    return {
        "chunks": len(sources),
        "unused": len(unused),
        "wasted_tokens": wasted_tokens,
        "waste_ratio": round(len(unused) / len(sources), 3),
    }
