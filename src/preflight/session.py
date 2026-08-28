"""Session risk accumulation.

This module is the reason Preflight is not another guardrail library.

A guardrail asks "is this response bad?". That question is unanswerable for
the failure modes that matter most in production, because they are not
properties of a response at all — they are properties of a trajectory:

  - A jailbreak that takes six turns, where each turn is individually benign.
  - A fabricated detail asserted at turn 2 and reasoned FROM at turn 6.
  - The same question re-asked four ways, quadrupling cost, signalling the
    model is failing at something nobody has noticed.
  - Per-response cost that never looks alarming and a session total that does.

None of these exist inside a single response. A stateless checker cannot see
them in principle, not just in practice.
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import re
import sys
import time

from .schemas import Claim, RequestContext, RiskVector, SessionRisk, TurnRecord

# How fast old turns stop counting. A probe 20 turns ago is weaker evidence
# than a probe 2 turns ago, but it is not zero evidence.
DECAY_HALF_LIFE_TURNS = 6.0

# Session is considered dormant after this long; state is evicted.
SESSION_TTL_S = 60 * 60 * 4


def _decay(turns_ago: int) -> float:
    return 0.5 ** (turns_ago / DECAY_HALF_LIFE_TURNS)


_STOP = {
    "the", "a", "an", "is", "are", "do", "does", "how", "what", "i", "to",
    "of", "for", "can", "you", "me", "my", "please", "and", "in", "on", "it",
    "that", "this", "with", "would", "could", "any", "there", "was", "were",
    "be", "been", "has", "have", "had", "will", "need", "want", "tell",
    "know", "get", "work", "works", "process", "way",
}

# Two prompts are "the same question" above this token-overlap threshold.
RETRY_SIMILARITY = 0.55


def content_tokens(text: str) -> frozenset[str]:
    """Content tokens, crudely stemmed to a 4-character prefix.

    Prefix stemming collapses close/closure and permanent/permanently, which
    is what a reformulated question actually looks like. It is blunt and will
    occasionally collide unrelated words — a documented limitation, not a
    hidden one. Embedding cosine is the production path; this keeps the
    prototype dependency-free.
    """
    t = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return frozenset(
        w[:4] for w in t.split() if w not in _STOP and len(w) > 2
    )


# Terms that recur across any same-domain conversation and therefore carry no
# contamination signal on their own. Reusing them is "staying on topic", not
# "reasoning from a fabrication". Excluding them is the difference between
# flagging the turn that inherits a fabricated fee and flagging every polite
# on-topic follow-up in the same thread.
_GENERIC_TERMS = frozenset({
    "about", "after", "again", "allow", "allows", "apply", "applies",
    "available", "based", "before", "being", "between", "could", "current",
    "customer", "customers", "detail", "details", "during", "eligible",
    "eligibility", "following", "general", "generally", "include", "includes",
    "including", "information", "month", "months", "needs", "period", "please",
    "policy", "product", "products", "program", "programme", "provide",
    "provided", "provides", "require", "required", "requires", "requirement",
    "service", "services", "should", "their", "there", "these", "those",
    "three", "under", "using", "value", "various", "which", "while", "would",
    "your", "within", "request", "account",
})


def distinctive_terms(text: str) -> set[str]:
    """Content terms that actually carry contamination signal.

    Drops generic same-domain vocabulary and keeps distinctive words plus any
    concrete figure — a fabricated fee, rate or deadline is precisely what a
    later turn reasons FROM, so numbers are signal, not noise.
    """
    low = text.lower()
    words = {w for w in re.findall(r"[a-z]{5,}", low) if w not in _GENERIC_TERMS}
    figures = set(re.findall(r"\d+(?:\.\d+)?%?", low))
    return words | figures


def semantic_fingerprint(text: str) -> str:
    """Stable id for a question's content tokens.

    Used only as a dictionary key. Matching is done by Jaccard similarity in
    `_match_fingerprint`, not by hash equality — an exact hash would miss the
    exact case we care about, where a user reformulates the same question
    four different ways and quadruples the bill.

    A production build swaps this for embedding cosine; the accumulator
    already accepts an embedding via `TurnRecord.prompt_embedding_id`. The
    lexical version is here so the prototype runs with zero model downloads.
    """
    toks = content_tokens(text)
    return hashlib.sha1(" ".join(sorted(toks)).encode()).hexdigest()[:16]


class SessionState:
    """Per-conversation risk memory."""

    def __init__(self, session_id: str, budget_usd: float = 0.50) -> None:
        self.session_id = session_id
        self.created_at = time.time()
        self.last_seen = time.time()
        self.budget_usd = budget_usd

        self.turns: list[TurnRecord] = []
        # claim_id -> claim, for claims that were never grounded
        self.ungrounded_ledger: dict[str, Claim] = {}
        # canonical fingerprint -> (token set, count)
        self.fingerprints: dict[str, tuple[frozenset[str], int]] = {}
        self.total_cost_usd = 0.0

    def _match_fingerprint(self, toks: frozenset[str]) -> str | None:
        """Find an existing question this one is a reformulation of."""
        best, best_sim = None, 0.0
        for fp, (known, _) in self.fingerprints.items():
            union = toks | known
            if not union:
                continue
            sim = len(toks & known) / len(union)
            if sim > best_sim:
                best, best_sim = fp, sim
        return best if best_sim >= RETRY_SIMILARITY else None

    # -- ingest ------------------------------------------------------------

    def record(
        self,
        ctx: RequestContext,
        risk: RiskVector,
        claims: list[Claim],
        boundary_proximity: float,
        cost_usd: float,
        action,
        prompt_text: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> TurnRecord:
        self.last_seen = time.time()
        toks = content_tokens(prompt_text)
        fp = self._match_fingerprint(toks) or semantic_fingerprint(prompt_text)
        known, count = self.fingerprints.get(fp, (toks, 0))
        self.fingerprints[fp] = (known, count + 1)
        self.total_cost_usd += cost_usd

        ungrounded = [c.id for c in claims if c.is_problem]
        for c in claims:
            if c.is_problem:
                self.ungrounded_ledger[c.id] = c

        rec = TurnRecord(
            turn=ctx.turn,
            risk=risk,
            action=action,
            boundary_proximity=boundary_proximity,
            cost_usd=cost_usd,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            prompt_fingerprint=fp,
            ungrounded_claim_ids=ungrounded,
        )
        self.turns.append(rec)
        return rec

    # -- trajectory signals ------------------------------------------------

    def escalation_gradient(self) -> float:
        """Detects a monotone walk toward a policy boundary.

        Each turn on its own may sit well under threshold. What matters is
        the SLOPE: a series that climbs 0.15 -> 0.28 -> 0.41 -> 0.55 is an
        attack in progress even though no single value trips a stateless
        gate.

        Score combines three things:
          slope      - least-squares gradient over recent boundary proximity
          monotonic  - fraction of steps that increased (accidents wobble,
                       attacks climb)
          level      - current proximity, so a flat-but-high session still
                       registers
        """
        pts = [(t.turn, t.boundary_proximity) for t in self.turns[-8:]]
        if len(pts) < 3:
            return 0.0

        n = len(pts)
        xs = [float(p[0]) for p in pts]
        ys = [p[1] for p in pts]
        mx = sum(xs) / n
        my = sum(ys) / n
        denom = sum((x - mx) ** 2 for x in xs)
        slope = 0.0 if denom == 0 else sum(
            (x - mx) * (y - my) for x, y in zip(xs, ys)
        ) / denom

        rises = sum(1 for a, b in zip(ys, ys[1:]) if b > a)
        monotonic = rises / max(1, len(ys) - 1)

        level = ys[-1]

        # Slope is normalised against "climbs 0.2 per turn" as a strong signal.
        slope_n = max(0.0, min(1.0, slope / 0.20))
        score = 0.45 * slope_n + 0.30 * monotonic + 0.25 * level
        return round(max(0.0, min(1.0, score)), 4)

    def contamination(self, current_claims: list[Claim]) -> float:
        """Detects reasoning built on earlier ungrounded claims.

        The failure: at turn 2 the model asserts an unverified fee waiver.
        Nobody blocks it — one soft claim, low stakes. By turn 6 the user is
        asking follow-ups that PRESUPPOSE the waiver, and the model is now
        computing amounts from it. The turn-6 response is internally
        consistent and fully "grounded" in the conversation. It is also
        wrong, and a stateless checker will pass it every time.

        We carry grounding status forward: a claim that reuses the language
        of an ungrounded ancestor inherits its contamination.
        """
        if not self.ungrounded_ledger:
            return 0.0

        prior_terms: set[str] = set()
        for c in self.ungrounded_ledger.values():
            prior_terms |= distinctive_terms(c.text)
        if not prior_terms:
            return 0.0

        hits = 0
        for c in current_claims:
            terms = distinctive_terms(c.text)
            if not terms:
                continue
            shared = terms & prior_terms
            overlap = len(shared) / len(terms)
            # Two guards against staying-on-topic false positives: the overlap
            # must clear the threshold AND rest on >=2 distinctive tokens, so a
            # single shared domain word can never brand a benign, grounded
            # follow-up as contaminated.
            if overlap >= 0.30 and len(shared) >= 2:
                hits += 1
                c.inherited_from = next(iter(self.ungrounded_ledger))

        if not current_claims:
            return 0.0

        share = hits / len(current_claims)
        # Depth matters: contamination compounds the longer it survives.
        depth = min(1.0, len(self.ungrounded_ledger) / 3.0)
        return round(min(1.0, 0.7 * share + 0.3 * depth), 4)

    def retry_burn(self) -> float:
        """Same question, re-asked. Cost multiplied, and usually a signal the
        model is failing at something."""
        if not self.fingerprints:
            return 0.0
        worst = max(c for _, c in self.fingerprints.values())
        if worst <= 1:
            return 0.0
        # 2 repeats -> 0.33, 3 -> 0.55, 4 -> 0.70, saturating.
        return round(min(1.0, 1.0 - math.exp(-0.4 * (worst - 1))), 4)

    def cost_creep(self) -> float:
        """Session total against session budget.

        Deliberately separate from per-request cost. The whole point is that
        no single invoice line looks wrong."""
        if self.budget_usd <= 0:
            return 0.0
        return round(min(1.0, self.total_cost_usd / self.budget_usd), 4)

    # -- output ------------------------------------------------------------

    def assess(self, current_claims: list[Claim]) -> SessionRisk:
        return SessionRisk(
            escalation_gradient=self.escalation_gradient(),
            contamination=self.contamination(current_claims),
            retry_burn=self.retry_burn(),
            cost_creep=self.cost_creep(),
        )

    def explain(self) -> list[str]:
        """Human-readable trajectory notes for the review queue."""
        out: list[str] = []
        eg = self.escalation_gradient()
        if eg > 0.35:
            seq = " -> ".join(f"{t.boundary_proximity:.2f}" for t in self.turns[-5:])
            out.append(f"boundary proximity climbing across turns: {seq}")
        if self.ungrounded_ledger:
            n = len(self.ungrounded_ledger)
            out.append(f"{n} ungrounded claim(s) still active in session context")
        worst_fp = (max(c for _, c in self.fingerprints.values())
                    if self.fingerprints else 0)
        if worst_fp > 1:
            out.append(f"same question re-asked {worst_fp}x (retry burn)")
        if self.cost_creep() > 0.6:
            out.append(
                f"session cost ${self.total_cost_usd:.4f} of "
                f"${self.budget_usd:.2f} budget"
            )
        return out


# --------------------------------------------------------------------------
# Backends
#
# The accumulator's trajectory logic lives entirely in SessionState. A backend
# only moves bytes, so swapping one for the other cannot change a single
# detection outcome — the property the old docstring promised but the old code
# never actually offered, because there was only ever one backend.
# --------------------------------------------------------------------------


class _InMemoryBackend:
    """Process-local dict.

    Fine for the prototype and a single worker, but the accumulator resets on
    restart and does NOT span workers: two uvicorn workers behind a load
    balancer keep separate copies, so a session's risk trajectory fragments
    across whichever worker happened to serve each turn. That silent failure
    is the entire reason the Redis backend exists.
    """

    kind = "memory"

    def __init__(self) -> None:
        self._d: dict[str, SessionState] = {}

    def load(self, sid: str) -> SessionState | None:
        self._evict()
        return self._d.get(sid)

    def save(self, st: SessionState) -> None:
        self._d[st.session_id] = st

    def delete(self, sid: str) -> None:
        self._d.pop(sid, None)

    def all(self) -> list[SessionState]:
        self._evict()
        return list(self._d.values())

    def _evict(self) -> None:
        now = time.time()
        for k in [k for k, v in self._d.items()
                  if now - v.last_seen > SESSION_TTL_S]:
            del self._d[k]


class _RedisBackend:
    """Redis-backed store — survives restarts, shared across workers.

    State is pickled: the accumulator holds enums, frozensets and pydantic
    models, all of which pickle natively, and the store is trusted (this
    service is the only writer). Redis enforces the TTL itself, so eviction is
    free and consistent across every worker instead of each worker running its
    own eviction clock.
    """

    kind = "redis"
    _PREFIX = "preflight:session:"

    def __init__(self, client) -> None:
        self._r = client

    def _key(self, sid: str) -> str:
        return f"{self._PREFIX}{sid}"

    def load(self, sid: str) -> SessionState | None:
        raw = self._r.get(self._key(sid))
        return pickle.loads(raw) if raw else None

    def save(self, st: SessionState) -> None:
        self._r.set(self._key(st.session_id), pickle.dumps(st), ex=SESSION_TTL_S)

    def delete(self, sid: str) -> None:
        self._r.delete(self._key(sid))

    def all(self) -> list[SessionState]:
        out: list[SessionState] = []
        for k in self._r.scan_iter(f"{self._PREFIX}*"):
            raw = self._r.get(k)
            if raw:
                out.append(pickle.loads(raw))
        return out


def _make_backend():
    """Redis if PREFLIGHT_REDIS_URL is set and reachable, else in-memory.

    A configured-but-unreachable Redis degrades to in-memory with a loud
    warning rather than crashing or pretending — the same explicit-degradation
    contract the detectors follow. Silently losing session state would be
    worse than saying so.
    """
    url = os.getenv("PREFLIGHT_REDIS_URL")
    if not url:
        return _InMemoryBackend()
    try:
        import redis  # optional dependency; only imported when configured
        client = redis.Redis.from_url(url, socket_connect_timeout=1)
        client.ping()
        return _RedisBackend(client)
    except Exception as exc:  # ImportError or connection failure
        print(
            f"[preflight] PREFLIGHT_REDIS_URL set but Redis is unavailable "
            f"({exc!r}); falling back to in-memory session store. Session risk "
            f"will NOT survive restart or span workers.",
            file=sys.stderr,
        )
        return _InMemoryBackend()


class SessionStore:
    """Session store with a pluggable backend.

    In-memory by default so the prototype runs with nothing installed; Redis
    when PREFLIGHT_REDIS_URL is set, so the accumulator survives restarts and
    is shared across workers. Because the backend only persists bytes, the
    detection semantics are identical either way.

    Note the read/mutate/`save` cycle: callers get() a state, mutate it via
    record(), then must call save() so the mutation is persisted. For the
    in-memory backend get() returns the live object and save() is idempotent;
    for Redis, save() is what actually writes the turn back. Making save()
    explicit keeps both backends honest instead of relying on object identity.
    """

    def __init__(self, backend=None) -> None:
        self._backend = backend if backend is not None else _make_backend()

    @property
    def kind(self) -> str:
        return getattr(self._backend, "kind", "custom")

    def get(self, session_id: str, budget_usd: float = 0.50) -> SessionState:
        st = self._backend.load(session_id)
        if st is None:
            st = SessionState(session_id, budget_usd)
            self._backend.save(st)
        return st

    def save(self, state: SessionState) -> None:
        state.last_seen = time.time()
        self._backend.save(state)

    def reset(self, session_id: str) -> None:
        self._backend.delete(session_id)

    def all(self) -> list[SessionState]:
        return sorted(
            self._backend.all(), key=lambda s: s.last_seen, reverse=True
        )
