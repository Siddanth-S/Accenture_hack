"""Session accumulator tests — the differentiator.

These pin the four trajectory signals that no stateless checker can see, plus
the new semantic retry path: paraphrased re-asks that share no words but mean
the same thing.
"""
from __future__ import annotations

import pytest

from preflight import session
from preflight.schemas import Action, Claim, ClaimVerdict, RequestContext, RiskVector
from preflight.session import SessionState


@pytest.fixture(autouse=True)
def _reset_embedder():
    """The embedder is a module-global; make sure no test leaks it into another."""
    session.set_embedder(None)
    yield
    session.set_embedder(None)


def _record(st: SessionState, prompt: str, *, boundary: float = 0.1,
            cost: float = 0.001, claims=None, turn: int = 1):
    ctx = RequestContext(session_id=st.session_id, turn=turn)
    st.record(
        ctx=ctx, risk=RiskVector(), claims=claims or [],
        boundary_proximity=boundary, cost_usd=cost, action=Action.PASS,
        prompt_text=prompt,
    )


def _ungrounded_claim(text: str) -> Claim:
    return Claim(text=text, start=0, end=len(text), verdict=ClaimVerdict.UNSUPPORTED)


# -- escalation gradient ---------------------------------------------------

def test_escalation_gradient_detects_climb():
    st = SessionState("climb")
    for i, b in enumerate([0.15, 0.28, 0.41, 0.55]):
        _record(st, f"probe number {i}", boundary=b, turn=i + 1)
    flat = SessionState("flat")
    for i in range(4):
        _record(flat, f"probe number {i}", boundary=0.12, turn=i + 1)
    assert st.escalation_gradient() > 0.35
    assert st.escalation_gradient() > flat.escalation_gradient()


def test_escalation_gradient_needs_three_points():
    st = SessionState("short")
    _record(st, "one", boundary=0.9, turn=1)
    _record(st, "two", boundary=0.9, turn=2)
    assert st.escalation_gradient() == 0.0


# -- contamination ---------------------------------------------------------

def test_contamination_carries_ungrounded_forward():
    st = SessionState("contam")
    prior = _ungrounded_claim("the fee waiver of 500 rupees applies automatically")
    _record(st, "does the waiver apply", claims=[prior], turn=1)

    # A later claim that reasons FROM the fabricated waiver: shares the
    # distinctive terms (waiver, 500) even though it is a new, on-topic sentence.
    current = Claim(
        text="the waiver of 500 also removes the penalty entirely",
        start=0, end=10, verdict=ClaimVerdict.UNVERIFIABLE,
    )
    risk = st.contamination([current])
    assert risk > 0
    assert current.inherited_from is not None


def test_contamination_ignores_generic_followup():
    st = SessionState("benign")
    prior = _ungrounded_claim("the fee waiver of 500 rupees applies automatically")
    _record(st, "does the waiver apply", claims=[prior], turn=1)
    # Generic on-topic follow-up sharing only stopword-ish domain vocab must not
    # be branded contaminated. A small residual floor remains from `depth`
    # (an ungrounded claim is still live in the ledger), but the follow-up must
    # register NO hit — inherited_from stays None — and the score stays well
    # under the 0.25 escalate threshold, so it never triggers on its own.
    benign = Claim(text="please provide the customer service details",
                   start=0, end=10, verdict=ClaimVerdict.UNVERIFIABLE)
    score = st.contamination([benign])
    assert benign.inherited_from is None
    assert score < 0.25


# -- retry burn (lexical) --------------------------------------------------

def test_retry_burn_lexical_same_wording():
    st = SessionState("retry")
    _record(st, "what is the late fee on a missed emi payment", turn=1)
    assert st.retry_burn() == 0.0  # first ask, no burn
    _record(st, "what is the late fee on a missed emi payment", turn=2)
    assert st.retry_burn() > 0.0   # re-asked -> burn


def test_retry_burn_lexical_misses_paraphrase():
    """Documents the limitation the embedder fixes: no shared words, no lexical
    match, so two fingerprints and zero burn."""
    st = SessionState("para-lex")
    _record(st, "what is the late fee on a missed emi", turn=1)
    _record(st, "how much extra do they charge if an instalment is unpaid", turn=2)
    assert len(st.fingerprints) == 2
    assert st.retry_burn() == 0.0


# -- retry burn (semantic) -------------------------------------------------

class _FakeEmbedder:
    """Maps any prompt containing a keyword to a fixed vector, so we can force
    two word-disjoint prompts to be 'the same meaning' deterministically —
    without downloading a real model."""

    name = "fake"

    def __init__(self, mapping: dict[str, list[float]]):
        self._mapping = mapping

    def encode(self, text: str) -> list[float]:
        for key, vec in self._mapping.items():
            if key in text:
                return vec
        return [0.0, 0.0, 1.0]


def test_retry_burn_semantic_catches_paraphrase():
    # 'fee' and 'charge' map to the SAME vector: same meaning, no shared words.
    session.set_embedder(_FakeEmbedder({"fee": [1.0, 0.0, 0.0],
                                        "charge": [1.0, 0.0, 0.0]}))
    st = SessionState("para-sem")
    _record(st, "what is the late fee on a missed emi", turn=1)
    _record(st, "how much extra do they charge if an instalment is unpaid", turn=2)
    # One fingerprint because the vectors matched, so the re-ask registers.
    assert len(st.fingerprints) == 1
    assert st.retry_burn() > 0.0


def test_semantic_falls_back_to_lexical_when_vectors_differ():
    # Unrelated prompts get distinct vectors -> no semantic match -> two asks.
    session.set_embedder(_FakeEmbedder({"loan": [1.0, 0.0, 0.0],
                                        "weather": [0.0, 1.0, 0.0]}))
    st = SessionState("distinct")
    _record(st, "am I approved for the loan", turn=1)
    _record(st, "what is the weather tomorrow", turn=2)
    assert len(st.fingerprints) == 2


def test_broken_embedder_degrades_not_crashes():
    class Boom:
        name = "boom"
        def encode(self, text):
            raise RuntimeError("model exploded")

    session.set_embedder(Boom())
    st = SessionState("boom")
    # Must not raise; falls back to lexical, so an identical re-ask still counts.
    _record(st, "same question here", turn=1)
    _record(st, "same question here", turn=2)
    assert st.retry_burn() > 0.0


# -- cost creep ------------------------------------------------------------

def test_cost_creep_against_budget():
    st = SessionState("cost", budget_usd=0.50)
    _record(st, "expensive question", cost=0.30, turn=1)
    assert st.cost_creep() == pytest.approx(0.6, abs=1e-3)


# -- assess bundles all four ----------------------------------------------

def test_assess_returns_all_signals():
    st = SessionState("assess")
    _record(st, "probe", boundary=0.4, cost=0.1, turn=1)
    sr = st.assess([])
    assert set(sr.as_dict()) == {
        "escalation_gradient", "contamination", "retry_burn", "cost_creep"
    }
