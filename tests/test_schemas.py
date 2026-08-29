"""Schema/risk-algebra tests.

The core invariant the project rests on: risk is a vector, combined by worst-case
per axis, never summed into one comfortable number. And a detector that did not
run must not contribute a (falsely clean) 0.0.
"""
from __future__ import annotations

from preflight.schemas import (
    Action,
    Finding,
    RiskVector,
    SessionRisk,
    Stakes,
)


def test_action_worst_orders_by_severity():
    assert Action.worst([Action.PASS, Action.ESCALATE, Action.REDACT]) is Action.ESCALATE
    assert Action.worst([Action.PASS, Action.PASS_WITH_CITATIONS]) is Action.PASS_WITH_CITATIONS


def test_action_worst_empty_is_pass():
    assert Action.worst([]) is Action.PASS


def test_risk_vector_merge_max_per_axis():
    a = RiskVector(groundedness=0.3, bias=0.9)
    b = RiskVector(groundedness=0.7, injection=0.5)
    merged = a.merge_max(b)
    assert merged.groundedness == 0.7  # max, not sum
    assert merged.bias == 0.9
    assert merged.injection == 0.5


def test_risk_vector_from_findings_ignores_not_ran():
    findings = [
        Finding(detector="a", dimension="uncertainty", score=0.8, ran=False),
        Finding(detector="b", dimension="groundedness", score=0.4, ran=True),
    ]
    v = RiskVector.from_findings(findings)
    # The skipped detector must NOT leak its score in — that's the whole point
    # of the ran flag.
    assert v.uncertainty == 0.0
    assert v.groundedness == 0.4


def test_session_risk_peak():
    sr = SessionRisk(escalation_gradient=0.2, contamination=0.75, cost_creep=0.1)
    assert sr.peak == 0.75


def test_stakes_are_ordered_tiers():
    assert {s.value for s in Stakes} == {"low", "medium", "high", "critical"}
