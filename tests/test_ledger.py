"""Tamper-evidence tests.

The ledger's whole promise is that editing a historical decision is *detectable*
— not merely discouraged. These tests hold that promise to account: an intact
chain verifies, and a single altered row makes verify_chain() name the exact
sequence it was altered at. This is the test behind the demo button.
"""
from __future__ import annotations

from preflight.ledger import Ledger
from preflight.schemas import (
    Action, Decision, RiskVector, SessionRisk, Stakes,
)


def _decision(seq: int, action: Action = Action.PASS) -> Decision:
    return Decision(
        request_id=f"req-{seq}",
        session_id="sess-1",
        turn=seq,
        use_case="default",
        stakes=Stakes.MEDIUM,
        action=action,
        reason=f"turn {seq} evaluated",
        risk=RiskVector(),
        session_risk=SessionRisk(),
        cost_avoided_usd=0.01,
    )


def _seed(ledger: Ledger, n: int = 5) -> None:
    for i in range(1, n + 1):
        ledger.append(_decision(i), policy_version="v1")


def test_intact_chain_verifies(tmp_path):
    led = Ledger(tmp_path / "led.db")
    _seed(led, 5)
    status = led.verify_chain()
    assert status.intact
    assert status.records == 5
    assert status.broken_at is None


def test_tamper_is_detected_at_exact_seq(tmp_path):
    led = Ledger(tmp_path / "led.db")
    _seed(led, 5)
    assert led.verify_chain().intact  # clean before

    result = led.tamper(3)
    assert result["tampered"] is True
    assert result["seq"] == 3

    status = led.verify_chain()
    assert not status.intact
    assert status.broken_at == 3


def test_tamper_defaults_to_first_record(tmp_path):
    led = Ledger(tmp_path / "led.db")
    _seed(led, 3)
    result = led.tamper()  # no seq -> oldest record
    assert result["tampered"] is True
    assert result["seq"] == 1
    assert not led.verify_chain().intact


def test_tamper_on_empty_ledger_is_noop(tmp_path):
    led = Ledger(tmp_path / "led.db")
    result = led.tamper()
    assert result["tampered"] is False
    assert led.verify_chain().intact


def test_stats_reports_cost_avoided(tmp_path):
    led = Ledger(tmp_path / "led.db")
    _seed(led, 4)  # each seeded decision avoided $0.01
    stats = led.stats()
    assert stats["records"] == 4
    assert round(stats["total_cost_avoided_usd"], 2) == 0.04
