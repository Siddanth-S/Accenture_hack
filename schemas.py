"""Core data model for Preflight.

Design note: risk is a VECTOR, never a scalar. Collapsing six independent
failure modes into one number destroys the information a policy needs to
choose between redacting a span and escalating to a human.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Risk dimensions
# --------------------------------------------------------------------------

RiskDimension = Literal[
    "groundedness",   # claims unsupported by retrieved evidence
    "uncertainty",    # model disagrees with itself / high semantic entropy
    "bias",           # answer moves under protected-attribute swap
    "privacy",        # PII present, or system-prompt / canary leakage
    "injection",      # prompt injection detected in input or retrieved context
    "cost",           # budget overrun, oversized model, retry burn
]

DIMENSIONS: tuple[str, ...] = (
    "groundedness",
    "uncertainty",
    "bias",
    "privacy",
    "injection",
    "cost",
)


class Stakes(str, Enum):
    """Stakes tier decides the bar, not the model.

    A marketing draft and a loan decision run the same detectors and get
    different thresholds. This is the governance knob.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Action(str, Enum):
    """The action ladder. Ordered by severity; `worst()` relies on order."""

    PASS = "pass"
    PASS_WITH_CITATIONS = "pass_with_citations"
    REDACT = "redact"
    REGENERATE = "regenerate"
    ESCALATE = "escalate"
    BLOCK = "block"

    @property
    def severity(self) -> int:
        return _ACTION_ORDER[self]

    @classmethod
    def worst(cls, actions: list["Action"]) -> "Action":
        if not actions:
            return cls.PASS
        return max(actions, key=lambda a: a.severity)


_ACTION_ORDER = {
    Action.PASS: 0,
    Action.PASS_WITH_CITATIONS: 1,
    Action.REDACT: 2,
    Action.REGENERATE: 3,
    Action.ESCALATE: 4,
    Action.BLOCK: 5,
}


class ClaimVerdict(str, Enum):
    """`UNVERIFIABLE` is a first-class verdict, not a failure.

    The brief notes there is often no real-time ground truth. Pretending
    otherwise is how checkers manufacture false confidence. Preflight says
    "I cannot check this" and lets policy decide what that means.
    """

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


# --------------------------------------------------------------------------
# Claims
# --------------------------------------------------------------------------


class Claim(BaseModel):
    """An atomic assertion extracted from a response.

    Character offsets are what make surgical action possible: we can redact
    or regenerate one clause instead of blocking a whole answer.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    text: str
    start: int
    end: int
    verdict: ClaimVerdict = ClaimVerdict.UNVERIFIABLE
    confidence: float = 0.0
    source_id: str | None = None
    source_span: tuple[int, int] | None = None
    entailment_score: float | None = None

    # Set when this claim rests on an ungrounded claim from an earlier turn.
    inherited_from: str | None = None

    @property
    def is_problem(self) -> bool:
        return self.verdict in (ClaimVerdict.UNSUPPORTED, ClaimVerdict.CONTRADICTED)


class Finding(BaseModel):
    """A single detector observation. Findings compose into the risk vector."""

    detector: str
    dimension: RiskDimension
    score: float  # 0.0 clean -> 1.0 maximal risk
    detail: str = ""
    span: tuple[int, int] | None = None
    latency_ms: float = 0.0
    # False when the detector could not run (capability missing, budget blown).
    ran: bool = True


# --------------------------------------------------------------------------
# Risk vector
# --------------------------------------------------------------------------


class RiskVector(BaseModel):
    """Six independent axes. Deliberately never summed."""

    groundedness: float = 0.0
    uncertainty: float = 0.0
    bias: float = 0.0
    privacy: float = 0.0
    injection: float = 0.0
    cost: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {d: getattr(self, d) for d in DIMENSIONS}

    def merge_max(self, other: "RiskVector") -> "RiskVector":
        """Combine by worst-case per axis. Two moderate risks on different
        axes must not average into one comfortable number."""
        return RiskVector(
            **{d: max(getattr(self, d), getattr(other, d)) for d in DIMENSIONS}
        )

    @classmethod
    def from_findings(cls, findings: list[Finding]) -> "RiskVector":
        v = cls()
        for f in findings:
            if not f.ran:
                continue
            setattr(v, f.dimension, max(getattr(v, f.dimension), f.score))
        return v


# --------------------------------------------------------------------------
# Session state — the differentiator
# --------------------------------------------------------------------------


class TurnRecord(BaseModel):
    """One turn's contribution to a session's risk history."""

    turn: int
    ts: float = Field(default_factory=time.time)
    risk: RiskVector
    action: Action
    prompt_embedding_id: str | None = None
    boundary_proximity: float = 0.0   # 0..1, closeness to known-unsafe region
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    prompt_fingerprint: str = ""      # semantic hash, for retry detection
    ungrounded_claim_ids: list[str] = Field(default_factory=list)


class SessionRisk(BaseModel):
    """Accumulated risk that does not exist inside any single response.

    Four trajectory signals:
      escalation_gradient  - monotone approach to a policy boundary
      contamination        - reasoning built on earlier ungrounded claims
      retry_burn           - same question re-asked, cost multiplied
      cost_creep           - session total vs budget, invisible per-response
    """

    escalation_gradient: float = 0.0
    contamination: float = 0.0
    retry_burn: float = 0.0
    cost_creep: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()

    @property
    def peak(self) -> float:
        return max(self.model_dump().values())


# --------------------------------------------------------------------------
# Request / decision envelope
# --------------------------------------------------------------------------


class RequestContext(BaseModel):
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str
    tenant: str = "default"
    use_case: str = "default"
    geo: str = "IN"
    stakes: Stakes = Stakes.MEDIUM
    turn: int = 1
    requested_model: str = ""
    routed_model: str = ""
    budget_usd: float = 0.05
    sources: list[dict[str, Any]] = Field(default_factory=list)
    stateless_mode: bool = False  # demo toggle: disable session accumulator


class Decision(BaseModel):
    """The full record of one oversight decision. This is what gets sealed
    into the ledger and what the console renders."""

    request_id: str
    session_id: str
    turn: int
    use_case: str
    stakes: Stakes

    action: Action
    reason: str
    triggered_by: list[str] = Field(default_factory=list)

    risk: RiskVector
    session_risk: SessionRisk
    thresholds: dict[str, float] = Field(default_factory=dict)

    claims: list[Claim] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)

    requested_model: str = ""
    routed_model: str = ""
    cost_usd: float = 0.0
    cost_avoided_usd: float = 0.0

    latency_ms: float = 0.0
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    degraded: list[str] = Field(default_factory=list)

    response_preview: str = ""
    ts: float = Field(default_factory=time.time)

    @property
    def expected_harm(self) -> float:
        """Ranks the human review queue. Probability x severity x
        irreversibility. A wrong answer that drives an irreversible action
        outranks a wrong answer nobody acts on."""
        p = max(self.risk.as_dict().values())
        sev = {Stakes.LOW: 0.2, Stakes.MEDIUM: 0.5,
               Stakes.HIGH: 0.8, Stakes.CRITICAL: 1.0}[self.stakes]
        return round(p * sev * (1.0 + self.session_risk.peak), 4)
