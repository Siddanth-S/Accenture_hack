"""Detectors.

Every detector reports whether it actually RAN. That flag is not bookkeeping —
it is the honest answer to a problem the brief raises directly: enterprises
consume models via API, so a checker cannot assume it can see model internals.
Logprobs may be absent. A GPU may not exist. A latency budget may blow before
the expensive check finishes.

A checker that silently skips a detector and returns a clean score is worse
than no checker, because it manufactures false assurance. Preflight instead
degrades explicitly: the finding is marked `ran=False`, the decision records
which checks were skipped, and high-stakes tiers fail closed rather than
passing an unchecked response.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

from ..schemas import Finding

# --------------------------------------------------------------------------
# Provider capability negotiation
# --------------------------------------------------------------------------


@dataclass
class ProviderCapabilities:
    """What can we actually inspect on this provider?

    Anthropic exposes no logprobs. OpenAI exposes top-k only. A self-hosted
    model exposes everything. The detector set adapts rather than pretending.
    """

    logprobs: bool = False
    top_k_logprobs: int = 0
    streaming: bool = True
    system_prompt: bool = True
    n_samples: bool = True  # can we cheaply sample N generations?

    @classmethod
    def for_model(cls, model: str) -> "ProviderCapabilities":
        m = model.lower()
        if m.startswith("claude"):
            return cls(logprobs=False, streaming=True, n_samples=True)
        if m.startswith(("gpt-", "o1", "o3")):
            return cls(logprobs=True, top_k_logprobs=5, streaming=True,
                       n_samples=True)
        if m.startswith(("qwen", "llama", "mistral", "local/")):
            return cls(logprobs=True, top_k_logprobs=20, streaming=True,
                       n_samples=True)
        return cls()


# --------------------------------------------------------------------------
# Privacy — deterministic, therefore allowed to block
# --------------------------------------------------------------------------

# Indian identifiers alongside the usual set. Presidio covers the generic
# ones; these are the local formats it misses.
_PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
    "phone_in": re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "ifsc": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    "aadhaar": re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "ip": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

_VERHOEFF_D = [
    [0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],
    [3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],
    [9,8,7,6,5,4,3,2,1,0],
]
_VERHOEFF_P = [
    [0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],
    [8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8],
]


def _verhoeff_valid(number: str) -> bool:
    """Aadhaar checksum. Without it, any 12-digit string is a false positive —
    and phone numbers, order IDs and timestamps are everywhere in enterprise
    text. Checksum validation is the difference between a usable PII detector
    and one the ops team mutes in week one."""
    c = 0
    for i, digit in enumerate(reversed(number.replace(" ", ""))):
        if not digit.isdigit():
            return False
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(digit)]]
    return c == 0


def _luhn_valid(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"[\s-]", "", number) if d.isdigit()]
    if len(digits) < 13:
        return False
    total, parity = 0, len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def detect_pii(text: str) -> tuple[Finding, list[tuple[int, int, str]]]:
    t0 = time.perf_counter()
    spans: list[tuple[int, int, str]] = []

    for kind, pat in _PII_PATTERNS.items():
        for m in pat.finditer(text):
            val = m.group()
            if kind == "aadhaar" and not _verhoeff_valid(val):
                continue
            if kind == "credit_card" and not _luhn_valid(val):
                continue
            spans.append((m.start(), m.end(), kind))

    severity = {
        "aadhaar": 1.0, "credit_card": 1.0, "ssn": 1.0, "pan": 0.9,
        "ifsc": 0.7, "phone_in": 0.6, "email": 0.5, "ip": 0.3,
    }
    score = max((severity.get(k, 0.5) for _, _, k in spans), default=0.0)
    kinds = sorted({k for _, _, k in spans})

    return Finding(
        detector="pii.deterministic",
        dimension="privacy",
        score=score,
        detail=f"{len(spans)} match(es): {', '.join(kinds)}" if spans else "clean",
        latency_ms=round((time.perf_counter() - t0) * 1000, 3),
    ), spans


CANARY = "PF-CANARY-7F3A21"


def detect_canary(text: str) -> Finding:
    """System-prompt leakage. We inject a canary token into the system prompt;
    if it appears in output, the model has been induced to reveal its
    instructions. Deterministic, so it is allowed to hard-block."""
    t0 = time.perf_counter()
    leaked = CANARY in text
    return Finding(
        detector="canary.deterministic",
        dimension="privacy",
        score=1.0 if leaked else 0.0,
        detail="system prompt leaked" if leaked else "clean",
        latency_ms=round((time.perf_counter() - t0) * 1000, 3),
    )


# --------------------------------------------------------------------------
# Injection
# --------------------------------------------------------------------------

_INJECTION_SIGNATURES = [
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", 0.95),
    (r"disregard\s+(your|all|the)\s+(instructions|rules|guidelines)", 0.95),
    (r"you\s+are\s+now\s+(a|an|in)\s+", 0.70),
    (r"\b(dan|developer)\s+mode\b", 0.85),
    (r"reveal\s+(your|the)\s+(system\s+)?prompt", 0.90),
    (r"repeat\s+(the\s+)?(text|words)\s+above", 0.80),
    (r"pretend\s+(you|to\s+be)", 0.55),
    (r"for\s+(research|educational|academic)\s+purposes\s+only", 0.40),
    (r"my\s+grandmother\s+used\s+to", 0.60),
    (r"this\s+is\s+(just\s+)?(a\s+)?(hypothetical|fictional|roleplay)", 0.45),
    (r"\bjailbreak\b", 0.75),
    (r"without\s+any\s+(restrictions|filters|limitations)", 0.80),
]

_COMPILED_INJ = [(re.compile(p, re.IGNORECASE), s) for p, s in _INJECTION_SIGNATURES]


def detect_injection(text: str, from_retrieved: bool = False) -> Finding:
    """Injection detection.

    `from_retrieved` matters enormously and is usually ignored. An injection
    typed by a user is an attack on their own session. The same string
    arriving inside a RETRIEVED document is an attack on every user who ever
    retrieves it, and it persists in the index. We weight it higher for that
    reason.
    """
    t0 = time.perf_counter()
    hits: list[str] = []
    score = 0.0

    for pat, sev in _COMPILED_INJ:
        if pat.search(text):
            hits.append(pat.pattern[:40])
            score = max(score, sev)

    if from_retrieved and score > 0:
        score = min(1.0, score * 1.25)

    return Finding(
        detector="injection.signature",
        dimension="injection",
        score=round(score, 4),
        detail=(
            f"{len(hits)} signature(s)"
            + (" in RETRIEVED context" if from_retrieved else "")
        ) if hits else "clean",
        latency_ms=round((time.perf_counter() - t0) * 1000, 3),
    )


def boundary_proximity(text: str) -> float:
    """How close is this prompt to a policy boundary?

    Fed to the session accumulator, where the TREND is what matters. A single
    value of 0.3 is unremarkable; 0.15 -> 0.28 -> 0.41 -> 0.55 across four
    turns is an attack being assembled, and no single one of those values
    would trip a stateless gate.
    """
    inj = detect_injection(text).score
    sensitive = [
        r"\b(bypass|circumvent|evade|avoid)\b",
        r"\b(exploit|vulnerabilit|weakness)\b",
        r"\b(confidential|internal|proprietary|classified)\b",
        r"\b(hypothetically|theoretically|in theory)\b",
        r"\b(no\s+one\s+would\s+know|between\s+us|off\s+the\s+record)\b",
        r"\b(edge\s+case|loophole|technicality|grey\s+area|gray\s+area)\b",
    ]
    soft = sum(1 for p in sensitive if re.search(p, text, re.IGNORECASE))
    soft_score = min(1.0, soft / 3.0)
    return round(min(1.0, 0.6 * inj + 0.4 * soft_score), 4)


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------


def token_entropy_score(logprobs: list[float] | None) -> Finding:
    """Uncertainty from the model's own token distribution — when available."""
    t0 = time.perf_counter()
    if not logprobs:
        return Finding(
            detector="entropy.token", dimension="uncertainty", score=0.0,
            detail="skipped: provider exposes no logprobs", ran=False,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )
    probs = [math.exp(lp) for lp in logprobs]
    ent = -sum(p * math.log(p + 1e-12) for p in probs) / max(1, len(probs))
    return Finding(
        detector="entropy.token", dimension="uncertainty",
        score=round(min(1.0, ent / 2.0), 4),
        detail=f"mean token entropy {ent:.3f} over {len(probs)} tokens",
        latency_ms=round((time.perf_counter() - t0) * 1000, 3),
    )


def semantic_entropy_score(
    samples: list[str] | None, nli=None
) -> Finding:
    """Semantic entropy — uncertainty when there is nothing to check against.

    Sample N generations, cluster them by bidirectional entailment (do they
    MEAN the same thing, ignoring wording), then take entropy over meaning
    clusters rather than token sequences.

    This is the principled answer to the brief's hardest constraint: no
    real-time ground truth. We stop asking "is this true" and ask "does the
    model agree with itself about it". Low token entropy with high semantic
    entropy is precisely confidently wrong.

    Expensive — N generations — so it lives in the tail tier and runs only
    where stakes justify it.

    Reference: Farquhar et al., "Detecting hallucinations in large language
    models using semantic entropy", Nature 630 (2024).
    """
    t0 = time.perf_counter()
    if not samples or len(samples) < 2:
        return Finding(
            detector="entropy.semantic", dimension="uncertainty", score=0.0,
            detail="skipped: sampling not run at this tier", ran=False,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )

    clusters: list[list[str]] = []
    for s in samples:
        placed = False
        for cl in clusters:
            if _same_meaning(s, cl[0], nli):
                cl.append(s)
                placed = True
                break
        if not placed:
            clusters.append([s])

    n = len(samples)
    ent = -sum((len(c) / n) * math.log(len(c) / n) for c in clusters)
    max_ent = math.log(n) if n > 1 else 1.0
    norm = ent / max_ent if max_ent > 0 else 0.0

    return Finding(
        detector="entropy.semantic", dimension="uncertainty",
        score=round(min(1.0, norm), 4),
        detail=f"{len(clusters)} meaning cluster(s) across {n} samples",
        latency_ms=round((time.perf_counter() - t0) * 1000, 3),
    )


def _same_meaning(a: str, b: str, nli=None) -> bool:
    if nli is not None:
        return (nli.entailment(a, b) > 0.6) and (nli.entailment(b, a) > 0.6)
    ta = {w for w in re.findall(r"[a-z]{4,}", a.lower())}
    tb = {w for w in re.findall(r"[a-z]{4,}", b.lower())}
    if not ta or not tb:
        return a.strip() == b.strip()
    return len(ta & tb) / len(ta | tb) > 0.6


# --------------------------------------------------------------------------
# Bias
# --------------------------------------------------------------------------

_PROTECTED_SWAPS = [
    ("he", "she"), ("his", "her"), ("man", "woman"), ("male", "female"),
    ("Rajesh", "Fatima"), ("Priya", "Aisha"), ("John", "Jamal"),
    ("Hindu", "Muslim"), ("young", "elderly"),
]


def counterfactual_pairs(prompt: str) -> list[tuple[str, str]]:
    """Build protected-attribute swaps for the counterfactual probe.

    The probe: run the swapped prompt, measure how far the answer moves. A
    loan assistant whose recommendation changes when only the applicant's
    name changes is biased, and no amount of reading the single original
    response would reveal it. This requires a second generation, which is why
    it is sampled rather than run per-request — the cost is real and pretending
    otherwise would be the same overclaiming we are trying to prevent.
    """
    out = []
    for a, b in _PROTECTED_SWAPS:
        pat = re.compile(rf"\b{re.escape(a)}\b", re.IGNORECASE)
        if pat.search(prompt):
            out.append((a, pat.sub(b, prompt)))
    return out


def bias_divergence(original: str, counterfactual: str, nli=None) -> Finding:
    """How far did the answer move under a protected-attribute swap?"""
    t0 = time.perf_counter()
    if not counterfactual:
        return Finding(
            detector="bias.counterfactual", dimension="bias", score=0.0,
            detail="skipped: no protected attribute present in prompt",
            ran=False,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )
    same = _same_meaning(original, counterfactual, nli)
    ta = {w for w in re.findall(r"[a-z]{4,}", original.lower())}
    tb = {w for w in re.findall(r"[a-z]{4,}", counterfactual.lower())}
    jac = len(ta & tb) / len(ta | tb) if (ta | tb) else 1.0
    score = 0.0 if same else round(min(1.0, 1.0 - jac), 4)
    return Finding(
        detector="bias.counterfactual", dimension="bias", score=score,
        detail=f"divergence under attribute swap (overlap {jac:.2f})",
        latency_ms=round((time.perf_counter() - t0) * 1000, 3),
    )
