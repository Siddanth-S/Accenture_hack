"""Claim-level verification.

Response-level scoring can only ever say yes or no. Claim-level scoring is
what unlocks the action ladder: if we know that claim 3 of 5 is unsupported
and exactly where it sits in the text, we can redact that clause or
regenerate it under constraint. Blocking the whole answer becomes the last
resort instead of the only tool.

Two deliberate design positions here:

1. UNVERIFIABLE is a verdict, not an error. The brief points out that the
   same knowledge gaps causing hallucination also make verification hard.
   A checker that forces every claim into supported/unsupported is
   manufacturing confidence it does not have.

2. Not every sentence is a claim. Hedged language ("this may vary",
   "you should confirm with...") is the model behaving correctly. Scoring it
   as unsupported punishes the exact behaviour we want and drives the false
   positive rate that causes users to bypass the system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .schemas import Claim, ClaimVerdict

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# Hedges signal the model is NOT asserting. These are good behaviour.
_HEDGES = (
    "may ", "might ", "could ", "typically", "generally", "usually",
    "i'm not sure", "i am not sure", "cannot confirm", "can't confirm",
    "please verify", "check with", "consult", "varies", "depends on",
    "i don't have", "i do not have", "unable to confirm", "approximately",
)

# Assertive markers raise the stakes of being wrong.
_ASSERTIVE = (
    "will ", "is ", "are ", "guarantees", "guaranteed", "always", "never",
    "must ", "definitely", "certainly", "confirmed", "entitled to",
    "you qualify", "waived", "no fee", "free of charge",
)

# Concrete specifics are what hallucinations attach to and what users act on.
_SPECIFIC = re.compile(
    r"(\d+(?:\.\d+)?\s*%|"          # percentages
    r"[₹$€£]\s?\d[\d,]*(?:\.\d+)?|" # money
    r"\b\d+\s*(?:days?|weeks?|months?|years?|hours?)\b|"
    r"\bsection\s+\d+|\bclause\s+\d+|\bpolicy\s+\w*\d+)",
    re.IGNORECASE,
)


@dataclass
class Source:
    id: str
    text: str


def extract_claims(response: str) -> list[Claim]:
    """Split a response into atomic, checkable assertions with offsets.

    Sentence segmentation with hedge filtering. A production build swaps this
    for a trained claim-decomposition model; the interface does not change,
    which is why this stays honest rather than a placeholder.
    """
    claims: list[Claim] = []
    cursor = 0

    for sent in _SENT_SPLIT.split(response.strip()):
        sent = sent.strip()
        if not sent:
            continue
        start = response.find(sent, cursor)
        if start < 0:
            start = cursor
        end = start + len(sent)
        cursor = end

        if len(sent) < 20:
            continue

        low = sent.lower()
        hedged = any(h in low for h in _HEDGES)
        assertive = any(a in low for a in _ASSERTIVE)
        specific = bool(_SPECIFIC.search(sent))

        # A hedged sentence with no hard specifics is not a claim we police.
        if hedged and not specific:
            continue
        if not (assertive or specific):
            continue

        claims.append(Claim(text=sent, start=start, end=end))

    return claims


def _token_set(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9%₹$]{3,}", text.lower())}


def verify_claim(
    claim: Claim, sources: list[Source], nli: Any | None = None
) -> Claim:
    """Check one claim against retrieved sources.

    With an NLI model loaded, this is entailment: P(source entails claim).
    Without one, we fall back to lexical + numeric overlap, which is weaker
    but honest about being weaker — the confidence it reports is capped so
    that a fallback verdict can never look as strong as a model verdict.

    The numeric check matters more than it looks. Hallucinations
    overwhelmingly attach to specifics — a rate, a fee, a deadline. A claim
    whose numbers appear in NO source is the single highest-yield signal
    available without a model.
    """
    if not sources:
        claim.verdict = ClaimVerdict.UNVERIFIABLE
        claim.confidence = 0.0
        return claim

    if nli is not None:
        # Take the best-entailing source, but track contradiction separately.
        # Low entailment alone does NOT mean contradiction — a source silent on
        # the claim also entails it weakly. Only an actively contradicting
        # source makes a claim CONTRADICTED; the rest is UNSUPPORTED/neutral.
        # A minimal NLI exposing only entailment() keeps the old behaviour.
        has_contra = hasattr(nli, "contradiction")
        best_id, best_entail, max_contra = None, 0.0, 0.0
        for s in sources:
            e = nli.entailment(premise=s.text, hypothesis=claim.text)
            if e > best_entail:
                best_id, best_entail = s.id, e
            if has_contra:
                max_contra = max(
                    max_contra, nli.contradiction(premise=s.text, hypothesis=claim.text)
                )
        claim.entailment_score = round(best_entail, 4)
        claim.source_id = best_id
        claim.confidence = round(best_entail, 4)
        if best_entail >= 0.70:
            claim.verdict = ClaimVerdict.SUPPORTED
        elif max_contra >= 0.55:
            claim.verdict = ClaimVerdict.CONTRADICTED
        elif not has_contra and best_entail <= 0.15:
            claim.verdict = ClaimVerdict.CONTRADICTED
        else:
            claim.verdict = ClaimVerdict.UNSUPPORTED
        return claim

    # ---- fallback: lexical + numeric grounding --------------------------
    claim_nums = set(_SPECIFIC.findall(claim.text))
    claim_toks = _token_set(claim.text)

    best_overlap, best_id = 0.0, None
    nums_found = False

    for s in sources:
        src_toks = _token_set(s.text)
        if claim_toks:
            overlap = len(claim_toks & src_toks) / len(claim_toks)
            if overlap > best_overlap:
                best_overlap, best_id = overlap, s.id
        if claim_nums:
            src_nums = set(_SPECIFIC.findall(s.text))
            norm = lambda xs: {re.sub(r"\s+", "", str(x)).lower() for x in xs}
            if norm(claim_nums) & norm(src_nums):
                nums_found = True

    claim.source_id = best_id
    # Cap at 0.65: a lexical match must never present as model-grade evidence.
    claim.confidence = round(min(0.65, best_overlap), 4)

    if claim_nums and not nums_found:
        # A specific figure appearing in NO source. Highest-yield signal
        # available without a model, and the one we are entitled to be
        # confident about.
        claim.verdict = ClaimVerdict.UNSUPPORTED
        claim.confidence = round(min(0.65, 0.35 + best_overlap * 0.3), 4)
    elif best_overlap >= 0.55:
        claim.verdict = ClaimVerdict.SUPPORTED
    else:
        # Everything else is UNVERIFIABLE, not UNSUPPORTED.
        #
        # This distinction is the difference between a usable checker and one
        # the ops team mutes in week one. Lexical overlap is weak evidence of
        # presence and almost no evidence of absence — plenty of correct
        # prose restates a source in different words. Calling that
        # "unsupported" manufactures exactly the alert fatigue the brief
        # warns about. With an NLI model loaded we CAN distinguish these,
        # which is precisely what the model buys us.
        claim.verdict = ClaimVerdict.UNVERIFIABLE

    return claim


def groundedness_score(claims: list[Claim]) -> float:
    """Aggregate claim verdicts into the groundedness risk axis.

    Weighted, not averaged: a contradicted claim is materially worse than an
    unverifiable one, and one bad claim in a long correct answer must not be
    diluted into invisibility by the length of the answer.
    """
    if not claims:
        return 0.0
    weights = {
        ClaimVerdict.SUPPORTED: 0.0,
        ClaimVerdict.UNVERIFIABLE: 0.25,
        ClaimVerdict.UNSUPPORTED: 0.75,
        ClaimVerdict.CONTRADICTED: 1.0,
    }
    scores = [weights[c.verdict] for c in claims]
    worst = max(scores)
    mean = sum(scores) / len(scores)
    # Worst-case dominates; mean stops a single soft flag reading as crisis.
    return round(min(1.0, 0.7 * worst + 0.3 * mean), 4)


def redact_spans(response: str, claims: list[Claim]) -> str:
    """Surgical redaction — remove offending spans, keep the rest."""
    bad = sorted(
        [c for c in claims if c.is_problem], key=lambda c: c.start, reverse=True
    )
    out = response
    for c in bad:
        out = out[:c.start] + "[removed: unverified claim]" + out[c.end:]
    return out
