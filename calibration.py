"""Conformal risk control for threshold selection.

The brief says over-flagging and under-flagging must be "deliberately tuned
rather than solved away". Most teams will tune by eye and ship a number they
cannot defend. This module replaces the guess with a guarantee.

The claim we can make at the end of it:

    "At this operating point, the probability that an unsafe response is
     passed is bounded at 5%, with 95% confidence, distribution-free."

Distribution-free means we assume nothing about the score distribution — only
that calibration data and production data are exchangeable. That assumption
is worth stating out loud, because it is also the failure mode: under
distribution shift the guarantee degrades, which is exactly why drift
monitoring feeds back into recalibration.

Reference: Angelopoulos & Bates, "A Gentle Introduction to Conformal
Prediction and Distribution-Free Uncertainty Quantification" (2021);
Bates et al., "Distribution-Free, Risk-Controlling Prediction Sets" (2021).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationResult:
    axis: str
    threshold: float
    target_risk: float          # alpha: max tolerated miss rate
    confidence: float           # 1 - delta
    empirical_miss_rate: float
    flag_rate: float            # fraction of traffic flagged (the cost)
    n_calibration: int
    valid: bool
    note: str = ""


def _hoeffding_upper_bound(empirical: float, n: int, delta: float) -> float:
    """Upper confidence bound on true risk from empirical risk.

    Hoeffding gives us: with probability >= 1 - delta,
        true_risk <= empirical_risk + sqrt(log(1/delta) / (2n))

    Conservative, but it holds for any distribution and needs no asymptotics,
    which is what makes the guarantee honest on a few hundred examples.
    """
    if n <= 0:
        return 1.0
    return empirical + math.sqrt(math.log(1.0 / delta) / (2.0 * n))


def calibrate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    axis: str = "groundedness",
    alpha: float = 0.05,
    delta: float = 0.05,
) -> CalibrationResult:
    """Find the LOWEST threshold whose risk bound still satisfies alpha.

    Args:
        scores: detector output, higher = riskier.
        labels: 1 = genuinely unsafe, 0 = safe.
        alpha:  maximum tolerated miss rate (unsafe items that pass).
        delta:  1 - confidence.

    We sweep candidate thresholds from strict to permissive and keep the most
    permissive one that still holds the bound. This matters: picking the
    strictest threshold that "works" would satisfy the guarantee while
    flagging half of all traffic, and alert fatigue is itself a failure mode
    the brief calls out explicitly.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(scores)

    if n == 0 or labels.sum() == 0:
        return CalibrationResult(
            axis=axis, threshold=0.5, target_risk=alpha, confidence=1 - delta,
            empirical_miss_rate=0.0, flag_rate=0.0, n_calibration=n,
            valid=False,
            note="insufficient calibration data (no positive examples)",
        )

    candidates = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    candidates.sort()

    best: CalibrationResult | None = None
    n_unsafe = int(labels.sum())

    for t in candidates:
        flagged = scores >= t
        # A miss = unsafe item that was NOT flagged.
        misses = int(((~flagged) & (labels == 1)).sum())
        emp = misses / n_unsafe
        bound = _hoeffding_upper_bound(emp, n_unsafe, delta)

        if bound <= alpha:
            cand = CalibrationResult(
                axis=axis,
                threshold=float(t),
                target_risk=alpha,
                confidence=1 - delta,
                empirical_miss_rate=round(emp, 4),
                flag_rate=round(float(flagged.mean()), 4),
                n_calibration=n,
                valid=True,
                note=(
                    f"risk bound {bound:.3f} <= alpha {alpha:.3f} "
                    f"(Hoeffding, n_unsafe={n_unsafe})"
                ),
            )
            # Sweep is ascending, so later valid candidates are more
            # permissive -> less alert fatigue for the same guarantee.
            best = cand

    if best is None:
        return CalibrationResult(
            axis=axis, threshold=0.0, target_risk=alpha, confidence=1 - delta,
            empirical_miss_rate=0.0, flag_rate=1.0, n_calibration=n,
            valid=False,
            note=(
                "no threshold satisfies the risk bound at this sample size — "
                "flag everything, or collect more calibration data"
            ),
        )
    return best


def risk_coverage_curve(
    scores: np.ndarray, labels: np.ndarray, steps: int = 40
) -> list[dict[str, float]]:
    """Points for the risk-coverage plot in the governance view.

    Coverage = fraction of traffic auto-handled (not flagged).
    Risk     = miss rate among that auto-handled traffic.

    This curve is the honest picture of the tradeoff the brief says must be
    tuned rather than solved. It shows a stakeholder exactly what they buy
    with each point of automation they give up.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    out: list[dict[str, float]] = []

    for t in np.linspace(0.0, 1.0, steps):
        passed = scores < t
        cov = float(passed.mean())
        if passed.sum() == 0:
            risk = 0.0
        else:
            risk = float(labels[passed].mean())
        out.append(
            {"threshold": round(float(t), 3),
             "coverage": round(cov, 4),
             "risk": round(risk, 4)}
        )
    return out


def expected_calibration_error(
    scores: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    """ECE — are the confidence numbers meaningful, or decorative?

    A detector that says 0.9 should be right about 90% of the time. Reporting
    ECE alongside AUC is a small thing that signals you know a score is not
    a probability until you check.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(scores) == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (scores >= lo) & (scores < hi)
        if m.sum() == 0:
            continue
        conf = scores[m].mean()
        acc = labels[m].mean()
        ece += (m.sum() / len(scores)) * abs(conf - acc)
    return round(float(ece), 4)
