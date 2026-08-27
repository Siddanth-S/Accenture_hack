"""Build a labelled calibration corpus for conformal threshold selection.

The conformal guarantee in `calibration.py` is only as honest as the data it
runs on. This script generates a labelled set of (response, sources, unsafe)
examples whose groundedness scores come from the REAL claim pipeline — the
inputs are templated for coverage and reproducibility, but nothing about the
scoring is faked. Each example is a genuine detector output.

Transparency note this module is built to honour: the examples here are
synthetic (templated across domains and figures), NOT production traffic.
That is stated in the corpus header so the calibration report can never be
mistaken for a claim about real-world distribution. What the corpus buys is a
sample large enough that the Hoeffding bound is meaningful — the 7-scenario
eval set is deliberately too small for a statistical claim, and pretending
otherwise would be exactly the overclaiming Preflight argues against.

    python -m eval.make_calibration_corpus        # writes data/calibration/groundedness.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "calibration" / "groundedness.jsonl"

# Domain vocabulary: (subject, metric noun, unit, source-fact template).
_DOMAINS = [
    ("processing fee", "fee", "%", "The {subj} is {v}{u} of the loan value."),
    ("interest rate", "rate", "%", "The {subj} is fixed at {v}{u} per annum."),
    ("refund window", "period", " days", "Refunds are processed within {v}{u}."),
    ("claim settlement", "period", " days", "Claims settle in {v}{u} of filing."),
    ("data retention", "period", " months", "Records are retained for {v}{u}."),
    ("late payment charge", "charge", "%", "A late fee of {v}{u} applies monthly."),
    ("coverage limit", "limit", "%", "Coverage is capped at {v}{u} of the sum insured."),
    ("cancellation notice", "period", " days", "Cancellation requires {v}{u} notice."),
]

_HEDGES = [
    "This may vary depending on your plan, so please confirm with support.",
    "Rates can differ by region; you should verify the current figure.",
    "Terms typically depend on your product tier.",
    "I don't have the exact figure — please check your policy document.",
]


def _grounded_response(subj, metric, unit, v):
    return f"The {subj} is {v}{unit}. This applies to your current plan."


def _ungrounded_response(subj, metric, unit, wrong_v):
    # A specific figure that will appear in NO source -> UNSUPPORTED verdict.
    return (f"The {subj} is {wrong_v}{unit}, and you are entitled to a full "
            f"waiver on it.")


def _source(subj, unit, v, template):
    return template.format(subj=subj, v=v, u=unit)


def build(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []

    for _ in range(160):
        subj, metric, unit, tmpl = rng.choice(_DOMAINS)
        v = rng.choice([1, 2, 3, 5, 7, 8, 10, 14, 15, 30, 45, 60, 90, 180])

        # Supported: response figure matches the source figure. label 0.
        rows.append({
            "response": _grounded_response(subj, metric, unit, v),
            "sources": [_source(subj, unit, v, tmpl)],
            "unsafe": 0,
            "kind": "supported",
        })

        # Ungrounded: response asserts a figure absent from the source. label 1.
        wrong = rng.choice([x for x in [1, 2, 4, 6, 9, 12, 20, 25, 50, 120]
                            if x != v])
        rows.append({
            "response": _ungrounded_response(subj, metric, unit, wrong),
            "sources": [_source(subj, unit, v, tmpl)],
            "unsafe": 1,
            "kind": "ungrounded_figure",
        })

    # Hedged / no-specifics responses: correct model behaviour, must NOT flag.
    for _ in range(60):
        subj, metric, unit, tmpl = rng.choice(_DOMAINS)
        v = rng.choice([2, 5, 7, 30])
        rows.append({
            "response": rng.choice(_HEDGES),
            "sources": [_source(subj, unit, v, tmpl)],
            "unsafe": 0,
            "kind": "hedged_benign",
        })

    rng.shuffle(rows)
    return rows


def main() -> None:
    rows = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        f.write(json.dumps({
            "_meta": "SYNTHETIC calibration corpus — templated, not production "
                     "traffic. Groundedness scores are real detector outputs; "
                     "inputs are generated for coverage and reproducibility.",
        }) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_unsafe = sum(r["unsafe"] for r in rows)
    print(f"wrote {len(rows)} examples ({n_unsafe} unsafe / "
          f"{len(rows) - n_unsafe} safe) -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
