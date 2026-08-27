"""Conformal calibration of policy thresholds — the runnable proof.

The policy YAML claims its thresholds are "produced by calibration, not
hand-picked". This module is what makes that claim true and inspectable:

    python -m eval.calibrate                 # report only
    python -m eval.calibrate --alpha 0.10    # choose the risk budget
    python -m eval.calibrate --write         # patch policies/default.yaml

It scores a labelled corpus through the REAL groundedness pipeline, runs the
distribution-free conformal procedure in `calibration.py`, and reports the
threshold the guarantee selects — alongside the threshold currently deployed
in the policy, so any drift between "what we claim" and "what we ship" is
visible rather than buried.

The honesty the whole project argues for shows up in the output directly: at
a small sample size the Hoeffding bound will refuse a 5% guarantee and say so,
rather than quietly shipping a number it cannot defend.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preflight.calibration import (                       # noqa: E402
    calibrate_threshold, expected_calibration_error, risk_coverage_curve,
)
from preflight.claims import (                            # noqa: E402
    Source, extract_claims, groundedness_score, verify_claim,
)

CORPUS = ROOT / "data" / "calibration" / "groundedness.jsonl"
POLICY = ROOT / "policies" / "default.yaml"

BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"
GREEN = "\033[32m"; AMBER = "\033[33m"; RED = "\033[31m"; CYAN = "\033[36m"


def _load_corpus() -> list[dict]:
    rows = []
    for line in CORPUS.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if "_meta" in obj:          # header line
            continue
        rows.append(obj)
    return rows


def _score_groundedness(response: str, sources: list[str]) -> float:
    """Real detector output — the same path the engine takes at stakes>=medium."""
    claims = extract_claims(response)
    srcs = [Source(id=f"s{i}", text=t) for i, t in enumerate(sources)]
    for c in claims:
        verify_claim(c, srcs, nli=None)
    return groundedness_score(claims)


def _deployed_threshold(axis: str) -> float | None:
    """The groundedness threshold currently in the default use case."""
    import yaml
    raw = yaml.safe_load(POLICY.read_text())
    for rule in raw["use_cases"]["default"]["rules"]:
        if rule["axis"] == axis:
            return float(rule["threshold"])
    return None


def _write_threshold(axis: str, threshold: float, alpha: float, n: int) -> None:
    """Patch the default-use-case threshold in place, recording provenance.

    Deliberately a targeted line rewrite rather than a YAML round-trip, so the
    file keeps its comments — the comments ARE the audit trail here.
    """
    text = POLICY.read_text()
    note = f"conformal alpha={alpha:g}, n={n}"
    # Match:  - {axis: groundedness, threshold: 0.55, action: regenerate}
    pat = re.compile(
        rf"(-\s*\{{axis:\s*{re.escape(axis)},\s*threshold:\s*)[0-9.]+(,\s*action:\s*\w+)([^}}]*)\}}"
    )
    n_sub = 0

    def _repl(m: re.Match) -> str:
        nonlocal n_sub
        n_sub += 1
        tail = m.group(3)
        tail = re.sub(r",?\s*note:\s*\"[^\"]*\"", "", tail)  # drop old note
        return f'{m.group(1)}{threshold:.2f}{m.group(2)}{tail}, note: "{note}"}}'

    new = pat.sub(_repl, text, count=1)
    if n_sub == 0:
        print(f"{RED}could not locate an editable '{axis}' rule in default "
              f"use case — not written{RESET}")
        return
    POLICY.write_text(new)
    print(f"{GREEN}wrote{RESET} {axis} threshold {threshold:.2f} into "
          f"{POLICY.relative_to(ROOT)} ({note})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Conformal calibration of policy thresholds")
    ap.add_argument("--axis", default="groundedness")
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="max tolerated miss rate (unsafe passing)")
    ap.add_argument("--delta", type=float, default=0.10, help="1 - confidence")
    ap.add_argument("--write", action="store_true",
                    help="patch the policy threshold with the calibrated value")
    args = ap.parse_args()

    rows = _load_corpus()
    scores = np.array([_score_groundedness(r["response"], r["sources"]) for r in rows])
    labels = np.array([int(r["unsafe"]) for r in rows])

    res = calibrate_threshold(scores, labels, axis=args.axis,
                              alpha=args.alpha, delta=args.delta)
    ece = expected_calibration_error(scores, labels)
    deployed = _deployed_threshold(args.axis)

    print(f"\n{BOLD}Conformal calibration — {args.axis}{RESET}")
    print(f"  {DIM}corpus{RESET}            {len(rows)} examples "
          f"({int(labels.sum())} unsafe / {len(rows) - int(labels.sum())} safe)"
          f"  {DIM}[synthetic]{RESET}")
    print(f"  {DIM}risk budget{RESET}       alpha={args.alpha:g} (max miss rate), "
          f"confidence={1 - args.delta:.0%}")
    print()

    if res.valid:
        col = GREEN
        verdict = "GUARANTEE HOLDS"
    else:
        col = AMBER
        verdict = "GUARANTEE REFUSED"
    print(f"  {col}{BOLD}{verdict}{RESET}")
    print(f"  {DIM}threshold{RESET}         {col}{res.threshold:.3f}{RESET}")
    print(f"  {DIM}empirical miss{RESET}    {res.empirical_miss_rate:.3f}")
    print(f"  {DIM}flag rate{RESET}         {res.flag_rate:.3f} "
          f"{DIM}(fraction of traffic sent for handling){RESET}")
    print(f"  {DIM}ECE{RESET}               {ece:.3f} "
          f"{DIM}(score↔accuracy gap){RESET}")
    print(f"  {DIM}note{RESET}              {res.note}")

    if deployed is not None:
        # Conformal reading: the calibrated value is the MOST PERMISSIVE
        # threshold that still holds the bound. A deployed threshold at or
        # below it holds the guarantee too (stricter -> more flagging, safer).
        # Only a deployed threshold ABOVE it can let the miss rate exceed alpha.
        if not res.valid:
            mark = f"{AMBER}no guarantee at this alpha{RESET}"
        elif deployed <= res.threshold + 1e-9:
            margin = res.threshold - deployed
            mark = (f"{GREEN}holds{RESET} "
                    f"{DIM}(stricter than required by {margin:.2f} — trades "
                    f"flag rate for margin){RESET}")
        else:
            mark = (f"{RED}VIOLATES guarantee{RESET} "
                    f"{DIM}(more permissive than the bound allows){RESET}")
        print(f"  {DIM}deployed{RESET}          {deployed:.3f}  "
              f"{DIM}(in policy now){RESET}  -> {mark}")

    # Risk-coverage: what each point of automation costs.
    curve = risk_coverage_curve(scores, labels, steps=11)
    print(f"\n  {DIM}risk-coverage (coverage = auto-handled, risk = miss rate there){RESET}")
    print(f"  {DIM}  thr    coverage   risk{RESET}")
    for p in curve:
        if p["threshold"] in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
            print(f"    {p['threshold']:.2f}     {p['coverage']:.2f}      {p['risk']:.2f}")

    if not res.valid:
        print(f"\n  {AMBER}At this sample size the {args.alpha:g} budget is not "
              f"defensible. Loosen --alpha or grow the corpus —{RESET}")
        print(f"  {AMBER}refusing to ship an undefended number is the honest "
              f"failure mode.{RESET}")

    if args.write:
        print()
        if res.valid:
            _write_threshold(args.axis, res.threshold, args.alpha, len(rows))
        else:
            print(f"{RED}not writing: guarantee refused at alpha={args.alpha:g}"
                  f"{RESET}")
    print()


if __name__ == "__main__":
    main()
