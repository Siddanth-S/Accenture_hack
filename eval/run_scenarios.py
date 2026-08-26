"""Scenario runner: stateless baseline vs session-aware Preflight.

This produces the table that decides whether the pitch lands. Every competing
approach is stateless by construction, so the comparison is not a strawman —
it is the actual state of the art in guardrail libraries.

    python -m eval.run_scenarios
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preflight.engine import Engine, InferenceResult          # noqa: E402
from preflight.policy import load_policy                      # noqa: E402
from preflight.schemas import Action, RequestContext          # noqa: E402
from preflight.session import SessionStore                    # noqa: E402

SCENARIOS = ROOT / "data" / "sessions" / "preflight-sessions-v1.jsonl"

BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"
GREEN = "\033[32m"; AMBER = "\033[33m"; RED = "\033[31m"; CYAN = "\033[36m"

ACTION_COLOUR = {
    Action.PASS: GREEN, Action.PASS_WITH_CITATIONS: GREEN,
    Action.REDACT: AMBER, Action.REGENERATE: AMBER,
    Action.ESCALATE: RED, Action.BLOCK: RED,
}

INTERVENED = {Action.REDACT, Action.REGENERATE, Action.ESCALATE, Action.BLOCK}


def load_scenarios() -> list[dict]:
    return [json.loads(l) for l in SCENARIOS.read_text().splitlines() if l.strip()]


async def run_session(engine: Engine, scen: dict, stateless: bool) -> list:
    engine.sessions.reset(scen["session_id"])
    decisions = []
    for t in scen["turns"]:
        ctx = RequestContext(
            session_id=scen["session_id"],
            use_case=scen["use_case"],
            turn=t["turn"],
            requested_model="claude-sonnet-5",
            sources=t.get("sources", []),
            stateless_mode=stateless,
        )
        result = InferenceResult(
            text=t["response"],
            tokens_in=len(t["prompt"].split()) * 4 // 3,
            tokens_out=len(t["response"].split()) * 4 // 3,
        )
        decisions.append(await engine.evaluate_response(ctx, t["prompt"], result))
    return decisions


async def main() -> None:
    policy = load_policy(ROOT / "policies" / "default.yaml")
    engine = Engine(policy, SessionStore())
    scenarios = load_scenarios()

    print(f"\n{BOLD}PREFLIGHT — scenario evaluation{RESET}")
    print(f"{DIM}policy {policy.id} @ {policy.version_hash} "
          f"| jurisdiction {policy.jurisdiction}{RESET}\n")

    # Metric note: "caught somewhere in the session" is too generous — a
    # checker that fires on turn 9 of a breach that began on turn 4 has not
    # prevented anything. We score TIMELY detection: intervened at or before
    # the turn the scenario becomes unsafe.
    tp_sess = tp_stateless = 0
    fp_sess = fp_stateless = 0
    n_unsafe = n_safe = 0
    session_only_turns = 0
    latencies: list[float] = []

    for scen in scenarios:
        stateless = await run_session(engine, scen, stateless=True)
        session = await run_session(engine, scen, stateless=False)
        latencies += [d.latency_ms for d in session]

        unsafe_from = scen["unsafe_from_turn"]
        is_unsafe = unsafe_from is not None
        n_unsafe += int(is_unsafe)
        n_safe += int(not is_unsafe)

        def timely(decisions) -> bool:
            return any(
                d.action in INTERVENED and d.turn <= unsafe_from
                for d in decisions
            )

        caught_sess = timely(session) if is_unsafe else any(
            d.action in INTERVENED for d in session)
        caught_stateless = timely(stateless) if is_unsafe else any(
            d.action in INTERVENED for d in stateless)

        session_only_turns += sum(
            1 for sl, se in zip(stateless, session)
            if se.action in INTERVENED and sl.action not in INTERVENED
        )

        if is_unsafe:
            tp_sess += int(caught_sess)
            tp_stateless += int(caught_stateless)
        else:
            fp_sess += int(caught_sess)
            fp_stateless += int(caught_stateless)

        print(f"{BOLD}{scen['name']}{RESET}  {DIM}[{scen['label']}]{RESET}")
        print(f"  {DIM}{scen['note']}{RESET}")

        for sl, se in zip(stateless, session):
            c_sl = ACTION_COLOUR[sl.action]
            c_se = ACTION_COLOUR[se.action]
            marker = ""
            if se.action in INTERVENED and sl.action not in INTERVENED:
                marker = f"  {CYAN}<- caught ONLY by session state{RESET}"
            print(
                f"    turn {se.turn}  "
                f"stateless {c_sl}{sl.action.value:<20}{RESET}  "
                f"session {c_se}{se.action.value:<20}{RESET}"
                f"{DIM}{se.latency_ms:>7.1f}ms{RESET}{marker}"
            )
            if se.action in INTERVENED:
                print(f"           {DIM}{se.reason[:100]}{RESET}")

        sr = session[-1].session_risk
        print(f"    {DIM}session risk: escalation={sr.escalation_gradient:.2f} "
              f"contamination={sr.contamination:.2f} "
              f"retry={sr.retry_burn:.2f} cost_creep={sr.cost_creep:.2f}{RESET}\n")

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    print(f"{BOLD}{'=' * 68}{RESET}")
    print(f"{BOLD}RESULTS{RESET}\n")
    print(f"  {'':<26}{'stateless':>12}{'session-aware':>16}")
    print(f"  {'-' * 54}")
    print(f"  {'unsafe caught':<26}{tp_stateless:>7}/{n_unsafe:<4}"
          f"{tp_sess:>11}/{n_unsafe:<4}")
    print(f"  {'detection rate':<26}"
          f"{tp_stateless / max(1, n_unsafe):>11.0%}"
          f"{tp_sess / max(1, n_unsafe):>15.0%}")
    print(f"  {'false positives':<26}{fp_stateless:>7}/{n_safe:<4}"
          f"{fp_sess:>11}/{n_safe:<4}")
    print()
    print(f"  {CYAN}{session_only_turns} turn(s) intervened on ONLY because of "
          f"session state{RESET}")
    print(f"  {DIM}these are structurally invisible to any stateless "
          f"checker{RESET}")
    print()
    print(f"  latency  p50 {p50:.1f}ms   p95 {p95:.1f}ms  "
          f"{DIM}(lexical fallback, no NLI model loaded){RESET}")
    print(f"{BOLD}{'=' * 68}{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
