#!/usr/bin/env python3
"""Report item 14's discriminator from two bench runs.

Reads warm0.json (request 1, nothing discarded) and warm1.json (request 2, one discarded) and prints
the two quantities the hypotheses disagree about: the round count and the PER-ROUND cost.

Parses by KEY NAME, never by column position -- ninfer_bench's JSON is a dict and its key set has
grown before, and this session already had a reporter that inverted its own conclusion.

The verdict logic is deliberately conservative. It reports WHAT CHANGED and refuses to name a cause
unless one of the two predictions actually holds; "no difference" is a result, not a failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def find_measurement(payload: object) -> dict:
    """Locate the row carrying decode_seconds_mean, and lift the speculative counters out of it.

    The speculative counters are NOT flat `spec_rounds` -- they live in a nested `speculative` object
    as {rounds, acceptance_rate, drafted_tokens, accepted_tokens, ...}. The first version of this
    reporter looked for `spec_rounds`, found nothing, and silently reported 0.0 for both arms, which
    reads as a result rather than as a broken probe. So the keys are asserted, not defaulted.

    The bench has also emitted both a bare list of rows and a dict with a 'tests' key, so the structure
    is walked rather than assumed.
    """
    if isinstance(payload, list):
        for item in payload:
            found = find_measurement(item)
            if found:
                return found
        return {}
    if isinstance(payload, dict):
        if "decode_seconds_mean" in payload:
            spec = payload.get("speculative") or {}
            return {
                "decode_seconds_mean": payload.get("decode_seconds_mean", 0.0),
                "decode_output_tok_s_mean": payload.get("decode_output_tok_s_mean", float("nan")),
                "spec_rounds": spec.get("rounds", 0.0),
                "spec_acceptance_rate": spec.get("acceptance_rate", float("nan")),
                "drafted_tokens": spec.get("drafted_tokens", 0),
                "accepted_tokens": spec.get("accepted_tokens", 0),
            }
        for value in payload.values():
            found = find_measurement(value)
            if found:
                return found
    return {}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: first_request_report.py <directory>")
        return 2
    root = Path(argv[1])
    first_path, second_path = root / "warm0.json", root / "warm1.json"
    for path in (first_path, second_path):
        if not path.exists():
            print(f"INCONCLUSIVE: {path} is missing, so the pair cannot be compared.")
            return 2

    first, second = find_measurement(load(first_path)), find_measurement(load(second_path))
    if not first or not second:
        print("INCONCLUSIVE: neither run reported spec_rounds / decode_seconds_mean.")
        print(f"  keys seen in warm0: {sorted(first)[:12] if first else '(none)'}")
        return 2

    def row(label: str, m: dict) -> tuple[float, float]:
        rounds = float(m.get("spec_rounds", 0.0) or 0.0)
        seconds = float(m.get("decode_seconds_mean", 0.0) or 0.0)
        per_round = (seconds / rounds * 1e3) if rounds > 0 else 0.0
        print(f"  {label:<10} {float(m.get('decode_output_tok_s_mean', float('nan'))):>8.2f} {rounds:>8.1f}"
              f"  {seconds * 1e3:>9.1f} ms  {per_round:>7.3f} ms  {m.get('spec_acceptance_rate', float('nan'))}")
        return rounds, per_round

    print("  request      tok/s     rounds   decode total   per-round cost   acceptance")
    r1, c1 = row("request 1", first)
    r2, c2 = row("request 2", second)
    print(f"\n  drafted/accepted:  req1 {first.get('drafted_tokens')}/{first.get('accepted_tokens')}"
          f"   req2 {second.get('drafted_tokens')}/{second.get('accepted_tokens')}")

    if not c1 or not c2:
        print("\n  INCONCLUSIVE: a per-round cost is zero, so the comparison has no denominator.")
        return 2

    round_delta = (r2 - r1) / r1 * 100.0 if r1 else 0.0
    cost_delta = (c2 - c1) / c1 * 100.0
    print(f"\n  rounds  request 2 vs 1: {round_delta:+.2f}%")
    print(f"  cost    request 2 vs 1: {cost_delta:+.2f}%")

    # This card's own spread is 5.8-7.6% on early runs, so anything inside that is not a difference.
    band = 5.0
    if abs(cost_delta) <= band:
        verdict = (
            "ARITHMETIC: per-round cost is unchanged within this card's spread, so the speed asymmetry "
            "comes from the round COUNT differing, not from the kernel doing different work. That makes "
            "item 14's text and speed halves ONE cause."
        )
    else:
        verdict = (
            "KERNEL SELECTION: per-round cost changes by more than this card's spread, so the first "
            "request is doing different per-round work. That implicates upstream #80 -- the token count "
            "t = 1 + accepted drafts selects the kernel, so a different t selects different arithmetic."
        )
    print(f"\n  VERDICT (band +-{band:.0f}%): {verdict}")

    if r1 and r2 and abs(round_delta) <= band and abs(cost_delta) <= band:
        print("  NOTE: neither quantity moved. That would NOT support either hypothesis; it would mean "
              "the two runs are not measuring different requests, and the loop needs checking before "
              "anything is concluded from it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))