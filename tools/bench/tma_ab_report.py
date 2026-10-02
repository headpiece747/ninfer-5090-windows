#!/usr/bin/env python3
"""Summarise the interleaved TMA-vs-MMA transport A/B from tma_ab.cmd.

Reports the MEDIAN of each arm's samples per token count and the ratio computed after the medians,
because this card's clocks drift between windows and a mean over a drifting window measures the
window. Parses every column BY HEADER NAME: the bench CSV has 39 columns and one empty field, so
column positions are not stable, and reading one by position has already produced a plausible wrong
answer in this project.

The 65..128 band is a CONTROL, not a measurement. Both arms dispatch to Fp8A8T64R64K128 there, so
they must agree. If they do not, the harness is measuring something other than transport and the
whole run is void -- that check runs first and reports VOID rather than a ratio.

usage: tma_ab_report.py <csv-dir> <repeats>
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

CONTROL_TOKENS = {65, 96}
CONTROL_TOLERANCE = 0.05  # 5% between arms on a band where both run the same schedule


def load(directory: Path, arm: str, repeats: int) -> dict[int, list[float]]:
    """Return token -> list of median_us samples, indexed by header name."""
    samples: dict[int, list[float]] = {}
    for index in range(1, repeats + 1):
        path = directory / f"{arm}_{index}.csv"
        if not path.exists():
            print(f"  missing {path.name}")
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if not row.get("T"):
                    continue
                token = int(row["T"])
                # Header name, never a fixed column index.
                samples.setdefault(token, []).append(float(row["median_us"]))
    return samples


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    directory = Path(argv[1])
    repeats = int(argv[2])

    tma = load(directory, "tma", repeats)
    mma = load(directory, "mma", repeats)
    if not tma or not mma:
        print("  INCONCLUSIVE: no samples for one or both arms")
        return 2

    print(f"\n  {'tokens':>7} {'n':>3} {'tma us':>11} {'mma us':>11} {'mma/tma':>9}  note")
    control_ok = True
    rows: list[tuple[int, float, float]] = []
    for token in sorted(set(tma) & set(mma)):
        a = statistics.median(tma[token])
        b = statistics.median(mma[token])
        ratio = b / a if a else float("inf")
        note = ""
        if token in CONTROL_TOKENS:
            note = "CONTROL (same schedule both arms)"
            if abs(ratio - 1.0) > CONTROL_TOLERANCE:
                control_ok = False
                note += " ** FAILED **"
        rows.append((token, a, b))
        print(f"  {token:>7} {min(len(tma[token]), len(mma[token])):>3} {a:>11.3f} {b:>11.3f} {ratio:>9.3f}  {note}")

    if not control_ok:
        print("\n  VERDICT: VOID. The control band disagrees, so the arms are not comparable and")
        print("  nothing here is evidence about TMA. Fix the harness before reading anything else.")
        return 1

    measured = [(t, a, b) for t, a, b in rows if t not in CONTROL_TOKENS]
    if not measured:
        print("\n  VERDICT: VOID. Only control bands were measured.")
        return 1

    ratios = [b / a for _, a, b in measured]
    print(f"\n  measured bands: {len(measured)}   mma/tma ratio  min {min(ratios):.3f}  "
          f"median {statistics.median(ratios):.3f}  max {max(ratios):.3f}")
    print("  ratio < 1 means TMA is FASTER; > 1 means TMA is SLOWER. 1.0 means no difference.")
    print("\n  Samples per point are low by construction (the bench's own median over --repeat).")
    print("  Treat a ratio inside roughly +/-0.05 as indistinguishable and re-run with more")
    print("  repeats before claiming a difference; this card's early-run spread has been 5.8-7.6%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
