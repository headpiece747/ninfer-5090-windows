#!/usr/bin/env python3
"""Release gate: refuse a release whose test suite has regressed.

The suite is green (122/122 since 2026-09-19), and a green run is not self-recording: it proves
today's tree, not that nothing was skipped or silently disabled. This gate compares the suite's
actual result against the recorded baseline in both directions:

  * a test that fails and is not baselined           -> FAIL (a regression)
  * a test that passes but is baselined as failing   -> FAIL (a stale baseline is worse than
                                                        none: it would hide the next regression)

Usage:
    python check_test_baseline.py                 # run ctest, then check
    python check_test_baseline.py --from-log f    # check a captured ctest log instead
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).resolve().parent / "test_baseline.json"
FAILURE_LINE = re.compile(r"^\s*\d+\s+-\s+(\S+)\s+\(Failed\)")


def find_ctest() -> str | None:
    """ctest is not on PATH outside a Visual Studio environment, so resolve it the same way the
    build recipe does rather than assuming the caller's shell."""
    found = shutil.which("ctest")
    if found:
        return found
    for root in (Path(r"C:\Program Files\Microsoft Visual Studio"),
                 Path(r"C:\Program Files (x86)\Microsoft Visual Studio")):
        if not root.exists():
            continue
        # Layout is <root>\<version>\<edition>\Common7\...\CMake\bin\ctest.exe, so iterate the
        # version/edition pairs rather than recursing over the whole (very large) install.
        for install in root.glob("*/*"):
            candidate = (install / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" /
                         "CMake" / "CMake" / "bin" / "ctest.exe")
            if candidate.exists():
                return str(candidate)
    return None


def run_ctest() -> str:
    ctest = find_ctest()
    if ctest is None:
        raise SystemExit(
            "  GATE ERROR: ctest was not found. Run this from a Visual Studio developer prompt, "
            "or pass --from-log with a captured ctest log.")
    print("  running ctest (this takes several minutes)...")
    result = subprocess.run(
        [ctest, "--test-dir", "build-test", "--output-on-failure"],
        cwd=REPO, capture_output=True, text=True)
    # ctest exits non-zero when tests fail, which is expected here; the log is what matters.
    return result.stdout + result.stderr


def parse_failures(log: str) -> set[str]:
    """Names of failing tests. ctest lists them under a 'The following tests FAILED:' header."""
    failures: set[str] = set()
    in_block = False
    for line in log.splitlines():
        if "The following tests FAILED:" in line:
            in_block = True
            continue
        if in_block:
            match = FAILURE_LINE.match(line)
            if match:
                failures.add(match.group(1))
            elif line.strip() and not line.startswith((" ", "\t")):
                in_block = False
    return failures


def parse_total(log: str) -> int | None:
    match = re.search(r"out of (\d+)", log)
    return int(match.group(1)) if match else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-log", type=Path,
                    help="check a captured ctest log instead of running the suite")
    args = ap.parse_args()

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    known = set(baseline.get("known_failures", {}))
    log = args.from_log.read_text(encoding="utf-8", errors="replace") \
        if args.from_log else run_ctest()

    total = parse_total(log)
    failing = parse_failures(log)
    if total is None:
        print("  GATE FAILED: could not read a ctest summary; the suite did not run to completion")
        print(log[-2000:])
        return 1

    new_failures = sorted(failing - known)
    fixed = sorted(known - failing)

    print(f"  suite        {total - len(failing)}/{total} passed "
          f"(baseline records {len(known)} known failure(s))")
    for name in sorted(known & failing):
        print(f"  expected     {name}  (baselined: {baseline['known_failures'][name]['case']})")

    if new_failures:
        print("\n  GATE FAILED: new test failure(s) not in the baseline:")
        for name in new_failures:
            print(f"    {name}")
        print("  Fix the regression, or record it in test_baseline.json with its reason.")
        return 1

    if fixed:
        print("\n  GATE FAILED: baselined failure(s) now pass, so the baseline is stale:")
        for name in fixed:
            print(f"    {name}")
        print("  Remove them from test_baseline.json. A stale baseline would hide the next "
              "regression.")
        return 1

    print("\n  GATE PASSED: no regression against the recorded baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
