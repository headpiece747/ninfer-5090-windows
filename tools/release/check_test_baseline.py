#!/usr/bin/env python3
"""Release gate: refuse a release whose test suite has regressed.

The suite is green (124/124 since 2026-09-19), and a green run is not self-recording: it proves
today's tree, not that nothing was skipped or silently disabled. This gate compares the suite's
actual result against the recorded baseline in both directions:

  * a test that fails and is not baselined           -> FAIL (a regression)
  * a test that passes but is baselined as failing   -> FAIL (a stale baseline is worse than
                                                        none: it would hide the next regression)
  * a suite that ran fewer tests than the baseline records
                                                     -> FAIL (a partial run is not a suite run)

The third check exists because the first two cannot see it: a run that executed 40 of 124 tests
reports no failures and no fixed baseline entries, so it passed. Any future scoping of the suite
depends on this being asserted.

Reuse: the suite executes the built test executables, so their size and modification time identify
the code under test more precisely than a source hash -- a rebuild changes both, and a stale build
is exactly the case a cached verdict must not cover. When every test executable is unchanged since
a green run, that verdict is reused instead of re-running the suite; the baseline comparison below
is still applied, so a changed baseline still takes effect. Pass --no-cache to force a run.

The ctest log is kept at build-test/Testing/Temporary/gate-ctest.log, so the run is auditable and
the per-test timings survive.

Usage:
    python check_test_baseline.py                 # run ctest (or reuse), then check
    python check_test_baseline.py --no-cache      # always run ctest
    python check_test_baseline.py --from-log f    # check a captured ctest log instead
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).resolve().parent / "test_baseline.json"
BUILD = REPO / "build-test"
LOG = BUILD / "Testing" / "Temporary" / "gate-ctest.log"
CACHE = BUILD / ".gate-cache.json"
# The slowest test measured 262 s (ninfer_context_kv_materialize_test) and ctest has no timeout by
# default, so this fails a hanging test rather than letting it stall a release indefinitely.
TEST_TIMEOUT_SECONDS = 900
# ctest reports a failing test as "(Failed)" or, when the process aborted, as "(Exit code 0xc0000409)".
# Matching only the first form hid four of five failures in one run.
FAILURE_LINE = re.compile(r"^\s*\d+\s+-\s+(\S+)\s+\((?:Failed|Exit code \S+|Exception[^)]*)\)")


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


def binary_key() -> str | None:
    """Identify the built test executables, or None when the build tree is absent."""
    executables = sorted(BUILD.rglob("*.exe"))
    if not executables:
        return None
    digest = hashlib.sha256()
    for executable in executables:
        stat = executable.stat()
        digest.update(f"{executable.relative_to(REPO).as_posix()}:{stat.st_size}:{stat.st_mtime_ns}\n"
                      .encode("utf-8"))
    return digest.hexdigest()


def run_ctest() -> str:
    ctest = find_ctest()
    if ctest is None:
        raise SystemExit(
            "  GATE ERROR: ctest was not found. Run this from a Visual Studio developer prompt, "
            "or pass --from-log with a captured ctest log.")
    print("  running ctest (this takes several minutes)...")
    result = subprocess.run(
        [ctest, "--test-dir", "build-test", "--output-on-failure",
         "--timeout", str(TEST_TIMEOUT_SECONDS)],
        cwd=REPO, capture_output=True, text=True)
    # ctest exits non-zero when tests fail, which is expected here; the log is what matters.
    log = result.stdout + result.stderr
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(log, encoding="utf-8", errors="replace")
    print(f"  ctest log : {LOG.relative_to(REPO)}")
    return log


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


def cached_log(key: str) -> str | None:
    if not CACHE.exists():
        return None
    try:
        record = json.loads(CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if record.get("key") != key:
        return None
    print(f"  reusing the recorded run of these binaries ({record.get('recorded', 'unknown')})")
    return record.get("log")


def store_cache(key: str, log: str, recorded: str) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({"key": key, "recorded": recorded, "log": log}, indent=2),
                     encoding="utf-8")


def parse_skipped(log: str) -> set[str]:
    """Tests ctest reported as Skipped.

    The suite-size assertion counts a skipped test as covered, so a run on a machine without the model
    artifacts can skip every real-model test and still read as green -- which is how the tokenizer's
    real vocabulary went unexercised while the gate passed. A release is cut where its artifacts are
    present, so a skip is absent evidence rather than a neutral outcome.
    """
    skipped: set[str] = set()
    for line in log.splitlines():
        stripped = line.strip()
        if stripped.endswith("(Skipped)") and " - " in stripped:
            skipped.add(stripped.split(" - ", 1)[1].rsplit(" (", 1)[0].strip())
    return skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-log", type=Path,
                    help="check a captured ctest log instead of running the suite")
    ap.add_argument("--no-cache", action="store_true",
                    help="always run the suite, ignoring a recorded run of the same binaries")
    args = ap.parse_args()

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    known = set(baseline.get("known_failures", {}))

    key = None if (args.from_log or args.no_cache) else binary_key()
    log = cached_log(key) if key else None
    reused = log is not None
    if log is None:
        log = args.from_log.read_text(encoding="utf-8", errors="replace") \
            if args.from_log else run_ctest()

    total = parse_total(log)
    failing = parse_failures(log)
    if total is None:
        print("  GATE FAILED: could not read a ctest summary; the suite did not run to completion")
        print(log[-2000:])
        return 1

    expected = baseline.get("suite_size")
    if expected is not None and total != expected:
        print(f"  GATE FAILED: the run covered {total} tests and the baseline records {expected}.")
        print("  A partial run cannot show a regression in the tests it skipped.")
        return 1

    new_failures = sorted(failing - known)
    fixed = sorted(known - failing)

    # Skips are reported after failures, never instead of them: this check used to return here and hide
    # a run that both skipped and failed.
    if new_failures:
        print("\n  GATE FAILED: new test failure(s) not in the baseline:")
        for name in new_failures:
            print(f"    {name}")
        print("  Fix the regression, or record it in test_baseline.json with its reason.")

    skipped = parse_skipped(log)
    required = set(baseline.get("required_tests", []))
    skipped_required = sorted(skipped & required)
    if skipped_required:
        print(f"\n  GATE FAILED: {len(skipped_required)} required test(s) were skipped, so their "
              f"coverage is missing:")
        for name in skipped_required:
            print(f"    {name}")
        print("  A required test is one this product's artifacts can satisfy; make it run.")
    if skipped:
        optional = baseline.get("optional_tests", {})
        print(f"\n  note: {len(skipped)} real-model test(s) not required on this product:")
        for name in sorted(skipped):
            print(f"    {name}  -- {optional.get(name, 'no reason recorded')}")

    if new_failures or skipped_required:
        return 1

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

    if key and not reused:
        store_cache(key, log, subprocess.run(["git", "log", "-1", "--format=%h %cI"], cwd=REPO,
                                             capture_output=True, text=True).stdout.strip())
    print("\n  GATE PASSED: no regression against the recorded baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
