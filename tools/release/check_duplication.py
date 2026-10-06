#!/usr/bin/env python3
"""Gate new copy-paste into src/ and include/, tolerating the duplication already there.

WHY THIS EXISTS. The 2026-10-03 code-health request asked for duplication, dead code and unused
includes. Dead code is `check_dead_types.py`. This is the duplication half, which is not buildable
sensibly by hand: jscpd tokenizes per language and matches with a rolling Rabin-Karp hash, and it
runs this tree -- 490 files, 91,284 lines -- in about 65 ms.

MEASURED ON THIS TREE, 2026-10-03, jscpd 5.4.0:
    490 files analyzed, 378 clones, 3,523 duplicated lines (3.86% of lines, 4.01% of tokens)
    At --min-lines 20: 9 clones, 243 duplicated lines (0.36% of C++ lines)
So the tree is not badly duplicated, and at default thresholds a third of the findings are 5-8 line
coincidences in the serving layer -- the kind a gate cries wolf over, which is how a gate gets muted.
`--min-lines 20` is what makes this signal rather than noise, and it is the one setting here that was
chosen by looking at the output rather than by convention.

WHY A BASELINE RATHER THAN A THRESHOLD. 378 clones already exist. A threshold gate would fail on
every commit from the first run and teach everyone to route around it. `--baseline` records what is
already there and `--fail-on-new-clones` fails only on an increase, so the gate is a ratchet: it stops
duplication growing without demanding a refactor nobody asked for. This is the same shape as
`check_production_stream_defaults.py`, and it is why the repo's own habit-hooks project ships a
snooze index rather than a limit.

WHY 5.0 AND NOT 0. `--min-lines 20` leaves a real gap: duplication can grow inside an existing
clone, or a new 19-line clone can appear. `--fail-on-new-clones 5` tolerates up to 5 new clones and
fails past it, so a commit adding six new 20-line copies is caught while one copying a helper for a
legitimate reason is not. Raise it if the ratchet proves loose; it is a number, not a principle.

THE THREE EXITS, and the middle one is the point:
    0  clean -- no new duplication past the allowance
    1  FAILING -- new duplication past the allowance; the message names the new clones
    2  BROKEN  -- jscpd could not run, or analyzed nothing
Exit 2 exists because a scan that silently finds nothing looks exactly like a clean tree, and this
repo has been bitten by a plausible zero before. jscpd's own `--fail-on-empty` covers the
analyzed-nothing case; it cannot cover its own absence, which is what this wrapper adds.

jscpd is not vendored and not on PATH by default. Install it once with `uv tool install jscpd==5.4.0`
(Rust binary, no runtime needed); this gate also accepts `npx --yes jscpd@5.4.0`. The version is
pinned in both paths, because a gate whose numbers move with a floating tag is a measurement with no
revision attached to it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

JSCPD_VERSION = "5.4.0"  # measured 2026-10-03; see module docstring
BASELINE = REPO / ".jscpd-baseline.json"
REPORT_DIR = REPO / ".jscpd-report"  # gitignored; the readable detail when the gate fails
SCOPE = ["src", "include"]
MIN_LINES = 20
ALLOW_NEW_CLONES = 5

# `tool.execute.before` in claims-gate.ts would read this shape, so keep it as jscpd writes it.


def jscpd_argv() -> list[str]:
    """Prefer a real binary; fall back to npx with the version pinned."""
    exe = shutil.which("jscpd") or shutil.which("cpd")
    if exe:
        return [exe]
    npx = shutil.which("npx")
    if npx:
        return [npx, "--yes", f"jscpd@{JSCPD_VERSION}"]
    return []


def main() -> int:
    argv = jscpd_argv()
    if not argv:
        print(
            "  GATE ERROR: jscpd not found. Install it with `uv tool install "
            f"jscpd=={JSCPD_VERSION}`, or put npx on PATH. Exiting 2 rather than reporting a "
            "clean tree, because a scan that did not run is not a pass."
        )
        return 2

    # Establish the baseline on first run, then gate against it.
    if not BASELINE.exists():
        seed = subprocess.run(
            argv
            + SCOPE
            + [
                "--baseline",
                str(BASELINE),
                # jscpd needs --update-baseline to CREATE the file; --baseline alone only reads
                # it, and its error ("not found -- run with --update-baseline") is how this was found.
                "--update-baseline",
                "--silent",
                "--fail-on-empty",
                "--min-lines",
                str(MIN_LINES),
            ],
            cwd=REPO,
            capture_output=True,
            text=True, encoding="utf-8",
            check=False,
        )
        if seed.returncode != 0 or not BASELINE.exists():
            print("  GATE ERROR: could not write the duplication baseline. jscpd said:")
            print("  " + (seed.stderr or seed.stdout).strip()[-800:])
            return 2
        print(f"  baseline written: {BASELINE.relative_to(REPO).as_posix()} ({BASELINE.stat().st_size} bytes)")
        print("  Duplication already in the tree is now recorded. This gate fails only on NEW clones.")
        return 0

    run = subprocess.run(
        argv
        + SCOPE
        + [
            "--baseline",
            str(BASELINE),
            "--fail-on-new-clones",
            str(ALLOW_NEW_CLONES),
            "--fail-on-empty",
            "--min-lines",
            str(MIN_LINES),
            "--reporters",
            "json",
            # `--output` names a DIRECTORY. Passing a .json path makes jscpd create a directory with
            # that name, and the report lands inside it as jscpd-report.json -- which is what the
            # first version of this gate did, and then failed to read with a permission error.
            "--output",
            str(REPORT_DIR),
        ],
        cwd=REPO,
        capture_output=True,
        text=True, encoding="utf-8",
        check=False,
    )

    report = REPORT_DIR / "jscpd-report.json"
    if run.returncode == 2 or not report.exists():
        print("  GATE ERROR: jscpd could not analyze the tree. Exiting 2 (broken), not 0 (clean).")
        print("  " + (run.stderr or run.stdout).strip()[-800:])
        return 2

    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  GATE ERROR: could not read jscpd's report: {exc}")
        return 2

    stats = data.get("statistics", {})
    total = stats.get("total", {})
    dup = total.get("percentage", 0)
    analyzed = total.get("lines", 0)
    if not analyzed:
        print("  GATE ERROR: jscpd analyzed 0 lines. Exiting 2, not 0.")
        return 2

    if run.returncode == 0:
        print(
            f"  lines={analyzed}  duplicated={dup}%  new clones within allowance "
            f"({ALLOW_NEW_CLONES}); see .jscpd-baseline.json"
        )
        print("  PASS: no new copy-paste past the allowance.")
        return 0

    print(f"  FAILING: new duplication past the allowance of {ALLOW_NEW_CLONES} clone(s).")
    # Name them. A gate that says "fail" without saying what is new is the bare-metric case that
    # habit-hooks' own study measured at a 5.6% genuine-fix rate; the finding has to be actionable.
    named = 0
    for dup_entry in data.get("duplicates", []):
        first = dup_entry.get("firstFile", {})
        second = dup_entry.get("secondFile", {})
        print(
            f"    {first.get('name', '?')}:{first.get('start', '?')} "
            f"<-> {second.get('name', '?')}:{second.get('start', '?')} "
            f"({dup_entry.get('lines', '?')} lines)"
        )
        named += 1
        if named >= 20:
            print("    ... (truncated; full report in .jscpd-report.json)")
            break
    if named == 0:
        print("    jscpd reported a failure it did not itemize; read .jscpd-report.json")
    print("  Refactor, or if the copy is deliberate, record it by re-seeding the baseline.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
