#!/usr/bin/env python3
"""Audit the port's delta against upstream, and ratchet it so it cannot grow.

Why this exists
---------------
The port's changes to upstream-owned files accumulate silently. A merge can take
upstream's version of a file and drop the port's with no error and no conflict, and
nothing notices until a test fails days later. Two such losses are recorded in this
tree already:

  * the incremental encode cache (ADR-0010), lost when a merge took upstream's
    ``src/models/qwen3_5/frontend/tokenizer.cpp``; its declaration survives in the
    port's ``tokenizer.h``, so the tree compiled and linked with the feature dead;
  * ``CausalTopk`` in ``include/ninfer/types.h``, whose own comment records that
    upstream's rewrite "did not carry it" and it had to be restored by hand.

The policy this enforces
------------------------
Upstream owns its files. The port changes one only to make the native Windows port
work, or for a product feature it has deliberately chosen, and every such change is
recorded in ``port_delta_baseline.json`` with a disposition. Filing an entry is the
review; the ratchet is what stops the list growing by accident.

Dispositions, after review:
  windows    keep -- the port's reason to exist; minimise and mark the hunk
  product    keep, but prefer a port-owned file so a merge cannot take it
  refactor   revert to upstream's version -- restructuring his code buys nothing
  unknown    not yet reviewed

A baseline entry carries two account fields, and the difference matters:

  disposition  the marker heuristic's guess, recomputed on every record. Deliberately broad, and
               a triage hint rather than a verdict.
  review       the verdict a review reached, carried forward like `reason`. The report prefers it,
               so a file the review KEPT cannot read as "refactor" -- a label that invites exactly
               the change the review decided against.
  reason       what the diff does and where to check it: a citation to an item, ADR, document or
               measurement, or a plain statement that the diff carries no provenance.

Usage
-----
    python tools/release/check_port_delta.py
    python tools/release/check_port_delta.py --against upstream/dev
    python tools/release/check_port_delta.py --check            # the gate
    python tools/release/check_port_delta.py --write-baseline   # after a review

The reference must be the upstream branch this tree is based on. That is normally `upstream/dev`,
because `dev` is where upstream works and what a merge takes; `master` is its release branch, and
while the port sat at `master` this defaulted to it. Once `dev` runs ahead, comparing against
`master` reports upstream's own newer files as the port's divergence -- 695 paths where the truth
was 541 -- so `--check` also fails when the recorded reference is not the one being compared.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).resolve().parent / "port_delta_baseline.json"
DEFAULT_UPSTREAM = "upstream/dev"

# Files the port added are its own and carry no merge risk; only these statuses diverge
# inside a file upstream owns.
DIVERGENT = ("M", "D")

# Windows / MSVC / port-mechanism markers. Deliberately broad: a false "windows" costs one
# human glance, while a false "not windows" buries a port-necessity under a wrong label.
MARKERS = re.compile(
    r"_WIN32|_MSC_VER|WIN32|Win32|WINDOWS|windows|MSVC|msvc|vcvars|__declspec|"
    r"FILE_FLAG|CreateFileW|OVERLAPPED|__int128|Uint128|_CRT_|\.cmd\b|\.bat\b|"
    r"Ninja|ninja|nmake|NMake|win64|Win64|QueryPerformance|"
    r"PkgConfig|STATUS_DLL|C2719|grid_constant",
    re.IGNORECASE,
)

# A heavily-removal change to an upstream file is the signature of restructuring his code
# rather than adding to it: helpers extracted to a new header, duplicated code deduplicated.
REFACTOR_MAX_ADDS = 4
REFACTOR_MIN_DELS = 10
# A heavily-addition change is the signature of a port feature bolted onto his file.
PRODUCT_MIN_ADDS = 20
PRODUCT_MAX_DELS = 2


class Entry(NamedTuple):
    path: str
    status: str
    added: int
    removed: int
    windows_marked: bool

    @property
    def size(self) -> int:
        return self.added + self.removed

    @property
    def disposition(self) -> str:
        if self.status == "A":
            return "port-added"
        if self.status == "D":
            return "deleted"
        if self.windows_marked:
            return "windows"
        if self.added <= REFACTOR_MAX_ADDS and self.removed >= REFACTOR_MIN_DELS:
            return "refactor"
        if self.added >= PRODUCT_MIN_ADDS and self.removed <= PRODUCT_MAX_DELS:
            return "product"
        return "unknown"


def git(*args: str) -> str:
    done = subprocess.run(
        ["git", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if done.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{done.stderr.strip()}")
    return done.stdout


def collect(against: str) -> list[Entry]:
    # Compared against the WORKING TREE, not HEAD. The pre-commit hook runs before the commit
    # exists, so a HEAD-based diff would miss the divergence that the very commit being guarded is
    # introducing -- reporting it one commit late, which is exactly how the encode cache was lost
    # without anything failing.
    #
    # And compared against the MERGE BASE, not the branch tip. `upstream/dev` moves every time it is
    # fetched, and a tip-based diff then reports upstream's own new work as the port's divergence --
    # on 2026-10-06 a fetch of 18 commits turned a green ratchet red for that reason alone. What the
    # ratchet is for is what the port changed relative to what it was based on, and that is the
    # merge base.
    #
    # During a merge the tree is based on MERGE_HEAD, not on HEAD: HEAD is still the port's tip until
    # the merge commit lands, so a HEAD-based base would be stale for exactly the commit this hook is
    # guarding and would read the incoming branch's own files as divergences.
    based_on = "MERGE_HEAD" if (REPO / ".git" / "MERGE_HEAD").exists() else "HEAD"
    base = git("merge-base", based_on, against).strip()
    counts: dict[str, tuple[int, int]] = {}
    for line in git("diff", "--numstat", base).splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            try:
                counts[parts[2]] = (int(parts[0]), int(parts[1]))
            except ValueError:
                counts[parts[2]] = (-1, -1)  # binary

    entries: list[Entry] = []
    for line in git("diff", "--name-status", base).splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status, path = parts[0][0], parts[1]
        added, removed = counts.get(path, (0, 0))
        marked = False
        if status == "M":
            diff = git("diff", "--unified=0", base, "--", path)
            added_lines = [
                text
                for text in diff.splitlines()
                if text.startswith("+") and not text.startswith("+++")
            ]
            marked = bool(MARKERS.search("\n".join(added_lines)))
        entries.append(Entry(path, status, added, removed, marked))
    return entries


def divergences(entries: list[Entry]) -> dict[str, Entry]:
    return {entry.path: entry for entry in entries if entry.status in DIVERGENT}


def report(entries: list[Entry], against: str, reviews: dict[str, str]) -> None:
    # A reviewed verdict wins over the marker heuristic. The heuristic is deliberately broad -- a
    # false "windows" costs one human glance -- so it is a triage hint, and a review that read the
    # diff is the better answer. Without this, twelve files the review KEPT read as "refactor",
    # which the docstring defines as "revert to upstream's version": a label that invites exactly
    # the change the review decided against.
    def verdict(entry: Entry) -> str:
        return reviews.get(entry.path, entry.disposition)

    by_disposition: dict[str, list[Entry]] = {}
    for entry in entries:
        by_disposition.setdefault(verdict(entry), []).append(entry)

    counts = Counter(verdict(entry) for entry in entries)
    print(f"delta against {against}: {len(entries)} paths")
    for name in ("windows", "product", "refactor", "unknown", "port-added", "deleted"):
        if name in counts:
            lines = sum(e.size for e in by_disposition[name])
            print(f"  {name:11s} {counts[name]:4d} paths  {lines:6d} lines")
    for name in ("refactor", "unknown"):
        group = by_disposition.get(name)
        if group:
            print(f"\n=== {name} (review: {counts[name]}) ===")
            for entry in sorted(group, key=lambda e: -e.size):
                print(f"  {entry.added:5d}+ {entry.removed:5d}-  {entry.path}")
    unreviewed = [e for e in entries if e.path not in reviews and e.status in DIVERGENT]
    if unreviewed:
        print(f"\n{len(unreviewed)} divergent path(s) carry no reviewed verdict; the disposition "
              f"shown is the marker heuristic.")


def write_baseline(paths: dict[str, Entry], against: str) -> None:
    # Carry the review forward. `reason` is the field a human writes and `review` is the verdict a
    # review reached; clearing either on every re-record is how a baseline rots: the entry survives,
    # the justification does not.
    previous: dict[str, dict[str, str]] = {}
    if BASELINE.exists():
        previous = json.loads(BASELINE.read_text(encoding="utf-8")).get("paths", {})
    payload = {
        "against": against,
        "recorded": datetime.date.today().isoformat(),
        "note": (
            "Every upstream-owned path the port diverges from, and why. Filed by review. "
            "check_port_delta.py --check fails when a path is added here without a commit "
            "recording it, and equally when an entry no longer diverges -- a stale entry is "
            "worse than none, because it hides the next real change under an expected one."
        ),
        "paths": {
            path: {
                "status": entry.status,
                "disposition": entry.disposition,
                "review": previous.get(path, {}).get("review", ""),
                "reason": previous.get(path, {}).get("reason", ""),
            }
            for path, entry in sorted(paths.items())
        },
    }
    BASELINE.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        # newline="\n" is load-bearing: Path.write_text translates \n to the platform separator
        # by default, so this file lands as CRLF on Windows and check_text_encoding.py fails the
        # commit. A generated file must not let the checkout decide the bytes it contains.
        newline="\n",
    )
    print(f"wrote {BASELINE.relative_to(REPO)}: {len(paths)} entries")


def check(paths: dict[str, Entry], against: str) -> int:
    if not BASELINE.exists():
        print(f"  FAIL: {BASELINE.name} is missing, so there is nothing to ratchet against")
        return 1
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    recorded: dict[str, object] = payload.get("paths", {})

    # The baseline is a statement about one upstream reference. Comparing it against a different one
    # makes every file that reference moved read as the port's divergence -- on 2026-10-05 this
    # reported 695 paths where the truth was 541 -- so a mismatch is its own failure rather than a
    # confusing path count that invites a re-record.
    recorded_against = payload.get("against")
    if recorded_against != against:
        print(f"  FAIL: {BASELINE.name} is filed against {recorded_against!r}, "
              f"but this run compares {against!r}")
        print("    Re-record deliberately:")
        print(f"      python tools/release/check_port_delta.py --write-baseline --against {against}")
        return 1

    grew = sorted(set(paths) - set(recorded))
    stale = sorted(set(recorded) - set(paths))

    if grew:
        print("  FAIL: new divergence in an upstream-owned file, not recorded in the baseline:")
        for path in grew:
            entry = paths[path]
            print(f"    {entry.status}  {path}  ({entry.added}+ {entry.removed}-)")
        print("  Change it under a port-owned file instead, revert it, or record it with a reason:")
        print("    python tools/release/check_port_delta.py --write-baseline")
    if stale:
        print("  FAIL: baseline entries that no longer diverge -- trim them in this commit:")
        for path in stale:
            print(f"    {path}")
    if grew or stale:
        return 1
    print(f"  port delta unchanged: {len(paths)} upstream-owned paths, all recorded")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--against", default=DEFAULT_UPSTREAM, help="upstream ref to compare")
    parser.add_argument("--json", type=Path, default=None, help="also write the entries as JSON")
    parser.add_argument(
        "--path",
        metavar="FILE",
        default=None,
        help=(
            "answer one question offline, without comparing against upstream: is this path a recorded "
            "divergence? Exits 0 when it is not (edit it freely; a NEW file of the port's own needs no "
            "entry, because a file upstream does not have cannot diverge from it) and 1 when it is "
            "(editing it needs that entry's reason updated in the same commit). Run it BEFORE editing an "
            "upstream-owned file, so the requirement arrives as an answer rather than as a blocked commit"
        ),
    )
    parser.add_argument("--check", action="store_true", help="ratchet: fail on any change")
    parser.add_argument(
        "--write-baseline", action="store_true", help="re-record the baseline after a review"
    )
    args = parser.parse_args()

    if args.path is not None:
        recorded_paths = (json.loads(BASELINE.read_text(encoding="utf-8")).get("paths", {})
                          if BASELINE.exists() else {})
        entry = recorded_paths.get(args.path)
        if entry is None:
            print(f"  {args.path}: not a recorded divergence -- edit it freely if this port owns it.")
            print("    A NEW file of the port's own needs no baseline entry: the gate rejects one,")
            print("    because a file upstream does not have cannot diverge from it.")
            return 0
        print(f"  {args.path}: a recorded divergence from upstream. Editing it needs this entry's")
        print("    reason updated in the same commit.")
        for key in ("status", "disposition", "review"):
            print(f"    {key:12s} {entry.get(key, '-')}")
        print(f"    reason       {' '.join(str(entry.get('reason', '')).split())[:500]}")
        return 1

    entries = collect(args.against)
    paths = divergences(entries)

    if args.write_baseline:
        write_baseline(paths, args.against)
        return 0
    if args.check:
        return check(paths, args.against)

    recorded: dict[str, dict[str, str]] = {}
    if BASELINE.exists():
        recorded = json.loads(BASELINE.read_text(encoding="utf-8")).get("paths", {})
    reviews = {path: entry["review"] for path, entry in recorded.items() if entry.get("review")}

    report(entries, args.against, reviews)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                [{**e._asdict(), "disposition": e.disposition,
                  "review": reviews.get(e.path, "")} for e in entries],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",  # see write_baseline: the checkout must not decide the bytes
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
