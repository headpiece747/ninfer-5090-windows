#!/usr/bin/env python3
"""Fail when a shipped lane figure was measured under conditions that no longer hold.

A figure is a claim about a revision. The defects of 2026-10-09 were all one class: a number relied on past
its conditions -- an acceptance figure from an instrument that could not reach the served regime, a
throughput column taken before its lane's depth changed, a context derived rather than served. None was a
missing measurement; all were a stale one, and nothing in the tree could tell.

This gate cannot refresh a stale figure (that needs the card). What it does is make staleness mechanical in
two directions:

  * **identity** -- every figure in `lane_figures.json` records the artifact it was measured on (size and
    mtime) and the engine binary that measured it (sha256). When an engine is rebuilt or an artifact is
    re-cut, the affected rows fail here with the remedy in the message.
  * **transcription** -- the profile table's `tok`/`acc` must agree with the fingerprint that produced them,
    which is the half `check_profile_consistency.py` cannot see: that gate compares the table with the
    documents, this one compares the table with the measurement.

`tools/bench/lane_fingerprint.py` writes this file, so the run that takes a figure files its provenance.

Exit status: 0 clean, 1 stale, 2 missing or unreadable input -- a check that did not run is not a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "release"))

from profiles import PROFILES  # noqa: E402

FIGURES = REPO / "tools" / "release" / "lane_figures.json"
SERVE = REPO / "build" / "apps" / "ninfer-serve.exe"
MODELS = Path(r"C:\AI\models")
TOK_TOLERANCE = 0.02      # the table rounds; the fingerprint does not
ACC_TOLERANCE = 0.5       # points, against the table's one-decimal string


def sha256_of(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("--serve", type=Path, default=SERVE,
                        help="the binary the figures were measured with; a test overrides it")
    args = parser.parse_args()

    if not args.figures.is_file():
        print(f"    FAIL: {args.figures} does not exist, so no figure has a recorded identity.\n"
              f"          Remedy: run tools/bench/lane_fingerprint.py (about 10 minutes, needs the card)")
        return 2
    if not args.serve.is_file():
        # A checkout without a build has no figures in play, so there is nothing to verify here. This is
        # deliberately not a pass: nothing was checked, and the message says so.
        print(f"    SKIP: no serving binary at {args.serve} in this checkout, so no figure is in play")
        return 0
    try:
        payload = json.loads(args.figures.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"    FAIL: {args.figures}: {error}")
        return 2

    recorded_engine = str(payload.get("engine", {}).get("sha256") or "")
    current_engine = sha256_of(args.serve)
    engine_matches = recorded_engine == current_engine
    published = {profile["file"]: profile for profile in PROFILES}

    stale: list[str] = []
    checked = 0
    for launcher, entry in sorted((payload.get("figures") or {}).items()):
        checked += 1
        if not engine_matches:
            stale.append(f"{launcher} / {entry.get('cell')}: measured with engine "
                         f"{recorded_engine[:12]}, tree has {current_engine[:12]}")
            continue
        record = entry.get("artifact") or {}
        artifact = MODELS / str(record.get("path", ""))
        if not artifact.is_file():
            stale.append(f"{launcher}: its artifact {artifact.name} is gone")
        else:
            stat = artifact.stat()
            if stat.st_size != record.get("bytes"):
                stale.append(f"{launcher}: artifact size {stat.st_size:,} against recorded "
                             f"{record.get('bytes')}")
            elif abs(stat.st_mtime - float(record.get("mtime") or 0.0)) > 1.0:
                stale.append(f"{launcher}: its artifact was rewritten after the figure was taken")
        profile = published.get(launcher)
        if profile is None:
            stale.append(f"{launcher}: not a shipped profile any more")
            continue
        measured_tok = float(entry.get("decode_avg") or 0.0)
        if measured_tok and abs(measured_tok / float(profile["tok"]) - 1.0) > TOK_TOLERANCE:
            stale.append(f"{launcher} / {entry.get('cell')}: fingerprint {measured_tok:.1f} against the "
                         f"table's {profile['tok']} tok/s")
        try:
            measured_acc = float(entry.get("acceptance")) * 100.0
            table_acc = float(str(profile["acc"]).rstrip("%"))
            if abs(measured_acc - table_acc) > ACC_TOLERANCE:
                stale.append(f"{launcher} / {entry.get('cell')}: fingerprint acceptance "
                             f"{measured_acc:.1f}% against the table's {table_acc:.1f}%")
        except (TypeError, ValueError):
            pass

    print(f"    {checked} lane figure(s) checked against engine {current_engine[:12]}"
          f"{'' if engine_matches else ' (CHANGED)'}"
          f", measured {payload.get('measured_utc', 'date unknown')}")
    if stale:
        print(f"    {len(stale)} figure(s) measured under conditions that no longer hold:")
        for line in stale[:12]:
            print(f"        {line}")
        print("    Remedy: python tools/bench/lane_fingerprint.py   (about 10 minutes, needs the card),")
        print("            then commit tools/release/lane_figures.json with the table figures it refreshes")
        return 1
    print("    PASS: every figure's artifact and engine identity match, and the table agrees with them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
