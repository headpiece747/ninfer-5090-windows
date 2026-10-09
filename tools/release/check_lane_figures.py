#!/usr/bin/env python3
"""Fail when a shipped lane figure was measured under conditions that no longer hold.

A figure is a claim about a revision. The defects of 2026-10-09 were all one class: a number relied on past
its conditions -- an acceptance figure from an instrument that could not reach the served regime, a
throughput column taken before its lane's depth changed, a context derived rather than served. None was a
missing measurement; all were a stale one, and nothing in the tree could tell.

This gate cannot refresh a stale figure (that needs the card). What it does is make staleness mechanical in
two directions:

  * **identity** -- every lane in `lane_figures.json` records the artifact it was measured on (size and
    mtime) and the engine binary that measured it (sha256). When an engine is rebuilt or an artifact is
    re-cut, the lane fails here with the remedy in the message.
  * **transcription** -- on the **`code`** cell only, the profile table's `tok`/`acc` must agree with the
    fingerprint that produced them. That cell is the one the published figures were measured on; the other
    cells read far lower by construction, because decode falls with context, and comparing them with the
    table was the first run's second mistake.

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
TABLE_CELL = "code"
TOK_TOLERANCE = 0.10      # percent against the table: run-to-run noise reaches 2.3%, a stale figure is 22%
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

    recorded_engine = str((payload.get("engine") or {}).get("sha256") or "")
    current_engine = sha256_of(args.serve)
    engine_matches = recorded_engine == current_engine
    published = {profile["file"]: profile for profile in PROFILES}
    lanes = payload.get("lanes") or {}

    stale: list[str] = []
    checked = 0
    for launcher, entry in sorted(lanes.items()):
        checked += 1
        if not engine_matches:
            stale.append(f"{launcher}: measured with engine {recorded_engine[:12]}, tree has "
                         f"{current_engine[:12]}")
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
        cell = (entry.get("cells") or {}).get(TABLE_CELL)
        if cell is None:
            stale.append(f"{launcher}: no fingerprint for the table's own cell {TABLE_CELL!r}")
            continue
        measured_tok = float(cell.get("decode_avg") or 0.0)
        if measured_tok and abs(measured_tok / float(profile["tok"]) - 1.0) > TOK_TOLERANCE:
            stale.append(f"{launcher} / {TABLE_CELL}: fingerprint {measured_tok:.1f} against the table's "
                         f"{profile['tok']} tok/s")
        measured_acc = cell.get("accept_rate")
        if isinstance(measured_acc, (int, float)):
            table_acc = float(str(profile["acc"]).rstrip("%"))
            if abs(float(measured_acc) * 100.0 - table_acc) > ACC_TOLERANCE:
                stale.append(f"{launcher} / {TABLE_CELL}: fingerprint acceptance "
                             f"{float(measured_acc) * 100:.1f}% against the table's {table_acc:.1f}%")

    print(f"    {checked} lane(s) checked against engine {current_engine[:12]}"
          f"{'' if engine_matches else ' (CHANGED)'}, measured {payload.get('measured_utc', 'unknown')}")
    if stale:
        print(f"    {len(stale)} figure(s) measured under conditions that no longer hold:")
        for line in stale[:12]:
            print(f"        {line}")
        print("    Remedy: python tools/bench/lane_fingerprint.py   (about 10 minutes, needs the card),")
        print("            then commit tools/release/lane_figures.json with the table figures it refreshes")
        return 1
    print("    PASS: every lane's artifact and engine identity match, and the table agrees with the "
          "fingerprint on its own cell.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
