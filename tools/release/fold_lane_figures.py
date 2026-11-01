"""Fold the fingerprint's own record into profiles.py's table -- dry run first.

The gate compares the table's `tok`/`acc` (the `code` cell) against the fingerprint, so the numbers must
come from tools/release/lane_figures.json, which the measurement wrote -- not from a hand-copy of a log
line, and not from the drift report, which only prints the drifted ones. This is the fold step of a lane
change (field -> launcher -> fingerprint -> fold -> gates): the table is what the launchers render from,
so it is folded first, and the docs follow (fold_doc_figures.py).

Dry by default: it prints exactly what it would change. Run with --apply to write. Idempotent: a lane
whose pair already matches is skipped, so a second run reports nothing to do. The result is verified by
check_lane_figures.py and check_profile_consistency.py, which are the owning gates for this edit --
that is the fold's verification, which is why it needs no test of its own.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIGURES = REPO / "tools" / "release" / "lane_figures.json"
PROFILES = REPO / "tools" / "release" / "profiles.py"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    record = json.loads(FIGURES.read_text(encoding="utf-8"))
    lanes = record.get("lanes", record)

    text = PROFILES.read_text(encoding="utf-8")
    changed = []
    for name, entry in lanes.items():
        cells = entry.get("cells", entry)
        code = cells.get("code") if isinstance(cells, dict) else None
        if not isinstance(code, dict):
            print(f"  {name}: no 'code' cell in the record (keys: {list(cells)[:6]})")
            continue
        tok = code.get("decode_avg")
        acc = code.get("accept_rate")
        if tok is None or acc is None:
            print(f"  {name}: code cell keys = {list(code)[:8]} (tok/accept not found)")
            continue
        # this lane's entry in profiles.py: its own file=, then the tok=/acc= pair
        pattern = re.compile(rf'(file="{re.escape(name)}",.*?)(tok=)([0-9.]+)(,\s*acc=")([0-9.]+%?)(")',
                             re.S)
        match = pattern.search(text)
        if not match:
            print(f"  {name}: no tok=/acc= pair matched in profiles.py")
            continue
        new_tok = f"{float(tok):.1f}"
        new_acc = f"{float(acc) * 100:.1f}%" if float(acc) <= 1.0 else f"{float(acc):.1f}%"
        if match.group(3) == new_tok and match.group(5) == new_acc:
            continue
        changed.append((name, match.group(3), new_tok, match.group(5), new_acc))
        text = text[:match.start(3)] + new_tok + text[match.end(3):match.start(5)] + new_acc + \
            text[match.end(5):]

    # The table's figures now come from this record, so the table's engine is this record's engine. Without
    # the rewrite, a fold leaves `check_lane_figures.py` certifying the record against a table that still
    # claims the engine it replaced.
    engine = str((record.get("engine") or {}).get("sha256") or "")
    engine_match = re.search(r'MEASURED_ENGINE = "([0-9a-f]*)"', text)
    engine_prev = engine_match.group(1) if engine_match is not None else ""
    engine_changed = bool(engine) and engine_match is not None and engine_prev != engine
    if engine_changed and engine_match is not None:
        text = text[:engine_match.start(1)] + engine + text[engine_match.end(1):]

    print(f"\nfigures to fold: {len(changed)}")
    for name, old_tok, new_tok, old_acc, new_acc in changed:
        print(f"  {name}: tok {old_tok} -> {new_tok}   acc {old_acc} -> {new_acc}")

    if engine_changed:
        print(f"  MEASURED_ENGINE: {engine_prev[:12]} -> {engine[:12]}")

    if args.apply and (changed or engine_changed):
        PROFILES.write_text(text, encoding="utf-8", newline="\n")
        print(f"\nwrote {PROFILES} ({len(changed)} lane(s) updated)")
    elif changed or engine_changed:
        print("\ndry run: nothing written (pass --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
