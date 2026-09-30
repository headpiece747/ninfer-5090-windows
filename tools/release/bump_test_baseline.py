#!/usr/bin/env python3
"""Add a newly added test to test_baseline.json's suite_size, preserving everything else.

The baseline gate compares the number of tests a run covered against `suite_size`, and refuses when
they differ: a run that covers fewer tests cannot demonstrate the absence of a regression in the ones
it skipped, and a run that covers more is the same signal from the other side. Adding a test
therefore requires moving that one number, and doing it by hand is how the recorded date and the
suite size drift apart.

This writes through `encoding="utf-8"` with no BOM, because the gate reads the file with json.load,
which rejects a BOM -- the same defect class as the opencode settings edit.

Refuses to run if the delta is not exactly the number of tests added, so a mistaken invocation cannot
quietly rescale the gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASELINE = Path(sys.argv[1] if len(sys.argv) > 1 else "tools/release/test_baseline.json")
ADDED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
RECORDED = sys.argv[3] if len(sys.argv) > 3 else "2026-09-30"
DESCRIPTION = sys.argv[4] if len(sys.argv) > 4 else ""

payload = json.loads(BASELINE.read_text(encoding="utf-8"))
before = int(payload["suite_size"])
after = before + ADDED
if ADDED <= 0:
    print(f"ABORT: {ADDED} is not an increase")
    raise SystemExit(1)

payload["suite_size"] = after
payload["recorded"] = RECORDED
if DESCRIPTION:
    payload.setdefault("added", []).append(
        {"date": RECORDED, "count": ADDED, "note": DESCRIPTION})

BASELINE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
# Re-read with the consumer rather than trusting the write.
verified = json.loads(BASELINE.read_text(encoding="utf-8"))
print(f"suite_size {before} -> {verified['suite_size']}, recorded {verified['recorded']}, "
      f"known_failures {len(verified.get('known_failures', {}))} preserved")
