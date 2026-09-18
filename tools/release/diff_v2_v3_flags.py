#!/usr/bin/env python3
"""Diff the v2 launcher flags against the v3 ones.

v2 was tuned over a long period and its .bat files are the record of that. Extracting both
flag sets and comparing them shows what v3 dropped, what it changed, and what v2 knew that
v3 does not -- which is more reliable than reading six files and remembering.
"""
from __future__ import annotations

import re
from pathlib import Path

V2_DIRS = [Path(r"C:\AI\ninfer-5090-windows"), Path(r"C:\AI\ninfer-quasar-5090")]
V3 = Path(r"C:\AI\ninfer-v3-windows")

FLAG = re.compile(r"(--[a-z][a-z0-9-]*)(?:[ =]+([^\s^\"]+))?")


def collect(paths: list[Path]) -> dict[str, set[str]]:
    """Map flag -> set of values seen, across the given files."""
    found: dict[str, set[str]] = {}
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        # launcher_env data lives in a set "NAME=..." line, so include it
        for match in FLAG.finditer(text):
            flag, value = match.group(1), match.group(2)
            bucket = found.setdefault(flag, set())
            if value and not value.startswith("%") and not value.startswith("$"):
                bucket.add(value.rstrip("^").strip('"'))
    return found


def main() -> int:
    v2_files = [p for d in V2_DIRS if d.exists() for p in sorted(d.glob("*.bat"))]
    v3_files = sorted(V3.glob("start_*_v3_*.bat")) + [V3 / "launcher_env.bat"]

    v2 = collect(v2_files)
    v3 = collect(v3_files)

    print(f"   v2 files: {len(v2_files)}   v3 files: {len(v3_files)}")
    print()
    print("   === in v2 but NOT in v3 ===")
    for flag in sorted(set(v2) - set(v3)):
        print(f"   {flag:<38} v2 values: {sorted(v2[flag])}")
    print()
    print("   === in v3 but NOT in v2 ===")
    for flag in sorted(set(v3) - set(v2)):
        print(f"   {flag:<38} v3 values: {sorted(v3[flag])}")
    print()
    print("   === present in both, different values ===")
    for flag in sorted(set(v2) & set(v3)):
        if v2[flag] != v3[flag]:
            print(f"   {flag:<38} v2 {sorted(v2[flag])}  ->  v3 {sorted(v3[flag])}")
    print()
    print("   === identical in both ===")
    same = [f for f in sorted(set(v2) & set(v3)) if v2[f] == v3[f]]
    print("   " + ", ".join(same) if same else "   (none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
