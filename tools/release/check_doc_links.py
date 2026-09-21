#!/usr/bin/env python3
"""Check that every relative link in the documentation resolves.

Two links in docs/maintainer/qwen3.8-27b-artifact.md pointed at qwen3.6-27b-model.md and
qwen3.8-27b-dflash2.md, both of which 9b884034 ("consolidate references for the v3 architecture")
deleted and replaced with qwen3_5-model.md and dflash.md. They stayed dead until a release was
about to carry them, because a rename leaves behind a link that still reads correctly and points
at nothing. package_release.py runs this in its pre-flight so a published archive cannot carry one.

Scope is the shipped documentation (README.md, RELEASE_NOTES.md, NOTICE, LICENSE) plus every
Markdown file under docs/. External URLs are counted and never fetched: this checks the
repository's own references, and packaging must not depend on the network.

A bracketed math subscript followed by a parenthesis -- W_down[e](SiLU(...)) -- reads as a link and
is not one: a link destination cannot contain whitespace or an unbalanced bracket, so a target
still holding one of those characters came from prose rather than from a link.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHIPPED = ["README.md", "RELEASE_NOTES.md", "NOTICE", "LICENSE"]

INLINE = re.compile(r"\[[^\]]*\]\(\s*([^)\s]+)")
REFERENCE = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)")
EXTERNAL = re.compile(r"^[a-z][a-z0-9+.-]*://|^mailto:")


def documents() -> list[Path]:
    candidates = [REPO / name for name in SHIPPED] + sorted((REPO / "docs").rglob("*.md"))
    return [path for path in candidates if path.exists()]


def targets(text: str):
    """Yield (line number, target) for every inline and reference-style link."""
    for number, line in enumerate(text.splitlines(), start=1):
        for match in INLINE.finditer(line):
            yield number, match.group(1)
        reference = REFERENCE.match(line)
        if reference:
            yield number, reference.group(1)


def main() -> int:
    found = documents()
    checked = 0
    external: set[str] = set()
    broken: list[str] = []

    for path in found:
        relative = path.relative_to(REPO).as_posix()
        for number, target in targets(path.read_text(encoding="utf-8", errors="replace")):
            target = target.strip()
            if not target or target.startswith(("#", "<")):
                continue
            if EXTERNAL.match(target):
                external.add(target.split("#")[0])
                continue
            if any(character.isspace() or character in "()[]" for character in target):
                continue
            file_part = target.split("#")[0].strip()
            if not file_part:
                continue
            checked += 1
            if not (path.parent / file_part).resolve().exists():
                broken.append(f"{relative}:{number} -> {target}")

    print(f"  documents      : {len(found)}")
    print(f"  relative links : {checked}")
    print(f"  external URLs  : {len(external)} (not fetched)")
    for item in broken:
        print(f"    DEAD  {item}")
    if broken:
        print("  FAIL: a relative link does not resolve.")
        return 1
    print("  PASS: every relative link resolves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
