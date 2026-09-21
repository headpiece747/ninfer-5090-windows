#!/usr/bin/env python3
"""Check that every relative link and heading anchor in the documentation resolves.

Two links in docs/maintainer/qwen3.8-27b-artifact.md pointed at qwen3.6-27b-model.md and
qwen3.8-27b-dflash2.md, both of which 9b884034 ("consolidate references for the v3 architecture")
deleted and replaced with qwen3_5-model.md and dflash.md. They stayed dead until a release was
about to carry them, because a rename leaves behind a link that still reads correctly and points
at nothing.

Anchors fail the same way and are resolved here too. Relabelling upstream's "## Quick start" as
"## Quick start (building the engine on Linux)" in 75eef593 changed its slug, and four documents
still linked ../../README.md#quick-start -- invisible to a check that only stats the file, which is
why the heading is resolved rather than assumed.

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
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")
# GitHub keeps word characters, spaces and hyphens when it derives a slug, and drops the rest.
SLUG_DROP = re.compile(r"[^\w\s-]", re.UNICODE)


def documents() -> list[Path]:
    candidates = [REPO / name for name in SHIPPED] + sorted((REPO / "docs").rglob("*.md"))
    return [path for path in candidates if path.exists()]


def slugs(path: Path) -> set[str]:
    """The slugs GitHub derives from a file's headings, including its -1 suffixes for repeats."""
    counts: dict[str, int] = {}
    result: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = HEADING.match(line)
        if not match:
            continue
        slug = SLUG_DROP.sub("", match.group(1)).strip().lower().replace(" ", "-")
        if not slug:
            continue
        seen = counts.get(slug, 0)
        counts[slug] = seen + 1
        result.add(slug if seen == 0 else f"{slug}-{seen}")
    return result


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
    links = 0
    anchors = 0
    external: set[str] = set()
    broken: list[str] = []
    slug_cache: dict[Path, set[str]] = {}

    for path in found:
        relative = path.relative_to(REPO).as_posix()
        for number, target in targets(path.read_text(encoding="utf-8", errors="replace")):
            target = target.strip()
            if not target or target.startswith("<"):
                continue
            if EXTERNAL.match(target):
                external.add(target.split("#")[0])
                continue
            if any(character.isspace() or character in "()[]" for character in target):
                continue
            file_part, _, fragment = target.partition("#")
            file_part = file_part.strip()
            destination = (path.parent / file_part).resolve() if file_part else path
            if file_part:
                links += 1
                if not destination.exists():
                    broken.append(f"{relative}:{number} -> {target} (no such file)")
                    continue
            if fragment:
                if destination.suffix != ".md":
                    continue
                anchors += 1
                if destination not in slug_cache:
                    slug_cache[destination] = slugs(destination)
                if fragment not in slug_cache[destination]:
                    broken.append(f"{relative}:{number} -> {target} (no such heading)")

    print(f"  documents      : {len(found)}")
    print(f"  relative links : {links}")
    print(f"  heading anchors: {anchors}")
    print(f"  external URLs  : {len(external)} (not fetched)")
    for item in broken:
        print(f"    DEAD  {item}")
    if broken:
        print("  FAIL: a link or an anchor does not resolve.")
        return 1
    print("  PASS: every relative link and heading anchor resolves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
