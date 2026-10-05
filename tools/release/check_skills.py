#!/usr/bin/env python3
"""Fail when a skill in this repository cannot be discovered or cannot be advertised.

A skill goes invisible two ways, and neither is loud:

  * **Undiscoverable.** OpenCode finds root-level ``*.md`` and files named exactly ``SKILL.md``
    at any depth. ``skills/foo/foo.md`` or ``skills/foo/notes/SKILL.md``-style mistakes are simply
    never read -- no error, no warning, the skill does not exist as far as the model is concerned.
  * **Undescribed.** A discovered skill with no ``description`` is registered but not advertised,
    so the model never learns it exists. The V2 skills page is explicit: "skills without one are
    not advertised".

Both are defects a commit can fix, so both fail here.

Two things are only *reported*, because both are legitimate configuration and a gate that guesses
gets muted:

  * a visibility flag (``disable-model-invocation: true``, ``metadata.opencode/autoinvoke: false``)
    hides a skill from the model while keeping it loadable by ID;
  * a duplicate ID, where a later source deliberately overrides an earlier one.

Scope, and why it is split. The project root is gated. The global root and the compatibility roots
are reported only: they are outside the repository, absent on a CI runner, and a commit cannot fix
them -- so failing on them would make this gate machine-dependent.

Usage
-----
    python tools/release/check_skills.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOME = Path.home()

# Gated: inside the repository, so a commit can fix what this reports.
PROJECT_ROOTS = [REPO / ".opencode" / "skills"]
# Reported only: outside the repository, and absent on CI.
OTHER_ROOTS = [
    HOME / ".claude" / "skills",
    HOME / ".agents" / "skills",
    HOME / ".config" / "opencode" / "skills",
    REPO / ".claude" / "skills",
    REPO / ".agents" / "skills",
]

FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
AUToinvoke = re.compile(r"opencode/autoinvoke\s*:\s*(\S+)")


def frontmatter(path: Path) -> dict[str, str]:
    """The fields this gate reads, without a YAML dependency."""
    match = FRONTMATTER.match(path.read_bytes().decode("utf-8", errors="replace"))
    if not match:
        return {}
    body = match.group(1)
    fields: dict[str, str] = {}
    for line in body.splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip().strip('"').strip("'")
    nested = AUToinvoke.search(body)
    if nested:
        fields["autoinvoke"] = nested.group(1).strip().strip('"').strip("'")
    return fields


def discover(root: Path) -> list[tuple[str, Path]]:
    """Root-level *.md, plus SKILL.md at any depth. The ID is the containing directory's name."""
    if not root.is_dir():
        return []
    found: list[tuple[str, Path]] = []
    for entry in sorted(root.iterdir()):
        if entry.is_file() and entry.suffix == ".md":
            found.append((entry.stem, entry))
        elif entry.is_dir():
            found.extend((p.parent.name, p) for p in sorted(entry.rglob("SKILL.md")))
    return found


def undiscoverable(root: Path) -> list[Path]:
    """Markdown that reads as a skill file but that OpenCode will never read.

    Deliberately narrow. Supporting files beside SKILL.md -- references/, notes, a README -- are
    normal and are loaded by the skill's own instructions, not discovered. The first version of
    this flagged all 25 of them in this repository and would have been red on every commit.

    What is a real mistake is a file named after its own directory, ``skills/foo/foo.md``, which
    looks like the skill's entry point and is silently not one: OpenCode reads root-level *.md and
    files named exactly SKILL.md, and nothing else.
    """
    if not root.is_dir():
        return []
    return sorted(
        p
        for p in root.rglob("*.md")
        if p.name != "SKILL.md" and p.stem == p.parent.name
    )


def hidden_by(meta: dict[str, str]) -> str:
    if str(meta.get("autoinvoke", "")).lower() == "false":
        return "metadata.opencode/autoinvoke: false"
    if str(meta.get("disable-model-invocation", "")).lower() == "true":
        return "disable-model-invocation: true"
    return ""


def main() -> int:
    failures: list[str] = []
    notes: list[str] = []
    resolved: dict[str, str] = {}

    for root in PROJECT_ROOTS:
        for path in undiscoverable(root):
            failures.append(
                f"{path.relative_to(REPO)}: not a root-level *.md and not named SKILL.md, "
                f"so OpenCode never reads it"
            )
        entries = discover(root)
        print(f"  {root.relative_to(REPO)}: {len(entries)} skill(s)")
        for skill_id, path in entries:
            meta = frontmatter(path)
            if not meta.get("description"):
                failures.append(
                    f"{path.relative_to(REPO)}: no description, so the model is never told "
                    f"this skill exists"
                )
            flag = hidden_by(meta)
            if flag:
                notes.append(f"{skill_id}: hidden from the model by {flag}")
            resolved[skill_id] = path.relative_to(REPO).as_posix()

    for root in OTHER_ROOTS:
        if not root.is_dir():
            continue
        entries = discover(root)
        print(f"  {root}: {len(entries)} skill(s) -- outside the repository, reported only")
        for skill_id, path in entries:
            meta = frontmatter(path)
            if not meta.get("description"):
                notes.append(f"{skill_id} ({root}): no description, so it is not advertised")
            flag = hidden_by(meta)
            if flag:
                notes.append(f"{skill_id} ({root}): hidden from the model by {flag}")
            if skill_id in resolved:
                notes.append(f"{skill_id}: {path} is overridden by {resolved[skill_id]}")
            resolved.setdefault(skill_id, str(path))

    print(f"\n  distinct skill IDs across every source: {len(resolved)}")
    for note in notes:
        print(f"  note: {note}")

    if failures:
        print(f"\n  FAIL: {len(failures)} skill(s) cannot be reached by the model:")
        for failure in failures:
            print(f"    {failure}")
        return 1
    print("\n  PASS: every skill in this repository is discoverable and advertised.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
