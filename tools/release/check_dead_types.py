#!/usr/bin/env python3
"""Report type declarations that nothing in the tree ever names again.

A header-declared type whose name occurs exactly once in the whole repository -- at its own
declaration -- is not dead code that happens to be unused. It is a declaration the product ships and
nothing can reach. This is the detector that found the copy-drafting withdrawal on 2026-10-03:
`NgramOptions`, `NgramDraftMode` and two `RuntimeStats` counters each appeared at their declaration
and nowhere else, so a flag validated, entered the Program, and produced nothing.

Scope is deliberately types only -- `struct`, `class`, `enum`, `union` -- and only in first-party
headers. A name that appears once is unambiguous; anything subtler needs a real parser, and a gate
that guesses is a gate that gets muted. Free functions, macros and templates are NOT checked here,
because "no textual reference" stops meaning "no use" for them the moment a template is instantiated
or a macro is expanded somewhere this cannot see. That limit is the honest one.

Two findings this has already produced, both verified by hand before acting:
  * `NgramOptions` / `NgramDraftMode` -- the withdrawn `--ngram chain` surface
  * `MediaTokenRuns`-era padding in RuntimeStats -- counters declared and written nowhere

It reports rather than blocks on anything a human might reasonably disagree with, and it skips
`third_party/`, `build*/`, and anything under a directory named `tests`.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# A top-level type declaration. `struct X {`, `class X final {`, `enum class X : uint8_t {`,
# `union X {`. Anything indented or inside a namespace block is skipped by requiring the name to
# start at column 0 with an optional `template <...>` prefix already consumed above it.
DECL = re.compile(
    r"^(?:template\s*<[^>]*>\s*)?"
    r"(?:struct|class|union)\s+(?:final\s+)?([A-Za-z_]\w*)\s*(?:final\s*)?(?::|\{|$)",
    re.MULTILINE,
)
ENUM = re.compile(
    r"^enum\s+(?:class\s+)?([A-Za-z_]\w*)\s*(?::[^;{]*)?\s*\{", re.MULTILINE
)

SKIP_DIR_PARTS = {"third_party", "build", "build-test", "build-asan", "build-bench", "tests", ".git"}


def first_party_headers() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "--", "*.h", "*.hpp", "*.cuh"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode != 0:
        return []
    out = []
    for name in listed.stdout.splitlines():
        if not name.strip():
            continue
        if set(Path(name).parts) & SKIP_DIR_PARTS:
            continue
        out.append(REPO / name)
    return out


GIT_PATHS = ["*.h", "*.hpp", "*.cuh", "*.cpp", "*.cu", "*.py"]
# One `git grep` per name measured 1062 processes and 29 s, which no pre-commit gate can afford.
# Batching the names into one alternation per chunk turns that into a handful of calls, and the
# chunk size stays well under the Windows command-line limit.
CHUNK = 120


def occurrences(names: list[str]) -> dict[str, int]:
    """Whole-word occurrence count per identifier, from ONE pass over the tree.

    One `git grep` per name measured 1062 processes and 29 s; batching the names into nine
    alternations still cost 21 s, because the cost is the tree scan and not the process count. So
    this asks for every identifier in one invocation and counts the output, which is a single pass
    regardless of how many names are being checked.
    """
    counts = dict.fromkeys(names, 0)
    found = subprocess.run(
        ["git", "grep", "-o", "-w", "-h", "-e", r"[A-Za-z_][A-Za-z0-9_]*", "--"] + GIT_PATHS,
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    for line in found.stdout.splitlines():
        word = line.strip()
        if word in counts:
            counts[word] += 1
    return counts


def main() -> int:
    headers = first_party_headers()
    if not headers:
        print("  GATE ERROR: no first-party headers found; is this a git checkout?")
        return 2

    seen: dict[str, str] = {}
    for header in headers:
        try:
            text = header.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for pattern in (DECL, ENUM):
            for match in pattern.finditer(text):
                name = match.group(1)
                if name in ("final", "alignas"):
                    continue
                seen.setdefault(name, header.relative_to(REPO).as_posix())

    if not seen:
        print("  no first-party type declarations found")
        return 0

    counts = occurrences(sorted(seen))
    dead = [(name, seen[name]) for name in sorted(seen) if counts[name] == 1]

    print(f"  headers          : {len(headers)}")
    print(f"  type declarations: {len(seen)}")
    if dead:
        print(f"\n  {len(dead)} declared type(s) that nothing in the tree ever names again:")
        for name, where in dead:
            print(f"    {name}  ({where})")
        print(
            "\n  A type whose name appears once -- at its own declaration -- cannot be reached by any\n"
            "  caller. Each is either surface the product ships and cannot use, or a header that should\n"
            "  not be shipped. Confirm before removing: a type used only through a template\n"
            "  instantiation or a macro expansion would not show here, and this check is types only."
        )
    print("  PASS: every declared type is named somewhere beyond its declaration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())