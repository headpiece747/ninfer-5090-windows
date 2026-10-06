#!/usr/bin/env python3
"""Check that every `file.ext:LINE` citation in the tracked Markdown still resolves.

This repository cites code by `path:line` constantly -- in ADRs, in maintainer notes, in
`AGENTS.md`, and in the comments inside tools and benches. That convention is what makes a
claim checkable, and it is also what rots: a line reference is a promise that the reader can
open the file and find the thing named. Two such citations were found stale in one session on
2026-10-03 (`check_shared_prefix_reuse.py:116` pointing at `native_render.cpp:550` -- a file since
withdrawn with ADR-0012 -- for code that had moved to `:605`, and ADR-0012's `native_render.cpp:629-631`
after that file shifted again), each of which sent a reader to the wrong place without any signal.

The check is deliberately mechanical and deliberately narrow. It answers one question -- does
the cited file exist, and is the cited line inside it -- because that is the class a script can
answer without a model and without a false positive. It does NOT decide whether the line says
what the document claims; that needs judgement and belongs to a review, not to a pre-commit
gate. A gate that guesses is a gate that cries wolf, and a gate that cries wolf is muted.

Deliberately NOT here, for the same reason `check_test_mutation.py` is not: a reference that
cannot be resolved mechanically is not evidence of drift. Three shapes are skipped rather than
guessed at:
  * a path with no extension, or one that is plainly prose rather than a source file
  * a line number that is a range spanning a file (`foo.cpp:10-20` checks the end of the range)
  * a citation inside a fenced code block, which is illustrative rather than a promise

One judgement call worth naming: a citation into a DIRECTORY, or into a file that exists but is
untracked, is reported. An untracked target means the document points at something the checkout
does not contain, which is the same reader-facing failure as a deleted file.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# A citation is a path that looks like a source file, a colon, and a line number. The extension
# list is what keeps prose ("the ratio 3:1") and durations ("kv-capacity 200000:30") out; it is
# the set of extensions this repository actually writes and cites.
CITATION = re.compile(
    r"(?<![\w/])((?:[\w.\-]+/)*[\w.\-]+\.(?:cpp|h|hpp|cu|cuh|py|cmd|ps1|json|yaml|yml|toml|md|jinja|cmake|txt))"
    r":(\d+)(?:-(\d+))?\b"
)

# Fenced blocks are illustrative. A path inside one is not a promise about the tree.
FENCE = re.compile(r"^\s*(?:```|~~~)")

# Extensions worth resolving. Anything else is prose that happens to contain a colon.
SOURCE_EXTENSIONS = {
    ".cpp", ".h", ".hpp", ".cu", ".cuh", ".py", ".cmd", ".ps1",
    ".json", ".yaml", ".yml", ".toml", ".cmake", ".jinja",
}

# Where a tracked Markdown file may legitimately cite something outside the repository.
EXTERNAL_PREFIXES = ("http://", "https://")


def tracked_files() -> list[str]:
    """Every tracked file, as repository-relative posix paths."""
    import subprocess

    listed = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=False
    )
    if listed.returncode != 0:
        return []
    return [line for line in listed.stdout.splitlines() if line.strip()]


def tracked_markdown(files: list[str]) -> list[Path]:
    """Tracked Markdown only.

    A gate reads tracked files so its verdict describes this commit rather than whatever
    happened to be in the working tree -- the same reasoning that moved the perturbation gate
    out of this hook. Untracked scratch notes are not promises anyone else can check.
    """
    return [REPO / name for name in files if name.endswith(".md")]


def line_counts(index: dict[str, list[str]]) -> dict[str, int]:
    """Lines per tracked file, computed once. A file that cannot be read is treated as absent
    from the index rather than as a failure: a binary or non-UTF-8 file is not something this
    gate can reason about, and guessing would be the crying-wolf failure mode."""
    counts: dict[str, int] = {}
    for name in index:
        try:
            counts[name] = len((REPO / name).read_text(encoding="utf-8").splitlines())
        except (OSError, UnicodeDecodeError):
            continue
    return counts


def main() -> int:
    files = tracked_files()
    if not files:
        print("  GATE ERROR: no tracked files found; is this a git checkout?")
        return 2

    # Citations name a bare filename far more often than a path -- `resource_manager.h:384` is
    # how this repository writes, and how a reader resolves it. So a bare name is matched
    # against every tracked file with that basename, exactly as a reader's search would.
    index: dict[str, list[str]] = {}
    for name in files:
        index.setdefault(name.rsplit("/", 1)[-1], []).append(name)
    counts = line_counts({name: files for name in files})

    documents = tracked_markdown(files)
    errors: list[str] = []
    notices: list[str] = []
    citations = 0

    for document in documents:
        if not document.is_file():
            continue
        # `docs/research/` holds notes on OTHER projects -- ModelOpt, TensorRT-LLM, vLLM,
        # llama.cpp -- so its citations name files this repository has never contained and cannot
        # be expected to. Checking them would report every line as broken, which is the crying-wolf
        # failure this gate exists to avoid. A citation there is a reference to someone else's
        # source, not a promise about this tree.
        if "docs/research/" in document.relative_to(REPO).as_posix():
            continue
        try:
            text = document.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        in_fence = False
        for number, line in enumerate(text.splitlines(), start=1):
            if FENCE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            for match in CITATION.finditer(line):
                path_text, first, last = match.group(1), int(match.group(2)), match.group(3)
                if Path(path_text).suffix not in SOURCE_EXTENSIONS:
                    continue
                citations += 1
                where = f"{document.relative_to(REPO).as_posix()}:{number}"
                reach = int(last) if last else first

                # A path-qualified citation is held to that path. A bare name is held to
                # "some tracked file of this name is at least that long" -- which is the claim a
                # reader is really relying on when they search for it.
                if "/" in path_text:
                    if path_text not in counts:
                        notices.append(
                            f"  {where}: `{path_text}` is not a tracked file (external or removed)"
                        )
                    elif reach > counts[path_text]:
                        errors.append(
                            f"  {where}: cites `{path_text}:{first}` but it has "
                            f"{counts[path_text]} lines"
                        )
                    continue

                candidates = index.get(path_text, [])
                if not candidates:
                    notices.append(
                        f"  {where}: `{path_text}` matches no tracked file (external or removed)"
                    )
                elif not any(counts.get(c, 0) >= reach for c in candidates):
                    longest = max((counts.get(c, 0), c) for c in candidates)
                    errors.append(
                        f"  {where}: cites `{path_text}:{first}` but the longest tracked file of "
                        f"that name is {longest[1]} at {longest[0]} lines"
                    )

    print(f"  documents      : {len(documents)} (docs/research/ skipped: external sources)")
    print(f"  citations      : {citations}")
    if notices:
        # Not blocking, and deliberately so. "No tracked file has that name" is genuinely
        # ambiguous: `docs/research/` cites ModelOpt, TensorRT-LLM and CUDA by design, and this
        # repository cannot tell an external reference from a deleted one. A gate that blocks on
        # that distinction blocks on a guess, and a gate that guesses is muted within a week.
        distinct = sorted({n.split("`")[1] for n in notices})
        print(f"\n  {len(notices)} citation(s) name no tracked file -- reported, not blocking:")
        print(f"    {len(distinct)} distinct name(s): " + ", ".join(distinct[:12]))
        if len(distinct) > 12:
            print(f"    ... and {len(distinct) - 12} more")
    if errors:
        print(f"\n  {len(errors)} citation(s) point PAST THE END of a file that does exist:")
        for finding in errors:
            print(finding)
        print(
            "\n  This is the one class that cannot be an external reference: the file is tracked and\n"
            "  the line is not in it, so a reader following the citation lands nowhere. Fix the\n"
            "  citation or the file. Do not widen this gate to cover the notices above."
        )
        return 1
    print("  PASS: every resolvable file:line citation in the tracked Markdown is in range.")
    return 0


if __name__ == "__main__":
    sys.exit(main())