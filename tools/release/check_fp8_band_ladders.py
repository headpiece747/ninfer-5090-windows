#!/usr/bin/env python3
"""Fail if any FP8 A8 ladder selects a different schedule than it did at a pinned base commit.

Every FP8 A8 ladder is a chain of `if (tokens <= B) return launch<S>(...)`, and every partials-capacity
function sizes its buffer from the same boundaries. Naming the boundaries made the two read one constant
instead of two literals, which removed a class of drift. It introduced another, immediately and
invisibly: `n5120_k17408.cu` and `fp8_linear_add_a8.cu` had a band bound substituted with the wrong
named constant, producing `tokens > 512 && tokens <= 384` -- false for every token count -- so tokens
513-768 fell through to Bulk instead of Wide.

**The test suite could not see it.** Both tiles are numerically correct and differ only in speed, and no
oracle in this tree asserts *which* tile a token count selects. A green suite is evidence about values,
not about selection.

So this resolves every branch of every ladder to its numeric meaning -- constants substituted, C++
operators translated, `x.ne[1]` and `tokens` unified -- and requires the result to match the base
commit. It compares resolved condition STRINGS index-by-index rather than walking the branches as one
chain, because fp8_linear_add_a8.cu holds two ladders chosen by K and walking them as a chain stops in
the first, which under-covers exactly the file that needs it most.

It carries its own control. An earlier version of this check substituted `tokens` for `0` before
evaluating, so every condition read `0 <= 64`, the first branch always won, and it reported a tree with
two dead branches as identical to a correct one. Run with --self-test to confirm it still has the power
to fail before believing a PASS.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The commit just before the band refactor. Every ladder below must still select what it selected then.
DEFAULT_BASE = "037b7a2c"

# The commit that shipped the dead-clause regression, used by --self-test as the control.
CONTROL_REV = "4244b0d9"

FILES: tuple[str, ...] = (
    "src/ops/linear/fp8/shapes/n14336_k5120.cu",
    "src/ops/linear/fp8/shapes/n16384_k5120.cu",
    "src/ops/linear/fp8/shapes/n34816_k5120.cu",
    "src/ops/linear/fp8/shapes/n5120_k17408.cu",
    "src/ops/linear/fp8/shapes/n5120_k6144.cu",
    "src/ops/linear_add/fp8/fp8_linear_add_a8.cu",
    "src/ops/attn_input_proj/fp8/fp8_attn_input_a8.cu",
    "src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu",
    "src/ops/linear_swiglu/fp8/fp8_linear_swiglu_a8.cu",
)

CONST = re.compile(r"constexpr\s+std::int32_t\s+(kBand\w+)\s*=\s*(\d+)\s*;")
LADDER = re.compile(
    r"if\s*\((?P<cond>[^)]*(?:\([^)]*\))?[^)]*)\)\s*\n?\s*"
    r"(?:return\s+)?launch(?:\.template)?\s*(?:_fp8_a8\w*)?\s*"
    r"(?:<Geometry,\s*|\.template\s+operator\(\)\s*<)(?P<sched>[\w:<>,\s]+?)[>;]",
    re.S,
)
CAPACITY = re.compile(
    r"(?:partial_capacity_bytes|fp8_\w*_partial_capacity_bytes)\s*\([^)]*\)\s*\{(.*?)\n\}", re.S
)
BOUNDARY_TOKENS = (1, 64, 129, 192, 193, 257, 385, 513, 769)


def blob(revision: str, relative: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative}"], capture_output=True, check=True
    )
    return result.stdout.decode("utf-8")


def constants(text: str) -> dict[str, int]:
    return {name: int(value) for name, value in CONST.findall(text)}


def branches(text: str) -> list[tuple[str, str]]:
    return [
        (m.group("cond").strip(), " ".join(m.group("sched").split())) for m in LADDER.finditer(text)
    ]


def normalise(expr: str, consts: dict[str, int]) -> str:
    """Resolve a condition to its numeric meaning, so only a real change can show as a difference."""
    resolved = expr.replace("x.ne[1]", "tokens").replace("&&", " and ").replace("||", " or ")
    for name, value in sorted(consts.items(), key=lambda kv: -len(kv[0])):
        resolved = re.sub(rf"\b{name}\b", str(value), resolved)
    resolved = re.sub(r"\s+", " ", resolved).strip()
    return re.sub(r"\b(tokens|max_tokens)\b", "T", resolved)


def capacity_branch(text: str, tokens: int) -> int:
    """Which branch of the capacity function `tokens` takes; -1 when there is no such function."""
    body = CAPACITY.search(text)
    if not body:
        return -1
    consts = constants(text)
    taken = 0
    for line in body.group(1).split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        condition = re.match(r"if\s*\(([^)]*)\)\s*return", stripped)
        if condition:
            resolved = normalise(condition.group(1), consts).replace("T", str(tokens))
            try:
                if eval(resolved, {"__builtins__": {}}, {}):  # noqa: S307
                    return taken
            except Exception:  # noqa: BLE001
                return -2
        taken += 1
    return taken


def compare(base: str, new: str) -> list[str]:
    """Differences between two revisions of one file, dispatch-agnostically."""
    old_branches, new_branches = branches(base), branches(new)
    notes: list[str] = []
    if len(old_branches) != len(new_branches):
        return [f"branch count {len(old_branches)} -> {len(new_branches)}"]
    old_consts, new_consts = constants(base), constants(new)
    for index, ((old_cond, old_sched), (new_cond, new_sched)) in enumerate(
        zip(old_branches, new_branches, strict=True)
    ):
        if normalise(old_cond, old_consts) != normalise(new_cond, new_consts):
            notes.append(
                f"ladder[{index}] {normalise(old_cond, old_consts)!r} -> "
                f"{normalise(new_cond, new_consts)!r}"
            )
        if old_sched != new_sched:
            notes.append(f"ladder[{index}] selects {old_sched} -> {new_sched}")
    for tokens in BOUNDARY_TOKENS:
        if capacity_branch(base, tokens) != capacity_branch(new, tokens):
            notes.append(f"capacity branch differs at t={tokens}")
            break
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE, help="revision the ladders must still match")
    parser.add_argument("--self-test", action="store_true", help="prove this check can still fail")
    arguments = parser.parse_args()

    if arguments.self_test:
        caught = 0
        for relative in FILES:
            # Both arguments to compare() are file CONTENT. Passing arguments.base here -- the
            # revision NAME -- made branches() see the string "037b7a2c", find no branches at all,
            # and report every file as changed. The control then passed for entirely the wrong
            # reason, which is the same failure this gate exists to prevent.
            notes = compare(
                blob(arguments.base, relative), blob(CONTROL_REV, relative)
            )
            if notes:
                caught += 1
                print(f"  control caught {relative.split('/')[-1]}: {notes[0]}")
        if caught != 2:
            print(
                f"  FAIL: the control caught {caught} file(s). Exactly 2 is correct -- those are the"
                f" two ladders that commit {CONTROL_REV} shipped with a dead band clause. A different"
                f" count means this check has lost or gained the power to see that defect."
            )
            return 1
        print(
            f"  SELF-TEST PASSED: the control caught exactly the {caught} file(s) with a dead clause."
        )
        return 0

    failures = 0
    for relative in FILES:
        try:
            notes = compare(blob(arguments.base, relative), Path(relative).read_bytes().decode("utf-8"))
        except subprocess.CalledProcessError:
            print(f"  FAIL {relative}: base revision {arguments.base} has no such file")
            failures += 1
            continue
        if notes:
            failures += 1
            print(f"  FAIL {relative}")
            for note in notes[:4]:
                print(f"         {note}")
    print(f"  files checked: {len(FILES)}   with divergences from {arguments.base}: {failures}")
    if failures:
        print("  A ladder changed which schedule it selects, or its capacity function changed shape.")
        print("  Both tiles are numerically correct, so the test suite will not catch this. If the")
        print("  change is intended, say so here and re-pin --base; otherwise restore the branch.")
        return 1
    print(f"  PASS: every FP8 A8 ladder still selects what it selected at {arguments.base}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())