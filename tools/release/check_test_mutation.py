#!/usr/bin/env python3
"""Release gate: refuse a test that passes because nothing can make it fail.

A green Op test is not evidence. The evidence is that the suite goes red when the code under test
is perturbed, and that proof was being produced by hand -- twice, for one Op, in a session -- which
means it stops happening the moment someone forgets. This gate makes it rerunnable.

The rule it enforces, from AGENTS.md: "A test result is only evidence if the setup could have passed
for the right reason." An assertion that cannot fail is worse than no assertion, and a selection
routine is exactly where that hides: a wrong tie-break or a reversed sort returns a well-formed,
plausible answer rather than crashing, so nothing but a deliberate perturbation distinguishes a
correct kernel from a subtly wrong one.

Each mutation costs one test invocation and no rebuild. The seam is compiled only where
BUILD_TESTING is on -- true in build-test, false in the apps tree -- so a shipping binary cannot
reach it. See the define in src/ops/CMakeLists.txt and the guarded branches in the Op kernels.

Two directions are checked, and the second is the one that makes the first mean anything:

  * the clean run must PASS
  * every declared mutation must make the suite FAIL

A suite that is red to begin with satisfies every mutation assertion trivially, so the clean run is
the control. A declared mutation that is not implemented in the kernel leaves the mutated run green
and fails here, which is a real finding: the manifest is claiming coverage that does not exist.

Usage:
    python check_test_mutation.py                    # gate every declared op
    python check_test_mutation.py --op topk_logprobs # one op
    python check_test_mutation.py --list             # show the manifest, run nothing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

MANIFEST = Path("tests/ops/mutations.json")
CTEST = Path(
    "C:/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions"
    "/Microsoft/CMake/CMake/bin/ctest.exe"
)
SCHEMA = 1
# ctest's own vocabulary, kept local so a typo in the manifest is a gate failure rather than a
# silently empty selection that exits 0 and reads as a pass.
SKIP_RETURN_CODE = 77


@dataclass(frozen=True)
class Mutation:
    id: int
    site: str
    why: str


@dataclass(frozen=True)
class Op:
    name: str
    env: str
    test: str
    mutations: tuple[Mutation, ...]


class ManifestError(RuntimeError):
    """The manifest is wrong. Always fatal: a gate that cannot read its own input must not pass."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def load(path: Path) -> dict[str, Op]:
    _require(path.is_file(), f"{path} not found")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ManifestError(f"{path} is not valid JSON: {error}") from error

    _require(raw.get("schema") == SCHEMA, f"{path}: expected schema {SCHEMA}, got {raw.get('schema')!r}")

    ops: dict[str, Op] = {}
    for name, entry in raw.get("ops", {}).items():
        _require(isinstance(entry, dict), f"{path}: ops.{name} must be an object")
        test = entry.get("test")
        env = entry.get("env")
        _require(isinstance(test, str) and bool(test),
                 f"{path}: ops.{name}.test must be a non-empty string")
        _require(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", test) is not None,
                 f"{path}: ops.{name}.test {test!r} is not a bare test name")
        _require(isinstance(env, str) and re.fullmatch(r"[A-Z][A-Z0-9_]*", env) is not None,
                 f"{path}: ops.{name}.env must be an UPPER_SNAKE name, got {env!r}")

        declared = entry.get("mutations")
        _require(isinstance(declared, list) and bool(declared),
                 f"{path}: ops.{name}.mutations must be a non-empty list -- an op with no "
                 "perturbation coverage should be absent, not present and empty")

        mutations: list[Mutation] = []
        seen: set[int] = set()
        for item in declared:
            _require(isinstance(item, dict), f"{path}: ops.{name}.mutations entries must be objects")
            identifier = item.get("id")
            _require(isinstance(identifier, int) and identifier > 0,
                     f"{path}: ops.{name} mutation id must be a positive int, got {identifier!r}")
            _require(identifier not in seen,
                     f"{path}: ops.{name} declares mutation id {identifier} twice")
            seen.add(identifier)
            site = item.get("site")
            why = item.get("why")
            # Stripped, not just truth-tested: a whitespace-only string is truthy, so "  " would
            # satisfy a bare `bool(why)` and record a mutation with no recorded purpose -- the exact
            # decoration this rejects. Caught by the gate's own test rather than by reading.
            _require(isinstance(site, str) and bool(site.strip()),
                     f"{path}: ops.{name} mutation {identifier} needs a site")
            _require(isinstance(why, str) and bool(why.strip()),
                     f"{path}: ops.{name} mutation {identifier} needs a why. A mutation whose "
                     "purpose is unrecorded is decoration, not coverage")
            mutations.append(Mutation(id=identifier, site=site, why=why))

        ops[name] = Op(name=name, env=env, test=test, mutations=tuple(mutations))
    return ops


# ninja's status prefix. Matched structurally because the dry run exits 0 whether or not there is work,
# so there is no exit code to branch on. A step line is not prose: "[3/4] Linking ..." is a fixed
# format, unlike a message that merely contains the word "build".
NINJA_STEP = re.compile(r"^\[\d+/\d+\]", re.MULTILINE)

# ctest's inventory line, from `ctest -N`. `ctest -N` exits 0 for a match and for no match alike, and
# prints "Total Tests: 0" either way, so the per-test lines are the structural signal.
CTEST_TEST_LINE = re.compile(r"^\s*Test\s+#\d+:", re.MULTILINE)


def test_is_registered(test: str) -> bool:
    """Whether ctest knows a test by this name, checked without running it.

    This is asked before the freshness check, not after, and the order is the point. The freshness
    check dry-runs a build for the named target, and a name that does not exist has no target -- so
    asking currency first reported a dry-run failure for what is actually a typo in the manifest,
    sending the reader after the build instead of the manifest.
    """
    environment = dict(os.environ)
    environment["PATH"] = str(CTEST.parent) + os.pathsep + environment.get("PATH", "")
    completed = subprocess.run(
        [str(CTEST), "--test-dir", "build-test", "-N", "-R", f"^{re.escape(test)}$"],
        env=environment,
        capture_output=True,
        text=True, encoding="utf-8",
        check=False,
    )
    return bool(CTEST_TEST_LINE.search(completed.stdout + completed.stderr))


def targets_are_current(targets: list[str]) -> tuple[bool, str]:
    """Report whether the tests this gate is about to perturb are already built from this source.

    This is the whole reason the gate can be trusted wherever it runs. Verified by demonstration,
    not assumed: with the kernel source edited and deliberately left unbuilt, an earlier version of
    this gate reported GATE PASSED -- a false green, because it was certifying the previous build. A
    gate that can certify code it never compiled is worse than no gate, since it converts an untested
    commit into an apparently tested one.

    It *verifies* rather than *builds*, deliberately. Building here was tried and does not work: nvcc
    needs cl.exe, which means the Visual Studio environment, which a Python gate cannot assume it
    has. The gate reached that failure and reported it, so it was safe -- but a gate that silently
    depends on an ambient environment is one that will be run in the wrong shell eventually.

    So the contract is: the tree must already be built, and this proves it rather than assuming it.
    A stale tree is a failure with the remedy named, which is a different claim from "this is
    correct" and the only one worth committing on. At the tail of test_v3.cmd the build has just
    happened, so this costs one no-op dry run.
    """
    environment = dict(os.environ)
    environment["PATH"] = str(CTEST.parent) + os.pathsep + environment.get("PATH", "")
    completed = subprocess.run(
        [str(CTEST.parent / "cmake.exe"), "--build", "build-test", "--target", *targets, "--", "-n"],
        env=environment,
        capture_output=True,
        text=True, encoding="utf-8",
        check=False,
    )
    if completed.returncode != 0:
        tail = " | ".join(line.strip() for line in completed.stdout.splitlines() if line.strip())[-300:]
        return False, f"the dry run failed (exit {completed.returncode}): {tail}"

    output = completed.stdout + completed.stderr
    steps = NINJA_STEP.findall(output)
    if steps:
        return False, (
            f"{len(steps)} build step(s) are pending for {', '.join(targets)}, so the binary under "
            "test is older than the source being checked"
        )
    return True, f"{', '.join(targets)} is built from the current source"


def run_once(test: str, env_value: str | None) -> tuple[bool, str, float]:
    """Run one test through ctest exactly as the release gate invokes it.

    ctest is used rather than the executable directly so the test runs with the environment the
    suite gives it, and so a skip is visible as a skip instead of passing as green.
    """
    environment = dict(os.environ)
    if env_value is None:
        environment.pop("NINFER_OP_MUTATION", None)
    else:
        environment["NINFER_OP_MUTATION"] = env_value
    # ctest and the CUDA runtime are not on this machine's PATH; the path is recorded in
    # build-test/CMakeCache.txt. Resolved here rather than assumed by the caller.
    environment["PATH"] = str(CTEST.parent) + os.pathsep + environment.get("PATH", "")

    started = time.monotonic()
    completed = subprocess.run(
        [str(CTEST), "--test-dir", "build-test", "-R", f"^{re.escape(test)}$"],
        env=environment,
        capture_output=True,
        text=True, encoding="utf-8",
        check=False,
    )
    elapsed = time.monotonic() - started

    output = completed.stdout + completed.stderr
    if "No tests were found" in output:
        # ctest exits 0 when its regex selects nothing, so a typo'd test name would otherwise reach
        # the mutation assertions and be reported as "the mutation left the suite GREEN" -- a true
        # failure caused by something else, which sends the next reader after the kernel instead of
        # the manifest. Detected here so the cause is named.
        return False, "NO TESTS MATCHED", elapsed
    if completed.returncode == SKIP_RETURN_CODE or "Skipped" in output:
        # A skipped test proves nothing in either direction, and treating skip as pass would let a
        # gate stay green with no coverage at all.
        return False, "SKIPPED", elapsed
    return completed.returncode == 0, f"exit {completed.returncode}", elapsed


def check_op(op: Op, verbose: bool) -> list[str]:
    problems: list[str] = []

    clean_ok, clean_note, clean_seconds = run_once(op.test, None)
    print(f"  {op.name}: control run, {op.env} unset -> {clean_note} ({clean_seconds:.1f}s)")
    if not clean_ok:
        if "NO TESTS MATCHED" in clean_note:
            problems.append(
                f"{op.name}: ctest found no test named {op.test!r}. Either the manifest names it "
                "wrongly, or the tree was not configured with BUILD_TESTING=ON. Note that ctest "
                "exits 0 when its regex selects nothing, so this would otherwise be misreported as a "
                "mutation that did not take effect."
            )
        else:
            problems.append(
                f"{op.name}: the control run did not pass ({clean_note}). Every mutation assertion "
                "below would be satisfied by a suite that is already red, so this is reported first "
                "and alone is a real failure."
            )
        return problems

    for mutation in op.mutations:
        mutated_ok, mutated_note, mutated_seconds = run_once(op.test, str(mutation.id))
        verdict = "PASSED (BAD)" if mutated_ok else f"failed ({mutated_note})"
        print(f"    mutation {mutation.id} {mutation.site} -> {verdict} ({mutated_seconds:.1f}s)")
        if verbose:
            print(f"      why: {mutation.why}")
        if mutated_ok:
            problems.append(
                f"{op.name}: mutation {mutation.id} ({mutation.site}) left the suite GREEN. Either "
                "the test does not cover that decision, or the mutation id is not implemented in "
                "the kernel. The manifest is claiming coverage that does not exist."
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--op", action="append", default=None, help="limit to one op (repeatable)")
    parser.add_argument("--list", action="store_true", help="print the manifest and run nothing")
    parser.add_argument("--verbose", action="store_true", help="print each mutation's recorded purpose")
    args = parser.parse_args()

    try:
        ops = load(args.manifest)
    except ManifestError as error:
        print(f"MANIFEST ERROR: {error}", file=sys.stderr)
        return 1

    if args.op:
        missing = [name for name in args.op if name not in ops]
        if missing:
            print(f"MANIFEST ERROR: no such op in the manifest: {', '.join(missing)}", file=sys.stderr)
            return 1
        ops = {name: ops[name] for name in args.op}

    if args.list:
        for op in ops.values():
            print(f"{op.name} -> {op.test} via {op.env}")
            for mutation in op.mutations:
                print(f"  {mutation.id}: {mutation.site}")
        return 0

    if not ops:
        print("GATE FAILED: the manifest declares no ops, so there is nothing to prove.")
        return 1

    # Pre-flight, and a failure rather than a skip. A gate that quietly declines to run looks exactly
    # like a gate that agreed, and "I could not check this" is not the same claim as "this is
    # correct". This is the one check wired into pre-commit that needs a built tree and the GPU, so
    # the remedy is named in the message rather than left to be discovered.
    if not CTEST.is_file():
        print(f"GATE FAILED: ctest not found at {CTEST}", file=sys.stderr)
        return 1
    tree = Path("build-test")
    if not tree.is_dir():
        print(
            "GATE FAILED: build-test does not exist, so there is no test to perturb. Build it "
            "first:\n"
            "  tools/scripts/test_v3.cmd\n"
            "A perturbation gate needs a test tree; it cannot prove anything from source alone.",
            file=sys.stderr,
        )
        return 1

    # Existence first, then currency. See test_is_registered: a name ctest does not know has no
    # build target, so checking currency ahead of this reports a dry-run failure for what is really a
    # typo in the manifest.
    unknown = [op.test for op in ops.values() if not test_is_registered(op.test)]
    if unknown:
        print(
            f"GATE FAILED: ctest found no test named {', '.join(repr(t) for t in unknown)}.\n"
            "  Either the manifest names it wrongly, or the tree was not configured with\n"
            "  BUILD_TESTING=ON. Note that ctest exits 0 when its regex selects nothing, so this\n"
            "  would otherwise be misreported as a mutation that did not take effect.",
            file=sys.stderr,
        )
        return 1

    # Prove the tree under test is the tree being checked. See targets_are_current: an earlier version
    # certified a stale binary and reported PASS.
    targets = sorted({op.test for op in ops.values()})
    current, currency_note = targets_are_current(targets)
    if not current:
        print(
            f"GATE FAILED: {currency_note}.\n"
            "  A perturbation gate must not certify a binary it did not compile, so this is a failure\n"
            "  rather than a run against the stale one. Build the tree, then re-run:\n"
            "    tools/scripts/test_v3.cmd\n"
            "  Or let test_v3.cmd do both, which is where this gate is meant to be reached from.",
            file=sys.stderr,
        )
        return 1
    print(f"tree is current: {currency_note}")

    print(f"running {sum(len(op.mutations) for op in ops.values())} perturbation assertions "
          f"across {len(ops)} op(s)")
    problems: list[str] = []
    for op in ops.values():
        problems.extend(check_op(op, args.verbose))
        print()

    if problems:
        print("GATE FAILED: a passing test here is not evidence.")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("GATE PASSED: every declared mutation turns the suite red, and the suite is green clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
