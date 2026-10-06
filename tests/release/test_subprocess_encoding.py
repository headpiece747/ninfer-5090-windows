"""The subprocess-encoding gate must flag what it names and pass what it does not.

A gate whose failure paths have never been run is indistinguishable from one that always passes, so
both directions are asserted here: a text-mode spawn with no encoding fails, and each of the three
legitimate forms passes. `spawned_text_calls` is the pure part -- it takes an AST and returns what it
found -- so these cases need no files on disk.

Written as pytest functions rather than as a script with module-level assertions. The first version
was a script while `.githooks/pre-commit` names it in its pytest line, so pytest collected NOTHING
from it: the assertions still ran at import, but the step reported "no tests collected", and a step
that reports no tests is one nobody can see fail.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GATE = REPO / "tools" / "release" / "check_subprocess_encoding.py"

spec = importlib.util.spec_from_file_location("check_subprocess_encoding", GATE)
assert spec is not None and spec.loader is not None
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)


def findings(source: str) -> list[tuple[int, bool, bool]]:
    return gate.spawned_text_calls(ast.parse(source))


def test_text_mode_without_an_encoding_is_reported() -> None:
    """The defect this gate exists for."""
    found = findings('import subprocess\nsubprocess.run(["x"], text=True, capture_output=True)\n')
    assert len(found) == 1 and found[0][1] and not found[0][2]


def test_universal_newlines_is_text_mode_too() -> None:
    found = findings('import subprocess\nsubprocess.run(["x"], universal_newlines=True)\n')
    assert len(found) == 1 and found[0][1] and not found[0][2]


def test_popen_is_a_spawner() -> None:
    found = findings('import subprocess\nsubprocess.Popen(["x"], text=True)\n')
    assert len(found) == 1 and not found[0][2]


def test_an_explicit_encoding_passes() -> None:
    found = findings('import subprocess\nsubprocess.run(["x"], text=True, encoding="utf-8")\n')
    assert len(found) == 1 and found[0][2]


def test_an_encoding_without_text_mode_passes() -> None:
    """encoding implies text, so this form is not the defect."""
    found = findings('import subprocess\nsubprocess.run(["x"], encoding="utf-8")\n')
    assert len(found) == 1 and not found[0][1] and found[0][2]


def test_binary_mode_passes() -> None:
    """Nothing is decoded, so the locale cannot matter."""
    found = findings('import subprocess\nsubprocess.run(["x"], capture_output=True)\n')
    assert len(found) == 1 and not found[0][1]


def test_an_explicit_errors_passes() -> None:
    found = findings('import subprocess\nsubprocess.run(["x"], text=True, errors="replace")\n')
    assert len(found) == 1 and found[0][2]


def test_a_non_subprocess_call_is_not_a_spawn() -> None:
    """The gate must not flag an unrelated call that happens to take text=."""
    assert not findings("def f(**kw): pass\nf(text=True)\n")


def test_the_real_tree_passes_the_gate() -> None:
    """The assertion that keeps the gate honest about itself."""
    assert gate.main() == 0
