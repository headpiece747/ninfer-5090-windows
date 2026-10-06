"""What the baseline gate's cached verdict must and must not cover.

On 2026-10-07 this gate read green on a tree it had never run. Upstream's merge added three test
registrations, the gate reused a recorded run of 145 tests, and the baseline's own `suite_size`
agreed with the stale log -- so the check that exists to catch a partial run could not see it. The
suite in the tree covered 149.

The cause was the cache key: it hashed every built test executable and nothing else. A configure
that adds or removes a registration relinks no executable, so the key matched, and the verdict
belonged to a different suite. Both properties are asserted here, and the control at the bottom
pins that the registration half is load-bearing rather than decoration -- an executable-only key,
reproduced inline, is shown to miss the change the real key catches.

The second half is the race. A suite takes minutes and so does a build, and this repository has
lost that race twice: relinking a test executable that ctest is executing fails the link with
LNK1104, and a relink landing mid-run means the failures describe no single tree. Neither is
visible in the ctest log, so the tree's identity is compared before and after the run.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "release"))

import check_test_baseline as gate  # noqa: E402
from check_test_baseline import binary_key, binary_key_changed_reason  # noqa: E402


def _tree(root: Path) -> Path:
    """A miniature build tree: one executable, one registration, and the top-level test file."""
    build = root / "build-test"
    (build / "tests").mkdir(parents=True)
    (build / "tests" / "ninfer_demo_test.exe").write_bytes(b"MZ" + b"\0" * 32)
    (build / "CTestTestfile.cmake").write_text("# subdirs\n", encoding="utf-8")
    (build / "tests" / "CTestTestfile.cmake").write_text(
        "add_test(NAME demo COMMAND demo)\n", encoding="utf-8")
    return build


def _point_at(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    """Redirect the gate at a temp tree. Both roots move together: the key is stored relative to
    REPO, so a BUILD outside it would raise rather than compare."""
    build = _tree(root)
    monkeypatch.setattr(gate, "REPO", root)
    monkeypatch.setattr(gate, "BUILD", build)
    return build


def test_no_build_tree_has_no_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """None is not "unchanged": it means no comparison is possible, and the cache is skipped."""
    monkeypatch.setattr(gate, "REPO", tmp_path)
    monkeypatch.setattr(gate, "BUILD", tmp_path / "build-test")
    assert binary_key() is None


def test_an_unchanged_tree_keeps_its_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The other direction: a key that changed on every call would make the cache useless."""
    _point_at(monkeypatch, tmp_path)
    assert binary_key() == binary_key()


def test_a_rebuilt_executable_changes_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The property the key already had, kept so the registration half cannot replace it."""
    build = _point_at(monkeypatch, tmp_path)
    before = binary_key()
    (build / "tests" / "ninfer_demo_test.exe").write_bytes(b"MZ" + b"\0" * 64)
    assert binary_key() != before


def test_a_changed_test_registration_changes_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The case that read green: upstream's merge added three registrations, relinked nothing."""
    build = _point_at(monkeypatch, tmp_path)
    before = binary_key()
    (build / "tests" / "CTestTestfile.cmake").write_text(
        "add_test(NAME demo COMMAND demo)\nadd_test(NAME other COMMAND other)\n", encoding="utf-8")
    assert binary_key() != before


def test_an_executable_only_key_would_miss_a_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control: the key as it was before this fix, reproduced here, does not move.

    Without this, "the key changed" would also be satisfied by a key that changes on any write at
    all -- and the fix would look correct while the defect it addresses was never reproduced.
    """
    build = _point_at(monkeypatch, tmp_path)

    def executable_only() -> str:
        digest = hashlib.sha256()
        for path in sorted(build.rglob("*.exe")):
            stat = path.stat()
            digest.update(f"{path.relative_to(tmp_path).as_posix()}:{stat.st_size}:"
                          f"{stat.st_mtime_ns}\n".encode("utf-8"))
        return digest.hexdigest()

    old_key = executable_only()
    real_key = binary_key()
    (build / "tests" / "CTestTestfile.cmake").write_text(
        "add_test(NAME demo COMMAND demo)\nadd_test(NAME other COMMAND other)\n", encoding="utf-8")
    assert executable_only() == old_key, "the control moved, so it is not the old behaviour"
    assert binary_key() != real_key, "the registration half is not load-bearing"


def test_a_tree_that_changed_during_the_run_is_not_certified() -> None:
    """The failure path has to fail, or the race check is decoration."""
    reason = binary_key_changed_reason("before", "after")
    assert reason is not None
    assert "no single tree" in reason


def test_the_same_tree_is_certified() -> None:
    """And it must be silent on the normal path, or every run reads as a race."""
    assert binary_key_changed_reason("same", "same") is None


def test_a_missing_key_is_not_a_race() -> None:
    """A host with no build tree, and --from-log, both legitimately have no key to compare."""
    assert binary_key_changed_reason(None, "after") is None
    assert binary_key_changed_reason("before", None) is None
