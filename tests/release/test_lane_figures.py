"""What the lane-figures gate must catch, and the two ways it must not pass for free.

Why this gate exists: the defects of 2026-10-09 were figures relied on past their conditions -- a throughput
column taken before its lane's depth changed and shipped beside `draft=9`, an acceptance figure from an
instrument that cannot reach the served regime, a context derived and never served. None was a missing
measurement, so what the gate adds is not measurement but *mechanical staleness*: every shipped figure
records the artifact and the engine binary it was measured on, and the gate compares those identities with
the tree.

Every case below is one failure path, driven through the real script with `--figures`, `--serve` and
`--models` pointed at a temporary tree, so nothing here touches the repository's artifacts or its binary.
The clean case is first, deliberately: an instrument whose failure path never runs is indistinguishable
from one that always passes, and the reverse is the same defect with the sign flipped -- a gate that always
fails is a gate that gets muted, so the passing case is asserted too.

The last two cases are the two ways a check can pass without checking anything: a missing provenance file
must be an error rather than a pass, and a checkout with no built binary must say it skipped rather than
claim success.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GATE = REPO / "tools" / "release" / "check_lane_figures.py"

# Anchored on __file__ for the same reason tests/release/test_text_encoding.py does it: a bare relative
# path would make these cases pass or fail with wherever pytest was invoked from.
sys.path.insert(0, str(REPO / "tools" / "release"))

from profiles import PROFILES  # noqa: E402

LAUNCHER = "start_nvidia_v3_dflash2_vision.bat"


def run_gate(figures: Path, serve: Path, models: Path) -> tuple[int, str]:
    completed = subprocess.run([sys.executable, str(GATE), "--figures", str(figures),
                                "--serve", str(serve), "--models", str(models)],
                               capture_output=True, encoding="utf-8", errors="replace", check=False)
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def fixture(tmp_path: Path, *, engine: str | None = None, bad_size: bool = False,
            bad_mtime: bool = False, tok: float | None = None, acc: float | None = None,
            with_serve: bool = True) -> tuple[Path, Path]:
    """One lane, one fake artifact, one fake binary: the smallest shape the gate reads."""
    profile = next(entry for entry in PROFILES if entry["file"] == LAUNCHER)
    artifact = tmp_path / profile["art"]
    artifact.write_bytes(b"x" * 16)
    stat = artifact.stat()
    serve = tmp_path / "ninfer-serve.exe"
    if with_serve:
        serve.write_bytes(b"not a real binary, only its hash matters here")
    digest = engine if engine is not None else (
        hashlib.sha256(serve.read_bytes()).hexdigest() if with_serve else "0" * 64)
    figures = tmp_path / "lane_figures.json"
    figures.write_text(json.dumps({
        "engine": {"sha256": digest, "bytes": serve.stat().st_size if with_serve else 0},
        "measured_utc": "2026-10-09T00:00:00Z",
        "lanes": {LAUNCHER: {
            "artifact": {"path": profile["art"],
                         "bytes": stat.st_size + (1 if bad_size else 0),
                         "mtime": stat.st_mtime + (60.0 if bad_mtime else 0.0)},
            "cells": {"code": {
                "decode_avg": profile["tok"] if tok is None else tok,
                "accept_rate": (float(str(profile["acc"]).rstrip("%")) / 100.0)
                if acc is None else acc,
                "sampling": "default"}},
        }},
    }, indent=2), encoding="utf-8")
    return figures, serve


def test_clean_tree_passes(tmp_path: Path) -> None:
    figures, serve = fixture(tmp_path)
    code, output = run_gate(figures, serve, tmp_path)
    assert code == 0, output
    assert "PASS" in output


def test_engine_change_fails(tmp_path: Path) -> None:
    figures, serve = fixture(tmp_path, engine="f" * 64)
    code, output = run_gate(figures, serve, tmp_path)
    assert code == 1, output
    assert "measured with engine" in output


def test_artifact_size_change_fails(tmp_path: Path) -> None:
    figures, serve = fixture(tmp_path, bad_size=True)
    code, output = run_gate(figures, serve, tmp_path)
    assert code == 1, output
    assert "artifact size" in output


def test_artifact_rewrite_fails(tmp_path: Path) -> None:
    figures, serve = fixture(tmp_path, bad_mtime=True)
    code, output = run_gate(figures, serve, tmp_path)
    assert code == 1, output
    assert "rewritten" in output


def test_table_throughput_mismatch_fails(tmp_path: Path) -> None:
    figures, serve = fixture(tmp_path)
    payload = json.loads(figures.read_text(encoding="utf-8"))
    payload["lanes"][LAUNCHER]["cells"]["code"]["decode_avg"] = (
        next(entry for entry in PROFILES if entry["file"] == LAUNCHER)["tok"] * 1.5)
    figures.write_text(json.dumps(payload), encoding="utf-8")
    code, output = run_gate(figures, serve, tmp_path)
    assert code == 1, output
    assert "against the table's" in output


def test_table_acceptance_mismatch_fails(tmp_path: Path) -> None:
    figures, serve = fixture(tmp_path, acc=None)
    payload = json.loads(figures.read_text(encoding="utf-8"))
    recorded = payload["lanes"][LAUNCHER]["cells"]["code"]["accept_rate"]
    payload["lanes"][LAUNCHER]["cells"]["code"]["accept_rate"] = recorded + 0.05
    figures.write_text(json.dumps(payload), encoding="utf-8")
    code, output = run_gate(figures, serve, tmp_path)
    assert code == 1, output
    assert "acceptance" in output


def test_missing_provenance_is_an_error(tmp_path: Path) -> None:
    figures = tmp_path / "absent.json"
    serve = tmp_path / "ninfer-serve.exe"
    serve.write_bytes(b"binary")
    code, output = run_gate(figures, serve, tmp_path)
    assert code == 2, output
    assert "does not exist" in output


def test_no_build_skips_with_a_note(tmp_path: Path) -> None:
    figures, serve = fixture(tmp_path, with_serve=False)
    absent = tmp_path / "not-built" / "ninfer-serve.exe"
    code, output = run_gate(figures, absent, tmp_path)
    assert code == 0, output
    assert "SKIP" in output and "no figure is in play" in output
