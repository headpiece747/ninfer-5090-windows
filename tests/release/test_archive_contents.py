"""Tests for release archive contents configuration.

Verifies that the packager and archive-contents gate consistently require the engine's shared
libraries and the converter's ranking data, and that non-binary staged sources exist in the tree.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "release"))

import check_archive_contents as checker  # noqa: E402
import package_release as packager  # noqa: E402
import pe_audit  # noqa: E402


def test_shared_engine_libraries_are_required() -> None:
    assert "ninfer_ops.dll" in checker.REQUIRED
    assert "ninfer_moe_cpu.dll" in checker.REQUIRED
    assert "ninfer_ops.dll" in packager.DLLS
    assert "ninfer_moe_cpu.dll" in packager.DLLS


def test_converter_ranking_fixture_is_required_and_present() -> None:
    ranking_path = "tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64"
    assert ranking_path in checker.REQUIRED
    assert "freq_corpus/fixtures/ranking/ranking.train.counts.i64" in packager.TOOL_DATA
    assert (REPO / ranking_path).is_file()


def test_pe_audit_system_dll_recognition() -> None:
    assert pe_audit.is_system_dll("kernel32.dll")
    assert pe_audit.is_system_dll("KERNEL32.DLL")
    assert pe_audit.is_system_dll("WINHTTP.dll")
    assert pe_audit.is_system_dll("api-ms-win-crt-runtime-l1-1-0.dll")
    assert pe_audit.is_system_dll("ext-ms-win-gdi-draw-l1-1-0.dll")
    assert not pe_audit.is_system_dll("ninfer_ops.dll")
    assert not pe_audit.is_system_dll("custom_engine.dll")


def test_staged_built_binaries_dependencies_satisfied() -> None:
    build_apps = REPO / "build" / "apps"
    if not (build_apps / "ninfer-serve.exe").exists():
        return
    staged_names = set(packager.EXES + packager.DLLS)
    binaries = {
        p.name: p.read_bytes()
        for p in build_apps.iterdir()
        if p.is_file() and p.name in staged_names
    }
    errors = pe_audit.audit_binary_dependencies(binaries)
    assert not errors, f"Unresolved dependencies in build/apps: {errors}"


if __name__ == "__main__":
    test_shared_engine_libraries_are_required()
    test_converter_ranking_fixture_is_required_and_present()
    test_pe_audit_system_dll_recognition()
    test_staged_built_binaries_dependencies_satisfied()
    print("  archive contents checks passed")

