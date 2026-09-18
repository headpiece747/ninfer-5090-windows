#!/usr/bin/env python3
"""Package a NInfer Windows release.

Reusable so the next release is one command. It stages an explicit file set (not a glob of
the build directory, which also holds cmake_install.cmake and unused DLLs), writes
SHA256SUMS over the staged files, archives with bsdtar so large files compress in parallel,
and prints the archive digest.

Layout follows the v1.0.x releases, with two deliberate changes:
  * one archive instead of a separate vision one -- the v3 engine enables Vision with a flag
    rather than a second binary;
  * five FFmpeg DLLs, matching the five components cmake/Dependencies.cmake links
    (avcodec, avformat, avutil, swscale, swresample), rather than everything staged locally.

Packaging runs tools/release/check_test_baseline.py first and refuses to build an archive when the
suite has regressed. The suite is not green -- one case is a documented upstream disagreement --
which is exactly why the gate compares against a recorded baseline instead of demanding green:
the previous release was cut while a test was failing and nobody noticed. Pass --skip-test-gate to
override deliberately.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(r"C:\AI\ninfer-v3-windows")
BUILD = REPO / "build" / "apps"
RELEASES = Path(r"C:\AI\releases")
NOTICE_SOURCE = Path(r"C:\AI\ninfer-5090-windows\NOTICE")

EXES = ["ninfer-serve.exe", "ninfer.exe", "ninfer-perplexity.exe"]
DLLS = ["avcodec-63.dll", "avformat-63.dll", "avutil-61.dll",
        "swscale-10.dll", "swresample-7.dll"]
ROOT_FILES = ["README.md", "RELEASE_NOTES.md", "LICENSE"]
LAUNCHER_FILES = [
    "launcher_env.bat", "download_model.bat", "download_model.py",
    "start_quasar_v3_dflash2_vision.bat", "start_quasar_v3_mtp4_vision.bat",
    "start_ninfer_v3_dflash2_vision.bat", "start_ninfer_v3_mtp5_vision.bat",
]

# Shipped from tools/ rather than the root. The QUASAR artifact is published as a v2
# container, so download_model.py sends the user to this offline upgrader; it has to be here.
TOOL_FILES = ["upgrade_ninfer_v2_to_v3.py", "chat_templates"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    positional = [argument for argument in sys.argv[1:] if not argument.startswith("--")]
    skip_gate = "--skip-test-gate" in sys.argv[1:]
    version = positional[0] if positional else "v1.1.0"

    if skip_gate:
        print("  test gate  : SKIPPED (--skip-test-gate)")
    else:
        print("  test gate  : running tools/release/check_test_baseline.py")
        gate = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "check_test_baseline.py")],
            cwd=REPO)
        if gate.returncode != 0:
            print("  RELEASE REFUSED: the suite regressed against the recorded baseline.")
            print("  Fix the regression, or pass --skip-test-gate to accept it deliberately.")
            return 1

    stage = RELEASES / f"stage-{version}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    missing: list[str] = []
    total = 0
    for group, location in ((EXES + DLLS, BUILD), (ROOT_FILES, REPO),
                            (LAUNCHER_FILES, REPO), (TOOL_FILES, REPO / "tools")):
        for name in group:
            source = location / name
            if not source.exists():
                missing.append(f"{name} (from {location})")
                continue
            if source.is_dir():
                shutil.copytree(source, stage / name, dirs_exist_ok=True)
                total += sum(f.stat().st_size for f in source.rglob("*") if f.is_file())
            else:
                shutil.copy2(source, stage / name)
                total += source.stat().st_size
    if NOTICE_SOURCE.exists():
        shutil.copy2(NOTICE_SOURCE, stage / "NOTICE")
    else:
        missing.append(f"NOTICE (from {NOTICE_SOURCE})")

    if missing:
        print("  MISSING:")
        for item in missing:
            print(f"    {item}")
        return 1

    names = sorted(p.name for p in stage.iterdir() if p.is_file())
    lines = [f"{sha256(stage / n)}  {n}" for n in names]
    (stage / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")

    archive = RELEASES / f"ninfer-windows-{version}-rtx5090.zip"
    archive.unlink(missing_ok=True)
    result = subprocess.run(["tar", "-a", "-c", "-f", str(archive), "-C", str(stage), "."],
                            capture_output=True, text=True)
    if result.returncode != 0 or not archive.exists():
        print(f"  ARCHIVE FAILED: {result.stderr[:300]}")
        return 1

    print(f"  staged {len(names) + 1} files from {total / (1024 ** 2):.0f} MB of inputs")
    for name in names:
        print(f"    {name}")
    print(f"    SHA256SUMS")
    print(f"  archive : {archive}")
    print(f"  size    : {archive.stat().st_size / (1024 ** 2):.1f} MB")
    print(f"  sha256  : {sha256(archive)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
