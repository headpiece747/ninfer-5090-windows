#!/usr/bin/env python3
"""Package a NInfer Windows release.

Reusable so the next release is one command. It stages an explicit file set (not a glob of
the build directory, which also holds cmake_install.cmake and unused DLLs), writes
SHA256SUMS over the staged files, archives with bsdtar so large files compress in parallel,
and prints the archive digest.

Layout follows the v1.0.x releases, with three deliberate changes:
  * one archive instead of a separate vision one -- the v3 engine enables Vision with a flag
    rather than a second binary;
  * five FFmpeg DLLs, matching the five components cmake/Dependencies.cmake links
    (avcodec, avformat, avutil, swscale, swresample), rather than everything staged locally;
  * NOTICE ships from the repository root like the other root files, not from a second
    checkout: the archive is the distribution, so the attribution has to travel with it, and
    reading it from another clone made the archive depend on that clone existing and current.

Packaging runs tools/release/check_doc_links.py and tools/release/check_test_baseline.py first, and
refuses to build an archive when either fails. The link check is here because a rename left two
links in docs/maintainer pointing at files that no longer existed and the release was about to
carry them. The test gate compares against a recorded baseline rather than trusting a bare run,
because a previous release was cut while a test was failing and nobody noticed; the baseline is
empty now that the suite is green, so any failure blocks the archive. Pass --skip-test-gate to
override deliberately.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiles import PROFILES  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "build" / "apps"
RELEASES = Path(r"C:\AI\releases")

EXES = ["ninfer-serve.exe", "ninfer.exe", "ninfer-perplexity.exe"]
DLLS = ["avcodec-63.dll", "avformat-63.dll", "avutil-61.dll",
        "swscale-10.dll", "swresample-7.dll"]
ROOT_FILES = ["README.md", "RELEASE_NOTES.md", "LICENSE", "NOTICE"]
# The four start_*.bat are generated from the table, so they are derived from it here rather
# than restated; the other three are hand-written and ship as they are.
LAUNCHER_FILES = [
    "launcher_env.bat", "download_model.bat", "download_model.py",
    *[profile["file"] for profile in PROFILES],
]

# Shipped from tools/ rather than the root. Both artifacts are published as v3 now, so nothing
# downloads into this path any more; the upgrader stays for a copy fetched before the republish,
# which a v3 engine rejects outright.
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

    print("  doc links  : running tools/release/check_doc_links.py")
    links = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "check_doc_links.py")],
        cwd=REPO)
    if links.returncode != 0:
        print("  RELEASE REFUSED: a relative link in the documentation does not resolve.")
        return 1

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
    # The DLLs are LGPL, so their licence text has to travel with them.
    ffmpeg_license = REPO / "ffmpeg" / "LICENSE.txt"
    if ffmpeg_license.exists():
        shutil.copy2(ffmpeg_license, stage / "FFMPEG-LICENSE.txt")
        total += ffmpeg_license.stat().st_size
    else:
        missing.append("ffmpeg/LICENSE.txt (the runtime DLLs' licence text)")

    if missing:
        print("  MISSING:")
        for item in missing:
            print(f"    {item}")
        return 1

    staged = sorted(p for p in stage.rglob("*") if p.is_file())
    names = [p.relative_to(stage).as_posix() for p in staged]
    lines = [f"{sha256(path)}  {name}" for path, name in zip(staged, names)]
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
