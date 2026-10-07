#!/usr/bin/env python3
"""Check a release archive's contents, and its own SHA256SUMS, without running it.

The packager writes SHA256SUMS over the staged files; this verifies every entry against that file
and asserts the expected set is present, so an archive that lost a launcher, a licence notice or an
FFmpeg DLL fails before it is published. It also reports files the archive should not carry: the
v1.0.x archives shipped avdevice/avfilter they never link, and the vision one carried the same
260 MB engine twice under two names.

check_release_archive.py covers the other half -- that the extracted archive starts and answers.

Usage: check_archive_contents.py ARCHIVE.zip
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiles import PROFILES  # noqa: E402

REQUIRED = [
    "ninfer-serve.exe",
    "ninfer.exe",
    "ninfer-perplexity.exe",
    "avcodec-63.dll",
    "avformat-63.dll",
    "avutil-61.dll",
    "swscale-10.dll",
    "swresample-7.dll",
    "launcher_env.bat",
    "download_model.bat",
    "download_model.py",
    # The build driver and the converter it runs. Without these an extracted archive can fetch the
    # source checkpoints and not turn them into an image, which is what made the earlier "sources only"
    # rewrite half a flow: the downloader was correct and nothing could consume what it fetched.
    "build_model.py",
    "tools/convert/__main__.py",
    "tools/convert/official_recipes.py",
    "tools/convert/qwen3_8_27b_nvfp4_calibration.json",
    "tools/artifact/writer.py",
    # Every launcher the packager stages, derived from the same table the packager derives it from.
    # This list was four hand-copied names, which meant the check could not fail for a launcher it
    # did not name -- so an archive that silently lost the Swift or NVIDIA lane passed, which is the
    # exact failure this file's own docstring says it exists to catch. Deriving it removes the copy.
    *[profile["file"] for profile in PROFILES],
    "upgrade_ninfer_v2_to_v3.py",
    "README.md",
    "RELEASE_NOTES.md",
    "LICENSE",
    "NOTICE",
    "SHA256SUMS",
    "FFMPEG-LICENSE.txt",
    "chat_templates/LICENSE",
    "chat_templates/qwen3_6.jinja",
    "chat_templates/qwen3_8.jinja",
]


def main() -> int:
    archive = Path(sys.argv[1])
    if not archive.exists():
        print(f"   missing archive: {archive}")
        return 1

    with zipfile.ZipFile(archive) as zf:
        by_name = {}
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            by_name[name[2:] if name.startswith("./") else name] = name
        print(f"{archive.name}: {len(by_name)} entries, {archive.stat().st_size / 2**20:.1f} MB")

        failures = [
            f"missing: {name}" for name in REQUIRED if name not in by_name
        ]
        extras = sorted(set(by_name) - set(REQUIRED))
        for name in extras:
            print(f"   extra: {name}")
        # Bytecode is never legitimate in an archive: it is version-specific, and it is what staging
        # the converter's package directory dragged in the first time the packager shipped one. The
        # gate printed it as an "extra" and still passed, which is a muted check.
        for name in sorted(by_name):
            if "__pycache__" in name or name.endswith(".pyc"):
                failures.append(f"bytecode in the archive: {name}")

        if "SHA256SUMS" in by_name:
            listed = {}
            for line in zf.read(by_name["SHA256SUMS"]).decode("utf-8").splitlines():
                if line.strip():
                    digest, _, name = line.partition("  ")
                    listed[name.strip()] = digest.strip()
            print(f"   SHA256SUMS lists {len(listed)} files")
            for name, expected in sorted(listed.items()):
                entry = by_name.get(name)
                if entry is None:
                    failures.append(f"listed but absent: {name}")
                    continue
                digest = hashlib.sha256(zf.read(entry)).hexdigest()
                if digest != expected:
                    failures.append(f"hash mismatch: {name}")
            unlisted = sorted(name for name in set(by_name) - set(listed) - {"SHA256SUMS"}
                              if "/" not in name)
            for name in unlisted:
                failures.append(f"not covered by SHA256SUMS: {name}")
            nested = sorted(name for name in set(by_name) - set(listed) if "/" in name)
            for name in nested:
                print(f"   note: nested file not covered by SHA256SUMS: {name}")

    if failures:
        print("   FAIL")
        for failure in failures:
            print(f"     {failure}")
        return 1
    print("   PASS: every required file present and every hash matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
