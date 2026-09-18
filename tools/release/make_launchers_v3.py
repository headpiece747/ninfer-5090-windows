#!/usr/bin/env python3
"""Render the v3 Windows launchers from the profile table.

The table lives in profiles.py. This file only turns it into cmd scripts, so a ceiling, a port
or a measured figure changes in one place. For an unchanged table, regeneration is expected to
be byte-identical, and tools/release/check_profile_consistency.py asserts that the shipped files
still match what this produces.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiles import PROFILES, ordered_flags  # noqa: E402

OUT = Path(r"C:\AI\ninfer-v3-windows")
V3 = r"C:\AI\ninfer-v3-windows"
MODELS = r"C:\AI\models"

TEMPLATE = """@echo off
REM ============================================================================
REM  {label}
REM
REM  Measured on this machine (32 GB RTX 5090), fp8 KV at the ceiling below:
REM      context {ctx_h}   decode {tok} tok/s   draft acceptance {acc}
REM      runtime {runtime}   free VRAM {free}
REM
REM  {note}
REM
REM  Requires the FFmpeg runtime DLLs beside the executable (staged by
REM  build_v3.cmd); without them the process exits 0xC0000135.
REM
REM  The three context-cache bounds are deliberate. With max-concurrency 1 the defaults
REM  are max(1,4) shared, 2 private and 2 anchors; measured on five distinct ~530-token
REM  prompts resent, that gave 1/5 round-2 hits at a 19.8% token-level hit rate, with
REM  four of five prompts re-prefilling in full on every call and no error. Raising the
REM  shared bound alone changed nothing; all three together gave 5/5 hits at 99.1%.
REM  They cost no context or VRAM: KV stays 262,144 and runtime stays 10.7 GiB.
REM ============================================================================
setlocal

REM Resolve beside this launcher first, so the released archive is portable wherever it is
REM extracted, then fall back to the source tree so the same file works while developing.
set "SERVE=%~dp0ninfer-serve.exe"
if not exist "%SERVE%" set "SERVE={v3}\\build\\apps\\ninfer-serve.exe"
set "MODEL=%~dp0models\\{art}"
if not exist "%MODEL%" set "MODEL={models}\\{art}"

if not exist "%SERVE%" (
    echo [ERROR] Engine not found.
    echo         Expected ninfer-serve.exe beside this launcher,
    echo         or a source build at {v3}\\build\\apps\\ninfer-serve.exe
    pause
    exit /b 1
)
if not exist "%MODEL%" (
    echo [ERROR] Artifact not found.
    echo         Expected %~dp0models\\{art}
    echo         or {models}\\{art}
    echo         Run download_model.bat to fetch it.
    pause
    exit /b 1
)

"%SERVE%" "%MODEL%" ^{flags}

pause
"""


def render_flags(profile: dict) -> str:
    """One flag per line, every line ending in a continuation caret except the last."""
    lines = [flag if value is None else f"{flag} {value}"
             for flag, value in ordered_flags(profile)]
    return "".join(f"\n  {line} ^" for line in lines[:-1]) + f"\n  {lines[-1]}"


def render(profile: dict) -> str:
    flags = render_flags(profile)
    return TEMPLATE.format(
        label=profile["label"], note=profile["note"], ctx=profile["ctx"],
        ctx_h=f"{profile['ctx']:,}", tok=profile["tok"], acc=profile["acc"],
        runtime=profile["runtime"], free=profile["free"], v3=V3, models=MODELS,
        art=profile["art"], flags=flags,
    )


def main() -> int:
    for profile in PROFILES:
        path = OUT / profile["file"]
        path.write_text(render(profile), encoding="utf-8", newline="\r\n")
        print(f"  wrote {profile['file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
