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

REPO = Path(__file__).resolve().parents[2]
OUT = REPO
V3 = str(REPO)
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
REM  build_windows.bat). This launcher checks for them and refuses with a readable
REM  message, rather than letting the process exit 0xC0000135 having printed nothing.
REM
REM  The three context-cache bounds are deliberate. With max-concurrency 1 the defaults
REM  are max(1,4) shared, 2 private and 2 anchors; measured on five distinct ~530-token
REM  prompts resent, that gave 1/5 round-2 hits at a 19.8% token-level hit rate, with
REM  four of five prompts re-prefilling in full on every call and no error. Raising the
REM  shared bound alone changed nothing; all three together gave 5/5 hits at 99.1%.
REM  They cost no context or VRAM: with the bounds raised, both the ceiling and the
REM  runtime are unchanged from the values above.
REM ============================================================================
setlocal

REM Resolve beside this launcher first, so the released archive is portable wherever it is
REM extracted, then fall back to the source tree so the same file works while developing.
set "SERVE=%~dp0ninfer-serve.exe"
if not exist "%SERVE%" set "SERVE={v3}\\build\\apps\\ninfer-serve.exe"
set "MODEL=%~dp0models\\{art}"
if not exist "%MODEL%" set "MODEL={models}\\{art}"
REM The lane's template travels with the archive; the source-tree copy is the fallback. Passing it
REM explicitly stops the lane inheriting whichever template its artifact embeds -- the two shipped
REM artifacts embed different ones, and the embedded pair predate the reasoning-effort alias mapping.
set "TEMPLATE=%~dp0chat_templates\\qwen3_8.jinja"
if not exist "%TEMPLATE%" set "TEMPLATE={v3}\\tools\\chat_templates\\qwen3_8.jinja"

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

REM --- Preflight ---------------------------------------------------------------------------
REM Each check replaces a failure that is otherwise cryptic: a missing FFmpeg DLL makes the
REM process exit 0xC0000135 before printing a reason, a busy port yields a bare bind error, and
REM a second model on this 32 GB card yields a runtime-reservation FATAL.
for %%F in ("%SERVE%") do set "SERVE_DIR=%%~dpF"

for %%D in (avcodec avformat avutil swscale swresample) do (
    if not exist "%SERVE_DIR%%%D-*.dll" (
        echo [ERROR] FFmpeg runtime DLL missing: %%D-*.dll
        echo         The engine needs all five beside the executable, in:
        echo             %SERVE_DIR%
        echo         build_windows.bat stages them from the ffmpeg\\bin directory it downloads.
        echo         Without them the engine exits 0xC0000135 without printing a reason.
        pause
        exit /b 1
    )
)

netstat -ano | findstr ":{port}" | findstr /I "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [ERROR] Port {port} is already in use.
    echo         Something is already listening there. Stop it, or change the --port flag
    echo         in this launcher. To see what holds it:
    echo             netstat -ano ^| findstr ":{port}"
    pause
    exit /b 1
)

tasklist /FI "IMAGENAME eq ninfer-serve.exe" 2>nul | find /I "ninfer-serve.exe" >nul
if not errorlevel 1 (
    echo [WARN]  A ninfer-serve.exe process is already running.
    echo         This card holds one model at a time, so starting another may fail with a
    echo         runtime-reservation error, or the running server may stop answering.
    choice /C YN /N /M "Continue anyway? [Y/N] "
    if errorlevel 2 exit /b 1
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
        art=profile["art"], flags=flags, port=profile["port"],
    )


def main() -> int:
    for profile in PROFILES:
        path = OUT / profile["file"]
        path.write_text(render(profile), encoding="utf-8", newline="\r\n")
        print(f"  wrote {profile['file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
