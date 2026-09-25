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
from profiles import PROFILES, launcher_env, ordered_flags  # noqa: E402

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
REM
REM  The CUDA wait schedule is pinned to blocking rather than left to the engine's default, which
REM  is upstream's spin (NINFER_CUDA_SYNC, see docs/cli.md). Measured 2026-09-25 over six
REM  interleaved Serve processes per condition: spin costs 0.18-0.31 of a core and buys no
REM  measurable latency, TTFT 1.3-1.9 ms apart against 3.0-15.5 ms spreads, idle and with half
REM  the machine's logical processors held busy by host work. The engine's default is untouched;
REM  docs/research/prompt-preparation-cost.md carries the measurement and its limits.
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
REM The lane's environment, from profiles.launcher_env. A harness that starts Serve has to start it
REM this way too, or what it measures is not what ships.
{env}

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

REM A stale debug-heap configuration on this executable's *name* is the most cryptic failure of
REM all, because nothing fails and nothing prints. It comes from an earlier session running gflags
REM or Application Verifier against the engine and leaving the entry behind:
REM     Image File Execution Options\\<name>   GlobalFlag 0x1000, heap tagging
REM     Image File Execution Options          USTEnabled = <name>, user-mode stack trace database
REM Either one makes the allocator capture a stack on every allocation. Prompt preparation is
REM allocation-dense, so it costs about thirty times what it should -- measured at 3.7 s against
REM 122 ms for a 229-message prompt -- and that is most of the time to first token on a long
REM conversation. Nothing in this tree sets either value, so a machine that has one is a machine
REM someone debugged on.
for %%F in ("%SERVE%") do set "SERVE_NAME=%%~nxF"
set "IFEO_ROOT=HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Image File Execution Options"
set "IFEO_STALE="
reg query "%IFEO_ROOT%\\%SERVE_NAME%" >nul 2>&1 && set "IFEO_STALE=1"
for /f "tokens=3" %%V in ('reg query "%IFEO_ROOT%" /v USTEnabled 2^>nul ^| findstr /i "USTEnabled"') do (
    if /i "%%V"=="%SERVE_NAME%" set "IFEO_STALE=1"
)
if defined IFEO_STALE if not defined NINFER_ALLOW_DEBUG_HEAP (
    echo [ERROR] %SERVE_NAME% carries an Image File Execution Options entry.
    echo         The operating system is making its heap tag every allocation and record a
    echo         stack trace for it. Nothing fails and no log says so, but host prompt
    echo         preparation costs about thirty times what it should.
    echo.
    echo         Remove the entry in an administrator shell, then start this launcher again:
    echo             reg delete "%IFEO_ROOT%\\%SERVE_NAME%" /f
    echo             reg delete "%IFEO_ROOT%" /v USTEnabled /f
    echo.
    echo         To start anyway, accepting the cost, set NINFER_ALLOW_DEBUG_HEAP=1 first.
    pause
    exit /b 1
)

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


def render_env(profile: dict) -> str:
    """The lane's environment as cmd `set` lines, sorted so regeneration stays byte-identical."""
    return "\n".join(f'set "{name}={value}"'
                     for name, value in sorted(launcher_env(profile).items()))


def render(profile: dict) -> str:
    flags = render_flags(profile)
    return TEMPLATE.format(
        label=profile["label"], note=profile["note"], ctx=profile["ctx"],
        ctx_h=f"{profile['ctx']:,}", tok=profile["tok"], acc=profile["acc"],
        runtime=profile["runtime"], free=profile["free"], v3=V3, models=MODELS,
        art=profile["art"], flags=flags, port=profile["port"], env=render_env(profile),
    )


def main() -> int:
    for profile in PROFILES:
        path = OUT / profile["file"]
        path.write_text(render(profile), encoding="utf-8", newline="\r\n")
        print(f"  wrote {profile['file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
