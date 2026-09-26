@echo off
REM ============================================================================
REM  NVIDIA ModelOpt + DFlash2 + Vision
REM
REM  Measured on this machine (32 GB RTX 5090), fp8 KV at the ceiling below:
REM      context 262,144   decode 322.4 tok/s   draft acceptance 59.2%
REM      runtime 10.7 GiB   free VRAM 2.45 GiB
REM
REM  NVIDIA's ModelOpt quantization of the base model, built by this port: its NVFP4 MLP imported on all 64 layers and its FP8 attention re-encoded from the BF16 base. Same full-corpus perplexity as the official stock at 20% smaller, with no FP8 tensor where that has 146, and this lane reaches the full context where it caps below.
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
REM extracted. The source-tree copy is the same shape one directory down: build/apps and
REM tools/chat_templates both sit beside the launcher in the repository, so the fallback is relative
REM too. An absolute path here bakes one machine's checkout into a file that ships, and it is what
REM made the generated launchers unverifiable anywhere else.
set "SERVE=%~dp0ninfer-serve.exe"
if not exist "%SERVE%" set "SERVE=%~dp0build\apps\ninfer-serve.exe"
set "MODEL=%~dp0models\qwen3_8_27b_nvfp4nvidia.v3.ninfer"
if not exist "%MODEL%" set "MODEL=C:\AI\models\qwen3_8_27b_nvfp4nvidia.v3.ninfer"
REM The lane's template travels with the archive; the source-tree copy is the fallback. Passing it
REM explicitly stops the lane inheriting whichever template its artifact embeds -- the two shipped
REM artifacts embed different ones, and the embedded pair predate the reasoning-effort alias mapping.
set "TEMPLATE=%~dp0chat_templates\qwen3_8.jinja"
if not exist "%TEMPLATE%" set "TEMPLATE=%~dp0tools\chat_templates\qwen3_8.jinja"
REM The lane's environment, from profiles.launcher_env. A harness that starts Serve has to start it
REM this way too, or what it measures is not what ships.
set "NINFER_CUDA_SYNC=blocking"

if not exist "%SERVE%" (
    echo [ERROR] Engine not found.
    echo         Expected ninfer-serve.exe beside this launcher,
    echo         or a source build at build\apps\ninfer-serve.exe in the repository
    pause
    exit /b 1
)
if not exist "%MODEL%" (
    echo [ERROR] Artifact not found.
    echo         Expected models\qwen3_8_27b_nvfp4nvidia.v3.ninfer beside this launcher,
    echo         or C:\AI\models\qwen3_8_27b_nvfp4nvidia.v3.ninfer
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
REM     Image File Execution Options\<name>   GlobalFlag 0x1000, heap tagging
REM     Image File Execution Options          USTEnabled = <name>, user-mode stack trace database
REM Either one makes the allocator capture a stack on every allocation. Prompt preparation is
REM allocation-dense, so it costs about thirty times what it should -- measured at 3.7 s against
REM 122 ms for a 229-message prompt -- and that is most of the time to first token on a long
REM conversation. Nothing in this tree sets either value, so a machine that has one is a machine
REM someone debugged on.
for %%F in ("%SERVE%") do set "SERVE_NAME=%%~nxF"
set "IFEO_ROOT=HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Image File Execution Options"
set "IFEO_STALE="
reg query "%IFEO_ROOT%\%SERVE_NAME%" >nul 2>&1 && set "IFEO_STALE=1"
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
    echo             reg delete "%IFEO_ROOT%\%SERVE_NAME%" /f
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
        echo         build_windows.bat stages them from the ffmpeg\bin directory it downloads.
        echo         Without them the engine exits 0xC0000135 without printing a reason.
        pause
        exit /b 1
    )
)

netstat -ano | findstr ":8092" | findstr /I "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [ERROR] Port 8092 is already in use.
    echo         Something is already listening there. Stop it, or change the --port flag
    echo         in this launcher. To see what holds it:
    echo             netstat -ano ^| findstr ":8092"
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

"%SERVE%" "%MODEL%" ^
  --vision ^
  --spec dflash2 ^
  --draft-tokens 7 ^
  --lm-head-draft ^
  --host 127.0.0.1 ^
  --port 8092 ^
  --model-id qwen3.8-27b-nvidia-v3-dflash2-vision ^
  --max-context 262144 ^
  --device-state-slots 1 ^
  --kv-capacity auto ^
  --kv-dtype fp8 ^
  --prefill-chunk 8192 ^
  --max-concurrency 1 ^
  --host-state-slots 16 ^
  --host-kv-mib 8192 ^
  --max-shared-prefixes 7 ^
  --max-private-continuations 8 ^
  --max-long-anchors-per-continuation 4 ^
  --context-cache-policy rolling ^
  --preserve-thinking ^
  --default-thinking-budget 4096 ^
  --pending-timeout-ms 600000 ^
  --chat-template %TEMPLATE%

pause
