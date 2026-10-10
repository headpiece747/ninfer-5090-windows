@echo off
REM ============================================================================
REM  Swift 1.5 + DFlash2 + Vision
REM
REM  Measured on this machine (32 GB RTX 5090), fp8 KV at the ceiling below:
REM      context 262,144   decode 280.7 tok/s   draft acceptance 51.8%
REM      runtime 8.83 GiB   free VRAM 4.20 GiB
REM
REM  Swift 1.5 replaces Swift 1.0 on both Swift lanes; these figures are measured 2026-09-30 with `profile` mode through this launcher's own flags. Width 7 was measured against every window 1-15 on the code domain and re-measured on four others: 13 is 33% faster on code and slower on chinese, prose and dialogue, so the width that maximises the worst domain is the one already shipped. This image encodes the DFlash2 draft to NVFP4 where Swift 1.0's left it at Q8, which is 0.77 GiB smaller. The encoding is measured per target and it reverses here: on Swift 1.0 the same change lost 3.2 acceptance points, so it was measured rather than assumed. An earlier revision of this note said the Q8-draft build of this same checkpoint was REFUSED at 262,144 with Vision. That was true when written and is no longer: the tiled saturation guard moved the memory picture, and the claim was not re-measured until 2026-10-04, when it was falsified. Both arms were measured with `verify` minutes apart on the q8draft image already in _superseded, and the Q8 build DOES serve at 262,144 with Vision. It is no longer a capacity question. On the default `code` domain the NVFP4 draft is also the better build on all three measured axes -- 348.3 tok/s and 63.0% acceptance against 290.3 and 49.3%, on 0.78 GiB MORE free memory (2.83 against 2.05 GiB) -- which reproduces this lane's recorded 2026-09-30 figures to within 0.7% five weeks later. That is the documented default domain, so the shipped encoding is settled by measurement rather than merely retained. It does NOT hold everywhere: on `chinese` the ordering reverses and Q8 is 11.7% faster with 4.6 points better acceptance. This note's figures are `code`; the per-domain split is in docs/research/swift15-lane-measurement.md. KV format changed from fp8 to k8v4 on 2026-10-09, interleaved on the three cells with each arm twice: the deciding cells' worst improves from 196.2 to 244.4 tok/s (long code +24.6%, acceptance 31.06 -> 42.69) while the published cell gives up 7.9% decode and 6.5 acceptance points (313.5 -> 287.3, 58.32% -> 51.83%) and long Chinese 4.2% and 2.5 points -- both disclosed. Adopted under the code-weighted rule's bound: the deciding worst case improves by more than the worst disclosed non-code cell gives up. Quality costs +0.098% corpus perplexity overall and +0.257% on the worst domain.
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
REM
REM  TDR (Timeout Detection and Recovery): the RTX 5090 is a GeForce, so it runs WDDM with a
REM  2-second TDR budget. No single GPU operation in any measured profile approaches this
REM  (prefill is chunked, decode uses CUDA graphs with a handful of tokens per round, weight
REM  upload is DMA). If a future profile ever contains a launch that could exceed 2 s, set
REM  TdrDelay in the registry (HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers) to
REM  raise the threshold. A TDR surfaces as a CUDA error and the engine's fail-stop latch
REM  terminates the process rather than continuing with a lost device.
REM ============================================================================
setlocal

REM Resolve beside this launcher first, so the released archive is portable wherever it is
REM extracted. The source-tree copy is the same shape one directory down: build/apps and
REM tools/chat_templates both sit beside the launcher in the repository, so the fallback is relative
REM too. An absolute path here bakes one machine's checkout into a file that ships, and it is what
REM made the generated launchers unverifiable anywhere else.
set "SERVE=%~dp0ninfer-serve.exe"
if not exist "%SERVE%" set "SERVE=%~dp0build\apps\ninfer-serve.exe"
set "MODEL=%~dp0models\qwen3_8_27b_nvfp4swift15.v3.ninfer"
if not exist "%MODEL%" set "MODEL=C:\AI\models\qwen3_8_27b_nvfp4swift15.v3.ninfer"
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
    echo         Expected models\qwen3_8_27b_nvfp4swift15.v3.ninfer beside this launcher,
    echo         or C:\AI\models\qwen3_8_27b_nvfp4swift15.v3.ninfer
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

netstat -ano | findstr ":8090" | findstr /I "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [ERROR] Port 8090 is already in use.
    echo         Something is already listening there. Stop it, or change the --port flag
    echo         in this launcher. To see what holds it:
    echo             netstat -ano ^| findstr ":8090"
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
  --kv-dtype k8v4 ^
  --host 127.0.0.1 ^
  --port 8090 ^
  --model-id qwen3.8-27b-swift15-v3-dflash2-vision ^
  --max-context 262144 ^
  --device-state-slots 1 ^
  --kv-capacity auto ^
  --prefill-chunk 8192 ^
  --max-concurrency 1 ^
  --host-context-mib 8192 ^
  --preserve-thinking ^
  --default-thinking-budget 4096 ^
  --pending-timeout-ms 600000 ^
  --chat-template %TEMPLATE%

pause
