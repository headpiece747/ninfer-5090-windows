@echo off
REM ============================================================================
REM  NVFP4-full + DFlash2 + Vision
REM
REM  Measured on this machine (32 GB RTX 5090), fp8 KV at the ceiling below:
REM      context 262,144   decode 326.6 tok/s   draft acceptance 63.7%
REM      runtime 10.7 GiB   free VRAM 1.74 GiB
REM
REM  Second artifact, same reach as QUASAR: 262,144 with Vision. This is the
REM  tightest profile in the set; dropping --lm-head-draft buys 0.33 GiB at ~2% slower.
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
if not exist "%SERVE%" set "SERVE=C:\AI\ninfer-v3-windows\build\apps\ninfer-serve.exe"
set "MODEL=%~dp0models\qwen3_8_27b_nvfp4full.v3.ninfer"
if not exist "%MODEL%" set "MODEL=C:\AI\models\qwen3_8_27b_nvfp4full.v3.ninfer"

if not exist "%SERVE%" (
    echo [ERROR] Engine not found.
    echo         Expected ninfer-serve.exe beside this launcher,
    echo         or a source build at C:\AI\ninfer-v3-windows\build\apps\ninfer-serve.exe
    pause
    exit /b 1
)
if not exist "%MODEL%" (
    echo [ERROR] Artifact not found.
    echo         Expected %~dp0models\qwen3_8_27b_nvfp4full.v3.ninfer
    echo         or C:\AI\models\qwen3_8_27b_nvfp4full.v3.ninfer
    echo         Run download_model.bat to fetch it.
    pause
    exit /b 1
)

"%SERVE%" "%MODEL%" ^
  --vision ^
  --spec dflash2 ^
  --draft-tokens 7 ^
  --lm-head-draft ^
  --host 127.0.0.1 ^
  --port 8088 ^
  --model-id qwen3.8-27b-nvfp4-v3-dflash2-vision ^
  --max-context 262144 ^
  --kv-capacity auto ^
  --kv-dtype fp8 ^
  --prefill-chunk 8192 ^
  --max-concurrency 1 ^
  --device-state-slots 1 ^
  --host-state-slots 8 ^
  --host-kv-mib 8192 ^
  --max-shared-prefixes 7 ^
  --max-private-continuations 8 ^
  --max-long-anchors-per-continuation 4 ^
  --preserve-thinking ^
  --default-thinking-budget 4096 ^
  --pending-timeout-ms 600000

pause
