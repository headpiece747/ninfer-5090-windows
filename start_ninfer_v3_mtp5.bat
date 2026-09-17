@echo off
REM ============================================================================
REM  NVFP4 + MTP5 (no Vision)
REM
REM  Measured on this machine (32 GB RTX 5090), fp8 KV at the ceiling below:
REM      context 240,000   decode 206.0 tok/s   draft acceptance 61.7%
REM      runtime 9.49 GiB   free VRAM 0.58 GiB
REM
REM  Highest-context nvfp4 profile; slower than DFlash2 by ~20% for 60k more context.
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

set "V3=C:\AI\ninfer-v3-windows"
set "MODEL=C:\AI\models\qwen3_8_27b_nvfp4.v3.ninfer"

if not exist "%V3%\build\apps\ninfer-serve.exe" (
    echo [ERROR] Engine not found at %V3%\build\apps\ninfer-serve.exe
    echo         Run build_windows.bat first.
    pause
    exit /b 1
)
if not exist "%MODEL%" (
    echo [ERROR] Artifact not found at %MODEL%
    pause
    exit /b 1
)

"%V3%\build\apps\ninfer-serve.exe" "%MODEL%" ^
  --spec mtp ^
  --draft-tokens 5 ^
  --lm-head-draft ^
  --host 127.0.0.1 ^
  --port 8090 ^
  --model-id qwen3.8-27b-nvfp4-v3-mtp5 ^
  --max-context 240000 ^
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
  --cors ^
  --preserve-thinking ^
  --default-thinking-budget 4096 ^
  --pending-timeout-ms 600000

pause
