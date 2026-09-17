@echo off
REM ============================================================================
REM  QUASAR QAT + MTP4 + Vision
REM
REM  Measured on this machine (32 GB RTX 5090), fp8 KV at the ceiling below:
REM      context 262,144   decode 225.4 tok/s   draft acceptance 65.3%
REM      runtime 10.4 GiB   free VRAM 3.08 GiB
REM
REM  Lower-VRAM QUASAR profile; MTP4 measured fastest of MTP 2-5 on QUASAR.
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
set "MODEL=C:\AI\models\qwen3_8_27b_nvfp4qat.v3.ninfer"

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
  --vision ^
  --spec mtp ^
  --draft-tokens 4 ^
  --lm-head-draft ^
  --host 127.0.0.1 ^
  --port 8087 ^
  --model-id qwen3.8-27b-quasar-v3-mtp4-vision ^
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
  --cors ^
  --preserve-thinking ^
  --default-thinking-budget 4096 ^
  --pending-timeout-ms 600000

pause
