@echo off
REM ============================================================================
REM  NVFP4 + DFlash2 (no Vision)
REM
REM  Measured on this machine (32 GB RTX 5090), fp8 KV at the ceiling below:
REM      context 180,224   decode 258.2 tok/s   draft acceptance 59.4%
REM      runtime 8.06 GiB   free VRAM 1.06 GiB
REM
REM  Highest-speed nvfp4 profile. --lm-head-draft is OFF: measured faster and
REM  a full ladder step more context than with it (258.2 @180,224 vs 225.0 @163,840).
REM
REM  Requires the FFmpeg runtime DLLs beside the executable (staged by
REM  build_v3.cmd); without them the process exits 0xC0000135.
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
  --spec dflash2 ^
  --draft-tokens 7 ^
  --host 127.0.0.1 ^
  --port 8088 ^
  --model-id qwen3.8-27b-nvfp4-v3-dflash2 ^
  --max-context 180224 ^
  --kv-capacity auto ^
  --kv-dtype fp8 ^
  --prefill-chunk 8192 ^
  --max-concurrency 1 ^
  --device-state-slots 1 ^
  --host-state-slots 8 ^
  --host-kv-mib 8192 ^
  --cors ^
  --preserve-thinking ^
  --default-thinking-budget 4096 ^
  --pending-timeout-ms 600000

pause
