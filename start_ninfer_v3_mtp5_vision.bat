@echo off
REM ============================================================================
REM  NVFP4 + MTP5 + Vision
REM
REM  Measured on this machine (32 GB RTX 5090), fp8 KV at the ceiling below:
REM      context 212,992   decode 204.8 tok/s   draft acceptance 61.7%
REM      runtime 8.77 GiB   free VRAM 1.03 GiB
REM
REM  Vision again costs context only, not throughput.
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
  --vision ^
  --spec mtp ^
  --draft-tokens 5 ^
  --lm-head-draft ^
  --host 127.0.0.1 ^
  --port 8091 ^
  --model-id qwen3.8-27b-nvfp4-v3-mtp5-vision ^
  --max-context 212992 ^
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
