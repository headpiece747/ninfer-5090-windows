@echo off
setlocal
echo =======================================================
echo Starting Qwen 3.8 27B QUASAR QAT on RTX 5090
echo Mode: Vision Multimodal + FP8 KV Cache (262k Context) + MTP-3
echo =======================================================

set "ROOT=%~dp0"
set "BIN=%ROOT%build\apps\ninfer-serve.exe"
set "MODEL=C:\ai\models\qwen3_8_27b_nvfp4qat.ninfer"

if not exist "%BIN%" (
    echo [ERROR] Cannot find ninfer-serve executable. Please run build_windows.bat first.
    pause
    exit /b 1
)

if not exist "%MODEL%" (
    echo [ERROR] Model artifact not found at:
    echo   %MODEL%
    echo Please run download_model.bat first to download the 17.35 GiB artifact.
    pause
    exit /b 1
)

echo Executable: %BIN%
echo Model:      %MODEL%
echo.

"%BIN%" "%MODEL%" ^
  --vision ^
  --host 127.0.0.1 ^
  --port 8080 ^
  --model-id qwen3.8-27b-nvfp4qat ^
  --max-context 262144 ^
  --kv-capacity auto ^
  --kv-dtype fp8 ^
  --spec mtp ^
  --draft-tokens 3 ^
  --lm-head-draft ^
  --prefill-chunk 1024 ^
  --max-concurrency 1 ^
  --device-state-slots 1 ^
  --host-state-slots 16 ^
  --host-kv-mib 16384 ^
  --cors ^
  --preserve-thinking ^
  --default-thinking-budget 4096 ^
  --pending-timeout-ms 600000

pause
