@echo off
setlocal
echo =======================================================
echo Starting Qwen 3.8 27B QUASAR QAT on RTX 5090
echo Mode: Vision Multimodal + FP8 KV Cache (262k Context) + MTP-3
echo =======================================================

call "%~dp0launcher_env.bat"
cd /d "%ROOT%"

if not exist "%SERVE_BIN%" (
    echo [ERROR] Cannot find ninfer-serve executable at %SERVE_BIN%
    echo Please run build_windows.bat first.
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

echo Executable: %SERVE_BIN%
echo Model:      %MODEL%
echo.

"%SERVE_BIN%" "%MODEL%" ^
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
