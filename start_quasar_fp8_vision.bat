@echo off
setlocal
call "%~dp0launcher_env.bat" verify_serve || exit /b 1
cd /d "%ROOT%"

echo =======================================================
echo Starting Qwen 3.8 27B QUASAR QAT on RTX 5090
echo Mode: Vision Multimodal + FP8 KV Cache (262k Context) + MTP-5
echo =======================================================
echo Executable: %SERVE_BIN%
echo Model:      %MODEL%
echo.

"%SERVE_BIN%" "%MODEL%" ^
  --vision ^
  --host 127.0.0.1 ^
  --port 8080 ^
  --model-id qwen3.8-27b-quasar ^
  %QUASAR_ARGS% ^
  --lm-head-draft ^
  --prefill-chunk 4096 ^
  --max-concurrency 1 ^
  --device-state-slots 1 ^
  --host-state-slots 16 ^
  --host-kv-mib 16384 ^
  --cors ^
  --preserve-thinking ^
  --default-thinking-budget 4096 ^
  --pending-timeout-ms 600000

pause
