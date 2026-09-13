@echo off
setlocal
call "%~dp0launcher_env.bat" verify_cli || exit /b 1
cd /d "%ROOT%"

echo =======================================================
echo Testing Qwen 3.8 27B QUASAR QAT via CLI (RTX 5090)
echo Mode: Vision Enabled + FP8 KV + MTP-3 Speculation
echo =======================================================
echo Executable: %BIN%
echo Model:      %MODEL%
echo.

"%BIN%" "%MODEL%" ^
  --vision ^
  --prompt "Explain quantum computing in three clear, concise bullet points." ^
  --max-new 256 ^
  %QUASAR_ARGS%

echo.
pause
