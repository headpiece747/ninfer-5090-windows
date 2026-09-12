@echo off
setlocal
echo =======================================================
echo Testing Qwen 3.8 27B QUASAR QAT via CLI (RTX 5090)
echo Mode: Vision Enabled + FP8 KV + MTP-3 Speculation
echo =======================================================

set "ROOT=%~dp0"
set "BIN=%ROOT%build\apps\ninfer.exe"
set "MODEL=C:\ai\models\qwen3_8_27b_nvfp4qat.ninfer"
if not exist "%MODEL%" set "MODEL=%ROOT%qwen3_8_27b_nvfp4qat.ninfer"

if not exist "%BIN%" (
    echo [ERROR] Cannot find ninfer CLI executable at %BIN%
    pause
    exit /b 1
)

if not exist "%MODEL%" (
    echo [ERROR] Model artifact not found at %MODEL%
    pause
    exit /b 1
)

echo Executable: %BIN%
echo Model:      %MODEL%
echo.

"%BIN%" "%MODEL%" ^
  --vision ^
  --prompt "Explain quantum computing in three clear, concise bullet points." ^
  --max-context 16384 ^
  --max-new 256 ^
  --kv-dtype fp8 ^
  --spec mtp ^
  --draft-tokens 3

echo.
pause
