@echo off
setlocal
echo =======================================================
echo Testing Qwen 3.8 27B QUASAR QAT Vision Input (RTX 5090)
echo Mode: Multimodal Vision + FP8 KV Cache (262k) + MTP-3
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
echo Image:      %ROOT%test_image.png
echo Messages:   %ROOT%test_vision_messages.json
echo.

"%BIN%" "%MODEL%" ^
  --vision ^
  --messages "%ROOT%test_vision_messages.json" ^
  --max-context 262144 ^
  --kv-capacity auto ^
  --max-new 128 ^
  --kv-dtype fp8 ^
  --spec mtp ^
  --draft-tokens 3

echo.
pause
