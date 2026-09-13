@echo off
setlocal
echo =======================================================
echo Testing Qwen 3.8 27B QUASAR QAT Vision Input (RTX 5090)
echo Mode: Multimodal Vision + FP8 KV Cache (262k) + MTP-3
echo =======================================================

set "ROOT=%~dp0"
cd /d "%ROOT%"
set "BIN=%ROOT%build\apps\ninfer.exe"
set "MODEL=C:\ai\models\qwen3_8_27b_nvfp4qat.ninfer"

if not exist "%BIN%" (
    echo [ERROR] Cannot find ninfer CLI executable at %BIN%
    echo Please run build_windows.bat first.
    pause
    exit /b 1
)

if not exist "%MODEL%" (
    echo [ERROR] Model artifact not found at:
    echo   %MODEL%
    echo Please run download_model.bat first to download the model artifact.
    pause
    exit /b 1
)

if not exist "%ROOT%test_image.png" call :create_image

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
exit /b 0

:create_image
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found in PATH to generate sample test image.
    exit /b 1
)
echo Generating sample test image...
python -c "from PIL import Image, ImageDraw; img = Image.new('RGB', (256, 256), color=(20, 40, 160)); draw = ImageDraw.Draw(img); draw.ellipse([64, 64, 192, 192], fill=(255, 215, 0), outline=(255, 255, 255), width=4); draw.rectangle([120, 80, 136, 176], fill=(255, 255, 255)); draw.rectangle([88, 120, 168, 136], fill=(255, 255, 255)); img.save(r'%ROOT%test_image.png')"
exit /b 0
