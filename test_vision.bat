@echo off
setlocal
call "%~dp0launcher_env.bat" verify_cli || exit /b 1
cd /d "%ROOT%"

if not exist "%ROOT%test_image.png" call :create_image

echo =======================================================
echo Testing Qwen 3.8 27B QUASAR QAT Vision Input (RTX 5090)
echo Mode: Multimodal Vision + FP8 KV Cache (262k) + MTP-3
echo =======================================================
echo Executable: %BIN%
echo Model:      %MODEL%
echo Image:      %ROOT%test_image.png
echo Messages:   %ROOT%test_vision_messages.json
echo.

"%BIN%" "%MODEL%" ^
  --vision ^
  --messages "%ROOT%test_vision_messages.json" ^
  --max-new 128 ^
  %QUASAR_ARGS%

echo.
pause
exit /b 0

:create_image
if not defined PYTHON_EXE (
    echo [ERROR] Python 3.11+ not found in PATH to generate sample test image.
    exit /b 1
)
echo Generating sample test image using %PYTHON_EXE%...
"%PYTHON_EXE%" -c "from PIL import Image, ImageDraw; img = Image.new('RGB', (256, 256), color=(20, 40, 160)); draw = ImageDraw.Draw(img); draw.ellipse([64, 64, 192, 192], fill=(255, 215, 0), outline=(255, 255, 255), width=4); draw.rectangle([120, 80, 136, 176], fill=(255, 255, 255)); draw.rectangle([88, 120, 168, 136], fill=(255, 255, 255)); img.save(r'%ROOT%test_image.png')"
exit /b 0
