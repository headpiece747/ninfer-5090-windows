@echo off
setlocal
call "%~dp0launcher_env.bat"
cd /d "%ROOT%"

echo =======================================================
echo Downloading Qwen 3.8 27B QUASAR QAT NVFP4 Model...
echo Source: https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4qat-NInfer
echo Destination: %MODEL%
echo File Size: ~17.35 GiB (18,638,209,796 bytes)
echo =======================================================

if defined PYTHON_EXE (
    echo Using Python + huggingface_hub (hf_transfer Rust engine) for maximum download speed...
    "%PYTHON_EXE%" "%ROOT%download_model.py"
    if %ERRORLEVEL% equ 0 (
        echo.
        echo =======================================================
        echo [SUCCESS] Model artifact downloaded and verified!
        echo =======================================================
        pause
        exit /b 0
    )
    echo Python download failed or module missing. Falling back to curl...
)

set "DEST_DIR=C:\ai\models"
if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"

set "URL=https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4qat-NInfer/resolve/main/qwen3_8_27b_nvfp4qat.ninfer"

echo.
echo Starting download (curl with resume support)...
curl.exe -L -C - --retry 5 --retry-delay 3 -o "%MODEL%" "%URL%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Download encountered an error. You can run this script again to resume.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo =======================================================
echo Download complete! Verifying file integrity...
if defined PYTHON_EXE (
    "%PYTHON_EXE%" "%ROOT%download_model.py" --verify-only
) else (
    echo [ERROR] Python 3.11+ is required to verify file integrity.
    pause
    exit /b 1
)
echo =======================================================
pause
