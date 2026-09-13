@echo off
setlocal
echo =======================================================
echo Downloading Qwen 3.8 27B QUASAR QAT NVFP4 Model...
echo Source: https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4qat-NInfer
echo Destination: C:\ai\models\qwen3_8_27b_nvfp4qat.ninfer
echo File Size: ~17.35 GiB (18,638,209,796 bytes)
echo =======================================================

where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo Using Python + huggingface_hub (hf_transfer Rust engine) for maximum download speed...
    python "%~dp0download_model.py"
    if %ERRORLEVEL% equ 0 goto done
    echo Python download failed or hf_transfer error. Falling back to curl...
)

set "DEST_DIR=C:\ai\models"
if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"

set "DEST_FILE=%DEST_DIR%\qwen3_8_27b_nvfp4qat.ninfer"
set "URL=https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4qat-NInfer/resolve/main/qwen3_8_27b_nvfp4qat.ninfer"

echo.
echo Starting download (curl with resume support)...
curl.exe -L -C - --retry 5 --retry-delay 3 -o "%DEST_FILE%" "%URL%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Download encountered an error. You can run this script again to resume.
    pause
    exit /b %ERRORLEVEL%
)

:done
echo.
echo =======================================================
echo Download complete!
echo Verifying file integrity (Size + SHA-256 Checksum)...
set "DEST_FILE=C:\ai\models\qwen3_8_27b_nvfp4qat.ninfer"
for %%I in ("%DEST_FILE%") do set "SIZE=%%~zI"
echo File size: %SIZE% bytes (Expected: 18638209796 bytes)

if not "%SIZE%"=="18638209796" (
    echo [WARNING] File size does not match expected 18638209796 bytes. Run again to resume or re-download.
    pause
    exit /b 1
)

echo [SUCCESS] File size matches exactly!
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python "%~dp0download_model.py" --verify-only
) else (
    echo Verifying SHA-256 checksum (Expected: df3c9c3a3660d688f0c2158d54fef8c66f21d723bd7a1b0b12acc908e86d12b7)...
    certutil -hashfile "%DEST_FILE%" SHA256
)
echo =======================================================
pause
