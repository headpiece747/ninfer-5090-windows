@echo off
setlocal
call "%~dp0launcher_env.bat"
cd /d "%ROOT%"

echo =======================================================
echo Downloading Qwen 3.8 27B QUASAR QAT NVFP4 Model...
echo Destination: %MODEL%
echo File Size: ~17.35 GiB (18,638,209,796 bytes)
echo =======================================================

if not defined PYTHON_EXE (
    echo [ERROR] Python 3.11 is required but not found.
    echo Please ensure Python 3.11 is installed or available in PATH.
    pause
    exit /b 1
)

for %%I in ("%MODEL%") do set "DEST_DIR=%%~dpI"
if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"

echo Using Python 3.11 + huggingface_hub for download and bit-for-bit SHA-256 verification...
"%PYTHON_EXE%" "%ROOT%download_model.py" --dest "%MODEL%"
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Download or verification failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo =======================================================
echo [SUCCESS] Model artifact downloaded and verified!
echo =======================================================
pause
