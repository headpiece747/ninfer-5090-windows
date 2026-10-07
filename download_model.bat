@echo off
REM ============================================================================
REM  NInfer v3 model build
REM
REM  Fetches the source checkpoints one line needs -- at the revisions pinned in
REM  download_model.py -- and converts them into the .ninfer image the launchers
REM  look for, with the converter this archive ships under tools\.
REM
REM  Nothing is downloaded prebuilt: the only artifact is the one this machine
REM  builds. Expect ~75-80 GiB of source checkpoints (they are shared between
REM  lines), 18-19 GiB for the image, and minutes of GPU time per line.
REM ============================================================================
setlocal
call "%~dp0launcher_env.bat"

echo =======================================================
echo  NInfer v3 model build
echo =======================================================
echo   1. QUASAR QAT       (recommended)   sources ~75 GiB
echo   2. NVFP4-full       (MTP lane)      sources ~77 GiB
echo   3. NVFP4-full noex  (DFlash2 lane)  sources ~77 GiB
echo   4. Swift 1.5                        sources ~80 GiB
echo   5. NVIDIA ModelOpt                  sources ~76 GiB
echo =======================================================
set "CHOICE="
set /p "CHOICE=Choose 1-5 [1]: "
if "%CHOICE%"=="" set "CHOICE=1"

if "%CHOICE%"=="1" set "LINE=quasar"
if "%CHOICE%"=="2" set "LINE=full"
if "%CHOICE%"=="3" set "LINE=noex"
if "%CHOICE%"=="4" set "LINE=swift15"
if "%CHOICE%"=="5" set "LINE=nvidia"
if not defined LINE (
    echo [ERROR] Invalid choice: %CHOICE%
    pause
    exit /b 1
)

if not defined PYTHON_EXE (
    echo [ERROR] Python with huggingface_hub is required but was not found.
    echo         Install Python 3.11+ and huggingface_hub, then retry.
    pause
    exit /b 1
)

echo.
"%PYTHON_EXE%" "%~dp0build_model.py" --line %LINE% --hf-src "%~dp0hf-src" --dest "%~dp0models"
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] The build failed; the converter's own message is above.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo =======================================================
echo [SUCCESS] %LINE% built. Start it with the matching start_*.bat.
echo =======================================================
pause
