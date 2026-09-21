@echo off
REM ============================================================================
REM  NInfer v3 model download
REM
REM  Fetches the artifact a launcher expects, verifies its size and SHA-256, and
REM  renames it to the .v3 form the launchers look for. Models land beside this
REM  file so an extracted release is self-contained; the launchers also accept
REM  C:\AI\models as a fallback.
REM
REM  Both repositories publish a v3 container now, so either artifact downloads
REM  and runs directly. The offline upgrader ships for anyone holding a copy
REM  fetched before the republish; the engine names it if a v2 file is loaded.
REM ============================================================================
setlocal
call "%~dp0launcher_env.bat"

echo =======================================================
echo  NInfer v3 model download
echo =======================================================
echo   1. QUASAR QAT   (recommended)   ~17.4 GiB
echo   2. NVFP4-full                    ~18.1 GiB
echo =======================================================
set "CHOICE="
set "ARTIFACT="
set /p "CHOICE=Choose 1 or 2 [1]: "
if "%CHOICE%"=="" set "CHOICE=1"

if "%CHOICE%"=="1" set "ARTIFACT=quasar"
if "%CHOICE%"=="2" set "ARTIFACT=nvfp4full"
if not defined ARTIFACT (
    echo [ERROR] Invalid choice: %CHOICE%
    pause
    exit /b 1
)

REM Each artifact downloads straight to the .v3 name the launchers look for. It used to stage
REM QUASAR under a .v2 name for the upgrader; both repositories publish v3 now.
if "%ARTIFACT%"=="quasar"     set "TARGET=%~dp0models\qwen3_8_27b_nvfp4qat.v3.ninfer"
if "%ARTIFACT%"=="nvfp4full"  set "TARGET=%~dp0models\qwen3_8_27b_nvfp4full.v3.ninfer"

if not defined PYTHON_EXE (
    echo [ERROR] Python with huggingface_hub is required but was not found.
    echo         Install Python 3.11+ and huggingface_hub, then retry.
    pause
    exit /b 1
)

echo.
echo Downloading %ARTIFACT% to %TARGET%
echo.
"%PYTHON_EXE%" "%~dp0download_model.py" --artifact %ARTIFACT% --dest "%TARGET%"
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Download or verification failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo =======================================================
echo [SUCCESS] %ARTIFACT% downloaded and verified.
echo           Start it with the matching start_*.bat in this folder.
echo =======================================================
pause
