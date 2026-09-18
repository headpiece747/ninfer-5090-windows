@echo off
set "ROOT=%~dp0"
set "BIN=%ROOT%build\apps\ninfer.exe"
set "SERVE_BIN=%ROOT%build\apps\ninfer-serve.exe"
if not defined MODEL REM Relative first so a released archive is self-contained; the absolute path keeps
REM the source-tree workflow unchanged.
set "MODEL=%~dp0models\qwen3_8_27b_nvfp4qat.v3.ninfer"
if not exist "%MODEL%" set "MODEL=C:\ai\models\qwen3_8_27b_nvfp4qat.v3.ninfer"
set "QUASAR_ARGS=--vision --spec mtp --draft-tokens 4 --lm-head-draft --max-context 262144 --kv-capacity auto --kv-dtype fp8 --prefill-chunk 8192"

:: Select Python 3.11 interpreter explicitly per AGENTS.md
set "PYTHON_EXE="
if exist "%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe" (
    set "PYTHON_EXE=%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe"
)
if not defined PYTHON_EXE (
    where py >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        py -3.11 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
        if %ERRORLEVEL% equ 0 REM Resolve to a single quotable path: "py -3.11" cannot be used inside quotes, so
REM download_model.bat would fail on a machine where the launcher is the only interpreter.
for /f "delims=" %%p in ('py -3.11 -c "import sys;print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%p"
    )
)
if not defined PYTHON_EXE (
    where python >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
        if %ERRORLEVEL% equ 0 set "PYTHON_EXE=python"
    )
)

:: Execute action if requested via argument
if "%~1"=="verify_cli" goto :verify_cli
if "%~1"=="verify_serve" goto :verify_serve
if "%~1"=="verify_model" goto :verify_model
exit /b 0

:verify_cli
if not exist "%BIN%" (
    echo [ERROR] Cannot find ninfer CLI executable at: %BIN%
    echo Please run build_windows.bat first.
    pause
    exit /b 1
)
goto :verify_model

:verify_serve
if not exist "%SERVE_BIN%" (
    echo [ERROR] Cannot find ninfer-serve executable at: %SERVE_BIN%
    echo Please run build_windows.bat first.
    pause
    exit /b 1
)
goto :verify_model

:verify_model
if not exist "%MODEL%" (
    echo [ERROR] Model artifact not found at: %MODEL%
    echo Please run download_model.bat first.
    pause
    exit /b 1
)
exit /b 0
