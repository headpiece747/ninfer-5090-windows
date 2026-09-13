@echo off
set "ROOT=%~dp0"
set "BIN=%ROOT%build\apps\ninfer.exe"
set "SERVE_BIN=%ROOT%build\apps\ninfer-serve.exe"
if not defined MODEL set "MODEL=C:\ai\models\qwen3_8_27b_nvfp4qat.ninfer"
set "QUASAR_ARGS=--max-context 262144 --kv-capacity auto --kv-dtype fp8 --spec mtp --draft-tokens 5"

:: Select Python 3.11 interpreter explicitly per AGENTS.md
set "PYTHON_EXE="
if exist "%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe" (
    set "PYTHON_EXE=%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe"
)
if not defined PYTHON_EXE (
    where py >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        py -3.11 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
        if %ERRORLEVEL% equ 0 set "PYTHON_EXE=py -3.11"
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
