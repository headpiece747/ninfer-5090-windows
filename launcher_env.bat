@echo off
set "ROOT=%~dp0"
set "BIN=%ROOT%build\apps\ninfer.exe"
set "SERVE_BIN=%ROOT%build\apps\ninfer-serve.exe"
REM Relative first, so a released archive is self-contained; the absolute path keeps the
REM source-tree workflow unchanged, and an externally set MODEL wins over both.
if defined MODEL goto :model_resolved
set "MODEL=%~dp0models\qwen3_8_27b_nvfp4qat.v3.ninfer"
if exist "%MODEL%" goto :model_resolved
set "MODEL=C:\ai\models\qwen3_8_27b_nvfp4qat.v3.ninfer"
:model_resolved
set "QUASAR_ARGS=--vision --spec mtp --draft-tokens 4 --lm-head-draft --max-context 262144 --kv-capacity auto --kv-dtype fp8 --prefill-chunk 8192"

:: Select Python 3.11 explicitly per AGENTS.md. Each candidate is checked in its own labelled
:: branch: a one-line `if <cond> for /f ...` corrupts cmd's label scan ("cannot find the batch
:: label"), and "py -3.11" cannot itself be stored in quotes, so it resolves to one quotable path.
set "PYTHON_EXE="
set "UV_PYTHON=%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe"
if exist "%UV_PYTHON%" goto :use_uv_python
where py >nul 2>&1
if errorlevel 1 goto :try_path_python
py -3.11 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
if errorlevel 1 goto :try_path_python
for /f "delims=" %%p in ('py -3.11 -c "import sys;print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%p"
goto :python_resolved
:try_path_python
where python >nul 2>&1
if errorlevel 1 goto :python_resolved
python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
if errorlevel 1 goto :python_resolved
set "PYTHON_EXE=python"
goto :python_resolved
:use_uv_python
set "PYTHON_EXE=%UV_PYTHON%"
:python_resolved

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
