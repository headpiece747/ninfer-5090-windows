@echo off
set "ROOT=%~dp0"
set "BIN=%ROOT%ninfer.exe"
if not exist "%BIN%" set "BIN=%ROOT%build\apps\ninfer.exe"
set "SERVE_BIN=%ROOT%ninfer-serve.exe"
if not exist "%SERVE_BIN%" set "SERVE_BIN=%ROOT%build\apps\ninfer-serve.exe"
REM Relative first, so a released archive is self-contained; the absolute path keeps the
REM source-tree workflow unchanged, and an externally set MODEL wins over both.
if defined MODEL goto :model_resolved
set "MODEL=%~dp0models\qwen3_8_27b_nvfp4qat.v3.ninfer"
if exist "%MODEL%" goto :model_resolved
set "MODEL=C:\ai\models\qwen3_8_27b_nvfp4qat.v3.ninfer"
:model_resolved
REM The lane's template, resolved the same way and for the same reason. It has to be a variable rather
REM than a literal path because QUASAR_ARGS is compared byte-for-byte between machines, and an absolute
REM path here would bake one checkout into a file that ships. The launchers resolve the identical
REM variable beside themselves, so the CLI harness and the server read the same template.
if defined TEMPLATE goto :template_resolved
set "TEMPLATE=%~dp0chat_templates\qwen3_8.jinja"
if exist "%TEMPLATE%" goto :template_resolved
set "TEMPLATE=%~dp0tools\chat_templates\qwen3_8.jinja"
:template_resolved
set "QUASAR_ARGS=--chat-template %TEMPLATE% --vision --spec mtp --draft-tokens 4 --proposal-head optimized --max-context 262144 --context-capacity auto --memory-headroom 1024MiB --prefill-chunk 8192 --kv-dtype fp8"

:: Select Python 3.11+ explicitly. Each candidate is checked in its own labelled
:: branch: a one-line `if <cond> for /f ...` corrupts cmd's label scan ("cannot find the batch
:: label"), and "py -3.11" / "py -3" cannot itself be stored in quotes, so it resolves to one quotable path.
:: Preserve caller-provided PYTHON_EXE, then prefer the maintainer venv, then Python 3.11+ on PATH or py launcher,
:: falling back to the bare uv base toolchain only if nothing else is available.
if defined PYTHON_EXE goto :python_resolved
set "VLLM_PYTHON=C:\vllm-env\Scripts\python.exe"
if exist "%VLLM_PYTHON%" goto :use_vllm_python
where python >nul 2>&1
if errorlevel 1 goto :try_py_311
call python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 goto :try_py_311
for /f "delims=" %%p in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%p"
goto :python_resolved
:try_py_311
where py >nul 2>&1
if errorlevel 1 goto :try_uv_python
call py -3.11 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 goto :try_py_3
for /f "delims=" %%p in ('py -3.11 -c "import sys;print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%p"
goto :python_resolved
:try_py_3
call py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 goto :try_uv_python
for /f "delims=" %%p in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%p"
goto :python_resolved
:try_uv_python
set "UV_PYTHON=%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe"
if exist "%UV_PYTHON%" set "PYTHON_EXE=%UV_PYTHON%"
goto :python_resolved
:use_vllm_python
set "PYTHON_EXE=%VLLM_PYTHON%"
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
