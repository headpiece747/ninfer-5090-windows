@echo off
REM Run the gate from a checkout at a DIFFERENT path, which is the one condition CI has that this
REM machine never reproduces.
REM
REM Why: paths derived from the module's own location -- `str(REPO)`, `template_path()` -- produce this
REM repository's directory when the gate runs here and the runner's directory when it runs there. A
REM reproduction that runs in the same checkout cannot see that difference, which is why the first
REM version of this script passed while CI kept failing.
REM
REM So the tree is copied to a scratch directory whose name shares no prefix with this one, and the
REM failing gate runs there. A gate that depends on where the repository lives fails here too.
REM
REM Usage:  tools\scripts\verify_as_ci.cmd [--remote-root]
setlocal enabledelayedexpansion

set "REPO=%~dp0..\.."
for %%I in ("%REPO%") do set "REPO=%%~fI"
cd /d "%REPO%" || exit /b 1

set "PY311=%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe"
if not exist "%PY311%" (
    echo [ERROR] Python 3.11 not found at %PY311%
    exit /b 1
)
set "VENV=%TEMP%\ninfer-ci-verify"
if not exist "%VENV%\Scripts\python.exe" (
    echo === building a scratch environment (once; reused after) ===
    "%PY311%" -m venv "%VENV%" || exit /b 1
    "%VENV%\Scripts\python.exe" -m pip install --quiet --upgrade pip || exit /b 1
    "%VENV%\Scripts\python.exe" -m pip install --quiet pytest pyyaml numpy safetensors ruff mypy || exit /b 1
    "%VENV%\Scripts\python.exe" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu || exit /b 1
)

set "BASH=bash"
where bash >nul 2>&1 || set "BASH=%ProgramFiles%\Git\usr\bin\sh.exe"

echo === 1. the hook in this checkout ===
set "NINFER_PYTHON=%VENV%\Scripts\python.exe"
"%BASH%" .githooks/pre-commit
set "LOCAL_STATUS=%ERRORLEVEL%"
echo     local hook exit: %LOCAL_STATUS%
echo.

REM The remote-path run. Only the files the gate reads are copied: the repository carries multi-gigabyte
REM build trees and artifacts, and copying all of it exhausts memory long before it tests anything.
set "REMOTE=%TEMP%\runner-path-check\ninfer-5090-windows"
if exist "%TEMP%\runner-path-check" rmdir /s /q "%TEMP%\runner-path-check"
mkdir "%REMOTE%\tools\release" || exit /b 1
mkdir "%REMOTE%\tools\chat_templates" || exit /b 1
mkdir "%REMOTE%\tools\scripts" || exit /b 1
copy /y "tools\release\*.py" "%REMOTE%\tools\release\" >nul || exit /b 1
copy /y "tools\chat_templates\*" "%REMOTE%\tools\chat_templates\" >nul || exit /b 1
copy /y "tests\convert\*.py" "%REMOTE%\" >nul 2>&1
copy /y "launcher_env.bat" "%REMOTE%\" >nul || exit /b 1
copy /y "*.bat" "%REMOTE%\" >nul || exit /b 1
echo === 2. the same gate from a checkout at a different path ===
echo     %REMOTE%
pushd "%REMOTE%"
"%VENV%\Scripts\python.exe" tools\release\check_profile_consistency.py
set "REMOTE_STATUS=%ERRORLEVEL%"
popd
echo     remote-path gate exit: %REMOTE_STATUS%
echo.

if not "%LOCAL_STATUS%"=="0" (
    echo [FAIL] the hook fails in this checkout
    exit /b 1
)
if not "%REMOTE_STATUS%"=="0" (
    echo [FAIL] the gate depends on where the repository lives -- this is what CI reports
    exit /b 1
)
echo [PASS] the hook passes here and from a different path
exit /b 0
