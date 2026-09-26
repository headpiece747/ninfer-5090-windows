@echo off
REM Run the commit hook under the conditions CI uses, so "will this pass on the runner?" is one command
REM instead of a push and a wait.
REM
REM Why this exists. The launcher comparison failed in CI for six consecutive pushes while passing here,
REM and every diagnosis was formed from the local machine rather than the runner's conditions. The
REM difference was never read; it was guessed. This script reproduces the conditions instead:
REM
REM   * Python 3.11, the version ci.yml pins -- not whatever is first on PATH;
REM   * the exact package set ci.yml installs, and nothing else;
REM   * NINFER_PYTHON as a bare command name, which is how the runner invokes the hook and what broke
REM     the first five runs;
REM   * a scratch virtual environment, so none of this touches the machine's tooling environment.
REM
REM It is slower than a local commit -- one environment build, about a minute -- which is the point:
REM the value is that it fails for the runner's reasons rather than this machine's.
REM
REM Usage:  tools\scripts\verify_as_ci.cmd
setlocal

set "REPO=%~dp0..\.."
cd /d "%REPO%" 2>nul
if errorlevel 1 (
    REM A relative fallback: an absolute workspace path does not resolve in every environment here.
    for %%I in ("%~dp0..\..") do set "REPO=%%~fI"
    cd /d "%REPO%"
)

REM The interpreter ci.yml pins. uv is what this machine already uses to hold several versions; if it is
REM absent the script says so rather than silently testing a different Python.
set "PY311=%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe"
if not exist "%PY311%" (
    echo [ERROR] Python 3.11 not found at %PY311%
    echo         ci.yml pins 3.11; testing another version would answer a different question.
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

echo === the hook, as the runner runs it ===
echo     interpreter: "%VENV%\Scripts\python.exe"
"%VENV%\Scripts\python.exe" -V
REM NINFER_PYTHON is the venv's interpreter, because the runner's hook sees a command name rather than a
REM path and the check on that name is one of the things this script exists to exercise.
set "NINFER_PYTHON=%VENV%\Scripts\python.exe"
REM Git ships bash and is how the runner has it on PATH. This machine does not put it there, so it is
REM located explicitly rather than assumed -- the hook is a POSIX script and needs a real bash.
set "BASH=bash"
where bash >nul 2>&1 || set "BASH=%ProgramFiles%\Git\usr\bin\sh.exe"
if not exist "%BASH%" if "%BASH%"=="bash" (
    echo [ERROR] no bash found; the hook is a POSIX script and cannot run without one
    exit /b 1
)
"%BASH%" .githooks/pre-commit
set "STATUS=%ERRORLEVEL%"
echo.
if "%STATUS%"=="0" (
    echo [PASS] the hook passes under CI's conditions
) else (
    echo [FAIL] the hook fails under CI's conditions -- this is what the runner will report
)
exit /b %STATUS%
