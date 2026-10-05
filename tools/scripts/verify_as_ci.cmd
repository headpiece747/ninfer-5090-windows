@echo off
REM Reproduce the two conditions CI has that this machine's own checkout never does: a repository at
REM a DIFFERENT path, and a checkout at the DEPTH ci.yml configures.
REM
REM Why the path matters: paths derived from the module's own location -- `str(REPO)`,
REM `template_path()` -- produce this repository's directory when the gate runs here and the runner's
REM directory when it runs there. A reproduction that runs in the same checkout cannot see that
REM difference, which is why the first version of this script passed while CI kept failing.
REM
REM Why the depth matters: actions/checkout clones one commit unless told otherwise, and a gate that
REM reads a pinned revision with `git show <rev>:<file>` then cannot run at all. That is how the
REM fast-gates tier sat red from 2026-10-04: check_fp8_band_ladders.py could not read its base, and
REM this script could not see it because its only arm ran in a full checkout.
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
REM The guard tests the property that matters, not a proxy. A venv can exist with a broken pytest
REM -- the package present but pytest.__main__ missing, which makes `python -m pytest` fail while the
REM directory check passes. That was this machine's state, and it failed every arm of this script for
REM a reason unrelated to what the script tests, with nothing saying so.
set "VENV_OK="
if exist "%VENV%\Scripts\python.exe" (
    "%VENV%\Scripts\python.exe" -m pytest --version >nul 2>&1 && set "VENV_OK=1"
)
if not defined VENV_OK (
    echo === building a scratch environment (once; reused after) ===
    if exist "%VENV%" rmdir /s /q "%VENV%"
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

REM === 3. the hook in a clone at the depth ci.yml configures ===
REM The depth is READ from the workflow, not written here. If fetch-depth: 0 is dropped from
REM .github/workflows/ci.yml, this arm becomes a shallow clone and reproduces the failure locally
REM instead of leaving it for CI to find.
set "DEPTH_ARGS=--depth 1"
REM Anchored with /r and ^ on purpose. The unanchored form also matched the explanatory COMMENT
REM above the setting, so deleting the real fetch-depth: 0 while leaving its prose would have kept
REM this arm on a full clone and hidden the regression -- the rule about guarding with string
REM presence over prose, which this repository already carries.
findstr /r /c:"^ *fetch-depth: *0" ".github\workflows\ci.yml" >nul 2>&1 && set "DEPTH_ARGS="
set "CLONED=%TEMP%\runner-depth-check\ninfer-5090-windows"
if exist "%TEMP%\runner-depth-check" rmdir /s /q "%TEMP%\runner-depth-check"
mkdir "%TEMP%\runner-depth-check" || exit /b 1
echo === 3. the hook in a clone at ci.yml's depth ===
echo     %CLONED%   (%DEPTH_ARGS%)
git clone %DEPTH_ARGS% --quiet "file:///%REPO:\=/%" "%CLONED%" || exit /b 1
pushd "%CLONED%"
"%BASH%" .githooks/pre-commit
set "DEPTH_STATUS=%ERRORLEVEL%"
popd
echo     depth-check hook exit: %DEPTH_STATUS%
echo.

if not "%LOCAL_STATUS%"=="0" (
    echo [FAIL] the hook fails in this checkout
    exit /b 1
)
if not "%REMOTE_STATUS%"=="0" (
    echo [FAIL] the gate depends on where the repository lives -- this is what CI reports
    exit /b 1
)
if not "%DEPTH_STATUS%"=="0" (
    echo [FAIL] the hook fails in a clone at ci.yml's configured depth -- this is what CI reports
    exit /b 1
)
echo [PASS] the hook passes here, from a different path, and in a clone at ci.yml's depth
exit /b 0
