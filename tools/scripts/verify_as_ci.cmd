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
REM The shell above needs Git's own usr\bin on PATH. The hook's gate() helper pipes a failed
REM gate's output through `tail -40`, and without that directory a failure prints
REM "tail: command not found" and exits 127 instead of the gate's status -- hiding the reason the
REM helper exists to show. CI never saw this because its runner has bash and coreutils on PATH.
set "PATH=%ProgramFiles%\Git\usr\bin;%PATH%"

echo === 1. the hook in this checkout ===
set "NINFER_PYTHON=%VENV%\Scripts\python.exe"
"%BASH%" .githooks/pre-commit
set "LOCAL_STATUS=%ERRORLEVEL%"
echo     local hook exit: %LOCAL_STATUS%
echo.

REM One clone serves both remaining arms: a clone IS a repository at a different path, and its depth
REM is whatever ci.yml configures.
REM
REM A clone rather than a hand-copied subset, which is what this was. That subset list had gone
REM stale -- it omitted README.md, RELEASE_NOTES.md and download_model.py, all of which
REM check_profile_consistency.py reads -- so every check read nothing, failed, and this script
REM reported "the gate depends on where the repository lives" for a cause that was missing files.
REM A clone cannot go stale that way.
REM
REM The depth is READ from the workflow, not written here, so dropping fetch-depth: 0 from
REM .github/workflows/ci.yml turns this arm into a shallow clone and reproduces the failure here
REM instead of leaving it for CI to find.
set "DEPTH_ARGS=--depth 1"
REM Anchored with /r and ^ on purpose. The unanchored form also matched the explanatory COMMENT
REM above the setting, so deleting the real fetch-depth: 0 while leaving its prose would have kept
REM this arm on a full clone and hidden the regression -- the rule about guarding with string
REM presence over prose, which this repository already carries.
findstr /r /c:"^ *fetch-depth: *0" ".github\workflows\ci.yml" >nul 2>&1 && set "DEPTH_ARGS="
set "CLONED=%TEMP%\runner-check\ninfer-5090-windows"
if exist "%TEMP%\runner-check" rmdir /s /q "%TEMP%\runner-check"
mkdir "%TEMP%\runner-check" || exit /b 1
echo === 2. the clone: a different path, at ci.yml's depth ===
echo     %CLONED%   (%DEPTH_ARGS%)
git clone %DEPTH_ARGS% --quiet "file:///%REPO:\=/%" "%CLONED%" || exit /b 1
echo.
echo === 3. the gate, then the hook, in that clone ===
pushd "%CLONED%"
"%VENV%\Scripts\python.exe" tools\release\check_profile_consistency.py
set "REMOTE_STATUS=%ERRORLEVEL%"
echo     remote-path gate exit: %REMOTE_STATUS%
"%BASH%" .githooks/pre-commit
set "DEPTH_STATUS=%ERRORLEVEL%"
echo     depth-check hook exit: %DEPTH_STATUS%
popd
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
