@echo off
REM Run a small device-test subset under NVIDIA Compute Sanitizer's memcheck.
REM
REM Why a subset, and why these tests. MSVC's AddressSanitizer cannot instrument device code, so
REM test_v3_asan.cmd covers the host side only; this is the device-side complement, and the skill at
REM Sanitizer runs are 10-20x slower than a normal one, and a kernel that outlives the Windows TDR
REM watchdog resets the driver on a card that is also driving the desktop, so this names five tests
REM that each take under a second unsanitized rather than the whole suite. Widen SUBSET when a
REM specific kernel is under suspicion, not for routine coverage.
REM
REM The binary is the ordinary release test build: the sanitizer instruments at run time, so
REM build-test must exist and be current -- tools\scripts\test_v3.cmd configures and builds it.
REM
REM compute-sanitizer is not on PATH on this machine; it lives in the toolkit under CUDA_PATH.
setlocal
set "REPO=%~dp0..\.."
if not exist "%REPO%\build-test\tests\CTestTestfile.cmake" (
    echo [ERROR] build-test is not configured. Run tools\scripts\test_v3.cmd first.
    exit /b 1
)

set "VCVARS="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
  for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VCVARS=%%I\VC\Auxiliary\Build\vcvars64.bat"
)
if not defined VCVARS set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul 2>&1

set "SANITIZER=%CUDA_PATH%\compute-sanitizer\compute-sanitizer.exe"
if not exist "%SANITIZER%" (
    echo [ERROR] compute-sanitizer not found at %SANITIZER%
    exit /b 1
)

cd /d "%REPO%"

REM The tests load the FFmpeg runtime DLLs from beside themselves.
xcopy /y /q ffmpeg\bin\*.dll build-test\tests\ >nul

set "SUBSET=ninfer_elementwise_test ninfer_embedding_test ninfer_indexing_test ninfer_reduction_test ninfer_rope_test"
set "FAILED="
for %%T in (%SUBSET%) do (
    echo === memcheck: %%T ===
    "%SANITIZER%" --tool memcheck --launch-timeout 300 --log-file "out\compute_sanitizer_%%T.log" "build-test\tests\%%T.exe"
    if errorlevel 1 set "FAILED=1"
)

if defined FAILED (
    echo [FAIL] compute-sanitizer reported an error; see out\compute_sanitizer_*.log
    exit /b 1
)
echo [PASS] memcheck clean on the subset; logs in out\compute_sanitizer_*.log
