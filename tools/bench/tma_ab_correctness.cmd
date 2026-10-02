@echo off
REM Correctness check for the transport A/B: both arms must produce the SAME result.
REM
REM The bench times but has no oracle, so the throughput numbers in item 12 are only admissible if the
REM MMA twin computes the same thing as the TMA schedule. This runs the real FP8 linear_add test --
REM which HAS an oracle, kA8Tolerance{0.04, ...} and verify_preserved -- once per arm.
REM
REM A throughput comparison between two kernels where only one is known to be correct is not a
REM measurement, so this runs BEFORE tma_ab.cmd is trusted and is reported as a gate.

setlocal enabledelayedexpansion
set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set "REPO=C:\AI\ninfer-v3-windows"
call "%VCVARS%" >nul 2>&1
if errorlevel 1 ( echo INCONCLUSIVE: vcvars64 did not initialise & exit /b 2 )

cd /d "%REPO%"
cmake --build build-test -j --target ninfer_linear_add_fp8_test > "%TEMP%\tma_abc_build.txt" 2>&1
if errorlevel 1 (
    echo BUILD FAILED
    powershell -NoProfile -Command "Get-Content -Tail 20 '%TEMP%\tma_abc_build.txt'"
    exit /b 2
)
copy /y "%REPO%\build\apps\*.dll" "%REPO%\build-test\tests\" >nul 2>&1

cd /d "%REPO%\build-test\tests"
set "FAILED=0"
for %%A in (tma mma unset) do (
    if "%%A"=="unset" (
        set "NINFER_FP8_TMA_ARM="
    ) else (
        set "NINFER_FP8_TMA_ARM=%%A"
    )
    echo.
    echo === arm: %%A ===
    ninfer_linear_add_fp8_test.exe > "%TEMP%\tma_abc_%%A.txt" 2>&1
    set "RC=!ERRORLEVEL!"
    findstr /C:"OK" /C:"FAIL" /C:"mismatch" /C:"failed" "%TEMP%\tma_abc_%%A.txt"
    echo     exit=!RC!
    if not "!RC!"=="0" set "FAILED=1"
)
set "NINFER_FP8_TMA_ARM="

echo.
if "%FAILED%"=="0" (
    echo VERDICT: GREEN - both arms pass the FP8 linear_add oracle, so the A/B compares two CORRECT
    echo kernels and the throughput ratio is admissible.
    exit /b 0
)
echo VERDICT: RED - at least one arm fails the oracle. The A/B is void until that is explained.
exit /b 1
