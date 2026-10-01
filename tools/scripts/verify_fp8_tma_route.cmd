@echo off
REM ============================================================================
REM FP8 A8 TMA route verification -- the loop that decided a5077adf, committed so the
REM measurement outlives the session it was taken in.
REM
REM THIS IS A GATE, NOT A DIAGNOSTIC. It exits 0 only if the FP8 A8 TMA route runs.
REM
REM WHY THE CONTROL ARM IS THE POINT OF THIS SCRIPT
REM Two harnesses in this project's history reported a finding that was not one:
REM   - a TMA probe spent a day asserting that cp.async.bulk.tensor "does not run on sm_120a",
REM     when its own mbarrier.try_wait.parity operands were mis-numbered and it faulted before any
REM     TMA instruction executed;
REM   - an earlier version of THIS script reported RED on a bench usage error, because it asserted
REM     "nonzero exit" instead of the symptom.
REM A gate that cannot tell "the bug is present" from "my harness is wrong" is worse than no gate,
REM because it is believed. So the control below runs FIRST and the verdict is one of three:
REM   GREEN        - control passed and the route ran
REM   RED          - control passed and the route faulted with cudaErrorIllegalInstruction
REM   INCONCLUSIVE - the control failed, so nothing this run says about the route is admissible
REM Never collapse INCONCLUSIVE into RED or GREEN.
REM
REM usage: verify_fp8_tma_route.cmd
REM ============================================================================
setlocal enabledelayedexpansion
set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set "ROOT=%TEMP%\ninfer_fp8_tma"
set "REPO=C:\AI\ninfer-v3-windows"
if not exist "%ROOT%" mkdir "%ROOT%"

call "%VCVARS%" >nul 2>&1
if errorlevel 1 (
    echo INCONCLUSIVE: vcvars64 did not initialise. Nothing about the route is established.
    exit /b 2
)

cd /d "%REPO%"
if errorlevel 1 (
    echo INCONCLUSIVE: cannot enter the repo. Nothing about the route is established.
    exit /b 2
)

rem ---------------------------------------------------------------- control arm
rem The control is the BF16 TMA route: ungated, same descriptor transport, known good. If this does
rem not run, the toolchain, the build or the harness is broken and the FP8 result means nothing.
echo === CONTROL: BF16 TMA route (ungated) ===
cmake --build build -j --target ninfer_linear_bench > "%ROOT%\build.txt" 2>&1
if errorlevel 1 (
    echo INCONCLUSIVE: the control's build failed.
    exit /b 2
)
copy /y build\apps\*.dll build\bench\ >nul 2>&1
cd build\bench
rem T=128 at n14336_k5120 routes to launch_bf16_tma_mma.
ninfer_linear_bench.exe --qtype BF16 --n 14336 --k 5120 --t 128 --policy a16 > "%ROOT%\control.txt" 2>&1
set "CRC=%ERRORLEVEL%"
type "%ROOT%\control.txt"
findstr /C:"BF16" "%ROOT%\control.txt" >nul 2>&1
if errorlevel 1 (
    echo.
    echo INCONCLUSIVE: the control produced no BF16 row. The harness is wrong, not the route.
    exit /b 2
)
if not "!CRC!"=="0" (
    echo.
    echo INCONCLUSIVE: the control exited !CRC!. The harness is wrong, not the route.
    exit /b 2
)
echo     control ok

rem ---------------------------------------------------------------- subject arm
cd /d "%REPO%"
echo.
echo === SUBJECT: FP8 A8 TMA route with the nine Windows gates flipped ===
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0flip_fp8_tma_gates.ps1"
if errorlevel 1 (
    echo INCONCLUSIVE: could not flip the gates. Nothing about the route is established.
    exit /b 2
)
cmake --build build-test -j --target ninfer_linear_fp8_a8_test > "%ROOT%\build2.txt" 2>&1
if errorlevel 1 (
    echo INCONCLUSIVE: the subject build failed.
    exit /b 2
)
copy /y build\apps\*.dll build-test\tests\ >nul 2>&1
cd build-test\tests
ninfer_linear_fp8_a8_test.exe > "%ROOT%\subject.txt" 2>&1
set "SRC=%ERRORLEVEL%"
type "%ROOT%\subject.txt"

cd /d "%REPO%"
git checkout -- src/ >nul 2>&1

echo.
findstr /C:"illegal instruction" "%ROOT%\subject.txt" >nul 2>&1
if not errorlevel 1 (
    echo RED: the FP8 A8 TMA route faulted with cudaErrorIllegalInstruction.
    exit /b 1
)
if not "!SRC!"=="0" (
    echo INCONCLUSIVE: nonzero exit with no illegal-instruction text. Read the output above.
    exit /b 2
)
echo GREEN: the FP8 A8 TMA route executed. The gates are the only thing still holding it back.
endlocal
exit /b 0
