@echo off
REM ============================================================================
REM Transport A/B for docs/active-work.md item 12: does the TMA load path cost throughput at a
REM MATCHED tile, or was the earlier 10x a schedule confound?
REM
REM Both arms live in one binary via NINFER_FP8_TMA_ARM, and the arms are INTERLEAVED inside one
REM window: tma, mma, tma, mma... with the ratio taken after per-arm medians. This card's clocks
REM drift enough between windows that measuring A then B would measure the window, not the kernel.
REM
REM CONTROL: tokens 65..96 dispatch to Fp8A8T64R64K128 in BOTH arms, so that band is the same route
REM in both. If the arms disagree there the harness is measuring something else and the run is VOID.
REM That check runs first in the reporter and refuses to print a ratio if it fails.
REM
REM This bench takes no --n: N is fixed at 5120. Passing --n makes it exit 1 with a usage error, which
REM reads as a failed run rather than a bad command line -- the same trap as the loop in item 12's history.
REM
REM No parentheses in these comment lines on purpose: cmd parses an unbalanced one in a REM as a
REM block delimiter and then tries to execute the following words. An earlier revision of this file
REM printed 'RANKS is not recognized' three times for exactly that reason.
REM
REM usage: tma_ab.cmd [repeats]
REM ============================================================================
setlocal enabledelayedexpansion
set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set "PY=C:\vllm-env\Scripts\python.exe"
set "REPO=C:\AI\ninfer-v3-windows"
set "REPEATS=%~1"
if "%REPEATS%"=="" set "REPEATS=5"
set "ROOT=%TEMP%\ninfer_tma_ab"
if not exist "%ROOT%" mkdir "%ROOT%"

call "%VCVARS%" >nul 2>&1
if errorlevel 1 (
    echo INCONCLUSIVE: vcvars64 did not initialise.
    exit /b 2
)
copy /y "%REPO%\build\apps\*.dll" "%REPO%\build\bench\" >nul 2>&1
cd /d "%REPO%\build\bench"

set "SWEEP=32,64,65,96,129,192,193,256,384,512,768,1025"
echo === interleaved A/B, %REPEATS% repeats, K=6144, tokens=%SWEEP% ===
echo     193 and above route to the split-K tiles (K6144MidBulk then K6144Bulk).
for /L %%R in (1,1,%REPEATS%) do (
    set "NINFER_FP8_TMA_ARM=tma"
    ninfer_fp8_linear_add_bench.exe --k 6144 --policy a8 --t-sweep !SWEEP! --csv-out "%ROOT%\tma_%%R.csv" >nul 2>&1
    set "NINFER_FP8_TMA_ARM=mma"
    ninfer_fp8_linear_add_bench.exe --k 6144 --policy a8 --t-sweep !SWEEP! --csv-out "%ROOT%\mma_%%R.csv" >nul 2>&1
    echo     repeat %%R done
)
set "NINFER_FP8_TMA_ARM="

"%PY%" "%REPO%\tools\bench\tma_ab_report.py" "%ROOT%" !REPEATS!
echo.
echo report exit=%ERRORLEVEL%
endlocal
exit /b 0
