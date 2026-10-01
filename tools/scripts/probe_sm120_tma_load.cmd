@echo off
REM Reproducer for docs/active-work.md item 12: a bare cp.async.bulk.tensor.2d does not run on
REM sm_120a on this machine.
REM
REM THIS IS A DIAGNOSTIC, NOT A GATE. It is EXPECTED TO FAIL (exit 1) today. Every arm reports
REM "an illegal memory access was encountered" from a descriptor that encoded successfully. If an arm
REM ever prints "TMA WORKS", that is a real change in the platform's behaviour and item 12 should be
REM re-examined -- do not treat a passing run as the suite being green.
REM
REM One process per arm: a TMA fault poisons the CUDA context, so the arms cannot share one.
REM Written as a file because the MSVC environment has to be initialised first, and inlining that
REM through PowerShell breaks argument splitting.
REM
REM Arms: BY_VALUE 1 is the form the BF16 and NVFP4 routes use and are not Windows-gated, so it is
REM the control; BY_VALUE 0 is the form this port's MSVC descriptor workaround forces on Windows.

setlocal
set "HERE=%~dp0"
set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"
set "OUT=%TEMP%\ninfer_tma_load"
if not exist "%OUT%" mkdir "%OUT%"

call "%VCVARS%" >nul 2>&1
if errorlevel 1 (
    echo FAILED: vcvars64 did not initialise. Run from a Developer Command Prompt, or set VCVARS.
    exit /b 1
)

call :arm 1 3
call :arm 0 3
call :arm 1 0
endlocal
exit /b 0

:arm
echo.
"%NVCC%" -arch=sm_120a -std=c++20 -DBY_VALUE=%1 -DSWIZZLE_MODE=%2 "%HERE%probe_sm120_tma_load.cu" -o "%OUT%\bv%1_sw%2.exe" -lcuda
if errorlevel 1 (
    echo BUILD FAILED for BY_VALUE=%1 SWIZZLE=%2
    goto :eof
)
"%OUT%\bv%1_sw%2.exe"
echo    exit=%ERRORLEVEL%
del "%OUT%\bv%1_sw%2.exe" >nul 2>&1
goto :eof
