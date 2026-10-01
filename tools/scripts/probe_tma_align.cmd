@echo off
REM Measures the two toolchain facts the TMA descriptor comments in src/ops/linear/*/ assert:
REM   1. what alignment CUtensorMap actually carries, and whether that is a property of the type
REM      or of the build flags;
REM   2. which by-value __grid_constant__ descriptor-block widths MSVC's ABI will accept.
REM
REM Run it after touching CUDA headers, the MSVC toolchain, or the C++ standard/flags in
REM CMakeLists.txt. The comments quote these numbers, so a change here invalidates them.
REM
REM Expected today (CUDA 13.3, MSVC, project default flags): alignof(CUtensorMap) == 8, and
REM 8/16/32/64 ACCEPTED with 128 and 256 rejected by C2719. With /Zc:__cplusplus the alignment
REM becomes 64, which is why both regimes are reported.
REM
REM A nonzero exit from a width in REGIME 2 is expected: it means MSVC rejects that width.

setlocal
set "SCRIPT_DIR=%~dp0"
set "HERE=%SCRIPT_DIR%"
set "OUT=%TEMP%\ninfer_tma_probe"
if not exist "%OUT%" mkdir "%OUT%"

set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"
set "CUDA_INC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\include"

call "%VCVARS%" >nul 2>&1
if errorlevel 1 (
    echo FAILED: vcvars64 did not initialise. Run this from a Developer Command Prompt, or set
    echo VCVARS above to the right path.
    exit /b 1
)

where cl.exe >nul 2>&1
if errorlevel 1 (
    echo FAILED: cl.exe is not on PATH after vcvars64.
    exit /b 1
)

echo ============================================================
echo REGIME 1: project default flags
echo ============================================================
call :align "%OUT%\align1.exe" ""
if errorlevel 1 exit /b 1

echo.
echo by-value __grid_constant__ parameter width:
call :sweep

echo.
echo ============================================================
echo REGIME 2: same source, /Zc:__cplusplus passed
echo ============================================================
echo (the header attribute becomes live here; the width sweep is unaffected because the
echo  descriptor block declares its own alignment)
call :align "%OUT%\align2.exe" "/Zc:__cplusplus"
if errorlevel 1 exit /b 1

endlocal
exit /b 0

:align
rem %1 = output exe, %2 = extra flags. cl.exe does not get CUDA's include path from nvcc,
rem so it is supplied here; without it the build fails on cuda.h.
cl.exe /nologo /EHsc /std:c++20 /I"%CUDA_INC%" %2 "%HERE%probe_tma_alignment.cpp" /Fe:%1 > "%OUT%\align.txt" 2>&1
if errorlevel 1 (
    type "%OUT%\align.txt"
    exit /b 1
)
%1
exit /b 0

:sweep
for %%W in (8 16 32 64 128 256) do call :one %%W
goto :eof

:one
rem No intermediate variable: a `set` inside this loop would need delayed expansion to be read
rem back by the echo, and that is a silent wrong answer rather than an error.
rem Every parenthesis in these echoes is escaped, INCLUDING the ones around %1. It expands to a bare
rem width, and an unescaped `alignas(8)` inside a parenthesized block ends the block early and
rem leaves cmd trying to run REJECTED as a command.
"%NVCC%" -std=c++20 -DPROBE_ALIGN=%1 -c "%HERE%probe_tma_param_align.cu" -o "%OUT%\sweep.obj" > "%OUT%\sweep.txt" 2>&1
if errorlevel 1 (
    findstr /C:"C2719" "%OUT%\sweep.txt" >nul 2>&1
    if errorlevel 1 (
        echo    alignas^(%1^)  REJECTED ^(see output, not C2719^)
        type "%OUT%\sweep.txt"
    ) else (
        echo    alignas^(%1^)  REJECTED ^(C2719^)
    )
) else (
    echo    alignas^(%1^)  ACCEPTED
)
del "%OUT%\sweep.obj" >nul 2>&1
goto :eof
