@echo off
REM ============================================================================
REM Item 14's discriminator, as ONE command.
REM
REM The open question is why this port's FIRST request is FASTER (261.7 tok/s against 169.8 for
REM requests 2-7) when every published first-request effect -- lazy module loading, graph capture, JIT,
REM allocator warm-up -- ADDS latency. Research ruled that family out by absence, so the answer has to be
REM measured, not argued.
REM
REM Two hypotheses, and they predict OPPOSITE things about per-round cost:
REM   arithmetic      -> per-round cost is UNCHANGED between request 1 and request 2; only the round
REM                       count differs, because the first request returns shorter text.
REM   #80 kernel sel. -> per-round cost DIFFERS, because upstream #80 selects the kernel from the token
REM                       count t = 1 + accepted drafts, and a different t selects different arithmetic.
REM
REM --warmup N discards N repetitions. So --warmup 0 measures the FIRST request and --warmup 1 measures
REM the second. Same binary, same artifact, same seed, same corpus; the only difference is whether the
REM first request ran. Per-round cost is decode_seconds_mean / spec_rounds.
REM
REM If per-round cost is flat and spec_rounds differs, the speed asymmetry is ARITHMETIC and the two
REM halves of item 14 are one cause. If per-round cost differs, #80's kernel selection is implicated and
REM the halves are separate.
REM ============================================================================
setlocal
set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set "PY=C:\vllm-env\Scripts\python.exe"
set "REPO=C:\AI\ninfer-v3-windows"
set "ARTIFACT=%NINFER_BENCH_ARTIFACT%"
if "%ARTIFACT%"=="" set "ARTIFACT=C:\AI\models\qwen3_8_27b_nvfp4qat.v3.ninfer"
set "OUT=%TEMP%\ninfer_first_request"

call "%VCVARS%" >nul 2>&1
if errorlevel 1 (
    echo INCONCLUSIVE: vcvars64 did not initialise.
    exit /b 2
)
if not exist "%OUT%" mkdir "%OUT%"
copy /y "%REPO%\build\apps\*.dll" "%REPO%\build\bench\" >nul 2>&1
cd /d "%REPO%"
if not exist "bench\fixtures\bench_corpus.ids" (
    echo INCONCLUSIVE: bench_corpus.ids is not where the bench looks for it. Run from the repo root.
    exit /b 2
)

REM first  = request 1 (nothing discarded).  second = request 2 (one discarded).
for %%W in (0 1) do (
    echo === warmup %%W : measuring the request after %%W discarded ===
    build\bench\ninfer_bench.exe --weights "%ARTIFACT%" --spec dflash2 --draft-tokens 7 ^
        --lm-head-draft -n 256 -r 1 --warmup %%W -o json --output-file "%OUT%\warm%%W.json"
    if errorlevel 1 (
        echo INCONCLUSIVE: the bench failed for warmup %%W - see the output above.
        exit /b 2
    )
)

"%PY%" "%REPO%\tools\bench\first_request_report.py" "%OUT%"
echo.
echo report exit=%ERRORLEVEL%
endlocal
exit /b 0