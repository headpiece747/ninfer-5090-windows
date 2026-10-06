@echo off
REM Configure, build and run the test suite in its own build tree.
REM
REM The suite needs a separate tree because it configures with BUILD_TESTING=ON, and the
REM FFmpeg runtime DLLs must be copied beside the test executables. Without that copy four
REM tests exit 0xC0000135 (STATUS_DLL_NOT_FOUND) before printing anything, which looks like a
REM crash rather than a missing DLL.
REM
REM Expected result on this port: every test passes except the by-construction failures listed in
REM tools\release\test_baseline.json. That file is the authority for the suite's size and its expected
REM failures; this recipe deliberately names no count, because a count written here drifts (it read 122
REM while the baseline recorded 124, and 130 after the 2026-09-25 upstream merge).
REM
REM Set NINFER_TEST_ARTIFACT before running if you also intend to run the baseline gate: three
REM real-model tests are required on this product, and CTest marks them skipped without it, so the
REM gate fails with "3 required test(s) were skipped" while this recipe's own pass count hides them.
REM
REM After a suite run, tools\release\check_test_baseline.py confirms no NEW failure appeared. The
REM recorded baseline is in tools\release\test_baseline.json.
setlocal
set "REPO=%~dp0..\.."

REM The project's Python, resolved before the configure because the configure needs it. The build
REM must use this interpreter and not whatever find_package(Python3) happens to find: on this machine
REM that resolves to C:/Python314, while the tests' declared dependencies (jsonschema, declared in
REM tests/text/requirements.txt for ninfer_json_schema_oracle_test) live in the selected environment.
REM Honours NINFER_PYTHON so CI and a developer override select the same interpreter here as elsewhere.
set "PY=%NINFER_PYTHON%"
if "%PY%"=="" set "PY=C:/vllm-env/Scripts/python.exe"

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%REPO%"

echo === CONFIGURE (BUILD_TESTING=ON) ===
cmake -B build-test -S . -G Ninja ^
  -DCMAKE_CUDA_ARCHITECTURES=120a ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DPython3_EXECUTABLE="%PY%" ^
  -DBUILD_TESTING=ON
echo CONFIGURE_EXIT=%ERRORLEVEL%

echo === BUILD TESTS ===
cmake --build build-test --config Release -j
echo BUILD_EXIT=%ERRORLEVEL%

echo === STAGE FFMPEG RUNTIME DLLS ===
xcopy /y /q ffmpeg\bin\*.dll build-test\tests\ >nul

REM --schedule-random because order-dependence is a real bug class here: a test that decorates itself
REM with an environment variable poisoned every test after it on 2026-09-25 (the empty NINFER_CUDA_SYNC
REM case, 36 failures), and a fixed order hides that. The baseline compares test names, so a random
REM order changes nothing about the verdict.
echo === CTEST ===
REM -j 8, and it is safe for a reason already in the tree rather than a new one: the real-model
REM tests carry RUN_SERIAL TRUE, so ctest runs nothing alongside them and the ~16 GiB artifact each
REM loads cannot collide. What -j adds is overlap between the op tests, which are host-oracle-bound
REM -- ninfer_softmax_attention_test measured 5% median GPU utilisation across 60 s of a 584 s test
REM -- so they spread across cores without changing anything they assert. Measured 2026-10-05: the
REM six softmax entries take 145.5 s at -j 6 against 511 s run serially.
ctest --test-dir build-test --output-on-failure --schedule-random -j 8
echo CTEST_EXIT=%ERRORLEVEL%
REM === OP PERTURBATION ===
REM Last, and separate from the suite, because it asks a different question: not "do the tests pass"
REM but "can they". Each declared mutation must turn its suite red. It was in .githooks/pre-commit and
REM was removed from there -- a hook that reads build-test\ certifies whatever binary was last built,
REM and with a kernel edited but not rebuilt it reported PASS. Here the build above is minutes old at
REM worst, and the gate verifies that for itself with a build dry run rather than assuming it, so the
REM verdict describes this tree rather than whatever happened to be lying around. It cannot build
REM here itself: nvcc needs cl.exe, which needs the Visual Studio environment a Python process does
REM not carry. Costs about a second per mutation, plus one no-op dry run.
"%PY%" tools\release\check_test_mutation.py
echo MUTATION_EXIT=%ERRORLEVEL%
