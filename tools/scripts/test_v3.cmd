@echo off
REM Configure, build and run the test suite in its own build tree.
REM
REM The suite needs a separate tree because it configures with BUILD_TESTING=ON, and the
REM FFmpeg runtime DLLs must be copied beside the test executables. Without that copy four
REM tests exit 0xC0000135 (STATUS_DLL_NOT_FOUND) before printing anything, which looks like a
REM crash rather than a missing DLL.
REM
REM Expected result on this port: 122 of 122 pass.
REM
REM Set NINFER_TEST_ARTIFACT before running if you also intend to run the baseline gate: three
REM real-model tests are required on this product, and CTest marks them skipped without it, so the
REM gate fails with "3 required test(s) were skipped" while this recipe's own pass count hides them.
REM
REM After a suite run, tools\release\check_test_baseline.py confirms no NEW failure appeared. The
REM recorded baseline is in tools\release\test_baseline.json.
setlocal
set "REPO=%~dp0..\.."

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%REPO%"

echo === CONFIGURE (BUILD_TESTING=ON) ===
cmake -B build-test -S . -G Ninja ^
  -DCMAKE_CUDA_ARCHITECTURES=120a ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_TESTING=ON
echo CONFIGURE_EXIT=%ERRORLEVEL%

echo === BUILD TESTS ===
cmake --build build-test --config Release -j
echo BUILD_EXIT=%ERRORLEVEL%

echo === STAGE FFMPEG RUNTIME DLLS ===
xcopy /y /q ffmpeg\bin\*.dll build-test\tests\ >nul

echo === CTEST ===
ctest --test-dir build-test --output-on-failure
echo CTEST_EXIT=%ERRORLEVEL%
