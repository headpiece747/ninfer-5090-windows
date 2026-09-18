@echo off
REM Configure, build and run the test suite in its own build tree.
REM
REM The suite needs a separate tree because it configures with BUILD_TESTING=ON, and the
REM FFmpeg runtime DLLs must be copied beside the test executables. Without that copy four
REM tests exit 0xC0000135 (STATUS_DLL_NOT_FOUND) before printing anything, which looks like a
REM crash rather than a missing DLL.
REM
REM Expected result on this port: 120 of 121 pass. ninfer_resource_manager_test asserts an
REM eviction ordering against a 5 ms wall-clock search budget, so it is timing-sensitive by
REM construction. See docs/upstream-reports/ninfer-tma-descriptor-graph-capture.md.
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
