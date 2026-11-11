@echo off
REM Configure and run the host-side test subset under MSVC AddressSanitizer in its own tree.
REM
REM Why a separate recipe: MSVC's /fsanitize=address defines the STL annotation markers
REM (annotate_string / annotate_vector / annotate_optional). If only the CXX objects are
REM instrumented, the linker rejects them against the uninstrumented CUDA host objects in
REM ninfer_core.lib with LNK2038 "mismatch detected for 'annotate_string'". Passing
REM -Xcompiler=/fsanitize=address on CMAKE_CUDA_FLAGS instruments nvcc's host pass too and the
REM mismatch disappears. Do not drop that flag.
REM
REM Scope: the host-only tests below. Device kernels are covered by compute-sanitizer instead
REM; ASan does not instrument device code.
REM
REM Leak reporting is off because MSVC's AddressSanitizer does not implement LeakSanitizer on
REM Windows: setting detect_leaks=1 makes every instrumented binary abort with
REM "AddressSanitizer: detect_leaks is not supported on this platform." It is not a choice about
REM reachable CUDA allocations, and no leak coverage is lost by leaving it off -- there was none.
REM
REM Passing -DCMAKE_CXX_FLAGS overwrites CMake's default /DWIN32 /D_WINDOWS /EHsc /Ob0 flags,
REM so /DWIN32 /D_WINDOWS /EHsc /bigobj MUST be included explicitly alongside /fsanitize=address:
REM omitting /EHsc previously caused exception-throwing host tests (ninfer_context_cost_test)
REM to abort with 0xC0000409 under C++ unwinding, and /bigobj avoids COFF section-limit C1128.
setlocal
set "REPO=%~dp0..\.."

set "VCVARS="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
  for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VCVARS=%%I\VC\Auxiliary\Build\vcvars64.bat"
)
if not defined VCVARS set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to initialize Visual Studio environment
    exit /b 1
)
cd /d "%REPO%"

set "TESTS=ninfer_context_cost_test ninfer_resource_manager_test ninfer_artifact_reader_test ninfer_context_capacity_test ninfer_read_queue_test ninfer_media_acquire_test"

REM The build must use the project's Python and not whatever find_package(Python3) finds: the tests'
REM declared dependencies live in the selected environment, and tests/CMakeLists.txt requires an
REM interpreter for ninfer_json_schema_oracle_test.
set "PY=%NINFER_PYTHON%"
if "%PY%"=="" set "PY=C:/vllm-env/Scripts/python.exe"

echo === CONFIGURE (AddressSanitizer) ===
cmake -B build-asan -S . -G Ninja ^
  -DCMAKE_CUDA_ARCHITECTURES=120a ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DPython3_EXECUTABLE="%PY%" ^
  -DBUILD_TESTING=ON ^
  -DCMAKE_CXX_FLAGS="/DWIN32 /D_WINDOWS /EHsc /bigobj /fsanitize=address /Zi" ^
  -DCMAKE_CUDA_FLAGS="-Xcompiler=/EHsc,/bigobj,/fsanitize=address" ^
  -DCMAKE_EXE_LINKER_FLAGS="/INCREMENTAL:NO /DEBUG"
echo CONFIGURE_EXIT=%ERRORLEVEL%
if errorlevel 1 exit /b 1

echo === BUILD ASAN TESTS ===
cmake --build build-asan --config Release -j --target %TESTS%
echo BUILD_EXIT=%ERRORLEVEL%
if errorlevel 1 exit /b 1

set "ASAN_OPTIONS=detect_leaks=0"
set "FAILED=0"
echo === RUN UNDER ASAN ===
REM || rather than "if errorlevel 1": that test is false for a negative exit code, which is exactly
REM what a crashed or unloadable process returns, so it reported FAILED=0 while a member did not run.
for %%T in (%TESTS%) do (
  echo ---- %%T ----
  build-asan\tests\%%T.exe || set "FAILED=1"
)
echo FAILED=%FAILED%
if not "%FAILED%"=="0" exit /b 1
echo ASAN SUBSET CLEAN
