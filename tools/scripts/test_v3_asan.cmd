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
REM (see .opencode/skills/cuda-debugging); ASan does not instrument device code.
REM
REM Leak reporting is off: CUDA and the driver keep reachable allocations that are not leaks in
REM this program's sense, and LeakSanitizer's exit code would otherwise mask a real finding.
setlocal
set "REPO=%~dp0..\.."

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%REPO%"

set "TESTS=ninfer_resource_manager_test ninfer_context_cost_test ninfer_artifact_reader_test ninfer_admission_policy_test ninfer_kv_capacity_test ninfer_materialization_budget_test"

echo === CONFIGURE (AddressSanitizer) ===
cmake -B build-asan -S . -G Ninja ^
  -DCMAKE_CUDA_ARCHITECTURES=120a ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_TESTING=ON ^
  -DCMAKE_CXX_FLAGS="/fsanitize=address /Zi" ^
  -DCMAKE_CUDA_FLAGS="-Xcompiler=/fsanitize=address" ^
  -DCMAKE_EXE_LINKER_FLAGS="/INCREMENTAL:NO /DEBUG"
echo CONFIGURE_EXIT=%ERRORLEVEL%

echo === BUILD ASAN TESTS ===
cmake --build build-asan --config Release -j --target %TESTS%
echo BUILD_EXIT=%ERRORLEVEL%

set "ASAN_OPTIONS=detect_leaks=0"
set "FAILED=0"
echo === RUN UNDER ASAN ===
for %%T in (%TESTS%) do (
  echo ---- %%T ----
  build-asan\tests\%%T.exe
  if errorlevel 1 set "FAILED=1"
)
echo FAILED=%FAILED%
if not "%FAILED%"=="0" exit /b 1
echo ASAN SUBSET CLEAN
