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
REM Leak reporting is off because MSVC's AddressSanitizer does not implement LeakSanitizer on
REM Windows: setting detect_leaks=1 makes every instrumented binary abort with
REM "AddressSanitizer: detect_leaks is not supported on this platform." It is not a choice about
REM reachable CUDA allocations, and no leak coverage is lost by leaving it off -- there was none.
REM
REM No host leak check exists on this platform today. The MSVC CRT debug heap
REM (_CrtSetDbgFlag(_CRTDBG_LEAK_CHECK_DF) + _CrtDumpMemoryLeaks) does detect leaks, but only
REM under the debug CRT (/MDd), and these tests link CUDA libraries built with the release CRT,
REM so mixing them fails with LNK2038. Dr. Memory was evaluated as the rebuild-free alternative
REM and cannot instrument on Windows 11 build 26200 (25H2): both 2.6.0 and the December 2025
REM weekly build abort with an internal crash before the target starts. Re-check Dr. Memory when
REM DynamoRIO ships a build for 25H2; until then, a Linux ASan run is the only route to host leak
REM coverage.
REM
REM KNOWN FAILURE, 2026-09-20: ninfer_context_cost_test dies with 0xC0000409
REM (STATUS_STACK_BUFFER_OVERRUN) and prints no report, while passing in the normal build. ASan
REM initialises first -- verbosity=1 prints the shadow layout -- so this is neither a startup nor a
REM PATH problem. A report-less fail-fast is what stack exhaustion under ASan's larger frames looks
REM like, but that is a hypothesis and not established; the test and its module are untouched by the
REM session that found it. Kept in the subset rather than dropped: a verification list that hides a
REM failing member is worse than one that names it. The other five pass.
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
