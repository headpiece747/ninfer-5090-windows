ninfer_add_test(ninfer_artifact_reader_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_reader.cpp"
  LIBRARIES ninfer_artifact)

ninfer_add_test(ninfer_artifact_backing_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_backing.cpp"
  LIBRARIES ninfer_artifact)
set_tests_properties(ninfer_artifact_backing_test PROPERTIES TIMEOUT 30)

ninfer_add_test(ninfer_artifact_materialization_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_materialization.cpp" "${CMAKE_CURRENT_LIST_DIR}/materialization_io.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/encoded_slice_checks.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/csf_selection_checks.cpp"
  LIBRARIES ninfer_artifact)

# The injected-failure cases need GNU --wrap, which MSVC does not provide. The test source reports
# that as SKIP rather than passing silently; see materialization_cuda_errors.cpp.
if(NOT MSVC)
ninfer_test_interpose(ninfer_artifact_materialization_test
  cudaMalloc cudaMallocHost cudaFree cudaFreeHost cudaEventCreateWithFlags cudaEventRecord
  cudaMemcpyAsync cudaStreamSynchronize statx)
endif()

add_test(NAME ninfer_artifact_writer_interop_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/writer_interop.py"
    $<TARGET_FILE:ninfer_artifact_materialization_test>)

set_tests_properties(
  ninfer_artifact_writer_interop_test
  PROPERTIES SKIP_RETURN_CODE 77)

set_tests_properties(
  ninfer_artifact_materialization_test
  PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME ninfer_artifact_cpu_materialization_test
  COMMAND ninfer_artifact_materialization_test --cpu-only)
set_tests_properties(ninfer_artifact_cpu_materialization_test
  PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=")

# The regression test for issue #7 (Windows Direct I/O). It reads NINFER_TEST_ARTIFACT through the
# materializer's 64 MiB direct pattern and locks the 4096-alignment contract plus a measured
# throughput floor. It skips with 77 when the artifact is missing or smaller than 4 GiB.
ninfer_add_test(ninfer_artifact_direct_io_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_direct_io.cpp"
  LIBRARIES ninfer_artifact)
set_tests_properties(ninfer_artifact_direct_io_test PROPERTIES
  TIMEOUT 120 SKIP_RETURN_CODE 77)
