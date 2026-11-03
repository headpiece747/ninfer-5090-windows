add_library(ninfer_op_test_support STATIC "${CMAKE_CURRENT_LIST_DIR}/suite.cpp")
ninfer_test_includes(ninfer_op_test_support)
ninfer_op_oracle_options(ninfer_op_test_support)
target_link_libraries(ninfer_op_test_support PUBLIC ninfer_cpu CUDA::cudart)

include("${CMAKE_CURRENT_LIST_DIR}/elementwise/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/embedding/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/indexing/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/reduction/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/normalization/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rope/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sequence/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sampling/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gated_residual/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/mhc/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kv_cache/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/tests.cmake")

# The port's DFlash2 NVFP4 route coverage. The source is this port's own -- upstream
# does not have it -- and the 2026-10-17 merge left it registered nowhere, so the test
# could not run. Restored from f89e05fc.
ninfer_add_op_test(ninfer_dflash2_nvfp4_routes_test
  SOURCES ops/test_dflash2_nvfp4_routes.cpp
  LIBRARIES ninfer_ops)

# The port's speculative-round coverage. Upstream deleted the file in the 2026-10-17 merge,
# taking the sparse-accept isolation test with it; restore it so the coverage is not lost
# silently. Restored from f89e05fc.
#
# The context-KV materialize test is deliberately NOT restored here, though it used to be.
# This block was a second add_executable() for the same target name that kv_cache/tests.cmake
# already registers through ninfer_add_op_test, so BUILD_TESTING=ON could not configure at
# all (CMP0002: duplicate target). The registered kv_cache/context_kv_materialize.cpp comes
# from the f30e661f suite reorganisation, runs the same batch sweep and the same workspace
# interval checks, and adds capacity-witness cases this copy lacks; the only thing the removed
# copy added was splitting that one sweep across four ctest entries via --batches, which is
# parallelism rather than coverage. tests/ops/test_context_kv_materialize.cpp is the now
# unreferenced source of that removed target, kept only so this decision stays revertable.
ninfer_add_op_test(ninfer_speculative_round_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_speculative_round.cpp"
  LIBRARIES ninfer_ops ninfer_core)

# The port's topk_logprobs coverage and perturbation seam. Restored after being
# dropped during upstream's suite reorganisation into ops subdirectories.
ninfer_add_op_test(ninfer_topk_logprobs_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_topk_logprobs.cpp"
  LIBRARIES ninfer_ops)
include("${CMAKE_CURRENT_LIST_DIR}/moe/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ple/tests.cmake")
