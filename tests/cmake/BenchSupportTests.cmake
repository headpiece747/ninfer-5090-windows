ninfer_add_test(ninfer_bench_support_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_ninfer_bench_support.cpp"
          ${PROJECT_SOURCE_DIR}/bench/inference/ninfer_bench_support.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer::json)

target_include_directories(ninfer_bench_support_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/inference)

ninfer_add_op_test(ninfer_bench_fixtures_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_bench_fixtures.cu"
  LIBRARIES ninfer_ops)
target_include_directories(ninfer_bench_fixtures_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/ops)

add_executable(ninfer_context_cost_measure_test
  "${CMAKE_CURRENT_LIST_DIR}/../test_context_cost_measure.cpp"
  ${PROJECT_SOURCE_DIR}/bench/context_cost/context_cost_measure.cpp)

target_include_directories(ninfer_context_cost_measure_test PRIVATE
  ${PROJECT_SOURCE_DIR}/bench/context_cost)

add_test(NAME ninfer_context_cost_measure_test COMMAND ninfer_context_cost_measure_test)

# The native renderer is a second implementation of one serialization, and it was compared only by a
# bench in no gate: seven `RenderedChat` fields had no coverage at all, and the reasoning-effort arms
# compared the native renderer with itself while reporting agreement. The suite compares 2 of the 9
# fields; these two entries run the differential oracle for the rest and fail the suite on any
# divergence, on a broken control, or on a retired native path. They need no FFmpeg runtime: this bench
# has no media-library link dependency (the dflash bench does, which is a separate recipe's problem).
# The media arm is separate because it is the only corpus carrying a multi-part user turn, which is
# where a part boundary sits between other parts rather than at the end of the content.
add_test(NAME ninfer_chat_render_native_parity_test
         COMMAND ninfer_qwen3_5_chat_render_bench
                 --template "${PROJECT_SOURCE_DIR}/tools/chat_templates/qwen3_8.jinja"
                 --native --sweep 5 --quiet)
add_test(NAME ninfer_chat_render_native_parity_media_test
         COMMAND ninfer_qwen3_5_chat_render_bench
                 --template "${PROJECT_SOURCE_DIR}/tools/chat_templates/qwen3_8.jinja"
                 --native --media --sweep 6 --quiet)
