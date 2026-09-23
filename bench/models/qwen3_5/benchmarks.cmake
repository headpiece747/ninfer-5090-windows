# Native target-private MTP proposal/verification microbenchmark. It enters through the target
# Program facade and executes the production round schedule over a real .ninfer artifact.
add_executable(ninfer_qwen3_5_mtp_round_bench
  "${CMAKE_CURRENT_LIST_DIR}/mtp_round_bench.cpp")
ninfer_internal_includes(ninfer_qwen3_5_mtp_round_bench)
target_link_libraries(ninfer_qwen3_5_mtp_round_bench PRIVATE
  ninfer_engine ninfer_model_runtime ninfer_core)

# Complete production DFlash steady round: confirmed-context append, next proposal, target
# verification/acceptance, and host publication over a real 35B artifact.
add_executable(ninfer_qwen3_5_dflash_round_bench
  "${CMAKE_CURRENT_LIST_DIR}/dflash_round_bench.cpp")
ninfer_internal_includes(ninfer_qwen3_5_dflash_round_bench)
target_link_libraries(ninfer_qwen3_5_dflash_round_bench PRIVATE
  ninfer_engine ninfer_model_runtime ninfer_core)

# Host-only chat-template render cost by conversation length. It reads a template and synthesizes
# its own conversation, so it needs no artifact and no GPU, and it separates the frontend's extra
# boundary-proving renders by turning the conditions that trigger them on and off.
add_executable(ninfer_qwen3_5_chat_render_bench
  "${CMAKE_CURRENT_LIST_DIR}/chat_render_bench.cpp")
ninfer_internal_includes(ninfer_qwen3_5_chat_render_bench)
target_link_libraries(ninfer_qwen3_5_chat_render_bench PRIVATE
  ninfer_model_runtime ninfer_core ninfer::json)
