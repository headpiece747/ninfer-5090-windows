# Native target-private MTP proposal/verification microbenchmark. It enters through the target
# Session and Executor and executes the production round schedule over a real .ninfer artifact.
add_executable(ninfer_qwen3_5_mtp_round_bench
  "${CMAKE_CURRENT_LIST_DIR}/mtp_round_bench.cpp")
ninfer_internal_includes(ninfer_qwen3_5_mtp_round_bench)
target_link_libraries(ninfer_qwen3_5_mtp_round_bench PRIVATE
  ninfer_engine ninfer_model_execution ninfer_core)

# Complete production DFlash steady round: confirmed-context append, next proposal, target
# verification/acceptance, and host publication over a real 35B artifact.
add_executable(ninfer_qwen3_5_dflash_round_bench
  "${CMAKE_CURRENT_LIST_DIR}/dflash_round_bench.cpp")
ninfer_internal_includes(ninfer_qwen3_5_dflash_round_bench)
target_link_libraries(ninfer_qwen3_5_dflash_round_bench PRIVATE
  ninfer_engine ninfer_model_execution ninfer_core)

# Token-exact per-token log probabilities for a supplied id file. It exists because the perplexity
# tool's --text front door re-tokenises, and re-tokenising decoded text does not reproduce the ids
# that produced it, so an index-aligned comparison between two decoding routes silently reads the
# wrong position. Engine::score_tokens takes ids directly; this is that path with a file front door.
add_executable(ninfer_score_token_logprobs
  "${CMAKE_CURRENT_LIST_DIR}/score_token_logprobs.cpp")
ninfer_internal_includes(ninfer_score_token_logprobs)
target_link_libraries(ninfer_score_token_logprobs PRIVATE ninfer_engine ninfer_core)

# Host-only chat-template render cost by conversation length. It reads a template and synthesizes
# its own conversation, so it needs no artifact and no GPU, and it separates the frontend's extra
# boundary-proving renders by turning the conditions that trigger them on and off.
add_executable(ninfer_qwen3_5_chat_render_bench
  "${CMAKE_CURRENT_LIST_DIR}/chat_render_bench.cpp")
ninfer_internal_includes(ninfer_qwen3_5_chat_render_bench)
target_link_libraries(ninfer_qwen3_5_chat_render_bench PRIVATE
  ninfer_model_control ninfer_core ninfer::json)

# Opt-in Segment Heap for that bench. Microsoft's documented replacement for the legacy NT heap is an
# application manifest; the NT heap serializes every allocation process-wide and scales badly with
# core count, while the segment heap has been measured *adding* VirtualAlloc traffic on an
# allocation-heavy workload. Neither direction can be assumed, so this is a build option whose two
# arms are measured against each other rather than a default.
option(NINFER_BENCH_SEGMENT_HEAP "Embed a Segment Heap manifest in the chat render bench" OFF)
if(NINFER_BENCH_SEGMENT_HEAP)
  target_link_options(ninfer_qwen3_5_chat_render_bench PRIVATE
    "/MANIFESTINPUT:${CMAKE_SOURCE_DIR}/cmake/segment_heap.manifest")
endif()

if(NINFER_TARGET STREQUAL "rtx5090")
  add_executable(ninfer_qwen3_5_moe_execution_bench
    "${CMAKE_CURRENT_LIST_DIR}/moe_execution_bench.cpp")
  ninfer_internal_includes(ninfer_qwen3_5_moe_execution_bench)
  target_link_libraries(ninfer_qwen3_5_moe_execution_bench PRIVATE ninfer_model_execution)
endif()
