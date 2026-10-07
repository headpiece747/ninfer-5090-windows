set(ninfer_op_tests
  add_bias
  gelu
  silu_mul
  residual_add
  sigmoid_mul
  rmsnorm
  rmsnorm_pack_tail
  gated_rmsnorm
  l2norm
  gated_delta_net
  kimi_delta_attention
  causal_conv1d_silu
  layer_norm
  embedding
  argmax
  gdn_gating
  gdn_gating_proj
  rope
  vision_pos_embed
  sampling
  scalar
  cast
  prepare_ragged_prefix
  scatter
  scatter_bf16_batch
  target_logprobs
  topk_logprobs
  position)
foreach(op IN LISTS ninfer_op_tests)
  ninfer_add_op_test(ninfer_${op}_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_${op}.cpp"
    LIBRARIES ninfer_ops)
endforeach()

ninfer_add_op_test(ninfer_linear_topk_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_topk.cu"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_candidate_selector_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_candidate_selector.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_dflash2_nvfp4_routes_test
  SOURCES ops/test_dflash2_nvfp4_routes.cpp
  LIBRARIES ninfer_ops)

# Built explicitly rather than through ninfer_add_op_test, because that helper registers one
# add_test with no arguments and CMake does not let a test's COMMAND be replaced afterwards.
# Everything else is what the helper does: ninfer_test_includes, the libraries, and the oracle
# options for the translation unit that carries the comparison.
#
# Why it is split. The causal sweep is five independent KV formats in one loop, and it is
# host-oracle-bound: median GPU utilisation measured 5% across 60 s of a 584 s test, so the card is
# idle while one process grinds formats in sequence. run_softmax_attention_causal_cache_tests takes
# the format as a filter over the same five run_storage_cases calls, with each format keeping its
# own criterion, so one entry per format is the identical work with the host phases spread across
# cores. Measured: 552.7 s sequential against 145.9 s for the slowest format, and the wall clock
# equalled the slowest, which is what shows they run concurrently.
#
# The unparameterised entry keeps plain_and_packed and context, which --kv-dtype cannot reach
# because it sets causal_only -- hence --no-causal, added for this. Without it this entry would
# re-run all five formats and duplicate 553 s.
#
# kv_cache_append is already registered this way, at --nvfp4-only and --k8v4-only.
add_executable(ninfer_softmax_attention_test
  "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/main.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/causal_cache.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/plain_and_packed.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/context.cpp")
ninfer_test_includes(ninfer_softmax_attention_test)
target_link_libraries(ninfer_softmax_attention_test PRIVATE ninfer_ops)
ninfer_op_oracle_options(ninfer_softmax_attention_test)

add_test(NAME ninfer_softmax_attention_test
  COMMAND ninfer_softmax_attention_test --no-causal)
set(ninfer_softmax_attention_kv_formats bf16 int8 fp8 nvfp4 k8v4)
foreach(format IN LISTS ninfer_softmax_attention_kv_formats)
  add_test(NAME ninfer_softmax_attention_${format}_test
    COMMAND ninfer_softmax_attention_test --kv-dtype ${format})
endforeach()
set_tests_properties(
  ninfer_softmax_attention_test
  ninfer_softmax_attention_bf16_test
  ninfer_softmax_attention_int8_test
  ninfer_softmax_attention_fp8_test
  ninfer_softmax_attention_nvfp4_test
  ninfer_softmax_attention_k8v4_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_op_test(ninfer_sliding_window_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sliding_window_attention.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_kv_cache_append_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kv_cache_append.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_rmsnorm_rope_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_rmsnorm_rope.cpp"
  LIBRARIES ninfer_ops)

# Split across four entries by batch, for the reason the softmax sweep above is split by KV format:
# this test is host-oracle-bound, so the card idles while one process walks the whole sweep (measured
# 166.1 s as one entry, the suite's longest).
#
# Unlike that sweep's filter, this one does not change what is asserted, and that is the point of
# choosing the batch as the axis: the check at the end of main compares each batch's workspace
# capacity against the peak that batch's own cases reached, and a batch's cases are exactly the ones
# --batches keeps or drops. The peak is a maximum over them, so a filtered run computes the number the
# whole-sweep run does. Verified by running all four here: the peaks match the unparameterised
# binary's batch for batch (41,943,040 / 655,360 / 983,040 / 1,310,720 / 1,638,400 / 1,966,080 /
# 1,863,680 / 1,966,080), and the four entries cost 82 / 25 / 21 / 33 s against 166 s in one.
#
# The batches are grouped so the heaviest (8, which carries the narrow cases and the graph replays,
# 82 s) stands alone; the rest are grouped by cost. A bare invocation still runs all eight, which is
# what a developer gets from the binary directly, so no entry re-runs another's work.
set(ninfer_context_kv_materialize_batch_groups "8" "1" "2,3,4" "5,6,7")
add_executable(ninfer_context_kv_materialize_test
  "${CMAKE_CURRENT_LIST_DIR}/test_context_kv_materialize.cpp")
ninfer_test_includes(ninfer_context_kv_materialize_test)
target_link_libraries(ninfer_context_kv_materialize_test PRIVATE ninfer_ops)
ninfer_op_oracle_options(ninfer_context_kv_materialize_test)
foreach(batch_group IN LISTS ninfer_context_kv_materialize_batch_groups)
  string(REPLACE "," "_" batch_suffix "${batch_group}")
  add_test(NAME "ninfer_context_kv_materialize_test_b${batch_suffix}"
    COMMAND ninfer_context_kv_materialize_test --batches "${batch_group}")
  set_tests_properties("ninfer_context_kv_materialize_test_b${batch_suffix}"
    PROPERTIES SKIP_RETURN_CODE 77)
endforeach()

add_test(NAME ninfer_kv_cache_append_nvfp4_test
  COMMAND ninfer_kv_cache_append_test --nvfp4-only)

add_test(NAME ninfer_kv_cache_append_k8v4_test
  COMMAND ninfer_kv_cache_append_test --k8v4-only)

set_tests_properties(
  ninfer_kv_cache_append_nvfp4_test
  ninfer_kv_cache_append_k8v4_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_op_test(ninfer_prepare_masked_block_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prepare_masked_block.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_sparse_moe_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sparse_moe.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_mtp_pack_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_pack.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_mtp_round_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_round.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_speculative_round_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_speculative_round.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_attn_input_proj_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_attn_input_proj.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_input_proj_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_dynamic_grouped_conv_prepare_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dynamic_grouped_conv_prepare.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_dynamic_grouped_conv_add_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_dynamic_grouped_conv_add.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_input_proj_conv_snapshot_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj_conv_snapshot.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_input_proj_conv_record_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj_conv_record.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gated_delta_net_replay_record_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gated_delta_net_replay_record.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_replay_fold_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_replay_fold.cpp"
  LIBRARIES ninfer_ops)

include("${CMAKE_CURRENT_LIST_DIR}/linear/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_add/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_pair/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_swiglu/tests.cmake")
