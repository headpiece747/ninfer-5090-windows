#include "ops/linear_attention/gated_delta_net/launch.h"

#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/chunked/launch.h"

#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

namespace ninfer::ops::detail::gated_delta_net {
std::size_t chunked_workspace_bytes(std::int32_t value_heads, std::int32_t tokens) {
    if (tokens <= 0) { return 0; }
    return chunked::workspace_bytes(value_heads, tokens);
}

void launch_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                    const Tensor& beta, float scale, const Tensor& ssm_state_in,
                    Tensor& ssm_state_out, Tensor& out, void* workspace,
                    std::size_t workspace_bytes, cudaStream_t stream) {
    const auto layout = chunked::compute_workspace_layout(v.ne[1], q.ne[2]);
    if (workspace == nullptr || workspace_bytes < layout.total_bytes) { throw std::bad_alloc(); }

    const DeviceSpan backing{workspace, workspace_bytes};
    const Tensor g_cumsum = layout.g_cumsum.bind(backing);
    const Tensor W        = layout.W.bind(backing);
    const Tensor U        = layout.U.bind(backing);
    const Tensor v_new    = layout.v_new.bind(backing);
    const Tensor h_chunk  = layout.h_chunk.bind(backing);

    auto* const cp_segment_begin  = static_cast<std::int32_t*>(layout.cp_segment_begin.bind(backing).data);
    auto* const cp_segment_count  = static_cast<std::int32_t*>(layout.cp_segment_count.bind(backing).data);
    auto* const cp_num_warmup     = static_cast<std::int32_t*>(layout.cp_num_warmup.bind(backing).data);
    auto* const cp_fallback       = static_cast<std::int32_t*>(layout.cp_fallback.bind(backing).data);
    auto* const cp_segment_fallback =
        static_cast<std::int32_t*>(layout.cp_segment_fallback.bind(backing).data);
    auto* const cp_segment_states = static_cast<float*>(layout.cp_segment_states.bind(backing).data);

    chunked::prepare_wy_wu_config prepare{};
    prepare.H_qk         = q.ne[1];
    prepare.H_v          = v.ne[1];
    prepare.L            = q.ne[2];
    prepare.k            = static_cast<const __nv_bfloat16*>(k.data);
    prepare.v            = static_cast<const __nv_bfloat16*>(v.data);
    prepare.g_in         = static_cast<const float*>(g.data);
    prepare.beta         = static_cast<const float*>(beta.data);
    prepare.W            = static_cast<__nv_bfloat16*>(W.data);
    prepare.U            = static_cast<__nv_bfloat16*>(U.data);
    prepare.g_cumsum_out = static_cast<float*>(g_cumsum.data);
    prepare.stream       = stream;
    CUDA_CHECK(chunked::launch_prepare_wy_wu(prepare));

    chunked::state_passing_config state{};
    state.H_qk      = q.ne[1];
    state.H_v       = v.ne[1];
    state.L         = q.ne[2];
    state.W         = static_cast<const __nv_bfloat16*>(W.data);
    state.U         = static_cast<const __nv_bfloat16*>(U.data);
    state.k         = static_cast<const __nv_bfloat16*>(k.data);
    state.g_cumsum  = static_cast<const float*>(g_cumsum.data);
    state.state_in  = static_cast<const float*>(ssm_state_in.data);
    state.v_new     = static_cast<__nv_bfloat16*>(v_new.data);
    state.h_chunk   = static_cast<__nv_bfloat16*>(h_chunk.data);
    state.state_out = static_cast<float*>(ssm_state_out.data);
    state.stream    = stream;

    // Gate-driven context parallelism. `prepare_wy_wu` and `output` are chunk-independent, so only
    // this stage is segmented: the chunk range is cut into independent segments and each runs from
    // its own incoming state. Segment 0 starts from the caller's state; later segments start from
    // zero and are corrected below when their gate does not decay enough.
    const std::int32_t chunks = q.ne[2] / kChunkSize;
    const chunked::cp_segment_plan plan =
        chunked::plan_cp_segments(v.ne[1], chunks, chunked::cp_device_sm_count());
    const std::int32_t segments = plan.segment_count();
    if (plan.use_cp && segments > 1) {
        const std::int64_t state_stride =
            static_cast<std::int64_t>(v.ne[1]) * kStateDim * kStateDim;
        // Capture-safe: the plan is a pure function of (H_v, chunks, SM count), so the device fills
        // its own segment table. A pageable host-to-device copy is illegal inside CUDA Graph
        // capture, which is how the chunked Op is exercised.
        CUDA_CHECK(cudaMemsetAsync(cp_segment_fallback, 0,
                                   static_cast<std::size_t>(segments) * sizeof(std::int32_t),
                                   stream));
        chunked::launch_cp_fill_segments(segments, v.ne[1], chunks, chunked::cp_device_sm_count(),
                                         cp_segment_begin, cp_segment_count, stream);
        // Slots 1..segments are the per-segment ends, produced from a zero start; slot 0 is the
        // caller's incoming state.
        CUDA_CHECK(cudaMemsetAsync(
            cp_segment_states + state_stride, 0,
            static_cast<std::size_t>(segments) * static_cast<std::size_t>(state_stride) *
                sizeof(float),
            stream));
        CUDA_CHECK(cudaMemcpyAsync(cp_segment_states, ssm_state_in.data,
                                   static_cast<std::size_t>(state_stride) * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));

        chunked::launch_cp_gate_warmup(state.g_cumsum, v.ne[1], kChunkSize, cp_segment_begin,
                                       cp_segment_count, segments, chunked::kCpWarmupThreshold,
                                       cp_num_warmup, cp_fallback, cp_segment_fallback, stream);

        state.segment_count       = segments;
        state.segment_begin       = cp_segment_begin;
        state.segment_chunk_count = cp_segment_count;
        state.segment_states      = cp_segment_states;
        CUDA_CHECK(chunked::launch_state_passing(state));

        // Exact correction, capture-safe: one guarded replay per segment, in order. Each reads the
        // segment's true carried state (slot i, already resolved by the replays launched before
        // it) and no-ops on the device when its fallback flag is clear - a host decision would
        // need a sync, which capture forbids. Replaying the whole segment when any head falls back
        // is exact; narrowing that per head is a later refinement, not a correctness change.
        // The replay launches one segment each, so the kernel's blockIdx.y is 0 and every array it
        // indexes must already be offset to segment i: `segment_begin[i]`/`segment_count[i]` give
        // the chunk range, `segment_states + i*stride` is the true incoming state (resolved by the
        // replay launched for i-1, or by the initial launch for i == 1), and the outgoing state
        // lands in slot i+1. `replay_flags` is offset the same way, so the guard reads segment i's
        // flag rather than segment 0's.
        for (std::int32_t i = 1; i < segments; ++i) {
            chunked::state_passing_config fix = state;
            fix.segment_count       = 1;
            fix.segment_begin       = cp_segment_begin + i;
            fix.segment_chunk_count = cp_segment_count + i;
            fix.segment_states =
                cp_segment_states + static_cast<std::int64_t>(i) * state_stride;
            fix.replay_flags        = cp_segment_fallback + i;
            CUDA_CHECK(chunked::launch_state_passing(fix));
        }
        CUDA_CHECK(cudaMemcpyAsync(
            ssm_state_out.data,
            cp_segment_states + static_cast<std::int64_t>(segments) * state_stride,
            static_cast<std::size_t>(state_stride) * sizeof(float), cudaMemcpyDeviceToDevice,
            stream));
    } else {
        CUDA_CHECK(chunked::launch_state_passing(state));
    }

    chunked::chunk_output_config output{};
    output.H_qk     = q.ne[1];
    output.H_v      = v.ne[1];
    output.L        = q.ne[2];
    output.q        = static_cast<const __nv_bfloat16*>(q.data);
    output.k        = static_cast<const __nv_bfloat16*>(k.data);
    output.v_new    = static_cast<const __nv_bfloat16*>(v_new.data);
    output.g_cumsum = static_cast<const float*>(g_cumsum.data);
    output.h_chunk  = static_cast<const __nv_bfloat16*>(h_chunk.data);
    output.attn_out = static_cast<__nv_bfloat16*>(out.data);
    output.scale    = scale;
    output.stream   = stream;
    CUDA_CHECK(chunked::launch_output(output));
}

} // namespace ninfer::ops::detail::gated_delta_net
