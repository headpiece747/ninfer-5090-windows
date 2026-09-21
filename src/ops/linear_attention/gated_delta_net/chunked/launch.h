#pragma once

#include "core/layout.h"
#include "ops/linear_attention/gated_delta_net/chunked/context_parallel.h"
#include "ops/linear_attention/gated_delta_net/common.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#define NINFER_GATED_DELTA_NET_PROPAGATE(expr)                                                     \
    do {                                                                                           \
        const cudaError_t ninfer_gated_delta_net_error = (expr);                                   \
        if (ninfer_gated_delta_net_error != cudaSuccess) { return ninfer_gated_delta_net_error; }  \
    } while (0)

namespace ninfer::ops::detail::gated_delta_net::chunked {

inline constexpr std::size_t kWorkspaceAlign = 256;

struct workspace_layout {
    TensorRegion g_cumsum;
    TensorRegion W;
    TensorRegion U;
    TensorRegion v_new;
    TensorRegion h_chunk;
    // Context-parallel scratch, sized for the plan's segment count at this token extent. Bound but
    // unused when the plan disables CP (segment_count == 1).
    LayoutRegion cp_segment_begin;   // int32[segments]
    LayoutRegion cp_segment_count;   // int32[segments]
    LayoutRegion cp_num_warmup;      // int32[segments * value_heads]
    LayoutRegion cp_fallback;        // int32[segments * value_heads]
    LayoutRegion cp_segment_fallback; // int32[segments]
    LayoutRegion cp_segment_states;  // fp32[(segments + 1) * value_heads * kStateDim * kStateDim]
    std::size_t total_bytes = 0;
};

inline workspace_layout compute_workspace_layout(std::int32_t value_heads, std::int32_t tokens) {
    const std::int32_t chunks = tokens / kChunkSize;
    LayoutBuilder builder;
    workspace_layout w{};
    w.g_cumsum =
        builder.add_tensor(DType::FP32, {value_heads, tokens}, kWorkspaceAlign, "g_cumsum");
    w.W = builder.add_tensor(DType::BF16, {kStateDim, value_heads, tokens}, kWorkspaceAlign, "W");
    w.U = builder.add_tensor(DType::BF16, {kStateDim, value_heads, tokens}, kWorkspaceAlign, "U");
    w.v_new =
        builder.add_tensor(DType::BF16, {kStateDim, value_heads, tokens}, kWorkspaceAlign, "v_new");
    w.h_chunk     = builder.add_tensor(DType::BF16, {kStateDim, kStateDim, value_heads, chunks},
                                       kWorkspaceAlign, "h_chunk");
    const std::int32_t segments = std::max<std::int32_t>(
        1, plan_cp_segments(value_heads, std::max<std::int32_t>(0, chunks), cp_device_sm_count())
               .segment_count());
    const std::size_t segment_bytes =
        static_cast<std::size_t>(segments) * sizeof(std::int32_t);
    const std::size_t per_head_bytes = static_cast<std::size_t>(segments) * value_heads *
                                       sizeof(std::int32_t);
    w.cp_segment_begin  = builder.add(segment_bytes, kWorkspaceAlign, "cp segment begin");
    w.cp_segment_count  = builder.add(segment_bytes, kWorkspaceAlign, "cp segment count");
    w.cp_num_warmup     = builder.add(per_head_bytes, kWorkspaceAlign, "cp warmup chunks");
    w.cp_fallback       = builder.add(per_head_bytes, kWorkspaceAlign, "cp fallback flags");
    w.cp_segment_fallback = builder.add(segment_bytes, kWorkspaceAlign, "cp segment fallback");
    w.cp_segment_states = builder.add(
        static_cast<std::size_t>(segments + 1) * kStateDim * kStateDim * value_heads * sizeof(float),
        kWorkspaceAlign, "cp segment states");
    w.total_bytes = builder.finish(kWorkspaceAlign, "Gated DeltaNet chunk workspace");
    return w;
}

inline std::size_t workspace_bytes(std::int32_t value_heads, std::int32_t tokens) {
    return compute_workspace_layout(value_heads, tokens).total_bytes;
}

struct prepare_wy_wu_config {
    std::int32_t H_qk = 0;
    std::int32_t H_v  = 0;
    std::int32_t L    = 0;

    const __nv_bfloat16* k = nullptr;
    const __nv_bfloat16* v = nullptr;
    const float* g_in      = nullptr;
    const float* beta      = nullptr;

    __nv_bfloat16* W    = nullptr;
    __nv_bfloat16* U    = nullptr;
    float* g_cumsum_out = nullptr;

    cudaStream_t stream = nullptr;
};

struct state_passing_config {
    std::int32_t H_qk = 0;
    std::int32_t H_v  = 0;
    std::int32_t L    = 0;

    const __nv_bfloat16* W = nullptr;
    const __nv_bfloat16* U = nullptr;
    const __nv_bfloat16* k = nullptr;
    const float* g_cumsum  = nullptr;
    const float* state_in  = nullptr;

    __nv_bfloat16* v_new   = nullptr;
    __nv_bfloat16* h_chunk = nullptr;
    float* state_out       = nullptr;

    // Context parallelism. With segment_count <= 1 the launch uses state_in/state_out over the
    // whole L, exactly as before. With segment_count > 1, blockIdx.y selects a segment, its chunk
    // range comes from segment_begin/segment_chunk_count, and the state lives in segment_states
    // ([segment_count + 1][H_v][kStateDim][kStateDim] FP32; slot 0 is incoming, slot i+1 is
    // segment i's outgoing state).
    std::int32_t segment_count              = 1;
    const std::int32_t* segment_begin       = nullptr;
    const std::int32_t* segment_chunk_count = nullptr;
    float* segment_states                   = nullptr;
    // Optional replay guard: when non-null and `replay_flags[blockIdx.y] == 0`, every block returns
    // immediately. The launcher launches one guarded replay per segment so the whole Op stays
    // CUDA-Graph-capturable (no host decision on the correction).
    const std::int32_t* replay_flags        = nullptr;

    cudaStream_t stream = nullptr;
};

struct chunk_output_config {
    std::int32_t H_qk = 0;
    std::int32_t H_v  = 0;
    std::int32_t L    = 0;

    const __nv_bfloat16* q       = nullptr;
    const __nv_bfloat16* k       = nullptr;
    const __nv_bfloat16* v_new   = nullptr;
    const float* g_cumsum        = nullptr;
    const __nv_bfloat16* h_chunk = nullptr;

    __nv_bfloat16* attn_out = nullptr;

    float scale = 0.0f;

    cudaStream_t stream = nullptr;
};

struct stage_validator {
    const char* name;
    std::int32_t H_qk;
    std::int32_t H_v;
    std::int32_t T;

    cudaError_t check_shape() const {
        if (T <= 0 || !are_head_counts_valid(H_qk, H_v)) {
            std::fprintf(stderr, "%s: invalid shape (H_qk=%d H_v=%d T=%d)\n", name, H_qk, H_v, T);
            return cudaErrorInvalidValue;
        }
        return cudaSuccess;
    }

    cudaError_t check_full_chunks() const {
        if ((T % kChunkSize) != 0) {
            std::fprintf(stderr,
                         "%s: Gated DeltaNet chunked path requires T to be a multiple of %d; "
                         "route tail tokens through AR instead (T=%lld)\n",
                         name, kChunkSize, static_cast<long long>(T));
            return cudaErrorInvalidValue;
        }
        return cudaSuccess;
    }

    cudaError_t check_grid(std::int64_t grid_x, std::int64_t grid_y,
                           std::int64_t grid_z = 1) const {
        if (grid_x > static_cast<std::int64_t>(0xffffffff)) {
            std::fprintf(stderr, "%s: grid.x too large (%lld)\n", name,
                         static_cast<long long>(grid_x));
            return cudaErrorInvalidConfiguration;
        }
        if (grid_y > static_cast<std::int64_t>(0xffff)) {
            std::fprintf(stderr, "%s: grid.y too large (%lld)\n", name,
                         static_cast<long long>(grid_y));
            return cudaErrorInvalidConfiguration;
        }
        if (grid_z > static_cast<std::int64_t>(0xffff)) {
            std::fprintf(stderr, "%s: grid.z too large (%lld)\n", name,
                         static_cast<long long>(grid_z));
            return cudaErrorInvalidConfiguration;
        }
        return cudaSuccess;
    }
};

cudaError_t launch_prepare_wy_wu(const prepare_wy_wu_config& cfg);
cudaError_t launch_state_passing(const state_passing_config& cfg);
cudaError_t launch_output(const chunk_output_config& cfg);

} // namespace ninfer::ops::detail::gated_delta_net::chunked
