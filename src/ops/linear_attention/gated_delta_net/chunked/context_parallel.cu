#include "ops/linear_attention/gated_delta_net/chunked/context_parallel.h"

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net::chunked {
namespace {

// One thread per (segment, head). Serial over the segment's chunks from the end, accumulating the
// chunk-local log-decay read at each chunk's last token. `g` is negative, so the running sum only
// decreases; crossing `threshold` means the incoming state has decayed below the tolerance.
__global__ void cp_gate_warmup_kernel(const float* __restrict__ g_cumsum, int value_heads, int bt,
                                      const int* __restrict__ segment_begin,
                                      const int* __restrict__ segment_count, int segment_count_n,
                                      float threshold, int* __restrict__ num_warmup_chunks,
                                      std::uint8_t* __restrict__ fallback) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= segment_count_n * value_heads) { return; }
    const int segment = index / value_heads;
    const int head    = index - segment * value_heads;

    const int begin = segment_begin[segment];
    const int count = segment_count[segment];
    if (count <= 0) {
        num_warmup_chunks[index] = 0;
        fallback[index]          = 0;
        return;
    }

    float running = 0.0f;
    int needed    = count; // whole segment when the decay never dominates
    std::uint8_t needs_correction = 1;
    for (int step = 0; step < count; ++step) {
        const int chunk = begin + count - 1 - step;
        const float g   = g_cumsum[(static_cast<std::int64_t>(chunk) * bt + (bt - 1)) * value_heads +
                                   head];
        running += g;
        if (running < threshold) {
            needed           = step + 1;
            needs_correction = 0;
            break;
        }
    }
    num_warmup_chunks[index] = needed;
    fallback[index]          = needs_correction;
}

} // namespace

void launch_cp_gate_warmup(const float* g_cumsum, std::int32_t value_heads, std::int32_t bt,
                           const std::int32_t* segment_begin, const std::int32_t* segment_count,
                           std::int32_t segment_count_n, float threshold,
                           std::int32_t* num_warmup_chunks, std::uint8_t* fallback,
                           cudaStream_t stream) {
    if (value_heads <= 0 || bt <= 0 || segment_count_n <= 0) { return; }
    const int total = segment_count_n * value_heads;
    const int block = 256;
    const int grid  = (total + block - 1) / block;
    cp_gate_warmup_kernel<<<grid, block, 0, stream>>>(
        g_cumsum, value_heads, bt, segment_begin, segment_count, segment_count_n, threshold,
        num_warmup_chunks, fallback);
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked
