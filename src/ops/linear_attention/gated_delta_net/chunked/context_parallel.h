#pragma once

// Gate-driven context parallelism (CP) planning for the chunked Gated DeltaNet pipeline.
//
// This is execution policy, not Op semantics: it decides how many independent segments a single
// sequence's chunk range is cut into so the per-chunk state recurrence can run with more than one
// wave of work. It never changes the mathematical result - unit 2 corrects each segment's incoming
// state against the true recurrence.
//
// The segmentation and the CP-enable gate mirror QwenLM/FlashQLA's sm120 path:
//   flash_qla/ops/gated_delta_rule/chunk/cp_context.py:63-72  (max_local_chunks)
//   flash_qla/ops/gated_delta_rule/chunk/cp_context.py:102-109 (use_cp gate)
// See docs/research/gdn-flashqla-context-parallelism.md for the citations, the warmup threshold,
// and the boundary-correction scheme.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ninfer::ops::detail::gated_delta_net::chunked {

// One segment of a sequence's chunk range. `chunk_begin + chunk_count` are in chunk units, and
// segment boundaries always fall on chunk boundaries.
struct cp_segment {
    std::int32_t chunk_begin = 0;
    std::int32_t chunk_count = 0;
};

struct cp_segment_plan {
    bool use_cp                    = false;
    std::int32_t max_local_chunks  = 0;
    std::vector<cp_segment> segments;

    [[nodiscard]] std::int32_t segment_count() const noexcept {
        return static_cast<std::int32_t>(segments.size());
    }
};

// FlashQLA cp_context.py:63-72: L_cp* is proportional to sqrt(B*H*Lc / P), scaled by 3 and rounded
// to a power of two, floored at 4. Python's round() is round-half-to-even, so nearbyint matches it.
[[nodiscard]] inline std::int32_t cp_max_local_chunks(std::int32_t value_heads,
                                                      std::int32_t total_chunks,
                                                      std::int32_t sm_count) noexcept {
    if (value_heads <= 0 || total_chunks <= 0 || sm_count <= 0) { return 4; }
    const double inner = std::sqrt(static_cast<double>(value_heads) * total_chunks / sm_count) * 3.0;
    if (!(inner > 0.0)) { return 4; }
    const int exponent = static_cast<int>(std::nearbyint(std::log2(inner)));
    const std::int32_t scaled =
        exponent <= 0 ? 1 : static_cast<std::int32_t>(1) << std::min(exponent, 30);
    return std::max<std::int32_t>(4, scaled);
}

// FlashQLA cp_context.py:102-109, for a single sequence (Be = sum(chunks)/max(chunks) = 1):
//   SM90/SM120: use_cp = Be*H <= 40 or (Be*H <= 56 and max(chunks) >= 128)
[[nodiscard]] inline bool cp_enabled(std::int32_t value_heads,
                                     std::int32_t total_chunks) noexcept {
    const std::int64_t be_h = value_heads; // Be == 1 for batch 1
    return be_h <= 40 || (be_h <= 56 && total_chunks >= 128);
}

// Cut `total_chunks` at chunk boundaries every max_local_chunks. Returns one whole-range segment
// with use_cp=false when CP is not enabled or there is nothing to split.
[[nodiscard]] inline cp_segment_plan plan_cp_segments(std::int32_t value_heads,
                                                     std::int32_t total_chunks,
                                                     std::int32_t sm_count) {
    cp_segment_plan plan;
    if (total_chunks <= 0) { return plan; }
    plan.max_local_chunks = cp_max_local_chunks(value_heads, total_chunks, sm_count);
    if (!cp_enabled(value_heads, total_chunks)) {
        plan.segments.push_back(cp_segment{0, total_chunks});
        return plan;
    }
    plan.use_cp = true;
    for (std::int32_t begin = 0; begin < total_chunks; begin += plan.max_local_chunks) {
        plan.segments.push_back(
            cp_segment{begin, std::min(plan.max_local_chunks, total_chunks - begin)});
    }
    return plan;
}

// The gate warmup threshold, in chunk-log-decay units. FlashQLA defaults to -10.0, i.e. the
// incoming state's retained fraction is below e^-10 (~4.5e-5). Validated at chunk 32 upstream;
// re-check at chunk 64 (see the research note's open question 3).
inline constexpr float kCpWarmupThreshold = -10.0f;

// Scan `g_cumsum` backward from each segment's last chunk, one value per chunk at its last token.
// For each (segment, head): `num_warmup_chunks` is how many trailing chunks must be recomputed from
// a zero start for the end state to be accurate, and `fallback` is 1 when the decay never crossed
// the threshold inside the segment, so the segment needs an exact boundary correction.
//
// All pointers are caller-owned device memory. `g_cumsum` is [chunk][BT][value_heads] with the
// chunk-local cumulative gate, matching the layout chunked/prepare_wy_wu.cuh writes.
void launch_cp_gate_warmup(const float* g_cumsum, std::int32_t value_heads, std::int32_t bt,
                           const std::int32_t* segment_begin, const std::int32_t* segment_count,
                           std::int32_t segment_count_n, float threshold,
                           std::int32_t* num_warmup_chunks, std::uint8_t* fallback,
                           cudaStream_t stream);

} // namespace ninfer::ops::detail::gated_delta_net::chunked
