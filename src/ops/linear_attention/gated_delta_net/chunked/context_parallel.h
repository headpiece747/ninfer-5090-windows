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
// `__host__ __device__` so the device-side segment fill agrees with the host planner bit for bit.
[[nodiscard]] inline __host__ __device__ std::int32_t
cp_max_local_chunks(std::int32_t value_heads, std::int32_t total_chunks,
                    std::int32_t sm_count) noexcept {
    if (value_heads <= 0 || total_chunks <= 0 || sm_count <= 0) { return 4; }
    const double inner = ::sqrt(static_cast<double>(value_heads) * total_chunks / sm_count) * 3.0;
    if (!(inner > 0.0)) { return 4; }
    // Python round() is round-half-to-even; nearbyint matches it. Keep double: the formula sits on
    // exact halves for some shapes and single precision would diverge from the reference.
    const int exponent = static_cast<int>(::nearbyint(::log2(inner)));
    if (exponent <= 0) { return 4; }
    const std::int32_t scaled = static_cast<std::int32_t>(1) << (exponent > 30 ? 30 : exponent);
    return scaled < 4 ? 4 : scaled;
}

// Context parallelism is gated OFF until the exact affine boundary correction exists.
//
// FlashQLA's predicate (cp_context.py:102-109, for a single sequence with Be == 1) is
//   SM90/SM120: use_cp = Be*H <= 40 or (Be*H <= 56 and max(chunks) >= 128)
// but that predicate is only sound together with its `M`-matrix correction, which this port does
// not implement. Without it a segment whose gate decayed below the warmup threshold keeps the
// state produced by running the segment from zero, and the dropped incoming state is not
// negligible at this Op's tolerance: at chunk 64 the residual is ~4.5e-5 against a 1.0e-5
// gross-absolute criterion, and FlashQLA itself validates CP only to RTOL = 0.02.
//
// The exact alternative - replaying every segment from its true predecessor state - is correct but
// serializes: measured ~632 us of replays at T=8192 against a 143 us CP win, a net loss. The
// segmented kernel, the planner, the warmup scan and the replay chain remain in the tree as the
// foundation the exact correction builds on; see docs/adr/ for the decision record and
// tests/ops/test_gated_delta_net.cpp for the contract that unit 3 must satisfy.
[[nodiscard]] inline bool cp_enabled(std::int32_t /*value_heads*/,
                                     std::int32_t /*total_chunks*/) noexcept {
    return false;
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

// The gate warmup threshold, in chunk-log-decay units: a segment whose incoming state has decayed
// below e^threshold may be seeded from zero without a boundary correction.
//
// FlashQLA uses -10.0, which is tuned for its sm120 chunk 32. The residual error of dropping the
// incoming state is e^threshold * |state|, and this Op's state magnitude reaches O(1), so -10.0
// leaves ~4.5e-5 - above the 1.0e-5 gross-absolute criterion the Op is qualified against. The
// threshold must therefore be tighter at chunk 64: -13.0 gives e^-13 * |state| <= 2.3e-6 * |state|,
// sound for the state magnitudes this recurrence produces. Raising the magnitude makes the guard
// skip fewer segments (more boundary replays, less parallelism), which is the correct trade: a
// wrong boundary state is a silent numerical error, a skipped replay is only lost speed.
inline constexpr float kCpWarmupThreshold = -13.0f;

// Scan `g_cumsum` backward from each segment's last chunk, one value per chunk at its last token.
// For each (segment, head): `num_warmup_chunks` is how many trailing chunks must be recomputed from
// a zero start for the end state to be accurate, and `fallback` is 1 when the decay never crossed
// the threshold inside the segment, so the segment needs an exact boundary correction.
//
// All pointers are caller-owned device memory. `g_cumsum` is [chunk][BT][value_heads] with the
// chunk-local cumulative gate, matching the layout chunked/prepare_wy_wu.cuh writes.
// Device SM count for the active device, cached after the first query. The CP split heuristic
// needs it; it is a hardware fact, not part of the Op's semantic inputs.
[[nodiscard]] std::int32_t cp_device_sm_count() noexcept;

// Fill `segment_begin[i]`/`segment_count[i]` for the plan on the device. Device-side because the
// chunked Op runs inside CUDA Graph capture, where a pageable host-to-device copy is not permitted.
void launch_cp_fill_segments(std::int32_t segments, std::int32_t value_heads, std::int32_t chunks,
                             std::int32_t sm_count, std::int32_t* segment_begin,
                             std::int32_t* segment_count, cudaStream_t stream);

// `fallback` is per (segment, head); `segment_fallback` is per segment (any head) and must be
// zero-initialised by the caller, so the replay guard is a single load.
void launch_cp_gate_warmup(const float* g_cumsum, std::int32_t value_heads, std::int32_t bt,
                           const std::int32_t* segment_begin, const std::int32_t* segment_count,
                           std::int32_t segment_count_n, float threshold,
                           std::int32_t* num_warmup_chunks, std::int32_t* fallback,
                           std::int32_t* segment_fallback, cudaStream_t stream);

} // namespace ninfer::ops::detail::gated_delta_net::chunked
