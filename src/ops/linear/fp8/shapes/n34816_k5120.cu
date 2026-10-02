#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry   = Fp8Geometry<34816, 5120>;
using C4         = Fp8A16SimtSchedule<4, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked,
                                      Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using Tma64x128  = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma64x256  = Fp8A8TmaMmaSchedule<64, 256, 128, 2, 4, 2, 1>;
using Tma128x256 = Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>;
using Bulk       = Fp8A8SplitKSchedule<Tma128x256, 170, 4, 8>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 4) return fp8_linear_a16_simt<Geometry, 4, C4>(x, weight, out, stream);
    if (tokens <= 8)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<8, 4, 1>>(x, weight, out,
                                                                             stream);
    if (tokens <= 16)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 2, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 24)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 32)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 1>>(x, weight, out,
                                                                              stream);
    if (tokens <= 64)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(
            x, weight, out, stream);
    if (tokens <= 96)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(
            x, weight, out, stream);
    fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, out,
                                                                               stream);
}


// Band boundaries, named once for this file. launch_a8 selects on them and partial_capacity_bytes
// sizes on them, so a band cannot be moved in one place and not the other.
constexpr std::int32_t kBandBulkSelectsAt = 192;

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    // UNGATED 2026-10-02 (docs/active-work.md item 12). Eighth of nine. Fp8N34816K5120 belongs to
    // linear_swiglu, whose oracle is already green on the ungated ladder with these same tiles
    // (64x128 Stages=2, 64x256, and the 128x256 split Bulk).
    //
    // This one needed the capacity function fixed BEFORE it could be ungated, not after: see the
    // static_assert below. Its split never engages, so allocating on the selection band would have
    // reserved 21.3 MiB per workspace for a buffer never read.
    if (x.ne[1] <= 64)
    return launch_fp8_a8<Geometry, Fp8A8T64R128K256>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 128)
    return launch_fp8_a8_tma<Geometry, Tma64x128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= kBandBulkSelectsAt)
    return launch_fp8_a8_tma<Geometry, Tma64x256>(x, weight, out, scratch, stream);
    // At T <= 256 the last wave is already well filled, so Bulk uses its ordinary
    // TMA kernel and needs no partials. Keep one compiled family for this region.
    launch_fp8_a8_tma<Geometry, Bulk>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 5; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    // CORRECTED 2026-10-02. An earlier version of this function claimed the split could NEVER engage
    // for this shape and returned 0 at every token count, guarded by a static_assert. That was wrong,
    // and the full suite caught it: ninfer_linear_fp8_a8_test threw "FP8 TMA split-K requires aligned
    // caller partials" at T = 257, 511, 512, 1024 and 1025 on this shape.
    //
    // The error was assuming for_each_token_slice caps a slice at kBlockTokens. It does not:
    //     const std::int64_t capacity = columns_per_block * kCudaGridYLimit;
    // so a slice spans kBlockTokens * kCudaGridYLimit tokens and blocks = rows / kBlockRows *
    // div_up(count, kBlockTokens) GROWS with the token count. At T=257 that is 136 * 3 = 408 CTAs, tail
    // = 408 % 170 = 68, which is inside the split range (<= 85). There is no compile-time property
    // here at all -- whether a split happens depends on the token count, so the buffer has to cover the
    // worst case across the whole selection band, which is what every other shape does.
    //
    // The static_assert is gone rather than repaired: it was a guard over a false invariant, which is
    // worse than no guard, because it read as a proof. Note also that it PASSED, which is the point --
    // an assertion over an assumption cannot detect the assumption being wrong.
    //
    // Selection band, not engagement band. The old threshold was 256, which left 193-256 selecting Bulk
    // with a null partials. linear_swiglu's test drives 193, 257 and 513, which is exactly this band.
    return max_tokens > kBandBulkSelectsAt ? Bulk::kPartialBytes : 0;
}
} // namespace

const Fp8LinearShape kFp8N34816K5120{34816,     5120,    launch_a16,
                                     launch_a8, uses_a8, partial_capacity_bytes};
} // namespace ninfer::ops::detail
