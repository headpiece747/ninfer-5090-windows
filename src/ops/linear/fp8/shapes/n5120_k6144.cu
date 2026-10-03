#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry  = Fp8Geometry<5120, 6144>;
using Gemv      = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using Tma32x64  = Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>;
using Tma64x128 = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>;
using MidBulk   = Fp8A8SplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, 170, 4, 8>;
using Bulk      = Fp8A8SplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

// Band boundaries, named once for the whole file. launch_a8 selects on them and
// partial_capacity_bytes sizes on them, so a band cannot be moved in one place and not the other --
// which is the whole defect class: six under-allocations, one over-allocation and a static_assert over
// a false invariant all came from these two functions restating the same numbers by hand.
//
// The capacity below is the WORST CASE over the bands, not the size of the band max_tokens itself
// selects. Whether a split engages at run time depends on the token count (fp8_split_k_plan), so a
// buffer sized for the selection at max_tokens can still be read by a different selection.
constexpr std::int32_t kBandT32x64Max  = 64;
constexpr std::int32_t kBandT64x64Max  = 128;
constexpr std::int32_t kBandT64x128Max = 192;
constexpr std::int32_t kBandMidBulkMax = 768;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 16)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 8, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 32)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 4, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 64)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 1>>(x, weight, out,
                                                                              stream);
    if (tokens <= 128)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 64, 128, 32, 16, 2, 2>>(
            x, weight, out, stream);
    fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, out,
                                                                               stream);
}

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    // UNGATED 2026-10-02 (docs/active-work.md item 12). Ninth and last of the nine. Fp8N5120K6144
    // owned by linear_add, whose oracle (ninfer_linear_add_fp8_test) is green on the ungated K-templated
    // ladder with these same tiles. Its partials thresholds were already correct and unconditional:
    // MidBulk at >192 and Bulk at >768 match the ladder's 193-768 and 769+ selection bands exactly.
    if (x.ne[1] <= kBandT32x64Max)
        return launch_fp8_a8_tma<Geometry, Tma32x64>(x, weight, out, scratch, stream);
    if (x.ne[1] <= kBandT64x64Max)
        return launch_fp8_a8<Geometry, Fp8A8T64R64K128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= kBandT64x128Max)
        return launch_fp8_a8_tma<Geometry, Tma64x128>(x, weight, out, scratch, stream);
    // The narrower row tile fills the GPU before the large-tile path reaches a full wave.
    if (x.ne[1] <= kBandMidBulkMax)
        return launch_fp8_a8_tma<Geometry, MidBulk>(x, weight, out, scratch, stream);
    launch_fp8_a8_tma<Geometry, Bulk>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    if (max_tokens > kBandMidBulkMax) return Bulk::kPartialBytes;
    return max_tokens > kBandT64x128Max ? MidBulk::kPartialBytes : 0;
}

} // namespace

const Fp8LinearShape kFp8N5120K6144{5120,      6144,    launch_a16,
                                    launch_a8, uses_a8, partial_capacity_bytes};
} // namespace ninfer::ops::detail
