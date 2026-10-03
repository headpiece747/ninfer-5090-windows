#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Fp8Geometry<5120, 17408>;
using Gemv     = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using Tma32x64 = Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>;
using Small = Fp8A8SplitKSchedule<Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>, 170, 4, 8>;
using Mid = Fp8A8SplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, 170, 4, 8>;
using Wide = Fp8A8SplitKSchedule<Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>, 170, 4, 8>;
using Bulk = Fp8A8SplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 8)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<8, 8, 2>>(x, weight, out,
                                                                             stream);
    if (tokens <= 16)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 8, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 32)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 8, 1>>(x, weight, out,
                                                                              stream);
    if (tokens <= 64)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<64, 2, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 128)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>(
            x, weight, out, stream);
    fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, out,
                                                                               stream);
}

// Band boundaries, named once for this file. launch_a8 selects on them and partial_capacity_bytes
// sizes on them, so a band cannot be moved in one place and not the other.
constexpr std::int32_t kBandT32x64Max = 64;
constexpr std::int32_t kBandSmallMax  = 128;
constexpr std::int32_t kBandMidMax    = 256;
constexpr std::int32_t kBandWideMax   = 384;
// Above kBandWideMax the ladder falls through to Bulk, except for the window below that returns to
// Wide. That makes this ladder NON-MONOTONIC, and it is why the capacity above is a worst case over
// the bands rather than the size of the band max_tokens itself selects: 385-512 runs Bulk and 513-768
// runs Wide, so a single buffer has to cover whichever one will be selected.
constexpr std::int32_t kBandBulkMin       = 512;
// The upper bound of that returning window, and it is NOT kBandWideMax. Naming them separately is
// load-bearing: an earlier revision of this refactor reused kBandWideMax here and produced
// `tokens > 512 && tokens <= 384`, which is false for every token count, so 513-768 silently fell
// through to Bulk and lost the wide tile. The suite stayed green because both tiles are correct and
// differ only in speed.
constexpr std::int32_t kBandWideReturnMax = 768;

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    // UNGATED 2026-10-02 (docs/active-work.md item 12). Fp8N5120K17408, like Fp8N5120K6144, is
    // owned by linear_add, whose oracle (ninfer_linear_add_fp8_test) is green on the ungated K-templated
    // ladder with these same tiles. Its partials thresholds were already correct and unconditional:
    // MidBulk at >192 and Bulk at >768 match the ladder's 193-768 and 769+ selection bands exactly.
    const int tokens = x.ne[1];
    if (tokens <= kBandT32x64Max)
        return launch_fp8_a8_tma<Geometry, Tma32x64>(x, weight, out, scratch, stream);
    if (tokens <= kBandSmallMax)
        return launch_fp8_a8_tma<Geometry, Small>(x, weight, out, scratch, stream);
    if (tokens <= kBandMidMax)
        return launch_fp8_a8_tma<Geometry, Mid>(x, weight, out, scratch, stream);
    // Wider token tiles avoid an extra wave in the gaps between the bulk anchors.
    if (tokens <= kBandWideMax || (tokens > kBandBulkMin && tokens <= kBandWideReturnMax))
        return launch_fp8_a8_tma<Geometry, Wide>(x, weight, out, scratch, stream);
    launch_fp8_a8_tma<Geometry, Bulk>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    if (max_tokens > kBandWideMax) return Bulk::kPartialBytes;
    if (max_tokens > kBandMidMax) return Wide::kPartialBytes;
    if (max_tokens > kBandSmallMax) return Mid::kPartialBytes;
    return max_tokens > kBandT32x64Max ? Small::kPartialBytes : 0;
}

} // namespace

const Fp8LinearShape kFp8N5120K17408{5120,      17408,   launch_a16,
                                     launch_a8, uses_a8, partial_capacity_bytes};
} // namespace ninfer::ops::detail
