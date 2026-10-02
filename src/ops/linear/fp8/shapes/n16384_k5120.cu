#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Fp8Geometry<16384, 5120>;
using Gemv     = Fp8A16GemvSchedule<4, 4, 16, 4, Fp8CodeCache::Default, 1, 1>;
// Seven resident CTAs keep the 1,024 row tiles in one wave without register spills.
using Sliced16 =
    Fp8A16SlicedKMmaSchedule<4, 16, 7, Cache::ca, Cache::cg, Fp8ActivationStage::PaddedZero, 1>;
using Tma64x128  = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma192x128 = Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>;
using MidBulk = Fp8A8SplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>, 170, 4, 8>;
using Bulk    = Fp8A8SplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 16) return fp8_linear_a16_sliced_k<Geometry, Sliced16>(x, weight, out, stream);
    if (tokens <= 24)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 8, 2>>(x, weight, out,
                                                                              stream);
    if (tokens <= 32)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 1>>(x, weight, out,
                                                                              stream);
    if (tokens <= 64)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>(
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
    // UNGATED 2026-10-02 (docs/active-work.md item 12). Fourth of the nine. Fp8N16384K5120 is the
    // gdn_input_proj shape, so the three gdn tests are this shape's oracle and were already green on
    // the ungated gdn ladder -- every tile here (64x128 Stages=2, 192x128, split 128x128 Stages=2, split
    // 128x256) is the same tile that route exercises at 65-192, 193-384, 385-512 and 513+.
    if (x.ne[1] <= 32)
    return launch_fp8_a8<Geometry, Fp8A8T32R32K128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 64)
    return launch_fp8_a8<Geometry, Fp8A8T64R128K256>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 128)
    return launch_fp8_a8_tma<Geometry, Tma64x128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= kBandBulkSelectsAt)
    return launch_fp8_a8_tma<Geometry, Tma192x128>(x, weight, out, scratch, stream);
    // Smaller output tiles leave only two full-K tiles to split near the 512-token anchor.
    if (x.ne[1] > 384 && x.ne[1] <= 512)
    return launch_fp8_a8_tma<Geometry, MidBulk>(x, weight, out, scratch, stream);
    launch_fp8_a8_tma<Geometry, Bulk>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    // Ungated with the ladder above, so this applies on every platform and the #ifdef that suppressed
    // it while the gate was in place is gone.
    //
    // Selection band, not engagement band. The old threshold was 256, which left 193-256 selecting a
    // split schedule with a null partials. This shape is Fp8N16384K5120 -- the gdn_input_proj shape -- so
    // run_fp8() in the gdn tests covers 193, 255, 256, 257, 385 and 512, which is exactly this band.
    return max_tokens > kBandBulkSelectsAt ? Bulk::kPartialBytes : 0;
}
} // namespace

const Fp8LinearShape kFp8N16384K5120{16384,     5120,    launch_a16,
                                     launch_a8, uses_a8, partial_capacity_bytes};
} // namespace ninfer::ops::detail
