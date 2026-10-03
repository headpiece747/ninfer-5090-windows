#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry  = Fp8Geometry<14336, 5120>;
using Gemv      = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using Tma64x128 = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma64x256 = Fp8A8TmaMmaSchedule<64, 256, 128, 2, 4, 2, 1>;
using Tma96x256 = Fp8A8TmaMmaSchedule<96, 256, 128, 3, 4, 2, 1>;
using Bulk      = Fp8A8SplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 16)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 4, 1>>(x, weight, out,
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
    if (tokens <= 128)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>>(
            x, weight, out, stream);
    fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, out,
                                                                               stream);
}


// Band boundaries, named once for this file. launch_a8 selects on them and partial_capacity_bytes
// sizes on them, so a band cannot be moved in one place and not the other.
constexpr std::int32_t kBandBulkSelectsAt = 288;

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    // UNGATED 2026-10-02 (docs/active-work.md item 12). Fifth of the nine.
    //
    // Fp8N14336K5120 belongs to attn_input_proj, whose oracle is already green on the ungated ladder
    // with these same tiles: 64x128 Stages=2 at 65-128, 64x256 at 129-192, 96x256 at 193-288 and the
    // 128x256 split Bulk at 289+. TMA medians on them: 1.107, 1.270, 1.288, 1.270.
    // Workload, metric and harness: docs/active-work.md item 12, the median tile table.
    //
    // This shape genuinely splits, which is why it needs the partials buffer: 14336/256 = 56 row
    // tiles, so the plan sees tail = 56 <= kSplitWaveCtas/2 = 85 and splits the underfilled wave.
    // (n34816 is the opposite case and is deliberately NOT ungated -- see its file.)
    if (x.ne[1] <= 32)
        return launch_fp8_a8<Geometry, Fp8A8T32R32K128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 96)
        return launch_fp8_a8<Geometry, Fp8A8T32R128K128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 128)
        return launch_fp8_a8_tma<Geometry, Tma64x128>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 192)
        return launch_fp8_a8_tma<Geometry, Tma64x256>(x, weight, out, scratch, stream);
    // Three 96-token tiles give 168 CTAs: one almost-full wave through T=288.
    if (x.ne[1] <= kBandBulkSelectsAt)
        return launch_fp8_a8_tma<Geometry, Tma96x256>(x, weight, out, scratch, stream);
    launch_fp8_a8_tma<Geometry, Bulk>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    // Ungated with the ladder above. Selection band: Bulk is the fall-through at >=289 tokens, so
    // 289-384 needs the buffer too. The old threshold was 384. attn_input_proj's test already drives
    // 289 and 385, the two boundaries of that band.
    return max_tokens > kBandBulkSelectsAt ? Bulk::kPartialBytes : 0;
}
} // namespace

const Fp8LinearShape kFp8N14336K5120{14336,     5120,    launch_a16,
                                     launch_a8, uses_a8, partial_capacity_bytes};
} // namespace ninfer::ops::detail
