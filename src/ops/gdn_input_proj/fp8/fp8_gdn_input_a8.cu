#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"

namespace ninfer::ops::detail {
namespace {
using Tma64x128  = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma192x128 = Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>;
using MidBulk = Fp8A8SplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>, 170, 4, 8>;
using Bulk    = Fp8A8SplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

// Band boundaries, named once for this file: the ladder below selects on them and
// fp8_*_partial_capacity_bytes sizes on them, so neither can move alone. The capacity is a worst case
// over the bands, since whether a split engages depends on the token count.
// The 385-512 window returns to the narrow split tile, which is what makes this ladder
// non-monotonic and the capacity above a worst case rather than a selection.
constexpr std::int32_t kBandT64x128Max    = 128;
constexpr std::int32_t kBandBulkSelectsAt = 192;
constexpr std::int32_t kBandWideBulkMin   = 384;
constexpr std::int32_t kBandWideBulkMax   = 512;
} // namespace


std::size_t fp8_gdn_input_partial_capacity_bytes(std::int32_t max_tokens) {
    // Ungated with the ladder below, so this applies on every platform and the #ifdef that suppressed
    // it while the gate was in place is gone.
    //
    // Selection band, not engagement band. The old threshold was 256, which left 193-256 selecting a
    // split schedule with a null partials. This route's own test drives 193, 255, 256 and 257, so the
    // band is covered as soon as the gate comes off.
    return max_tokens > kBandBulkSelectsAt ? Bulk::kPartialBytes : 0;
}

void fp8_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const Fp8GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                                   static_cast<__nv_bfloat16*>(z.data)};
    const auto operands = fp8_a8_operands(weight, workspace, x.ne[1]);
    const auto launch   = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, 5120>;
        if constexpr (S::kTmaSwizzle)
            launch_fp8_a8_tma_mma<S>(operands, output, LinearIdentityEpilogue{}, stream,
                                       workspace.partials);
        else
            launch_fp8_a8_mma<S>(operands, output, LinearIdentityEpilogue{}, stream,
                             Fp8IdentityRows{}, workspace.partials);
    };
    // UNGATED 2026-10-02 (docs/active-work.md item 12). Third of the nine, and the one that closes the
    // oracle gap on the split 128x128 Stages=2 tile.
    //
    // That tile previously had NO oracle coverage anywhere, which blocked this route and
    // n16384_k5120. It does not need test scaffolding to be covered: this route's ladder assigns
    // MidBulk to 385-512 tokens, where the partials buffer IS allocated, and run_fp8() already drives
    // 383, 384, 385, 511, 512 and 513 -- plus 385 and 512 again through CUDA Graph replay. The forced-
    // tile experiment that crashed was measuring a configuration this ladder never selects.
    //
    // All four tiles are oracle-checked on both arms: 64x128 Stages=2 (1.107), 192x128 (1.241), the split
    // 128x128 Stages=2 (1.047), and the 128x256 split Bulk (1.270). TMA leads every median. Note the
    // 128x128 tile is the one where MMA wins two bands (448 and 512) -- 1.047 median is the thinnest
    // margin of the five, so it is recorded here rather than left to be discovered.
    // Workload, metric and harness: docs/active-work.md item 12, the median tile table.
    if (x.ne[1] <= 32) return launch.template operator()<Fp8A8T32R32K128>();
    if (x.ne[1] <= 64) return launch.template operator()<Fp8A8T64R128K256>();
    if (x.ne[1] <= kBandT64x128Max) return launch.template operator()<Tma64x128>();
    if (x.ne[1] <= kBandBulkSelectsAt) return launch.template operator()<Tma192x128>();
    // Smaller output tiles leave only two full-K tiles to split near the 512-token anchor.
    if (x.ne[1] > kBandWideBulkMin && x.ne[1] <= kBandWideBulkMax)
        return launch.template operator()<MidBulk>();
    launch.template operator()<Bulk>();
}
} // namespace ninfer::ops::detail
