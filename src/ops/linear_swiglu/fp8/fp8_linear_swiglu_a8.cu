#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear_swiglu/token_major_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
using Tma64x128 = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma64x256 = Fp8A8TmaMmaSchedule<64, 256, 128, 2, 4, 2, 1>;
using Bulk      = Fp8A8SplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;
} // namespace

// Band boundaries, named once for this file. The ladder below selects on them and
// fp8_*_partial_capacity_bytes sizes on them, so a band cannot be moved in one place and not
// the other. The capacity is a worst case over the bands: whether a split engages depends on
// the token count, so a buffer sized for one selection can still be read by another.
constexpr std::int32_t kBandT64x128Max    = 128;
constexpr std::int32_t kBandBulkSelectsAt = 192;

std::size_t fp8_linear_swiglu_partial_capacity_bytes(std::int32_t max_tokens) {
    // Ungated with the ladder below, so this threshold now applies on every platform and the #ifdef that
    // suppressed it while the gate was in place is gone -- exactly as intended.
    //
    // Selection band, not engagement band: fp8_tma_split_k_plan decides at run time whether the tail
    // wave splits, but the ladder decides whether a split schedule is SELECTED, and the buffer has to
    // exist across the whole selection band. The old threshold was 256, which left 193-256 selecting
    // Bulk with a null partials and the launcher throwing "FP8 split-K requires aligned caller
    // partials". Pre-existing upstream; the Windows gate had been masking it.
    return max_tokens > kBandBulkSelectsAt ? Bulk::kPartialBytes : 0;
}

void fp8_linear_swiglu_a8_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                 WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto scratch = allocate_fp8_a8_workspace(
        workspace, x.ne[1], weight.k, fp8_linear_swiglu_partial_capacity_bytes(x.ne[1]));
    launch_fp8_a8_quantize(x, weight, scratch, stream);
    const auto operands = fp8_a8_operands(weight, scratch, x.ne[1]);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2};
    const auto launch = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, 5120>;
        if constexpr (S::kTmaSwizzle)
            launch_fp8_a8_tma_mma<S>(operands, output, SwiGluTokenMajorMmaEpilogue{}, stream,
                                     scratch.partials, SwiGluTokenMajorMmaRows<S>{});
        else
            launch_fp8_a8_mma<S>(operands, output, SwiGluTokenMajorMmaEpilogue{}, stream,
                                 SwiGluTokenMajorMmaRows<S>{}, scratch.partials);
    };
    // UNGATED 2026-10-02 (docs/active-work.md item 12). This was PORT-DISPATCH-gated on Windows because
    // upstream's TMA route faulted here with cudaErrorIllegalInstruction. That fault was a descriptor
    // ABI defect -- alignas(128) on a by-value kernel parameter, which MSVC cannot lay out -- and it is
    // fixed at alignas(64), which is the width cuda.h asks for. tools/scripts/verify_fp8_tma_route.cmd
    // reproduces the route end to end, and PyGPUkit #107 reports the same misalignment defect on this
    // same GPU and OS.
    //
    // Ungating is per ROUTE and on three pieces of evidence, not on the transport being faster:
    //   - all three tiles this ladder selects are oracle-checked on both arms, at the tokens they serve
    //     (64x128 Stages=2, 64x256, and the 128x256 split Bulk);
    //   - TMA leads on the median at all three -- 1.107, 1.270 and 1.270 -- and the only bands where MMA
    //     wins on the 64x128 tile are at large token counts, outside this ladder's split-tile region;
    //   - scratch.partials is allocated for the whole selection band, so no rung can select a split tile
    //     with a null buffer.
    if (x.ne[1] <= 16) return launch.template operator()<Fp8A8T16R64K128>();
    if (x.ne[1] <= 32) return launch.template operator()<Fp8A8T32R128K128>();
    if (x.ne[1] <= 64) return launch.template operator()<Fp8A8T64R128K256>();
    if (x.ne[1] <= kBandT64x128Max) return launch.template operator()<Tma64x128>();
    if (x.ne[1] <= kBandBulkSelectsAt) return launch.template operator()<Tma64x256>();
    launch.template operator()<Bulk>();
}
} // namespace ninfer::ops::detail
