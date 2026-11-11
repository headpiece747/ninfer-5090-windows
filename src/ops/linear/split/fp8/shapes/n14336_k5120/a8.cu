#include "ops/linear/split/fp8/shapes/n14336_k5120/launch.h"
#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"
#include "ops/linear/kernels/fp8/fp8_instances.cuh"

namespace ninfer::ops::detail::linear_split::fp8_n14336_k5120 {

namespace {
using Tma64x128 = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma64x256 = Fp8A8TmaMmaSchedule<64, 256, 128, 2, 4, 2, 1>;
using Tma96x256 = Fp8A8TmaMmaSchedule<96, 256, 128, 3, 4, 2, 1>;
using Bulk      = Fp8A8SplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

// Band boundaries, named once for this file: the ladder below selects on them and
// fp8_*_partial_capacity_bytes sizes on them, so neither can move alone. The capacity is a worst case
// over the bands, since whether a split engages depends on the token count.
constexpr std::int32_t kBandT64x128Max    = 128;
constexpr std::int32_t kBandT64x256Max    = 192;
constexpr std::int32_t kBandBulkSelectsAt = 288;
} // namespace

std::size_t partial_capacity(std::int32_t max_tokens) {
    // Selection band, not engagement band: fp8_tma_split_k_plan decides at run time whether the tail
    // wave splits, but the ladder decides whether a split schedule is SELECTED. The upstream threshold
    // of 384 left 289-384 selecting Bulk with a null partials. This route's test already exercises 289
    // and 385, the two boundaries of that band.
    return max_tokens > kBandBulkSelectsAt ? Bulk::kPartialBytes : 0;
}

void launch_a8(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
               Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const LinearBf16SegmentedOutput<6144, 1024, 6144, 1024> output{
        static_cast<__nv_bfloat16*>(outputs[0].data), static_cast<__nv_bfloat16*>(outputs[1].data),
        static_cast<__nv_bfloat16*>(outputs[2].data), static_cast<__nv_bfloat16*>(outputs[3].data)};
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
    // UNGATED 2026-10-02 (measured: docs/maintainer/fp8-ungated-bands.md). Second of the nine, ungated on the same three
    // grounds as linear_swiglu -- the cudaErrorIllegalInstruction that justified the gate was a
    // descriptor ABI defect (alignas(128) on a by-value kernel parameter rejected by MSVC C2719),
    // now resolved via common::TmaKernelParam<Fp8TmaDescriptors> preserving .align 128 in PTX -- and this
    // route's coverage is the best of the nine:
    //   - all four tiles are oracle-checked on both arms: 64x128 Stages=2, 64x256, 96x256, and the
    //     128x256 split Bulk. TMA leads the median on each (1.107, 1.270, 1.288, and 1.270 on the
    //     linear_add Bulk measurement of the same tile). Workload, metric and harness:
    //     docs/maintainer/fp8-ungated-bands.md.
    //   - run_fp8_target() already drives tokens 1..128 and then {129, 191, 192, 193, 256, 257, 287,
    //     288, 289, 383, 384, 385, 512, 513, 1024, 1025}, which reaches EVERY rung including Bulk and
    //     the 289 boundary where the partials buffer starts.
    //   - and it drives the same list again through CUDA Graph replay, which matters here specifically:
    //     the TMA descriptors are launch-owned values copied into kernel parameters during capture, so
    //     capture correctness is a separate question from eager correctness, and this route answers it.
    if (x.ne[1] <= 32) return launch.template operator()<Fp8A8T32R32K128>();
    if (x.ne[1] <= 96) return launch.template operator()<Fp8A8T32R128K128>();
    if (x.ne[1] <= kBandT64x128Max) return launch.template operator()<Tma64x128>();
    if (x.ne[1] <= kBandT64x256Max) return launch.template operator()<Tma64x256>();
    // Three 96-token tiles give 168 CTAs: one almost-full wave through T=288.
    if (x.ne[1] <= kBandBulkSelectsAt) return launch.template operator()<Tma96x256>();
    launch.template operator()<Bulk>();
}

} // namespace ninfer::ops::detail::linear_split::fp8_n14336_k5120
