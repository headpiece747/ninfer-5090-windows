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

std::size_t fp8_linear_swiglu_partial_capacity_bytes(std::int32_t max_tokens) {
    return max_tokens > 256 ? Bulk::kPartialBytes : 0;
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
                                 SwiGluTokenMajorMmaRows<S>{});
    };
    // PORT-DISPATCH: pre-merge FP8 A8 dispatch on Windows (docs/active-work.md item 12)
    //
    // Upstream routed every FP8 A8 path to its TMA kernel, which faults on this target with
    // cudaErrorIllegalInstruction. A TMA schedule and an MMA schedule are different tile shapes, so
    // the schedule selection itself has to differ rather than being forwarded at one seam. This is
    // the pre-merge dispatch verbatim, because that is the one that passed the suite; upstream's
    // stays selected on other platforms and stays in the tree.
#ifdef _WIN32
    if (x.ne[1] <= 16) return launch.template operator()<Fp8A8T16R64K128>();
    if (x.ne[1] <= 32) return launch.template operator()<Fp8A8T32R128K128>();
    if (x.ne[1] <= 64) return launch.template operator()<Fp8A8T64R128K128>();
    if (x.ne[1] <= 96) return launch.template operator()<Fp8A8T32R128K128>();
    launch.template operator()<Fp8A8T64R128K128>();
#else
    if (x.ne[1] <= 16) return launch.template operator()<Fp8A8T16R64K128>();
    if (x.ne[1] <= 32) return launch.template operator()<Fp8A8T32R128K128>();
    if (x.ne[1] <= 64) return launch.template operator()<Fp8A8T64R128K256>();
    if (x.ne[1] <= 128) return launch.template operator()<Tma64x128>();
    if (x.ne[1] <= 192) return launch.template operator()<Tma64x256>();
    launch.template operator()<Bulk>();
#endif
}
} // namespace ninfer::ops::detail
