#include "ops/linear_add/fp8/fp8_linear_add_plan.h"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"

#include <cstdlib>
#include <string_view>

namespace ninfer::ops::detail {
namespace {
// This Op admits dense, even-width BF16 residuals. Adjacent MMA rows share one load.
struct Fp8ResidualAddEpilogue : LinearResidualAddEpilogue {
    __device__ __forceinline__ float2 apply_row_pair(int row, int token, float2 value) const {
        const auto* pointer =
            residual.data + static_cast<std::int64_t>(token) * residual.leading_dim + row;
        const float2 add = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(pointer));
        return make_float2(value.x + add.x, value.y + add.y);
    }
};

using K6144Tma32x64  = Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>;
using K6144Tma64x128 = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>;

// MMA twins of the two non-split-K TMA tiles above, for the transport A/B in
// docs/active-work.md item 12.
//
// Fp8A8TmaMmaSchedule derives from Fp8A8MmaSchedule and overrides only kTmaSwizzle, the producer
// warp and the cache hint, so the twin is the SAME instantiation of the base template: same
// BlockTokens, BlockRows, BlockK, WarpsTokens, WarpsRows, Stages and MinBlocksPerSm. Only the load
// path differs. That is the controlled comparison the earlier ~10x figure was not -- that figure came
// from a bench that ran ONE non-TMA schedule against FOUR TMA schedules with different tiles, so it
// confounded transport with schedule and could not attribute anything to TMA.
//
// The split-K tiles (K6144MidBulk, K6144Bulk) ARE twinned, and that needed split-K on the MMA side
// first: Fp8A8SplitKSchedule never depended on a tensor map, so the wrapper and the reduction work
// unchanged over an MMA base (fp8_split_k.cuh). Those two tiles serve tokens 193-768 and beyond --
// most real prefill traffic -- so leaving them untested left the gate question open exactly where it
// matters most. The split geometry is decoded by one shared helper (fp8_split_k_range), so the two
// arms cannot drift apart in how they split K.
//
// Both arms are instantiated in one binary so a measurement can interleave them; see the
// NINFER_FP8_TMA_ARM selector in launch_problem. This is a measurement hook, not a product option:
// remove it once item 12's question is answered.
using K6144Mma32x64  = Fp8A8MmaSchedule<32, 64, 128, 1, 2, 3, 2, Cache::cg, Cache::cg,
                                               Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast, 1>;
using K6144Mma64x128 = Fp8A8MmaSchedule<64, 128, 128, 2, 4, 3, 1, Cache::cg, Cache::cg,
                                               Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast, 1>;
using K6144MidBulk =
    Fp8A8SplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, 170, 4, 8>;
using K6144Bulk = Fp8A8SplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;
using K6144MidBulkMma =
    Fp8A8SplitKSchedule<Fp8A8MmaSchedule<128, 128, 128, 2, 4, 3, 1, Cache::cg, Cache::cg,
                                          Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast, 1>,
                         170, 4, 8>;
using K6144BulkMma =
    Fp8A8SplitKSchedule<Fp8A8MmaSchedule<128, 256, 128, 2, 4, 2, 1, Cache::cg, Cache::cg,
                                          Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast, 1>,
                         170, 4, 8>;

// Twins of the three tiles the OTHER gated ladders use and nothing has measured: 64x256 and 96x256 in
// linear_swiglu / attn_input_proj / the shape instantiations, and 192x128 in gdn_input_proj. codegraph
// showed the gates do not share one ladder -- these are the shapes the K=6144 result cannot reach by
// construction. Selected by NINFER_FP8_TMA_TILE, which overrides the ladder for both arms so each is
// measured at the same tile rather than at whatever the ladder would have picked.
using K6144Wide64x256 = Fp8A8TmaMmaSchedule<64, 256, 128, 2, 4, 2, 1>;
using K6144Wide64x256Mma = Fp8A8MmaSchedule<64, 256, 128, 2, 4, 2, 1, Cache::cg, Cache::cg,
                                            Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast, 1>;
using K6144Wide96x256 = Fp8A8TmaMmaSchedule<96, 256, 128, 3, 4, 2, 1>;
using K6144Wide96x256Mma = Fp8A8MmaSchedule<96, 256, 128, 3, 4, 2, 1, Cache::cg, Cache::cg,
                                            Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast, 1>;
using K6144Tall192x128 = Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>;
using K6144Tall192x128Mma = Fp8A8MmaSchedule<192, 128, 128, 3, 4, 2, 1, Cache::cg, Cache::cg,
                                             Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast, 1>;

// The two Stages=2 variants. 64x128x128 is the highest-reuse tile in the gated set -- six files select
// it -- and the K=6144 ladder measured Stages=3, which is a different schedule, so its result could not
// transfer. 128x128x128 split Stages=2 is gdn_input_proj's MidBulk and n16384's.
using K6144Mid64x128S2 = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using K6144Mid64x128S2Mma = Fp8A8MmaSchedule<64, 128, 128, 2, 4, 2, 1, Cache::cg, Cache::cg,
                                              Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast, 1>;
using K6144MidBulkS2 = Fp8A8SplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>, 170, 4, 8>;
using K6144MidBulkS2Mma =
    Fp8A8SplitKSchedule<Fp8A8MmaSchedule<128, 128, 128, 2, 4, 2, 1, Cache::cg, Cache::cg,
                                          Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast, 1>,
                         170, 4, 8>;

using K17408Tma32x64 = Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>;
using K17408Small =
    Fp8A8SplitKSchedule<Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>, 170, 4, 8>;
using K17408Mid = Fp8A8SplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, 170, 4, 8>;
using K17408Wide =
    Fp8A8SplitKSchedule<Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>, 170, 4, 8>;
using K17408Bulk =
    Fp8A8SplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

template <int K>
void launch_problem(const Tensor& x, const Weight& weight, Tensor& residual,
                    Fp8A8Workspace workspace, cudaStream_t stream) {
    auto* data = static_cast<__nv_bfloat16*>(residual.data);
    const LinearBf16Output output{data, weight.n};
    const Fp8ResidualAddEpilogue epilogue{{{data, weight.n}}};
    const auto operands = fp8_a8_operands(weight, workspace, x.ne[1]);
    const auto launch   = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, K>;
        if constexpr (S::kTmaSwizzle)
            launch_fp8_a8_tma_mma<S>(operands, output, epilogue, stream, workspace.partials);
        else
            launch_fp8_a8_mma<S>(operands, output, LinearResidualAddEpilogue{{data, weight.n}},
                                   stream, Fp8IdentityRows{}, workspace.partials);
    };
    // Transport A/B for docs/active-work.md item 12. Selected by NINFER_FP8_TMA_ARM=tma|mma at K=6144
    // only, where both a TMA schedule and its MMA twin exist. Unset, the platform dispatch below runs
    // completely unchanged, so this cannot affect a shipped configuration.
    //
    // A runtime branch over two compile-time instantiations on purpose: interleaving the arms needs
    // both in ONE binary, because this card's clocks drift enough between windows that measuring A
    // and then B would measure the window rather than the kernel.
    if constexpr (K == 6144) {
        // NINFER_FP8_TMA_TILE pins ONE tile for both arms, overriding the ladder, so a tile the
        // Windows route never selects can still be measured at a matched tile. Without this the only
        // measurable tiles are the ones the ladder happens to choose, and codegraph showed the other
        // gates use 64x256, 96x256 and 192x128 -- none of which this ladder contains.
        if (const char* tile = std::getenv("NINFER_FP8_TMA_TILE")) {
            const std::string_view want(tile);
            const char* arm_env   = std::getenv("NINFER_FP8_TMA_ARM");
            const bool mma = arm_env && std::string_view(arm_env) == "mma";
            if (want == "wide64")
                return mma ? launch.template operator()<K6144Wide64x256Mma>()
                           : launch.template operator()<K6144Wide64x256>();
            if (want == "wide96")
                return mma ? launch.template operator()<K6144Wide96x256Mma>()
                           : launch.template operator()<K6144Wide96x256>();
            if (want == "tall192")
                return mma ? launch.template operator()<K6144Tall192x128Mma>()
                           : launch.template operator()<K6144Tall192x128>();
            if (want == "mid128")
                return mma ? launch.template operator()<K6144Mid64x128S2Mma>()
                           : launch.template operator()<K6144Mid64x128S2>();
            if (want == "midbulk128")
                return mma ? launch.template operator()<K6144MidBulkS2Mma>()
                           : launch.template operator()<K6144MidBulkS2>();
        }
        if (const char* arm = std::getenv("NINFER_FP8_TMA_ARM")) {
            // Each arm runs the schedule ladder the OTHER transport would run, tile for tile, so the
            // only difference between the arms is cp.async.bulk.tensor versus cp.async. Mirrors the
            // #else dispatch below one-for-one. Tokens 65..128 are the CONTROL: both arms land on
            // Fp8A8T64R64K128 and must agree.
            if (std::string_view(arm) == "mma") {
                if (x.ne[1] <= 64) return launch.template operator()<K6144Mma32x64>();
                if (x.ne[1] <= 128) return launch.template operator()<Fp8A8T64R64K128>();
                if (x.ne[1] <= 192) return launch.template operator()<K6144Mma64x128>();
                if (x.ne[1] <= 768) return launch.template operator()<K6144MidBulkMma>();
                return launch.template operator()<K6144BulkMma>();
            }
            if (x.ne[1] <= 64) return launch.template operator()<K6144Tma32x64>();
            if (x.ne[1] <= 128) return launch.template operator()<Fp8A8T64R64K128>();
            if (x.ne[1] <= 192) return launch.template operator()<K6144Tma64x128>();
            if (x.ne[1] <= 768) return launch.template operator()<K6144MidBulk>();
            return launch.template operator()<K6144Bulk>();
        }
    }
    // PORT-DISPATCH: pre-merge FP8 A8 dispatch on Windows (docs/active-work.md item 12)
    //
    // Upstream routed every FP8 A8 path to its TMA kernel, which faults on this target with
    // cudaErrorIllegalInstruction. A TMA schedule and an MMA schedule are different tile shapes, so
    // the schedule selection itself has to differ rather than being forwarded at one seam. These are
    // the pre-merge dispatches verbatim, because those are the ones that passed the suite;
    // upstream's stay selected on other platforms and stay in the tree.
#ifdef _WIN32
    if constexpr (K == 6144) {
        if (x.ne[1] <= 64) return launch.template operator()<Fp8A8T32R32K128>();
        if (x.ne[1] <= 128) return launch.template operator()<Fp8A8T64R64K128>();
        launch.template operator()<Fp8A8T64R128K128>();
    } else {
        if (x.ne[1] <= 64) return launch.template operator()<Fp8A8T32R32K128>();
        if (x.ne[1] <= 128) return launch.template operator()<Fp8A8T64R64K128>();
        launch.template operator()<Fp8A8T64R128K128>();
    }
#else
    if constexpr (K == 6144) {
        if (x.ne[1] <= 64) return launch.template operator()<K6144Tma32x64>();
        if (x.ne[1] <= 128) return launch.template operator()<Fp8A8T64R64K128>();
        if (x.ne[1] <= 192) return launch.template operator()<K6144Tma64x128>();
        // The narrower row tile fills the GPU before the large-tile path reaches a full wave.
        if (x.ne[1] <= 768) return launch.template operator()<K6144MidBulk>();
        launch.template operator()<K6144Bulk>();
    } else {
        const int tokens = x.ne[1];
        if (tokens <= 64) return launch.template operator()<K17408Tma32x64>();
        if (tokens <= 128) return launch.template operator()<K17408Small>();
        if (tokens <= 256) return launch.template operator()<K17408Mid>();
        // Wider token tiles avoid an extra wave in the gaps between the bulk anchors.
        if (tokens <= 384 || (tokens > 512 && tokens <= 768))
            return launch.template operator()<K17408Wide>();
        launch.template operator()<K17408Bulk>();
    }
#endif
}
} // namespace

std::size_t fp8_linear_add_partial_capacity_bytes(std::int32_t k, std::int32_t max_tokens) {
    if (k == 6144) {
        if (max_tokens > 768) return K6144Bulk::kPartialBytes;
        return max_tokens > 192 ? K6144MidBulk::kPartialBytes : 0;
    }
    if (max_tokens > 384) return K17408Bulk::kPartialBytes;
    if (max_tokens > 256) return K17408Wide::kPartialBytes;
    if (max_tokens > 128) return K17408Mid::kPartialBytes;
    return max_tokens > 64 ? K17408Small::kPartialBytes : 0;
}

void fp8_linear_add_a8_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                              WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto scratch = allocate_fp8_a8_workspace(
        workspace, x.ne[1], weight.k, fp8_linear_add_partial_capacity_bytes(weight.k, x.ne[1]));
    launch_fp8_a8_quantize(x, weight, scratch, stream);
    if (weight.k == 6144)
        launch_problem<6144>(x, weight, residual, scratch, stream);
    else
        launch_problem<17408>(x, weight, residual, scratch, stream);
}
} // namespace ninfer::ops::detail
