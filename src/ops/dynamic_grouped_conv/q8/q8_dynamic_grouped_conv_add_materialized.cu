#include "core/weight.h"
#include "ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_kernels.h"
#include "core/device.h"
#include "ops/dynamic_grouped_conv/dynamic_grouped_conv_add_finish.h"
#include "ops/linear/q8/q8_ksplit_config.h"
#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/q8/q8_rowsplit_output.cuh"
#include "ops/linear/q8/q8_ksplit_mma.cuh"
#include <cuda_bf16.h>
#include <array>
#include <algorithm>
#include <utility>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kRows = 5120;

using Launch = Q8Launch;

template <int InputRows, int TileColumns>
void tiled_projection(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int Warps =
        InputRows == 4096 ? (TileColumns <= 40 ? 8 : 4) : (TileColumns <= 32 ? 8 : 4);
    constexpr Cache Activation =
        InputRows == 4096 && ((TileColumns > 24 && TileColumns <= 40) || TileColumns > 48)
            ? Cache::cg
            : Cache::ca;
    using Geometry            = Q8LinearGeometry<kRows, InputRows>;
    using Schedule            = Q8KSplitSchedule<Warps, TileColumns, Warps == 8 ? 2 : 3,
                                                 Q8KSplitScaleAccess::Shared, Activation>;
    constexpr int SharedBytes = TileColumns > 64 ? sizeof(Q8KSplitSharedStorage<Schedule>) : 0;
    if constexpr (SharedBytes > 0) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            q8_ksplit_mma_kernel<Geometry, TileColumns, Schedule, Q8ContiguousOutput,
                                 Q8KSplitStoreEpilogue, Q8KSplitIdentityRows, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes);
        CUDA_CHECK(attribute);
    }
    const int columns = x.ne[1];
    Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), kRows};
    const dim3 grid(kRows / 16, (columns + TileColumns - 1) / TileColumns);
    q8_ksplit_mma_kernel<Geometry, TileColumns, Schedule, Q8ContiguousOutput, Q8KSplitStoreEpilogue,
                         Q8KSplitIdentityRows, false, true>
        <<<grid, Schedule::kThreads, SharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, Q8KSplitStoreEpilogue{},
            Q8KSplitIdentityRows{}, columns);
    CUDA_CHECK(cudaGetLastError());
}

// Live columns stay dynamic; only the eight-column MMA accumulator layout is specialized.
template <int C, std::size_t... I>
constexpr auto make_launchers(std::index_sequence<I...>) {
    return std::array<Launch, sizeof...(I)>{&tiled_projection<C, 8 * (1 + static_cast<int>(I))>...};
}

constexpr auto attention = make_launchers<4096>(std::make_index_sequence<11>{});
constexpr auto mlp       = make_launchers<17408>(std::make_index_sequence<11>{});

__global__ void finish_kernel(const __nv_bfloat16* projected, const __nv_bfloat16* base,
                              const __nv_bfloat16* delta, __nv_bfloat16* residual, int width) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y;
    if (row >= kRows) return;
    const int index = col * kRows + row;
    finish_value(row, col, width, __bfloat162float(projected[index]),
                 col % width ? __bfloat162float(projected[index - kRows]) : 0.0f, base, delta,
                 residual);
}

void materialized(Q8DynamicConvAddSchedule schedule, const Tensor& x, const Weight& weight,
                  const Tensor& base, const Tensor& delta, Tensor& residual, Tensor& projected,
                  cudaStream_t stream) {
    const int tokens  = x.ne[1] * x.ne[2];
    const Tensor flat = x.view({x.ne[0], tokens});
    Tensor result     = projected.view({kRows, tokens});
    switch (schedule) {
    case Q8DynamicConvAddSchedule::TiledMma: {
        const auto& launchers = x.ne[0] == 4096 ? attention : mlp;
        launchers[(tokens - 1) / 8](flat, weight, result, stream);
        break;
    }
    case Q8DynamicConvAddSchedule::MmaK128:
        launch_q8_mma_r64x32_c64_k128_a1(flat, weight, result, stream);
        break;
    }
    dynamic_grouped_conv_add_finish_launch(projected, base, delta, residual, stream);
}
} // namespace

void q8_dynamic_grouped_conv_add_materialized_launch(Q8DynamicConvAddSchedule schedule,
                                                     const Tensor& x, const Weight& weight,
                                                     const Tensor& base, const Tensor& delta,
                                                     Tensor& residual, Tensor& projected,
                                                     cudaStream_t stream) {
    materialized(schedule, x, weight, base, delta, residual, projected, stream);
}
} // namespace ninfer::ops::detail
