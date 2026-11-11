#pragma once

#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/tma_param.cuh"
#include "ops/common/host/launch.h"
#include "ops/common/numeric/math.h"

#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/kernels/fp8/fp8_a8_mma_common.cuh"
#include "ops/linear/kernels/fp8/fp8_operands.h"
#include "ops/linear/kernels/fp8/fp8_split_k.cuh"

#include <cuda.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops::detail {

struct alignas(128) Fp8TmaDescriptors {
    CUtensorMap activation;
    CUtensorMap weight;
};

static_assert(alignof(Fp8TmaDescriptors) == 128);
static_assert(sizeof(Fp8TmaDescriptors) == 2 * sizeof(CUtensorMap),
              "two descriptors back to back; the fp8_tma_acquire_map size argument assumes this");

inline CUtensorMap fp8_tma_map(const std::uint8_t* pointer, int rows, int k, int block_rows,
                               int block_k) {
    CUtensorMap result{};
    const std::uint64_t dimensions[]{static_cast<std::uint64_t>(k),
                                     static_cast<std::uint64_t>(rows)};
    const std::uint64_t strides[]{static_cast<std::uint64_t>(k)};
    const std::uint32_t box[]{static_cast<std::uint32_t>(block_k),
                              static_cast<std::uint32_t>(block_rows)};
    const std::uint32_t steps[]{1, 1};
    const auto swizzle = block_k == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B;
    // Copy the represented E4M3 bytes; the row scales remain explicit FP32 epilogue operands.
    const auto status = cuTensorMapEncodeTiled(
        &result, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<std::uint8_t*>(pointer), dimensions,
        strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("FP8 TMA descriptor: ") +
                                 (name ? name : "CUDA error"));
    }
    return result;
}

template <class Schedule, class Epilogue>
inline constexpr int fp8_tma_scratch_bytes = [] {
    constexpr int epilogue_bytes = [] {
        if constexpr (requires { Epilogue::template kSharedBytes<Schedule>; })
            return Epilogue::template kSharedBytes<Schedule>;
        else
            return 0;
    }();
    constexpr int bytes =
        Schedule::kStorageBytes > epilogue_bytes ? Schedule::kStorageBytes : epilogue_bytes;
    return (bytes + 127) / 128 * 128;
}();

template <class Schedule, bool FullTokens, class Output, class Epilogue, bool SplitK = false,
          class RowPolicy = Fp8IdentityRows>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_a8_tma_mma_kernel(
    const __grid_constant__ common::TmaKernelParam<Fp8TmaDescriptors> descriptor_param,
    Fp8A8Operands operands, Output output, Epilogue epilogue, RowPolicy row_policy, int token_offset,
    int count, Fp8SplitKPlan plan, float* partials) {
    const auto& descriptors = descriptor_param.get();
    constexpr int BT = Schedule::kBlockTokens, BR = Schedule::kBlockRows;
    constexpr int BK = Schedule::kBlockK, S = Schedule::kStages;
    const int k = Schedule::kStaticK ? Schedule::kStaticK : operands.k;
    int tile = blockIdx.x, partial = -1, k_begin = 0, tiles_k = k / BK;
    if constexpr (SplitK) {
        tile = fp8_split_k_range<Schedule>(plan, tile, partial, k_begin, tiles_k);
    }
    const int row_tiles = operands.rows / BR, token_tiles = common::div_up(count, BT);
    int row_tile, token_tile;
    fp8_mma_tile_coordinates<Schedule>(tile, row_tiles, token_tiles, row_tile, token_tile);
    const int row_begin   = row_tile * (BR / (RowPolicy::kPaired ? 2 : 1));
    const int token_begin = token_offset + token_tile * BT;

    extern __shared__ __align__(128) unsigned char fp8_tma_shared[];
    auto* activation = fp8_tma_shared;
    auto* weight     = activation + S * BT * BK;
    auto* full       = reinterpret_cast<std::uint64_t*>(fp8_tma_shared +
                                                        fp8_tma_scratch_bytes<Schedule, Epilogue>);
    auto* empty      = full + S;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < S; ++stage) {
            common::cta_mbarrier_init(full + stage, 1);
            common::cta_mbarrier_init(empty + stage, Schedule::kConsumerWarps);
        }
        common::cta_mbarrier_fence_init();
    }
    __syncthreads();

    if (threadIdx.x < Schedule::kProducerThreads) {
        if (threadIdx.x == 0) {
            for (int kt = 0; kt < tiles_k; ++kt) {
                const int stage = kt % S;
                common::cta_mbarrier_wait(empty + stage, 1U ^ ((kt / S) & 1U));
                common::cta_mbarrier_arrive_expect_tx(full + stage, (BT + BR) * BK);
                common::tma_load_2d(activation + stage * BT * BK, &descriptors.activation,
                                    (k_begin + kt) * BK, token_begin, full + stage);
                if constexpr (RowPolicy::kPaired) {
                    // Each consumer warp owns both gate/up fragments. Load their contiguous
                    // weight spans into that warp's logical shared rows without repacking.
                    constexpr int span = Schedule::kWarpRows / 2;
#pragma unroll
                    for (int local = 0; local < BR; local += span) {
                        common::tma_load_2d(weight + (stage * BR + local) * BK, &descriptors.weight,
                                            (k_begin + kt) * BK,
                                            row_policy.weight_row(row_begin, local, operands.rows),
                                            full + stage);
                    }
                } else {
                    common::tma_load_2d(weight + stage * BR * BK, &descriptors.weight,
                                        (k_begin + kt) * BK, row_begin, full + stage);
                }
            }
        }
        return;
    }

    const int tid  = static_cast<int>(threadIdx.x) - Schedule::kProducerThreads;
    const int warp = tid >> 5, lane = tid & 31;
    float accumulators[Schedule::kMmaTokens][Schedule::kMmaRows][4] = {};
    for (int kt = 0; kt < tiles_k; ++kt) {
        const int stage = kt % S;
        common::cta_mbarrier_wait(full + stage, (kt / S) & 1U);
        fp8_mma_compute_stage<Schedule>(activation + stage * BT * BK, weight + stage * BR * BK,
                                        accumulators, warp, lane);
        // Release only after every lane in this consumer warp has finished its shared reads.
        __syncwarp();
        if (lane == 0) common::cta_mbarrier_arrive(empty + stage);
    }
    if constexpr (SplitK) {
        if (partial >= 0) {
            fp8_store_split_partials<Schedule>(partials, partial, accumulators, warp, lane);
            return;
        }
    }
    // All consumers must finish reading staged inputs before the epilogue reuses the storage.
    __syncthreads();
    fp8_finish_mma_tile<Schedule, FullTokens>(
        output, epilogue, row_policy, fp8_tma_shared, accumulators, operands.x_scales,
        operands.scales, row_begin, token_begin, operands.rows, token_offset + count, warp, lane);
}

// The split-K plan, partial store and reduce live in fp8_split_k.cuh: none of them reads a tensor
// map, so the split-K route is shared with the non-TMA MMA transport rather than owned by TMA.

template <class Schedule, class Output, class Epilogue, class RowPolicy = Fp8IdentityRows>
void launch_fp8_a8_tma_mma(const Fp8A8Operands& p, Output output, Epilogue epilogue,
                           cudaStream_t stream, float* partials = nullptr,
                           RowPolicy row_policy = {}) {
    validate_fp8_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK)
        throw std::invalid_argument("FP8 TMA requires complete row/K tiles");
    if constexpr (RowPolicy::kPaired) {
        static_assert(RowPolicy::kWarpPaired && RowPolicy::kContiguousPairs);
        static_assert(Schedule::kWarpRows % 16 == 0);
    }
    constexpr int weight_span = RowPolicy::kPaired ? Schedule::kWarpRows / 2 : Schedule::kBlockRows;
    // Descriptors are launch-owned values, copied into kernel parameters during Graph capture.
    const Fp8TmaDescriptors descriptors{
        fp8_tma_map(p.x, p.tokens, p.k, Schedule::kBlockTokens, Schedule::kBlockK),
        fp8_tma_map(p.codes, p.rows, p.k, weight_span, Schedule::kBlockK)};
    common::for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const int blocks =
            p.rows / Schedule::kBlockRows * common::div_up(count, Schedule::kBlockTokens);
        const auto plan   = fp8_split_k_plan<Schedule>(blocks, p.k);
        const auto launch = [&]<bool Full, bool Split>() {
            constexpr auto kernel =
                fp8_a8_tma_mma_kernel<Schedule, Full, Output, Epilogue, Split, RowPolicy>;
            constexpr int bytes =
                fp8_tma_scratch_bytes<Schedule, Epilogue> + Schedule::kBarrierBytes;
            const int dynamic = common::prepare_shared<bytes, kernel, true>();
            const int grid    = Split ? plan.full_tiles + plan.split_ctas : blocks;
            kernel<<<grid, Schedule::kThreads, dynamic, stream>>>(
                common::make_tma_kernel_param(descriptors), p, output, epilogue, row_policy, offset,
                count, plan, partials);
            CUDA_CHECK(cudaGetLastError());
            if constexpr (Split) {
                fp8_a8_split_k_reduce<Schedule>
                    <<<plan.tail_tiles * Schedule::kReductionBlocks, 256, 0, stream>>>(
                        p, partials, output, epilogue, row_policy, plan, offset, count);
                CUDA_CHECK(cudaGetLastError());
            }
        };
        if constexpr (Schedule::kSplitWaveCtas > 0) {
            static_assert(
                (!RowPolicy::kPaired && requires { epilogue.apply(0, 0, 0.0f); }) ||
                    (RowPolicy::kPaired && requires { epilogue.apply_pair(0, 0, 0.0f, 0.0f); }),
                "FP8 TMA split-K requires a scalar or paired epilogue after reduction");
            if (plan.split_ctas) {
                if (!partials || reinterpret_cast<std::uintptr_t>(partials) % 16)
                    throw std::invalid_argument("FP8 TMA split-K requires aligned caller partials");
                if (count % Schedule::kBlockTokens == 0)
                    launch.template operator()<true, true>();
                else
                    launch.template operator()<false, true>();
                return;
            }
        }
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true, false>();
        else
            launch.template operator()<false, false>();
    });
}

} // namespace ninfer::ops::detail
