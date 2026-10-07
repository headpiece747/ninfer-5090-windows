#pragma once

#include "ops/common/mbarrier.cuh"
#include "ops/linear/bf16/bf16_mma_common.cuh"
#include "ops/linear/bf16/bf16_operands.h"
#include "ops/common/token_slices.h"
#include "ops/common/math.h"
#include <cuda.h>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

// No alignas override, and the reason is NOT the one an earlier revision of this comment gave. It
// used to say CUtensorMap "already declares alignas(TENSOR_MAP_ALIGN), which is 64 under MSVC and 128
// elsewhere, so inheriting it leaves a by-value kernel parameter MSVC can lay out". The macro is 64.
// The attribute is not applied at all. cuda.h:3749-3753 selects TENSOR_MAP_ALIGN = 64 under _MSC_VER,
// but cuda.h:3756 applies it only inside `#if defined(__cplusplus) && (__cplusplus >= 201103L)`, and
// nvcc's host pass on MSVC reports __cplusplus = 199711L unless /Zc:__cplusplus is passed. This build
// does not pass it, so CUtensorMap carries no alignment attribute and inherits alignof == 8 from its
// `cuuint64_t[16]` payload. It is the ABSENCE of the attribute, not an inherited 64, that leaves the
// by-value parameter below layable.
//
// Measured on this toolchain (CUDA 13.3, MSVC) by tools/scripts/probe_tma_align.cmd:
//   alignof(CUtensorMap) = 8, TENSOR_MAP_ALIGN = 64, __cplusplus = 199711, attribute applied: NO
//   with /Zc:__cplusplus: alignof = 64, __cplusplus = 202002, attribute applied: yes
// So this is a property of the build flags, not of the type. Re-run that tool after touching the CUDA
// headers, the MSVC toolchain, or the C++ flags in CMakeLists.txt; these comments quote its numbers.
//
// Overriding to alignas(128) produced C2719 ("formal parameter with requested alignment of 128 won't
// be aligned") in nvcc's cudafe1 host stub on Windows. The same sweep shows a by-value
// __grid_constant__ descriptor block accepted at 8/16/32/64 and rejected at 128 and 256 with four
// C2719 sites each, so 128 is past what MSVC's ABI will honour for a by-value parameter -- which is
// why the header picks 64 and why this struct declares nothing.
//
// If /Zc:__cplusplus is ever added to this build, the struct below inherits alignas(64) and the
// reasoning above has to be re-checked rather than assumed to carry over. That combination is not
// built today, so it is not measured here.
//
// This is the port's fix from 1218d574, which also makes the descriptor safe under CUDA Graph capture
// -- the bytes travel in the kernel node, so a replay sees them again, where a device buffer filled by
// cudaMemcpyAsync would read a caller stack frame that is gone.
struct Bf16TmaDescriptors {
    CUtensorMap weight;
    CUtensorMap activation;
};

template <class Schedule>
inline CUtensorMap bf16_tma_map(const __nv_bfloat16* pointer, int rows, int k, int block_rows,
                                int block_k) {
    // Factor K into 64-element sectors. The contiguous 128-byte dimension matches the
    // hardware swizzle while the next box dimension permits larger K tiles without repacking.
    // `alignas(64)` is this port's change (1218d574): MSVC's host stub rejects a by-value descriptor
    // at 128 with C2719, and the measured acceptance ceiling is 64 -- see the struct above, which
    // carries the probe's numbers.
    alignas(64) CUtensorMap result{};
    std::uint64_t dimensions[]{64, static_cast<std::uint64_t>(k / 64),
                               static_cast<std::uint64_t>(rows)};
    std::uint64_t strides[]{128, static_cast<std::uint64_t>(k) * 2};
    std::uint32_t box[]{64, static_cast<std::uint32_t>(block_k / 64),
                        static_cast<std::uint32_t>(block_rows)};
    if constexpr (bf16_predicated_k<Schedule>) {
        // A logical 2D K axis lets TMA zero-fill its partial final sector, preserving row stride.
        static_assert(Schedule::kBlockK == 64, "K-tail TMA requires a 128-byte inner box");
        dimensions[0] = static_cast<std::uint64_t>(k);
        dimensions[1] = static_cast<std::uint64_t>(rows);
        strides[0]    = static_cast<std::uint64_t>(k) * 2;
        box[1]        = static_cast<std::uint32_t>(block_rows);
    }
    const std::uint32_t steps[]{1, 1, 1};
    const auto status = cuTensorMapEncodeTiled(
        &result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, bf16_predicated_k<Schedule> ? 2 : 3,
        const_cast<__nv_bfloat16*>(pointer), dimensions, strides, box, steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("BF16 TMA descriptor: ") +
                                 (name ? name : "CUDA error"));
    }
    return result;
}

template <class Schedule>
Bf16TmaDescriptors make_bf16_tma_descriptors(const Bf16A16Operands& p) {
    validate_bf16_operands<Schedule>(p);
    if ((!bf16_predicated_rows<Schedule> && p.rows % Schedule::kBlockRows) ||
        p.k % (bf16_predicated_k<Schedule> ? 8 : Schedule::kBlockK))
        throw std::invalid_argument("BF16 TMA requires compatible row/K tiles");
    return {bf16_tma_map<Schedule>(p.weight, p.rows, p.k, Schedule::kBlockRows, Schedule::kBlockK),
            bf16_tma_map<Schedule>(p.x, p.tokens, p.k, Schedule::kBlockTokens, Schedule::kBlockK)};
}

template <class Schedule>
__device__ __forceinline__ void bf16_tma_load(void* destination, const CUtensorMap* map,
                                              int k_sector, int row, std::uint64_t* barrier) {
    if constexpr (bf16_predicated_k<Schedule>) {
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                     "[%0], [%1, {%2, %3}], [%4];"
                     :
                     : "r"(smem_addr(destination)), "l"(map), "r"(k_sector * 64), "r"(row),
                       "r"(smem_addr(barrier))
                     : "memory");
    } else {
        asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                     "[%0], [%1, {%2, %3, %4}], [%5];"
                     :
                     : "r"(smem_addr(destination)), "l"(map), "r"(0), "r"(k_sector), "r"(row),
                       "r"(smem_addr(barrier))
                     : "memory");
    }
}

template <class Schedule, class Epilogue>
inline constexpr int bf16_tma_scratch_bytes =
    ((Schedule::kTensorBytes > bf16_epilogue_bytes<Schedule, Epilogue>
          ? Schedule::kTensorBytes
          : bf16_epilogue_bytes<Schedule, Epilogue>)+127) /
    128 * 128;

template <class Schedule, bool FullTokens, class Output, class Epilogue>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void bf16_a16_tma_mma_kernel(
    const __grid_constant__ Bf16TmaDescriptors descriptors, Output output, Epilogue epilogue,
    int rows, int input_rows, int token_offset, int count) {
    constexpr int BR = Schedule::kBlockRows, BT = Schedule::kBlockTokens, BK = Schedule::kBlockK;
    constexpr int S   = Schedule::kStages;
    const int K       = Schedule::kStaticK ? Schedule::kStaticK : input_rows;
    const int tiles_r = bf16_predicated_rows<Schedule> ? (rows + BR - 1) / BR : rows / BR;
    const int tiles_t = (count + BT - 1) / BT;
    int tile_r, tile_t;
    bf16_mma_tile_coordinates<Schedule>(blockIdx.x, tiles_r, tiles_t, tile_r, tile_t);
    const int row_begin = tile_r * BR, token_begin = token_offset + tile_t * BT;
    extern __shared__ __align__(128) unsigned char storage[];
    auto* a = reinterpret_cast<__nv_bfloat16*>(storage);
    auto* b = a + S * BR * BK;
    auto* full =
        reinterpret_cast<std::uint64_t*>(storage + bf16_tma_scratch_bytes<Schedule, Epilogue>);
    auto* empty = full + S;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < S; ++stage) {
            cta_mbarrier_init(full + stage, 1);
            cta_mbarrier_init(empty + stage, Schedule::kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();
    const int tiles_k = K / BK + (bf16_predicated_k<Schedule> && K % BK != 0);
    if (threadIdx.x < Schedule::kProducerThreads) {
        if (threadIdx.x == 0) {
            for (int kt = 0; kt < tiles_k; ++kt) {
                const int stage = kt % S;
                cta_mbarrier_wait(empty + stage, 1U ^ ((kt / S) & 1U));
                cta_mbarrier_arrive_expect_tx(full + stage, (BR + BT) * BK * 2);
                bf16_tma_load<Schedule>(a + stage * BR * BK, &descriptors.weight, kt * (BK / 64),
                                        row_begin, full + stage);
                bf16_tma_load<Schedule>(b + stage * BT * BK, &descriptors.activation,
                                        kt * (BK / 64), token_begin, full + stage);
            }
        }
        return;
    }
    const int tid  = threadIdx.x - Schedule::kProducerThreads;
    const int warp = tid / 32, lane = tid & 31;
    float accum[Schedule::kMmaRows][Schedule::kMmaTokens][4] = {};
#pragma unroll Schedule::kConsumerKUnroll
    for (int kt = 0; kt < tiles_k; ++kt) {
        const int stage = kt % S;
        cta_mbarrier_wait(full + stage, (kt / S) & 1U);
        bf16_mma_compute_stage<Schedule>(a + stage * BR * BK, b + stage * BT * BK, accum, warp,
                                         lane);
        // Every lane must finish its shared reads before the elected lane releases this stage.
        __syncwarp();
        if (lane == 0) cta_mbarrier_arrive(empty + stage);
    }
    bf16_finish_mma_tile<Schedule, FullTokens>(output, epilogue, storage, accum, row_begin,
                                               token_begin, rows, token_offset + count, warp, lane);
}

template <class Schedule, class Output, class Epilogue>
void launch_bf16_a16_tma_mma(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                             cudaStream_t stream) {
    // Descriptors are launch-owned values, copied into kernel parameters during Graph capture.
    const auto descriptors = make_bf16_tma_descriptors<Schedule>(p);
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const auto blocks = static_cast<std::int64_t>(bf16_predicated_rows<Schedule>
                                                          ? div_up(p.rows, Schedule::kBlockRows)
                                                          : p.rows / Schedule::kBlockRows) *
                            div_up(count, Schedule::kBlockTokens);
        if (blocks > 2147483647LL)
            throw std::invalid_argument("BF16 TMA grid exceeds CUDA grid.x capacity");
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = bf16_a16_tma_mma_kernel<Schedule, Full, Output, Epilogue>;
            constexpr int bytes =
                bf16_tma_scratch_bytes<Schedule, Epilogue> + Schedule::kBarrierBytes;
            bf16_prepare_shared<bytes, kernel>();
            kernel<<<static_cast<unsigned>(blocks), Schedule::kThreads, bytes, stream>>>(
                descriptors, output, epilogue, p.rows, p.k, offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}

} // namespace ninfer::ops::detail
