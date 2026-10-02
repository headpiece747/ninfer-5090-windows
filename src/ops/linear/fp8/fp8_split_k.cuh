#pragma once

// Split-K machinery shared by both FP8 A8 transports.
//
// This was TMA-only and lived inside fp8_a8_tma_mma.cuh, which made split-K unreachable from the MMA
// route for a reason that had nothing to do with split-K: the plan, the partial store and the
// reduction touch only schedule constants and FP8 operands. They never read a tensor map, so the
// TMA schedule was not what made them work.
//
// That mattered for item 12. Deciding the nine PORT-DISPATCH gates needs a matched-tile comparison on
// K6144MidBulk (128x128x128) and K6144Bulk (128x256x128), which cover tokens 193-768 and beyond --
// most real prefill traffic. Both are split-K, so without this they could not be twinned at all and
// the transport question stayed unanswerable for the tiles that carry most prefill traffic.
//
// All three names below were used only by fp8_a8_tma_mma.cuh, so the move has no external call site.

#include "ops/common/math.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/common/vector_output.cuh"
#include "ops/linear/fp8/fp8_a8_mma_common.cuh"
#include "ops/linear/fp8/fp8_operands.h"

#include <algorithm>

namespace ninfer::ops::detail {

struct Fp8SplitKPlan {
    int full_tiles = 0;
    int tail_tiles = 0;
    int split_ctas = 0;

    __host__ __device__ int parts(int tail) const {
        return split_ctas / tail_tiles + (tail < split_ctas % tail_tiles);
    }

    __host__ __device__ int first_part(int tail) const {
        const int extra = split_ctas % tail_tiles;
        return tail * (split_ctas / tail_tiles) + (tail < extra ? tail : extra);
    }
};

template <class Schedule>
inline Fp8SplitKPlan fp8_split_k_plan(int tiles, int k) {
    Fp8SplitKPlan plan{tiles, 0, 0};
    if constexpr (Schedule::kSplitWaveCtas > 0) {
        const int tail = tiles % Schedule::kSplitWaveCtas;
        if (tail && tail <= Schedule::kSplitWaveCtas / 2) {
            const int max_parts = std::min(Schedule::kMaxParts, k / Schedule::kBlockK);
            if (max_parts >= 2)
                plan = {tiles - tail, tail, std::min(Schedule::kSplitWaveCtas, tail * max_parts)};
        }
    }
    return plan;
}

// Decode a split CTA's block index. Returns the tile it should compute (the same value for an
// unsplit CTA) and writes the split index through partial -- -1 when the CTA keeps the full K loop --
// plus its K range through k_begin and tiles_k. Both transports use this verbatim, so the split
// geometry cannot drift between them; that would make a matched-tile A/B meaningless.
template <class Schedule>
__device__ inline int fp8_split_k_range(Fp8SplitKPlan plan, int tile, int& partial, int& k_begin,
                                       int& tiles_k) {
    partial = -1;
    if (tile < plan.full_tiles) return tile;
    partial                = tile - plan.full_tiles;
    const int base         = plan.split_ctas / plan.tail_tiles;
    const int extra        = plan.split_ctas % plan.tail_tiles;
    const int longer       = extra * (base + 1);
    const int tail         = partial < longer ? partial / (base + 1) : extra + (partial - longer) / base;
    const int part         = partial - plan.first_part(tail);
    const int parts        = plan.parts(tail);
    const int first_tile   = tiles_k * part / parts;
    k_begin                = first_tile;
    tiles_k                = tiles_k * (part + 1) / parts - first_tile;
    return plan.full_tiles + tail;
}

// Store raw FP32 accumulators. Scales, epilogue and the final BF16 cast belong after the complete K
// reduction, including for fused epilogues -- so a partial carries no scale and this is the only
// thing a split CTA writes.
template <class Schedule>
__device__ inline void fp8_store_split_partials(float* partials, int partial,
                                     const float (&acc)[Schedule::kMmaTokens]
                                                            [Schedule::kMmaRows][4],
                                     int warp, int lane) {
    constexpr int BT = Schedule::kBlockTokens, BR = Schedule::kBlockRows;
#pragma unroll
    for (int mt = 0; mt < Schedule::kMmaTokens; ++mt) {
        const int token = warp / Schedule::kWarpsRows * Schedule::kWarpTokens + mt * 16 + lane / 4;
#pragma unroll
        for (int mr = 0; mr < Schedule::kMmaRows; ++mr) {
            const int row = warp % Schedule::kWarpsRows * Schedule::kWarpRows + mr * 8 + 2 * (lane % 4);
            float* destination = partials + std::size_t(partial) * BT * BR + token * BR + row;
            *reinterpret_cast<float2*>(destination) =
                make_float2(acc[mt][mr][0], acc[mt][mr][1]);
            *reinterpret_cast<float2*>(destination + 8 * BR) =
                make_float2(acc[mt][mr][2], acc[mt][mr][3]);
        }
    }
}

template <class Schedule, class Output, class Epilogue, class RowPolicy>
__global__ void fp8_a8_split_k_reduce(Fp8A8Operands p, const float* partials, Output output,
                                      Epilogue epilogue, RowPolicy row_policy, Fp8SplitKPlan plan,
                                      int token_offset, int count) {
    constexpr int BT = Schedule::kBlockTokens, BR = Schedule::kBlockRows;
    constexpr int stored_rows    = BR / (RowPolicy::kPaired ? 2 : 1);
    constexpr int chunk_elements = BT * stored_rows / Schedule::kReductionBlocks;
    const int tail               = blockIdx.x / Schedule::kReductionBlocks;
    const int chunk              = blockIdx.x % Schedule::kReductionBlocks;
    int row_tile, token_tile;
    fp8_mma_tile_coordinates<Schedule>(plan.full_tiles + tail, p.rows / BR, div_up(count, BT),
                                       row_tile, token_tile);
    const int first = plan.first_part(tail), parts = plan.parts(tail);
    const int row_begin    = row_tile * stored_rows;
    const auto tile_output = linear_output_tile<stored_rows>(output, row_begin);
    for (int index = chunk * chunk_elements + threadIdx.x * 2; index < (chunk + 1) * chunk_elements;
         index += blockDim.x * 2) {
        const int token_local = index / stored_rows;
        const int token       = token_offset + token_tile * BT + token_local;
        const int output_row  = index % stored_rows;
        const int row         = row_begin + output_row;
        if (token < token_offset + count) {
            const auto sum_pair = [&](int local_row) {
                float2 sum{};
                for (int part = 0; part < parts; ++part) {
                    const auto value = *reinterpret_cast<const float2*>(
                        partials + std::size_t(first + part) * BT * BR + token_local * BR +
                        local_row);
                    sum.x += value.x;
                    sum.y += value.y;
                }
                const int parent  = row_policy.weight_row(row_begin, local_row, p.rows);
                const float scale = p.x_scales[token];
                sum.x             = sum.x * scale * __bfloat162float(p.scales[parent]);
                sum.y             = sum.y * scale * __bfloat162float(p.scales[parent + 1]);
                return sum;
            };
            float2 value;
            if constexpr (RowPolicy::kPaired) {
                constexpr int half_warp = Schedule::kWarpRows / 2;
                const int gate_row =
                    (output_row / half_warp) * Schedule::kWarpRows + output_row % half_warp;
                const float2 gate = sum_pair(gate_row);
                const float2 up   = sum_pair(gate_row + half_warp);
                value             = make_float2(epilogue.apply_pair(row, token, gate.x, up.x),
                                                epilogue.apply_pair(row + 1, token, gate.y, up.y));
            } else {
                value = fp8_apply_row_pair(epilogue, row, row + 1, token, sum_pair(output_row));
            }
            if constexpr (std::is_same_v<Output, LinearBf16Output>) {
                *reinterpret_cast<__nv_bfloat162*>(tile_output.at(row, token)) =
                    __floats2bfloat162_rn(value.x, value.y);
            } else {
                tile_output.store(row, token, value.x);
                tile_output.store(row + 1, token, value.y);
            }
        }
    }
}

} // namespace ninfer::ops::detail
