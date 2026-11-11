// NVFP4 projection kernels of context_kv_materialize: the weight-only NVFP4 [1024,5120]
// key/value parents (row slices of the draft module's query_key_value payload). One MMA family
// serves every column count: e2m1 codes stage raw and decode to exactly-representable BF16, the
// stored E4M3 scales apply in FP32 per 16-value group after each group's MMA accumulation, and
// the payload divisor folds into the captured scales. The activation staging, envelope filtering,
// key scratch, and cache stores are shared with the W8 family through epilogue.cuh.
#include "ops/kv_cache/context_materialize/dispatch.h"
#include "core/device.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/kv_cache/context_materialize/epilogue.cuh"
#include "ops/common/quant/nvfp4.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops::detail {
namespace {

using namespace common;
using namespace context_kv;

constexpr int kHidden = 5120;

// The port's own staging swizzle, from the pre-merge tree's
// src/ops/context_kv_materialize/context_kv_common.cuh:35 (`swizzle_128`, removed by this merge --
// `git show 133bff91:src/ops/context_kv_materialize/context_kv_common.cuh`). It permutes in
// 8-element units: (((column >> 3) ^ (row & 7)) << 3) | (column & 7). It is NOT
// common::swizzle_xor_128, which permutes in 16-element units (col ^ ((row & 7) << 4)); this kernel
// was written against the 8-element form, and substituting the 16-element one while moving the file
// is what broke the >256 route: with the substitution in place the FP64 oracle passes at W=256 and
// fails at W=257 and W=512, which is the same boundary the served lanes showed as acceptance
// collapsing to exactly zero above 256 context tokens.
__device__ __forceinline__ int swizzle_stage_128(int row, int column) {
    return (((column >> 3) ^ (row & 7)) << 3) | (column & 7);
}

// Eight BF16 values decoded from one K16 NVFP4 group, in the alignment the MMA's operand
// swizzle expects.
union alignas(16) ContextKVBf16x8 {
    uint4 raw;
    __nv_bfloat162 pair[4];
};

template <int Rows, int Columns, int BlockK>
union alignas(16) MaterializeNvfp4Storage {
    struct {
        __nv_bfloat16 code_values[Rows][BlockK];
        __nv_bfloat16 activations[Columns][BlockK];
        std::uint8_t codes[Rows][BlockK / 2];
        std::uint8_t scales[Rows][BlockK / 16];
    } mainloop;

    float scores[Columns][Rows];
};

template <int Rows, int Columns, int BlockK, int ColumnWarps>
__global__ __launch_bounds__(Rows / 16 * ColumnWarps * 32, 1) void context_kv_mma_nvfp4_kernel(
    const __nv_bfloat16* hidden, const int* positions, const int* counts, DeviceLayers layers,
    float* key_scratch, int width, int batch, int min_count, int max_count) {
    constexpr int kBlockRows = Rows, kBlockK = BlockK;
    constexpr int kBlockColumns = Columns;
    constexpr int kColumnWarps  = ColumnWarps;
    constexpr int kWarps = Rows / 16 * kColumnWarps, kThreads = kWarps * 32;
    constexpr int kWarpColumns = Columns / kColumnWarps, kTokenMmas = kWarpColumns / 8;
    constexpr int kKTiles = kHidden / kBlockK;
    constexpr int kGroupsPerTile = kBlockK / 16;
    static_assert((Rows == 64 || Rows == 128) && Columns % (8 * ColumnWarps) == 0);
    static_assert(kHidden % kBlockK == 0 && kThreads <= 1024);
    static_assert((kBlockK % 16) == 0 && kBlockK <= 128);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int warp_row = warp / kColumnWarps, warp_col = warp % kColumnWarps;
    const int gid = lane >> 2, lid = lane & 3;
    const int a_matrix = lane >> 3, a_rowoff = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_coloff = (a_matrix >> 1) << 3, b_row = lane & 7, b_coloff = ((lane >> 3) & 1) << 3;
    const int column_begin = blockIdx.y * Columns,
              live_columns = min(Columns, max_count * batch - column_begin);
    const int row_begin = blockIdx.x * Rows, layer_index = blockIdx.z >> 1;
    const bool value          = (blockIdx.z & 1) != 0;
    const auto layer          = layers.layer[layer_index];
    const auto* weight_codes  = value ? layer.value_codes : layer.key_codes;
    const auto* weight_scales = value ? layer.value_scales : layer.key_scales;
    const float inverse_divisor = value ? layer.value_inverse_divisor : layer.key_inverse_divisor;
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    auto& storage  = *reinterpret_cast<MaterializeNvfp4Storage<Rows, Columns, BlockK>*>(shared_bytes);
    auto& mainloop = storage.mainloop;
    {
        float accumulators[kTokenMmas][4] = {};

        const auto stage_activation = [&](int k_tile) {
            const int k_begin    = k_tile * kBlockK;
            constexpr int kItems = kBlockColumns * (kBlockK / 8);
            for (int item = tid; item < kItems; item += kThreads) {
                const int column  = item / (kBlockK / 8);
                const int k8      = item - column * (kBlockK / 8);
                auto* destination = &mainloop.activations[column][swizzle_stage_128(column, k8 * 8)];
                if (column < live_columns) {
                    cp_async<16, Cache::ca>(
                        destination,
                        hidden +
                            static_cast<std::int64_t>(context_column(column_begin + column, width,
                                                                    max_count)) *
                                kHidden +
                            k_begin + k8 * 8);
                } else {
                    cp_async_zfill<16, Cache::ca>(destination, hidden + k_begin + k8 * 8, 0);
                }
            }
        };

        const auto stage_weight = [&](int k_tile) {
            const int k_begin     = k_tile * kBlockK;
            constexpr int kChunks = kBlockRows * (kBlockK / 32);
            for (int item = tid; item < kChunks; item += kThreads) {
                const int local_row = item / (kBlockK / 32);
                const int chunk     = item - local_row * (kBlockK / 32);
                cp_async<16, Cache::cg>(
                    &mainloop.codes[local_row][chunk * 16],
                    weight_codes + static_cast<std::int64_t>(row_begin + local_row) * (kHidden / 2) +
                        k_begin / 2 + chunk * 16);
            }
            // The scale plane is the registered K16M128x4 blocked arrangement: each k-tile's
            // groups span BlockK/64 consecutive 512-byte tiles, four group bytes per row and tile.
            for (int local_row = tid; local_row < kBlockRows; local_row += kThreads) {
                const int row       = row_begin + local_row;
                const int in_tile   = (row % 32) * 16 + ((row % 128) / 32) * 4;
                const std::int64_t tile_base =
                    static_cast<std::int64_t>(row / 128) * (kHidden / 64) + k_begin / 64;
#pragma unroll
                for (int half = 0; half < kBlockK / 64; ++half) {
                    cp_async<4>(&mainloop.scales[local_row][half * 4],
                                weight_scales + (tile_base + half) * 512 + in_tile);
                }
            }
        };

        // E2M1 magnitudes are exactly representable in BF16; decode pairs straight to registers
        // and keep the stored E4M3 scale application in FP32.
        const auto decode_e2m1_codes = [&]() {
            constexpr int kChunksPerRow = kBlockK / 32;
            for (int item = tid; item < kBlockRows * kChunksPerRow; item += kThreads) {
                const int row      = item / kChunksPerRow;
                const int chunk    = item - row * kChunksPerRow;
                const int col      = chunk * 32;
                const uint4 packed = *reinterpret_cast<const uint4*>(&mainloop.codes[row][col / 2]);
                ContextKVBf16x8 decoded[4];
                const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
                for (int word = 0; word < 4; ++word) {
#pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        const float2 values = decode_nvfp4_e2m1x2(
                            static_cast<std::uint8_t>((words[word] >> (byte * 8)) & 0xffu));
                        decoded[word].pair[byte] =
                            __floats2bfloat162_rn(values.x, values.y);
                    }
                }
#pragma unroll
                for (int vec = 0; vec < 4; ++vec) {
                    store_vec(&mainloop.code_values[row][swizzle_stage_128(row, col + vec * 8)],
                              decoded[vec].raw);
                }
            }
        };

        stage_activation(0);
        stage_weight(0);
        cp_commit();

#pragma unroll 1
        for (int k_tile = 0; k_tile < kKTiles; ++k_tile) {
            cp_wait<0>();
            __syncthreads();
            decode_e2m1_codes();
            __syncthreads();

            // Capture the group scales (with the payload divisor folded in) before the next
            // async weight stage reuses their shared plane.
            float top_scales[kGroupsPerTile], bottom_scales[kGroupsPerTile];
#pragma unroll
            for (int g = 0; g < kGroupsPerTile; ++g) {
                top_scales[g] = decode_e4m3(mainloop.scales[warp_row * 16 + gid][g]) *
                                inverse_divisor;
                bottom_scales[g] =
                    decode_e4m3(mainloop.scales[warp_row * 16 + gid + 8][g]) * inverse_divisor;
            }
            __syncthreads();
            const int next = k_tile + 1;
            if (next < kKTiles) {
                stage_weight(next);
                cp_commit();
            }

            const auto load_fragments = [&](int k_step, unsigned(&a)[4],
                                            unsigned(&b)[kTokenMmas][2]) {
                const int weight_row = warp_row * 16 + a_rowoff;
                const int weight_col = k_step * 16 + a_coloff;
                ldmatrix_x4(
                    a[0], a[1], a[2], a[3],
                    smem_addr(
                        &mainloop.code_values[weight_row][swizzle_stage_128(weight_row, weight_col)]));
#pragma unroll
                for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                    const int activation_row = warp_col * kWarpColumns + token_mma * 8 + b_row;
                    const int activation_col = k_step * 16 + b_coloff;
                    ldmatrix_x2(b[token_mma][0], b[token_mma][1],
                                smem_addr(&mainloop.activations[activation_row][swizzle_stage_128(
                                    activation_row, activation_col)]));
                }
            };

            unsigned a_fragments[4];
            unsigned b_fragments[kTokenMmas][2];
#pragma unroll
            for (int group = 0; group < kGroupsPerTile; ++group) {
                float group_acc[kTokenMmas][4] = {};
                load_fragments(group, a_fragments, b_fragments);
#pragma unroll
                for (int t = 0; t < kTokenMmas; ++t)
                    mma_bf16(group_acc[t][0], group_acc[t][1], group_acc[t][2], group_acc[t][3],
                             a_fragments[0], a_fragments[1], a_fragments[2], a_fragments[3],
                             b_fragments[t][0], b_fragments[t][1]);
#pragma unroll
                for (int t = 0; t < kTokenMmas; ++t) {
                    accumulators[t][0] =
                        fmaf(group_acc[t][0], top_scales[group], accumulators[t][0]);
                    accumulators[t][1] =
                        fmaf(group_acc[t][1], top_scales[group], accumulators[t][1]);
                    accumulators[t][2] =
                        fmaf(group_acc[t][2], bottom_scales[group], accumulators[t][2]);
                    accumulators[t][3] =
                        fmaf(group_acc[t][3], bottom_scales[group], accumulators[t][3]);
                }
            }

            if (next < kKTiles) {
                __syncthreads();
                stage_activation(next);
                cp_commit();
            }
        }

        __syncthreads();
        auto& scores         = storage.scores;
        const int local_row0 = warp_row * 16 + gid;
        const int local_row1 = local_row0 + 8;
#pragma unroll
        for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
            const int column0 = warp_col * kWarpColumns + token_mma * 8 + 2 * lid;
            if (column0 < Columns) {
                scores[column0][local_row0] = accumulators[token_mma][0];
                scores[column0][local_row1] = accumulators[token_mma][2];
            }
            if (column0 + 1 < Columns) {
                scores[column0 + 1][local_row0] = accumulators[token_mma][1];
                scores[column0 + 1][local_row1] = accumulators[token_mma][3];
            }
        }
    }
    __syncthreads();
    for (int local = warp; local < live_columns; local += kWarps) {
        const int packed_column = column_begin + local;
        const int column        = context_column(packed_column, width, max_count);
        const int request       = column / width;
        const int count         = counts[request];
        if (count < min_count || count > max_count || column % width >= count) continue;
        if constexpr (Rows == 128) {
            if (!value) {
                store_key_head(storage.scores[local], layer, positions, layer.offsets, column, width,
                               row_begin / 128, layers.epsilon, layers.frequencies);
                continue;
            }
        }
        for (int r = lane; r < Rows; r += 32) {
            const int row      = row_begin + r;
            const float result = storage.scores[local][r];
            if (!value) {
                key_scratch[row + 1024LL * (packed_column + max_count * batch * layer_index)] =
                    result;
            } else {
                // Uses store_value() from epilogue.cuh with per-layer offsets and kv::cyclic_offset.
                store_value(row, column, layer.offsets[request], positions, layer, result);
            }
        }
    }
}

template <int Rows, int Columns, int BlockK, int ColumnWarps>
void launch_mma(const Tensor& x, const Tensor& positions, const Tensor& counts, DeviceLayers layers,
                ContextKVMaterializeExecutionEnvelope envelope, const Tensor& scratch,
                cudaStream_t stream) {
    constexpr int bytes = sizeof(MaterializeNvfp4Storage<Rows, Columns, BlockK>);
    if constexpr (bytes > 48 * 1024)
        CUDA_CHECK(cudaFuncSetAttribute(
            context_kv_mma_nvfp4_kernel<Rows, Columns, BlockK, ColumnWarps>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    context_kv_mma_nvfp4_kernel<Rows, Columns, BlockK, ColumnWarps>
        <<<dim3(kRows / Rows, (envelope.max_count * x.ne[2] + Columns - 1) / Columns,
                2 * kContextKVMaterializeLayers),
           Rows / 16 * ColumnWarps * 32, bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const int*>(positions.data),
            static_cast<const int*>(counts.data), layers, static_cast<float*>(scratch.data),
            x.ne[1], x.ne[2], envelope.min_count, envelope.max_count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

// The W8 table's route boundaries, kept private to the NVFP4 family. `columns` counts the
// envelope's live prefix capacity across requests, not physical padding, exactly as the dense
// selector reads it.
namespace {
enum class Route : std::uint8_t { KSplit16, KSplit24, Mma32, Mma80, Mma96, Fused64, Mma64 };

constexpr Route route_for(std::int32_t columns) {
    if (columns <= 16) return Route::KSplit16;
    if (columns <= 24) return Route::KSplit24;
    if (columns <= 32) return Route::KSplit16;
    if (columns <= 64) return Route::Mma32;
    if (columns <= 80) return Route::Mma80;
    if (columns <= 96) return Route::Mma96;
    if (columns <= 256) return Route::Fused64;
    return Route::Mma64;
}
} // namespace

void context_kv_materialize_nvfp4_launch(
    const Tensor& context, const Tensor& positions, const Tensor& counts,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    const ContextKVMaterializeProfile& profile, ContextKVMaterializeExecutionEnvelope envelope,
    const Tensor& key_scratch, cudaStream_t stream) {
    const DeviceLayers device_layers = make_device_layers(layers, profile);
    // Every column count serves from the MMA family; the small-column grouped schedules of the
    // W8 table land on the 32-column MMA tile.
    switch (route_for(envelope.max_count * context.ne[2])) {
    case Route::KSplit16:
    case Route::KSplit24:
    case Route::Mma32:
        launch_mma<64, 32, 128, 2>(context, positions, counts, device_layers, envelope, key_scratch,
                                   stream);
        break;
    case Route::Mma80:
        launch_mma<64, 80, 128, 5>(context, positions, counts, device_layers, envelope, key_scratch,
                                   stream);
        break;
    case Route::Mma96:
        launch_mma<64, 96, 128, 6>(context, positions, counts, device_layers, envelope, key_scratch,
                                   stream);
        break;
    case Route::Fused64:
        launch_mma<128, 64, 128, 2>(context, positions, counts, device_layers, envelope,
                                    key_scratch, stream);
        return;
    case Route::Mma64:
        // The port's own shape for this route, restored: it was never the defect. The >256 route
        // failed because the staging swizzle this file was moved onto (common::swizzle_xor_128) is
        // not the one the kernel was written against -- see swizzle_stage_128 above. Changing this
        // shape to the 128-wide K tile "fixed" the oracle by routing around the wrong swizzle, which
        // would have left the same defect waiting at any shape that reached the differing bit.
        launch_mma<64, 64, 64, 2>(context, positions, counts, device_layers, envelope, key_scratch,
                                  stream);
        break;
    }
    context_kv::launch_key_post(static_cast<const float*>(key_scratch.data),
                                static_cast<const int*>(positions.data),
                                static_cast<const int*>(counts.data), device_layers, context.ne[2],
                                context.ne[1], envelope.min_count, envelope.max_count, stream);
}

ContextKVMaterializeImplementation select_context_kv_materialize_nvfp4(std::int32_t columns) {
    // The port's contract, restored: the fused route needs no key scratch, every other route runs
    // the key-post pass and does.
    return {context_kv_materialize_nvfp4_launch, route_for(columns) != Route::Fused64};
}

} // namespace ninfer::ops::detail
