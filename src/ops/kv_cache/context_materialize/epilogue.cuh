#pragma once

#include "ninfer/ops/context_kv_materialize.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/host/operands.h"
#include "ops/kv_cache/kernels/layout.cuh"
#include "ops/rope/kernels/phase.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops::detail::context_kv {

constexpr int kLayers  = static_cast<int>(kContextKVMaterializeLayers);
constexpr int kRows    = 1024;
constexpr int kHeadDim = 128;

__device__ __forceinline__ int context_column(int column, int width, int prefix) {
    return width == prefix ? column : column / prefix * width + column % prefix;
}

struct DeviceLayerView {
    const std::uint8_t* key_codes;
    const std::uint8_t* key_scales;
    const std::uint8_t* value_codes;
    const std::uint8_t* value_scales;
    const __nv_bfloat16* key_norm;
    __nv_bfloat16* cache_k;
    __half* cache_v;
    std::int32_t padded_capacity;
    const std::int64_t* offsets;
    // The port's NVFP4 key/value parents are slices of a packed draft payload whose stored divisor
    // the captured E4M3 scales do not include, so the decode has to fold `1 / weight_scale` back
    // in. Both are 1.0 for every registered non-NVFP4 format.
    float key_inverse_divisor   = 1.0F;
    float value_inverse_divisor = 1.0F;
};

struct DeviceLayers {
    DeviceLayerView layer[kContextKVMaterializeLayers];
    float epsilon;
    RopeD128Frequencies frequencies;
};

// Builds the device-side views from the Op's host layer array. Both the registered projections here
// and the port's NVFP4 materialize route stage the same caches, so the builder is shared rather
// than duplicated; the divisors stay 1.0 for every non-NVFP4 format.
inline DeviceLayers make_device_layers(
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    const ContextKVMaterializeProfile& profile) {
    DeviceLayers result{};
    result.epsilon     = profile.epsilon;
    result.frequencies = rope_d128_frequencies(profile.theta);
    const auto inverse_divisor = [](const Weight& weight) {
        return weight.qtype == QType::NVFP4 ? 1.0F / weight.weight_scale : 1.0F;
    };
    for (int index = 0; index < kLayers; ++index) {
        const auto& source  = layers[index];
        result.layer[index] = {
            static_cast<const std::uint8_t*>(source.key_weight.qdata),
            static_cast<const std::uint8_t*>(source.key_weight.scales),
            static_cast<const std::uint8_t*>(source.value_weight.qdata),
            static_cast<const std::uint8_t*>(source.value_weight.scales),
            static_cast<const __nv_bfloat16*>(source.key_norm_weight.data),
            common::operand::state_field_base<__nv_bfloat16>(source.cache.k),
            common::operand::state_field_base<__half>(source.cache.v),
            std::int32_t(source.cache.padded_capacity),
            static_cast<const std::int64_t*>(source.cache.k.records.byte_offsets.data),
            inverse_divisor(source.key_weight),
            inverse_divisor(source.value_weight),
        };
    }
    return result;
}

__device__ __forceinline__ void store_key_head(const float* input, DeviceLayerView layer,
                                               const int* positions, const std::int64_t* offsets,
                                               int column, int width, int head, float epsilon,
                                               const RopeD128Frequencies& frequencies) {
    const int lane = threadIdx.x & 31;
    const int j    = lane * 2;
    float x0 = input[j], x1 = input[j + 1], y0 = input[j + 64], y1 = input[j + 65];
    float sum           = common::warp_reduce_sum(x0 * x0 + x1 * x1 + y0 * y0 + y1 * y1);
    const float inverse = rsqrtf(__shfl_sync(0xffffffffU, sum, 0) / 128.0f + epsilon);
    x0 *= inverse * __bfloat162float(layer.key_norm[j]);
    x1 *= inverse * __bfloat162float(layer.key_norm[j + 1]);
    y0 *= inverse * __bfloat162float(layer.key_norm[j + 64]);
    y1 *= inverse * __bfloat162float(layer.key_norm[j + 65]);
    float sin0, cos0, sin1, cos1;
    rope_d128_sincos(frequencies, positions[column], j, &sin0, &cos0);
    rope_d128_sincos(frequencies, positions[column], j + 1, &sin1, &cos1);
    const auto dst = offsets[column / width] / sizeof(__nv_bfloat16) +
                     kv::cyclic_offset<128, 2048>(positions[column], head, layer.padded_capacity);
    auto* out      = reinterpret_cast<__nv_bfloat162*>(layer.cache_k + dst);
    out[lane]      = __floats2bfloat162_rn(x0 * cos0 - y0 * sin0, x1 * cos1 - y1 * sin1);
    out[lane + 32] = __floats2bfloat162_rn(y0 * cos0 + x0 * sin0, y1 * cos1 + x1 * sin1);
}

__device__ __forceinline__ void store_value(int row, int column, std::int64_t offset,
                                            const int* positions, DeviceLayerView layer,
                                            float result) {
    const auto dst =
        offset / sizeof(__half) + kv::cyclic_offset<128, 2048>(positions[column], row / 128,
                                                               layer.padded_capacity, row % 128);
    layer.cache_v[dst] = __float2half_rn(__bfloat162float(__float2bfloat16_rn(result)));
}

struct MaterializeProjectionEpilogue {
    DeviceLayerView layer;
    const int* positions;
    const int* counts;
    const std::int64_t* offsets;
    float* scratch;
    int layer_index, width, batch, min_count, max_count;
    bool value;

    __device__ void store(int row, int packed_column, float result) const {
        if (packed_column >= max_count * batch) return;
        const int column  = context_column(packed_column, width, max_count);
        const int request = column / width, count = counts[request];
        if (count < min_count || count > max_count || column % width >= count) return;
        if (!value)
            scratch[row + 1024LL * (packed_column + max_count * batch * layer_index)] = result;
        else { store_value(row, column, offsets[request], positions, layer, result); }
    }

    template <class Output>
    __device__ void store_fragment(const Output&, int row, int col, float4 sum, int,
                                   int columns) const {
        if (col < columns) {
            store(row, col, sum.x);
            store(row + 8, col, sum.z);
        }
        if (col + 1 < columns) {
            store(row, col + 1, sum.y);
            store(row + 8, col + 1, sum.w);
        }
    }
};

struct ContextPrefixColumns {
    int width, prefix;

    __device__ __forceinline__ int operator()(int column) const {
        return context_column(column, width, prefix);
    }
};

void launch_key_post(const float* key_scratch, const std::int32_t* positions,
                     const std::int32_t* counts, const DeviceLayers& layers,
                     std::int32_t batch_size, std::int32_t width, std::int32_t min_count,
                     std::int32_t max_count, cudaStream_t stream);

} // namespace ninfer::ops::detail::context_kv
