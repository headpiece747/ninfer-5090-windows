#pragma once

#include "ninfer/ops/context_kv_materialize.h"

namespace ninfer::ops::detail {

struct ContextKVMaterializeImplementation {
    void (*launch)(const Tensor& context, const Tensor& positions, const Tensor& counts,
                   const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>&,
                   const ContextKVMaterializeProfile&, ContextKVMaterializeExecutionEnvelope,
                   const Tensor& key_scratch, cudaStream_t stream);
    bool key_scratch;
};

// Columns count the envelope's live prefix capacity across requests, not physical padding.
ContextKVMaterializeImplementation select_context_kv_materialize(const ContextKVMaterializeProfile&,
                                                                 std::int32_t columns);

ContextKVMaterializeImplementation select_bf16_context_kv_materialize(std::int32_t columns);

// The NVFP4-encoded key/value parents' implementation family. Port-only: upstream registers no
// NVFP4 parent here. Selected from the layer weights' format rather than from `columns`, because
// the two families are exclusive per call.
ContextKVMaterializeImplementation select_context_kv_materialize_nvfp4(std::int32_t columns);

} // namespace ninfer::ops::detail
