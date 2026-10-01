#pragma once

#include "ops/softmax_attention/common/causal_operands.h"
#include "ops/softmax_attention/common/causal_partition.h"
#include <algorithm>

namespace ninfer::ops::detail {

inline constexpr int kMxfp8TiledQueryRows = 128;
inline constexpr int kMxfp8TiledMaxSplits = 8;

// Minimize waves per KV partition, retaining fewer partitions on a tie.
// The bound limits FP32 partial traffic; live rows cap the count at
// ceil(visible_keys / 512). Count changes work and storage, never kernel topology.
//
// The occupancy guard is the term the wave-balance search below is missing. That search compares
// only wave counts, and more splits always balance better, so it walks to kMxfp8TiledMaxSplits
// whenever the improvement is a single unit -- measured on this product's shapes, all eight steps
// from 1 to 8 qualify at a prefill width of 8192. Each split costs a full FP32 accumulator of
// kCausalHeadDim depth per head per width (4*256+8 = 1032 bytes here), so at 24 heads that cap
// costs 1.62 GiB of the 2.246 GiB workspace to hold partials for work that was never short of
// occupancy.
//
// FlashAttention's num_splits_heuristic carries the missing term and states the reason: it returns
// 1 split once the query tiles already fill 80% of the SMs, "however, we also don't want too many
// splits as that would incur more HBM reads/writes". Here ctas is 1536 against a threshold of 136,
// so the guard is not marginal -- it is the difference between splitting a saturated launch and not.
// Below the threshold the wave-balance search still runs, because there the splits are what fill
// the machine.
inline CausalKvPartition mxfp8_tiled_partition(int heads, int width, int visible_capacity,
                                               int multiprocessor_count) {
    const std::int64_t tiles =
        (static_cast<std::int64_t>(width) + kMxfp8TiledQueryRows - 1) / kMxfp8TiledQueryRows;
    const std::int64_t ctas = heads * tiles;
    const auto saturated    = multiprocessor_count > 0 &&
                           ctas * 5 >= static_cast<std::int64_t>(multiprocessor_count) * 4;
    int selected            = 1;
    auto waves              = (ctas + multiprocessor_count - 1) / multiprocessor_count;
    for (int splits = 2; !saturated && splits <= kMxfp8TiledMaxSplits; ++splits) {
        const auto next = (ctas * splits + multiprocessor_count - 1) / multiprocessor_count;
        if (next * selected < waves * splits) {
            selected = splits;
            waves    = next;
        }
    }
    CausalKvPartition partition{1, selected, 9};
    partition.capacity = partition.active(visible_capacity);
    return partition;
}

inline std::size_t mxfp8_tiled_workspace_bytes(int heads, int min_width, int max_width,
                                               int visible_capacity, int multiprocessor_count) {
    std::size_t maximum = 0;
    // A query-tile interval has one split target and increasing partial storage.
    // Check each interval's last width; checking max_width alone would miss a
    // larger allocation immediately before the split target decreases.
    for (std::int64_t begin = std::max(min_width, 17); begin <= max_width;) {
        const auto last =
            ((begin + kMxfp8TiledQueryRows - 1) / kMxfp8TiledQueryRows) * kMxfp8TiledQueryRows;
        const int end = static_cast<int>(std::min<std::int64_t>(max_width, last));
        const auto partition =
            mxfp8_tiled_partition(heads, end, visible_capacity, multiprocessor_count);
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, end, partition.capacity, 1);
        maximum = std::max(maximum, layout.peak_bytes(1));
        begin   = static_cast<std::int64_t>(end) + 1;
    }
    return maximum;
}

} // namespace ninfer::ops::detail
