#pragma once

#include "ninfer/types.h"
#include "runtime/contract/resources.h"

#include <cstddef>

namespace ninfer::runtime {

[[nodiscard]] KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                                       const SequenceCapacityCurve& curve,
                                                       std::size_t available_runtime_bytes);

// Publishes the resolution into the summary both execution cores report.
//
// This lives here rather than in each core because the two cores held identical copies of these
// ten assignments. A field added on one side of that pair could be added to one core and silently
// report zero from the other, and nothing would fail: the value is only ever read out.
inline void publish_kv_capacity(const KvCapacityResolution& resolution, MemorySummary& out) noexcept {
    out.kv_capacity_mode                  = resolution.mode;
    out.kv_capacity_page_groups           = resolution.main_page_groups;
    out.kv_capacity_max_page_groups       = resolution.maximum_main_page_groups;
    out.minimum_runtime_reservation_bytes = resolution.minimum_runtime_reservation_bytes;
    out.kv_capacity_increment_bytes       = resolution.bytes_per_additional_main_page_group;
    out.runtime_reservation_bytes         = resolution.runtime_reservation_bytes;
    out.available_after_weights_bytes     = resolution.available_after_weights_bytes;
    out.available_after_startup_bytes     = resolution.available_after_startup_bytes;
    out.kv_capacity_headroom_bytes        = resolution.automatic_headroom_bytes;
    out.planned_slack_bytes               = resolution.planned_slack_bytes;
}

} // namespace ninfer::runtime
