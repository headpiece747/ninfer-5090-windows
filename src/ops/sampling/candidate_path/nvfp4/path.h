#pragma once

#include "ops/sampling/candidate_path/plan.h"

namespace ninfer::ops::candidate_path_detail {

// Weight-only NVFP4 codebook route of the selector walk; see the BF16 route in the same family for
// the route structure and the shared SelectorWorkspace contract.
void execute_candidate_path_nvfp4(const Tensor& candidate_ids, const Tensor& unary_scores,
                                  const Tensor& projected_hidden, const Tensor& anchors,
                                  const Weight& predecessor_codebook,
                                  const Weight& successor_codebook, const Tensor& base_positions,
                                  const SamplingConfig* configs, Tensor& drafts, Tensor& proposal_q,
                                  const Tensor& live_rows, const SelectorWorkspace& workspace,
                                  cudaStream_t stream);

} // namespace ninfer::ops::candidate_path_detail
