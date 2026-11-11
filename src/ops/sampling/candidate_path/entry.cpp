#include "ops/common/host/operands.h"
#include "ninfer/ops/candidate_selector.h"
#include "ops/sampling/candidate_path/plan.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {
constexpr std::int32_t kRank              = 256;
constexpr std::int32_t kNvfp4CodebookRows = 248320;
constexpr const char* kOp                 = "candidate_selector_path";

void require_codebook(const Weight& codebook, std::int32_t expected_rows, Device device,
                      const char* label) {
    if (codebook.device != device) {
        throw std::invalid_argument(std::string(kOp) + ": wrong device for " + label);
    }
    if (codebook.qtype == QType::NVFP4) {
        if (codebook.n != kNvfp4CodebookRows || codebook.n != expected_rows ||
            codebook.k != kRank) {
            throw std::invalid_argument(std::string(kOp) + ": invalid " + label);
        }
        (void)common::validate_nvfp4_weight(codebook, kOp);
        return;
    }
    const std::uint64_t payload_bytes =
        expected_rows > 0 ? static_cast<std::uint64_t>(expected_rows) * kRank * sizeof(std::uint16_t)
                          : 0ULL;
    if (codebook.qtype != QType::BF16 || codebook.layout != QuantLayout::Contiguous ||
        codebook.ndim != 2 || expected_rows <= 0 || codebook.n != expected_rows ||
        codebook.k != kRank || codebook.shape[0] != expected_rows || codebook.shape[1] != kRank ||
        codebook.padded_shape[0] != expected_rows || codebook.padded_shape[1] != kRank ||
        codebook.payload_bytes < payload_bytes || codebook.qhigh != nullptr ||
        codebook.high_plane_bytes != 0 || !common::operand::aligned(codebook.qdata, 16)) {
        throw std::invalid_argument(std::string(kOp) + ": invalid " + label);
    }
}

void validate_common(int steps, int batch, const Tensor& hidden, const Tensor& anchors,
                     const Weight& predecessor, const Weight& successor, int token_domain,
                     const Tensor& positions, const SamplingConfig* configs, const Tensor& drafts,
                     const Tensor& q, MemoryArena& workspace, CudaExecutionView execution) {
    if (steps < 1 || steps > 15 || batch < 1 || batch > 8)
        throw std::invalid_argument("candidate_selector_path: K must be 1..15, B must be 1..8");
    if (workspace.device() != execution.device())
        throw std::invalid_argument("candidate_selector_path: workspace belongs to another device");
    const std::int32_t rows = predecessor.n;
    require_codebook(predecessor, rows, execution.device(), "predecessor codebook");
    require_codebook(successor, rows, execution.device(), "successor codebook");
    // Both codebooks are dispatched together from predecessor.qtype.
    if (predecessor.qtype != successor.qtype) {
        throw std::invalid_argument("candidate_selector_path: codebook formats must match");
    }
    if (token_domain < 1 || token_domain > rows)
        throw std::invalid_argument("candidate_selector_path: invalid token domain");
    const auto tensor = [&](const Tensor& t, DType type, std::initializer_list<int> shape,
                            const char* name) {
        common::operand::tensor(t, type, shape, 16, kOp, name, execution.device());
    };
    tensor(hidden, DType::BF16, {256, steps, batch}, "projected hidden");
    tensor(anchors, DType::I32, {batch}, "anchors");
    tensor(positions, DType::I32, {batch}, "positions");
    tensor(drafts, DType::I32, {steps, batch}, "drafts");
    tensor(q, DType::FP32, {16, steps, batch}, "proposal q");
    if (!common::operand::aligned(configs, alignof(SamplingConfig)))
        throw std::invalid_argument("candidate_selector_path: invalid configs");
}

void execute(CandidateSelectorInput input, const Tensor& ids, const Tensor& unary,
             const Tensor& keys, const Tensor& hidden, const Tensor& anchors,
             const Weight& predecessor, const Weight& successor, int token_domain,
             const Tensor& positions, const SamplingConfig* configs, Tensor& drafts, Tensor& q,
             MemoryArena& workspace, CudaExecutionView execution) {
    const int steps = ids.ne[1], batch = ids.ne[2];
    validate_common(steps, batch, hidden, anchors, predecessor, successor, token_domain, positions,
                    configs, drafts, q, workspace, execution);
    if (input == CandidateSelectorInput::GatheredKeys && predecessor.qtype == QType::NVFP4) {
        throw std::invalid_argument(
            "candidate_selector_path: NVFP4 codebooks require explicit candidate IDs");
    }
    common::operand::tensor(ids, DType::I32, {16, steps, batch}, 16, kOp, "candidate IDs",
                            execution.device());
    if (input == CandidateSelectorInput::Candidates)
        common::operand::tensor(unary, DType::FP32, {16, steps, batch}, 16, kOp, "unary scores",
                                execution.device());
    else
        common::operand::tensor(keys, DType::I64, {16, steps, batch, 2}, 8, kOp, "gathered keys",
                                execution.device());
    if (predecessor.qtype == QType::NVFP4) {
        const auto elements = static_cast<std::size_t>(predecessor.n) * kRank;
        const std::array ranges{
            common::operand::range(ids, kOp),
            common::operand::range(unary, kOp),
            common::operand::range(hidden, kOp),
            common::operand::range(anchors, kOp),
            common::operand::range(predecessor.qdata, elements / 2, kOp),
            common::operand::range(predecessor.scales, elements / 16, kOp),
            common::operand::range(successor.qdata, elements / 2, kOp),
            common::operand::range(successor.scales, elements / 16, kOp),
            common::operand::range(positions, kOp),
            common::operand::range(drafts, kOp),
            common::operand::range(q, kOp),
            common::operand::range(configs, batch * sizeof(SamplingConfig), kOp)};
        common::operand::disjoint(ranges, kOp);
    } else {
        const std::array ranges{
            common::operand::range(ids, kOp),
            common::operand::range(input == CandidateSelectorInput::Candidates ? unary : keys, kOp),
            common::operand::range(hidden, kOp),
            common::operand::range(anchors, kOp),
            common::operand::range(predecessor.qdata, predecessor.payload_bytes, kOp),
            common::operand::range(successor.qdata, successor.payload_bytes, kOp),
            common::operand::range(positions, kOp),
            common::operand::range(drafts, kOp),
            common::operand::range(q, kOp),
            common::operand::range(configs, batch * sizeof(SamplingConfig), kOp)};
        common::operand::disjoint(ranges, kOp);
    }
    candidate_path_detail::execute_candidate_path(input, ids, unary, keys, hidden, anchors,
                                                  predecessor, successor, positions, configs,
                                                  drafts, q, workspace, execution.stream());
}
} // namespace

std::size_t candidate_selector_path_workspace_capacity_bytes(CandidateSelectorInput input,
                                                             int min_steps, int max_steps,
                                                             int min_batch, int max_batch) {
    if (min_steps < 1 || max_steps > 15 || max_steps < min_steps || min_batch < 1 ||
        max_batch > 8 || max_batch < min_batch)
        throw std::invalid_argument("selector workspace: invalid K/B interval");
    WorkspaceLayoutBuilder layout;
    for (int k = min_steps; k <= max_steps; ++k)
        for (int b = min_batch; b <= max_batch; ++b) {
            auto scope = layout.scope();
            (void)candidate_path_detail::allocate_selector_workspace(
                layout, candidate_path_detail::select_candidate_path(input, k, b), k, b);
        }
    return layout.peak_bytes();
}

void candidate_selector_path(const Tensor& ids, const Tensor& unary, const Tensor& hidden,
                             const Tensor& anchors, const Weight& predecessor,
                             const Weight& successor, int token_domain, const Tensor& positions,
                             const SamplingConfig* configs, Tensor& drafts, Tensor& q,
                             MemoryArena& workspace, CudaExecutionView execution) {
    execute(CandidateSelectorInput::Candidates, ids, unary, {}, hidden, anchors, predecessor,
            successor, token_domain, positions, configs, drafts, q, workspace, execution);
}

void candidate_selector_path(const Tensor& keys, const Tensor& hidden, const Tensor& anchors,
                             const Weight& predecessor, const Weight& successor, int token_domain,
                             const Tensor& positions, const SamplingConfig* configs, Tensor& ids,
                             Tensor& drafts, Tensor& q, MemoryArena& workspace,
                             CudaExecutionView execution) {
    execute(CandidateSelectorInput::GatheredKeys, ids, {}, keys, hidden, anchors, predecessor,
            successor, token_domain, positions, configs, drafts, q, workspace, execution);
}
} // namespace ninfer::ops
