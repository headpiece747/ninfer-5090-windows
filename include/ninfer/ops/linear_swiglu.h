#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include "core/device.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Capacity for every extent in the inclusive interval; invalid profiles or intervals throw.
[[nodiscard]] std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype,
                                                                 std::int32_t gate_up_rows,
                                                                 std::int32_t input_rows,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens);

/**
 * Policy-bearing capacity query. Q4/Q8 use A16 under every policy. NVFP4 uses A16 under
 * A16Only/AllowA8 at every positive T — fused through T=16, then the port's wide-T A16 MMA
 * route — and AllowA4 accepts every positive T. Row-scaled FP8 accepts all policies, with
 * A8 permitted by AllowA8/AllowA4.
 * A permissive policy covers whichever qualified route the private resolver selects across the
 * requested interval.
 */
[[nodiscard]] std::size_t
linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                       std::int32_t input_rows, LinearPolicy policy,
                                       std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * out[i,t] = SiLU((W_gate * x)[i,t]) * (W_up * x)[i,t].
 *
 * x [K,T] and out [M,T] are contiguous, 16-byte-aligned BF16 matrices.
 * gate_up_weight [2*M,K] stores all M gate rows followed by their matching M up rows.
 * Weights use their native stored representation. Inputs and weight storage must not overlap out.
 * Writes the entire output and has no persistent state.
 *
 * Policy controls permitted activation quantization. The overload without a policy uses A16Only.
 * Caller workspace covers the complete operation, is scoped to the call and overlaps no operand.
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                   MemoryArena& ws, CudaExecutionView execution);

/**
 * A16-only convenience form. Q4/Q8 and row-scaled FP8 retain their complete positive-T domain.
 * NVFP4 keeps the complete positive-T domain: the fused small-T family through T=16, then the
 * port's wide-T fused A16 MMA route (no workspace).
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, MemoryArena& ws,
                   CudaExecutionView execution);

// Explicit clamped product: SiLU(min(gate, limit)) * clamp(up, -limit, limit).
// limit must be finite and positive. Gate/up accumulators feed the product without
// an intermediate BF16 rounding boundary. Other operands/effects follow linear_swiglu.
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, float limit,
                   LinearPolicy policy, MemoryArena& ws, CudaExecutionView execution);

/**
 * g = W_gate*x + bias_gate; u = W_up*x + bias_up;
 * out[i,t] = SiLU(min(g[i,t],limit)) * clamp(u[i,t],-limit,limit).
 *
 * Represented [2*M,K] weights and naturally aligned BF16 bias [2*M] both use [gate;up]
 * row order. The clamp limit is finite and positive.
 * The projection and bias feed the product without intermediate BF16 rounding.
 * x/out follow linear_swiglu's matrix and alias requirements; out also overlaps no bias.
 * Inputs are preserved. Policy/workspace and Graph semantics follow linear_swiglu.
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, const Tensor& bias, Tensor& out,
                   float limit, LinearPolicy policy, MemoryArena& workspace,
                   CudaExecutionView execution);

// A16 execution requiring no workspace.
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, const Tensor& bias, Tensor& out,
                   float limit, CudaExecutionView execution);

[[nodiscard]] std::size_t
linear_swiglu_bias_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                            std::int32_t input_rows, LinearPolicy policy,
                                            std::int32_t min_tokens, std::int32_t max_tokens);

} // namespace ninfer::ops
