#pragma once

// Internal argument-validation primitives shared by the Op implementations.
//
// This module exists because a rule should have one spelling. aligned_to was written out
// verbatim in seventeen files -- measured, not estimated: one distinct body, 63 call sites --
// so a change to what "aligned" means had seventeen places it had to be made, and any one of
// them could have been missed without a test noticing.
//
// Only primitives belong here. An Op's own admissibility rules -- a shape that is legal for
// one epilogue and not another -- stay with that Op, where the reason for the rule lives.

#include "core/tensor.h"

#include <cstdint>

namespace ninfer::ops {

// True when pointer is non-null and its address is a multiple of alignment, which must be a
// power of two. The null test is part of the rule rather than a guard around it: a null
// pointer is not aligned to anything, and callers rely on that.
inline bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

// True when two tensors' byte ranges overlap. Ranges, not shapes: two views of the same storage
// overlap even when their extents differ, and a caller checking operand aliasing wants that.
//
// The name is deliberately not `overlaps`. An unnamed namespace has an implicit using-directive
// in its enclosing namespace, so a TU-local `overlaps(const Tensor&, const Tensor&)` becomes
// visible in ninfer::ops alongside anything declared here -- same signature, same level, and the
// call is ambiguous. aligned_to never met this because every local copy of it was removed; the
// other overlaps variants are kept, so this one needs a name no TU uses locally.
//
// This is the majority spelling. target_logprobs.cpp holds a variant written with subtraction
// rather than addition, which cannot overflow for addresses near the top of the address space.
// That one is deliberately left alone: adopting it here would be a behaviour change dressed up
// as a deduplication, and the two only differ where the addition would wrap.
inline bool tensors_overlap(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

} // namespace ninfer::ops
