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

#include <cstdint>

namespace ninfer::ops {

// True when pointer is non-null and its address is a multiple of alignment, which must be a
// power of two. The null test is part of the rule rather than a guard around it: a null
// pointer is not aligned to anything, and callers rely on that.
inline bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

} // namespace ninfer::ops
