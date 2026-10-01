// Sweep subject: which by-value __grid_constant__ descriptor-block alignments does MSVC accept?
//
// The port's C2719 fix rests on a threshold -- that a by-value `__grid_constant__` descriptor block is
// layable up to some width and not above it. tools/scripts/probe_tma_align.cmd compiles this once per
// width and reports ACCEPTED or C2719. This exercises the real shape, a by-value __grid_constant__
// kernel parameter holding CUtensorMap members, because the failure is in nvcc's generated host stub
// and not in C++'s own layout rules.
//
// PROBE_ALIGN is supplied by the caller (-DPROBE_ALIGN=...). It is deliberately NOT named `N`:
// cuda.h declares cuMemsetD8(CUdeviceptr, unsigned char, size_t N), so -DN=8 macro-replaces that
// parameter into `size_t 8` and every width fails with "expected a )" inside cuda.h -- a result that
// looks like a finding and is an artefact of the harness.

#include <cuda.h>
#include <cstdint>

#ifndef PROBE_ALIGN
#error "define PROBE_ALIGN to the alignment under test"
#endif

struct alignas(PROBE_ALIGN) Desc {
    CUtensorMap a;
    CUtensorMap b;
};

__global__ void probe_kernel(const __grid_constant__ Desc d, float alpha) {
    // Touch the payload so the parameter cannot be optimised out of the signature.
    if (alpha == 12345.0f) {
        volatile std::uint64_t x = d.a.opaque[0] + d.b.opaque[0];
        (void)x;
    }
}

void launch(const Desc* d, float alpha, cudaStream_t s) {
    probe_kernel<<<1, 1, 0, s>>>(*d, alpha);
}
