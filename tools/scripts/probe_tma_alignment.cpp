// Prints the alignment facts behind the port's TMA descriptor comments.
//
// The kernel headers state that CUtensorMap carries no alignment attribute in this build, and that a
// by-value __grid_constant__ descriptor block is layable up to 64 and not at 128. Both are claims
// about the toolchain, so this prints the values they rest on rather than leaving them to be
// re-derived from cuda.h with a guess about which branch of it is live.
//
// Run it through tools/scripts/probe_tma_align.cmd, which also sweeps the by-value parameter width.
// The point of interest is that alignof(CUtensorMap) is a property of the BUILD FLAGS, not of the
// type: cuda.h applies alignas(TENSOR_MAP_ALIGN) only inside
// `#if defined(__cplusplus) && (__cplusplus >= 201103L)`, and nvcc's host pass on MSVC reports
// __cplusplus = 199711L unless /Zc:__cplusplus is passed. Compare the two runs.

#include <cuda.h>

#include <cstdio>

int main() {
    std::printf("alignof(CUtensorMap) = %zu\n", alignof(CUtensorMap));
    std::printf("sizeof(CUtensorMap)  = %zu\n", sizeof(CUtensorMap));
    std::printf("TENSOR_MAP_ALIGN     = %d\n", TENSOR_MAP_ALIGN);
    std::printf("__cplusplus          = %ld\n", static_cast<long>(__cplusplus));
    std::printf("attribute applied    = %s\n",
                (alignof(CUtensorMap) == TENSOR_MAP_ALIGN ? "yes" : "NO"));
    return 0;
}
