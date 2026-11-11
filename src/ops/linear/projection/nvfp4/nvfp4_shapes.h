#pragma once
#include "ops/linear/kernels/nvfp4/nvfp4_launch.h"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_plan.h"

namespace ninfer::ops::detail {
struct Nvfp4LinearShape {
    std::int32_t n, k;
    Nvfp4Launch a16;
    void (*a4)(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace, cudaStream_t);
    bool (*uses_a4)(std::int32_t min_tokens, std::int32_t max_tokens);
    // Maximum over the interval, including routes that need less storage at larger T.
    std::size_t (*partial_capacity_bytes)(std::int32_t min_tokens,
                                          std::int32_t max_tokens) = nullptr;
};

extern const Nvfp4LinearShape kNvfp4N12288K4096;
extern const Nvfp4LinearShape kNvfp4N4096K6144;
extern const Nvfp4LinearShape kNvfp4N14336K5120;
extern const Nvfp4LinearShape kNvfp4N16384K5120;
extern const Nvfp4LinearShape kNvfp4N28672K5120;
extern const Nvfp4LinearShape kNvfp4N34816K5120;
extern const Nvfp4LinearShape kNvfp4N5120K6144;
extern const Nvfp4LinearShape kNvfp4N5120K14336;
extern const Nvfp4LinearShape kNvfp4N5120K17408;
// DFlash2 drafter module shapes (weight-only A16 routes). Port-only: upstream registers five dense
// NVFP4 geometries and no drafter shapes, and these serve the NVFP4 draft module our v3 artifacts
// carry -- confirmed against the artifact's conversion report, which registers
// dflash2/layers/0/attention/{query,key,value} with NVFP4 among the emitted formats.
extern const Nvfp4LinearShape kNvfp4DFlash2Feature;
extern const Nvfp4LinearShape kNvfp4DFlash2Qkv;
extern const Nvfp4LinearShape kNvfp4DFlash2AttnOut;
extern const Nvfp4LinearShape kNvfp4DFlash2ConvProj;
extern const Nvfp4LinearShape kNvfp4DFlash2Selector;
} // namespace ninfer::ops::detail
