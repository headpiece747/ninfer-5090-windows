#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// DFlash2 drafter matrices are weight-only NVFP4: they carry no activation-quant
// divisor sites, so only the A16 routes are eligible (uses_a4 below is false).
using Geometry = Nvfp4Geometry<6144, 5120>;
using Gemv =
    Nvfp4GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
template <int Tokens>
using Exact     = Nvfp4SimtSchedule<(Tokens >= 14 ? 16 : 4), 1, 2, 16, Tokens, 1,
                                    Nvfp4SimtActivationAccess::TokenPacked, Nvfp4ScaleAccess::Direct,
                                    Nvfp4CodeCache::Default, 1, Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using C2        = Nvfp4SimtSchedule<4, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                    Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                    Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using C4        = Nvfp4SimtSchedule<4, 1, 2, 16, 4, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                    Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                    Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using C12       = Nvfp4SimtSchedule<4, 1, 2, 16, 12, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                    Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                    Nvfp4SimtBlockOrder::RowsContiguous, 4>;
using C20       = Nvfp4SimtSchedule<4, 1, 2, 8, 20, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                    Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                    Nvfp4SimtBlockOrder::RowsContiguous, 4>;
using C24       = Nvfp4SimtSchedule<4, 1, 2, 16, 24, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                    Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                    Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using C28       = Nvfp4SimtSchedule<4, 1, 2, 16, 8, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                    Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                    Nvfp4SimtBlockOrder::TokenTilesContiguous, 4>;
using C32       = Nvfp4SimtSchedule<4, 1, 2, 16, 8, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                    Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                    Nvfp4SimtBlockOrder::TokenTilesContiguous, 4>;
using FullChunk = Nvfp4SimtSchedule<4, 1, 2, 16, 32, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                    Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                    Nvfp4SimtBlockOrder::RowsContiguous, 1>;

Nvfp4Launch select_a16(std::int32_t tokens) {
    if (tokens == 1) return launch_nvfp4_gemv<Geometry, Gemv>;
    if (tokens == 32) return launch_nvfp4_simt<Geometry, 32, FullChunk, true>;
    if (tokens >= 5 && tokens <= 8) return select_nvfp4_exact<Geometry, 5, 8, Exact>(tokens);
    if (tokens >= 13 && tokens <= 16) return select_nvfp4_exact<Geometry, 13, 16, Exact>(tokens);
    if (tokens <= 2) return launch_nvfp4_simt<Geometry, 2, C2, true>;
    if (tokens <= 4) return launch_nvfp4_simt<Geometry, 4, C4, false>;
    if (tokens <= 12) return launch_nvfp4_simt<Geometry, 12, C12, false>;
    if (tokens <= 20) return launch_nvfp4_simt<Geometry, 20, C20, false>;
    if (tokens <= 24) return launch_nvfp4_simt<Geometry, 24, C24, false>;
    if (tokens <= 28) return launch_nvfp4_simt<Geometry, 28, C28, false>;
    if (tokens <= 32) return launch_nvfp4_simt<Geometry, 32, C32, false>;
    throw std::logic_error("nvfp4 A16 chunk exceeds drafter shape capacity");
}

// Weight-only drafter: never eligible for the A4 (activation-quantized) routes.
bool uses_a4(std::int32_t, std::int32_t) { return false; }
} // namespace

const Nvfp4LinearShape kNvfp4DFlash2Qkv{6144, 5120, launch_nvfp4_a16_chunks<32, select_a16>,
                               nullptr, uses_a4};
} // namespace ninfer::ops::detail
