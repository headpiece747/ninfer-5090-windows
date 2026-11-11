#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace ninfer::ops::common {

/// By-value `__grid_constant__` kernel parameter carrier for 128-byte-aligned TMA
/// descriptor bundles.
///
/// On Windows, `nvcc` emits formal kernel parameters in the generated `.cudafe1.stub.c`
/// host wrapper, where MSVC rejects formal by-value parameters aligned above 64 bytes
/// (`error C2719`). Keeping `T` itself `alignas(128)` on both host and device while
/// aligning this host/device carrier to 64 bytes only on the MSVC host pass avoids `C2719`
/// without downgrading `T` or its device-side `.param .align 128` ABI.
template <class T>
struct
#if defined(__CUDA_ARCH__) || !defined(_MSC_VER)
    alignas(128)
#else
    alignas(64)
#endif
    TmaKernelParam {
    static_assert(std::is_trivially_copyable_v<T>,
                  "TmaKernelParam requires a trivially copyable descriptor type");
    static_assert(sizeof(T) % 128 == 0,
                  "TmaKernelParam requires size divisible by 128 bytes");
    static_assert(alignof(T) == 128,
                  "TmaKernelParam requires 128-byte alignment on the carried type");

    std::uint8_t bytes[sizeof(T)];

    __host__ static TmaKernelParam pack(const T& value) noexcept {
        TmaKernelParam param{};
        std::memcpy(param.bytes, &value, sizeof(T));
        return param;
    }

    __device__ __forceinline__ const T& get() const noexcept {
        return *reinterpret_cast<const T*>(bytes);
    }
};

template <class T>
__host__ inline TmaKernelParam<T> make_tma_kernel_param(const T& value) noexcept {
    return TmaKernelParam<T>::pack(value);
}

}  // namespace ninfer::ops::common
