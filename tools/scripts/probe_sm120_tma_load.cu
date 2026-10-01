// REPRODUCER, NOT A TEST. This currently FAILS on sm_120a (RTX 5090) with "an illegal memory access
// was encountered", in every arm. That failure is the finding: docs/active-work.md item 12 records the
// project's FP8 TMA route dying here, and compute-sanitizer --tool memcheck spent 30 minutes on the
// smallest failing case without naming an instruction. This reproduces the same class of fault in a
// standalone program of about 150 lines, in about two seconds, with no project code in the path.
//
// Measured 2026-10-01 on CUDA 13.3, MSVC, driver 617.14, sm_120a. Every arm below faults:
//   BY_VALUE=1 SWIZZLE=128B    -> illegal memory access   (the form BF16 and NVFP4 use, ungated)
//   BY_VALUE=0 SWIZZLE=128B    -> illegal memory access   (the form this port forces on Windows)
//   BY_VALUE=1 SWIZZLE=NONE    -> illegal memory access
// so the cause is not the by-pointer descriptor form, not the swizzle mode, not the missing tensormap
// proxy fence, and not the mbarrier expect_tx ordering -- each was varied and each is ruled out.
//
// The descriptor is built the way src/ops/linear/fp8/fp8_a8_tma_mma.cuh's fp8_tma_map builds one,
// because that one is known to encode successfully: rank 2, dimensions {k, rows} with K FIRST, a
// single globalStrides entry {k}, elementStrides {1,1}, L2_PROMOTION_NONE. Three earlier versions of
// this probe were rejected by the driver with CUDA_ERROR_INVALID_VALUE -- rank 3 with three stride
// entries and L2_PROMOTION_L2_128B. That rejection is the driver correctly validating a malformed
// descriptor and says nothing about TMA's runtime behaviour.
//
// See docs/research/sm120-tma-illegal-instruction-evidence.md: sm_120 does support TMA
// (cp.async.bulk.tensor requires sm_90 or higher), supports neither wgmma nor tcgen05, and CUTLASS,
// vLLM and TensorRT-LLM all ship working SM120 TMA kernels -- so this is not "TMA is unavailable on
// consumer Blackwell". CUTLASS issue #2728 reports illegal instruction inside cp.async.bulk.tensor on
// an RTX 5090 on Ubuntu, which is the same instruction on the same GPU class on a different OS.
//
// usage: tools\scripts\probe_sm120_tma_load.cmd

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#ifndef SWIZZLE_MODE
#define SWIZZLE_MODE 3  // 0 = NONE, 1 = 32B, 2 = 64B, 3 = 128B
#endif

#define CU_CHECK(x)                                                             \
    do {                                                                       \
        CUresult s_ = (x);                                                     \
        if (s_ != CUDA_SUCCESS) {                                              \
            const char* n_ = nullptr;                                          \
            (void)cuGetErrorName(s_, &n_);                                     \
            std::printf("  driver %s -> %s\n", #x, n_ ? n_ : "?");             \
            return 2;                                                          \
        }                                                                      \
    } while (0)

#define RT_CHECK(x)                                                            \
    do {                                                                       \
        cudaError_t e_ = (x);                                                  \
        if (e_ != cudaSuccess) {                                               \
            std::printf("  runtime %s -> %s\n", #x, cudaGetErrorString(e_));   \
            return 2;                                                          \
        }                                                                      \
    } while (0)

__device__ __forceinline__ std::uint32_t smem_u32(const void* p) {
    return static_cast<std::uint32_t>(__cvta_generic_to_shared(p));
}

#ifndef BY_VALUE
#define BY_VALUE 0  // 1 = pass the tensor map BY VALUE as __grid_constant__, 0 = by pointer
#endif

// The whole point of the A/B. This port's Windows path passes the descriptor by POINTER into a
// device buffer, because a by-value alignas(128) parameter cannot be laid out by the MSVC ABI (C2719).
// The BF16 and NVFP4 routes pass it BY VALUE and are not Windows-gated, so if by-value works here and
// by-pointer does not, the failure is the pointer form rather than TMA.
//
// And by-value with no alignas is available on Windows after all: this toolchain gives CUtensorMap
// alignof == 8 (cuda.h gates its own alignas on __cplusplus >= 201103L, which nvcc's MSVC host pass
// reports as 199711), so it was always the alignas(128), not the by-value passing, that C2719 rejected.
#if BY_VALUE
__global__ void tma_probe(const __grid_constant__ CUtensorMap map, std::uint8_t* out,
                          int expect_first) {
#else
__global__ void tma_probe(const CUtensorMap* map, std::uint8_t* out, int expect_first) {
#endif
    __shared__ alignas(128) std::uint8_t tile[128 * 32];
    __shared__ alignas(8) std::uint64_t bar;

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&bar)));
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Fence only makes sense for the pointer form: a map in parameter space is already in the
        // right proxy. Kept under the guard so the by-value arm is not testing a second variable.
#if !BY_VALUE
        asm volatile("fence.proxy.acquire.tensormap::generic.cta [%0], 128;" ::"l"(map) : "memory");
#endif

        // ORDER MATTERS: expect_tx declares how many bytes will complete this barrier's transaction,
        // so it must be issued BEFORE the copy that consumes it. Issuing it afterwards leaves the
        // transaction counted against a barrier with no pending count, which is undefined and is the
        // most likely reason this probe faulted in every arm.
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&bar)),
                                                                    "r"(128 * 32));

        // Same instruction and operand shape as fp8_tma_load.
#if BY_VALUE
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {0, 0}], [%2];"
            :
            : "r"(smem_u32(tile)), "l"(&map), "r"(smem_u32(&bar))
            : "memory");
#else
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {0, 0}], [%2];"
            :
            : "r"(smem_u32(tile)), "l"(map), "r"(smem_u32(&bar))
            : "memory");
#endif
    }

    // Bounded wait. mbarrier.try_wait.parity takes a .pred destination plus FOUR operands
    // (done, [addr], phase, suspendHint); the syntax is taken from src/ops/common/mbarrier.cuh.
    if (threadIdx.x == 0) {
        constexpr std::uint32_t kSuspendTicks = 0x989680;
        std::uint32_t arrived = 0;
        for (std::uint32_t i = 0; i < 2000000 && arrived == 0; ++i) {
            asm volatile("{\n"
                         ".reg .pred p;\n"
                         "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n"
                         "selp.b32 %3, 1, 0, p;\n"
                         "}\n"
                         : "=r"(arrived)
                         : "r"(smem_u32(&bar)), "r"(0), "r"(kSuspendTicks)
                         : "memory");
        }
        out[3] = static_cast<std::uint8_t>(arrived);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        out[0] = tile[0];
        out[1] = tile[1];
        out[2] = (tile[0] == static_cast<std::uint8_t>(expect_first)) ? 1 : 0;
    }
}

int main() {
    const int k = 128;    // innermost extent, in bytes; also the swizzle-128B box width
    const int rows = 32;  // outer extent
    const int block_k = k;
    const int block_rows = rows;

    const std::size_t elements = static_cast<std::size_t>(k) * rows;
#if BY_VALUE
    std::printf("descriptor: rank 2, dims {%d, %d}, strides {%d}, box {%d, %d}, swizzle %d, "
                "passed BY VALUE\n",
                k, rows, k, block_k, block_rows, SWIZZLE_MODE);
#else
    std::printf("descriptor: rank 2, dims {%d, %d}, strides {%d}, box {%d, %d}, swizzle %d, "
                "passed BY POINTER\n",
                k, rows, k, block_k, block_rows, SWIZZLE_MODE);
#endif

    std::uint8_t* device = nullptr;
    RT_CHECK(cudaMalloc(reinterpret_cast<void**>(&device), elements));
    std::uint8_t* host = new std::uint8_t[elements];
    for (std::size_t i = 0; i < elements; ++i) { host[i] = static_cast<std::uint8_t>(i & 0xFF); }
    RT_CHECK(cudaMemcpy(device, host, elements, cudaMemcpyHostToDevice));

    const std::uint64_t dimensions[2] = {static_cast<std::uint64_t>(k),
                                         static_cast<std::uint64_t>(rows)};
    const std::uint64_t strides[1] = {static_cast<std::uint64_t>(k)};
    const std::uint32_t box[2] = {static_cast<std::uint32_t>(block_k),
                                  static_cast<std::uint32_t>(block_rows)};
    const std::uint32_t steps[2] = {1, 1};

    alignas(64) CUtensorMap map{};
    CU_CHECK(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, device, dimensions,
                                    strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                    static_cast<CUtensorMapSwizzle>(SWIZZLE_MODE),
                                    CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    std::printf("  descriptor encoded OK\n");

    // cuda.h: "tensorMap address must be aligned to 64 bytes".
    void* map_device = nullptr;
    RT_CHECK(cudaMalloc(&map_device, 256));
    const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(map_device);
    std::printf("  map_device %p  (mod 64 = %zu)\n", map_device,
                static_cast<std::size_t>(addr % 64));
    if (addr % 64 != 0) {
        std::printf("  alignment precondition violated -> would fault for that reason alone\n");
        return 2;
    }
    RT_CHECK(cudaMemcpy(map_device, &map, sizeof(map), cudaMemcpyHostToDevice));

    std::uint8_t* out = nullptr;
    RT_CHECK(cudaMalloc(reinterpret_cast<void**>(&out), 8));
    RT_CHECK(cudaMemset(out, 0, 8));

#if BY_VALUE
    tma_probe<<<1, 128>>>(map, out, 0);
#else
    tma_probe<<<1, 128>>>(static_cast<const CUtensorMap*>(map_device), out, 0);
#endif

    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        std::printf("  LAUNCH FAILED: %s\n", cudaGetErrorString(launch));
        return 1;
    }
    const cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        std::printf("  EXECUTION FAILED: %s\n", cudaGetErrorString(sync));
        return 1;
    }

    std::uint8_t got[4] = {0, 0, 0, 0};
    RT_CHECK(cudaMemcpy(got, out, 4, cudaMemcpyDeviceToHost));
    std::printf("  arrived=%u  out[0]=%u out[1]=%u  matched_expected=%u\n", got[3], got[0], got[1],
                got[2]);
    if (got[2] != 1) {
        std::printf("  TMA RAN but did not deliver the source bytes\n");
        return 1;
    }
    std::printf("  TMA WORKS on sm_120a with the project's descriptor shape\n");
    return 0;
}
