# Windows port: what the research found, and what it is deferred behind

> ## ⚠️ SUPERSEDED 2026-10-04 — three of the five deferred items are already done
>
> This document is a dated research record, not current guidance. Its ranked *"Deferred, in the
> order they are worth doing"* table (`:106`) lists three fixes that have since landed. Verified
> against source, not against another document:
>
> | this document claims | source, 2026-10-04 |
> |---|---|
> | `cmake/Dependencies.cmake:30` guards libcurl behind `if(NOT MSVC)` | `Dependencies.cmake:33` is `find_package(CURL 7.85)` — **done** |
> | `src/product/CMakeLists.txt:9` still links `ws2_32` | no first-party `ws2_32` link anywhere — **done** |
> | `SO_EXCLUSIVEADDRUSE` is never set | `src/serve/http_transport.cpp:109` sets it — **done** |
> | `README.md` says "memory-mapped"; the code does not map | **still open** |
> | the longest launch has never been measured | **still open** — see `server-open-items.md:15` |
>
> The authoritative list is `docs/research/server-open-items.md`. Read this file for how the findings
> were reached, not for what is outstanding. A reader taking the table below as current re-derives
> three fixes that already shipped.

From `windows-port-best-practice.md` (the cited research) and the verification that followed it. Each
item here is scoped rather than open-ended: what it is, the evidence, and what finishing it means.

## The port's platform surface is nine files

`git grep -l "_WIN32\|_MSC_VER" -- src apps include` returns exactly nine:

```
apps/perplexity/main.cpp                         src/product/logging/startup_log.cpp
src/artifact/file_io.cpp                         src/product/media_acquire/acquire.cpp
src/artifact/file_io.h                           src/runtime/engine/context_cache/context_cost.cpp
src/product/logging/logging.cpp                  src/serve/http_transport.cpp
                                                 src/serve/request_log.cpp
```

That is a small, reviewable surface, and the research confirmed most of it is already correct.

## Confirmed correct -- no change recommended

- **The `alignas` fix.** Removing the local `alignas(128)` so the wrapper inherits `TENSOR_MAP_ALIGN` is
  the recommended fix, not a workaround: NVIDIA's `cuda.h` v13.3 defines `TENSOR_MAP_ALIGN 64` under
  `_MSC_VER` and `128` otherwise, and MSVC rejects over-aligned by-value parameters
  ([C2719](https://learn.microsoft.com/en-us/cpp/error-messages/compiler-errors-2/compiler-error-c2719)).
  The pointer alternative is strictly worse -- it would reintroduce the graph-capture bug the port
  diagnosed in `1218d574`.
- **Artifact I/O.** `CreateFileW` + `FILE_FLAG_NO_BUFFERING` + `SEQUENTIAL_SCAN` + positional
  `OVERLAPPED` is Microsoft's documented `O_DIRECT` equivalent
  ([File Buffering](https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering)). Already
  correct.
- **cpp-httplib.** Windows-supported; no rewrite warranted.
- **The `_WIN32` branches themselves.** No standard API covers unbuffered I/O, `isatty`, terminal width
  or process id, so the branches are the right tool.

## The dead `ws2_32` link and the compiled-out SSRF guard

**What is true.** `cmake/Dependencies.cmake:30` guards libcurl behind `if(NOT MSVC)`, so no
`PkgConfig::LIBCURL` target exists on Windows, so `src/product/CMakeLists.txt:5` never sets
`NINFER_HAVE_LIBCURL=1`, so every block guarded by it in `acquire.cpp` (lines 46, 96, 362) compiles out
-- including the SSRF guard `private_address`/`private_ipv4`. Remote HTTP media therefore throws at
runtime:

```cpp
throw std::invalid_argument("ninfer compiled without curl. Remote URLs are unsupported.");
```

**What is also true, and the research missed it.** This is *documented*: `docs/serving.md:473` says
"HTTP media URLs are not supported". And the port *tests the refusal* deliberately --
`tests/test_media_acquire.cpp:226` is headed "Url: only the compile-time refusal is reachable without a
network". So this is a **deliberate deferral, not a regression.**

**What is genuinely wrong:** two things are vestigial, and the reason is recorded nowhere.

1. `src/product/CMakeLists.txt:9` still links `ws2_32` for a path that cannot exist.
2. The CMake mechanism above is not mentioned in the one place that states the limitation, so a reader
   of `docs/serving.md:473` cannot tell whether it is a policy choice or a Windows gap.

**Finishing it means:** one paragraph at `docs/serving.md:473` giving the reason, and either removing
the `ws2_32` link or commenting why it stays. Cheap, and it stops the next reader re-deriving this.

## TDR: safe by workload shape, not by measurement

**The primary source.** Microsoft's WDDM documentation: *"The default timeout period in Windows is two
seconds. If the GPU can't complete or preempt the current task within the TDR timeout period, the OS
diagnoses that the GPU is frozen"* and *"hardware vendors should ensure that graphics operations ... take
no more than two seconds"*
([WDDM support for TDR](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/timeout-detection-and-recovery)).
The trigger is a failed **preempt** wait, not simply a long kernel. Mitigations are
[`TdrDelay`](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys)
(the threshold in seconds; Intel's own install guide recommends 20 for long kernels) and
`TdrLevel = 0` to disable detection. Both need administrator rights and a reboot -- the same shape as
the IFEO entry the launchers already document.

**The code already behaves correctly under it.** `src/core/device.cu:61`:

```cpp
void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
    std::fprintf(stderr, "%s:%d: CUDA_CHECK(%s) failed: %s: %s\n", ...);
    std::abort();
}
```

A TDR reset surfaces as a CUDA error, and this terminates the process rather than pretending to
continue. That is the supervisory behaviour the issue-5 reporter asked for in *that* case, and it is
already present here.

**What is missing is a measurement, not a mechanism.** The research's own limit: no single launch is
known to approach 2 s, but nothing has measured the longest one. That is answerable from the stats
interval output or an nsys trace of a shipped lane at its shipped context.

**Finishing it means:** a `TdrDelay` paragraph in the launcher header beside the existing IFPE one, and
the longest-launch figure from a profile run recorded with the profile it came from.

## Two smaller documented items

- **`SO_EXCLUSIVEADDRUSE` is never set.** cpp-httplib leaves `SO_REUSEADDR` on Windows; Microsoft's
  guidance is that servers "must set `SO_EXCLUSIVEADDRUSE`". The launchers' `netstat` pre-check is a
  TOCTOU check, not protection. A port-level change to the transport wrapper, or a documented decision
  to accept it on a single-owner local machine -- this project's own stated trust model is a local,
  single-owner deployment, which is the argument for documenting rather than changing.
- **`README.md` says "memory-mapped"; the code does not map.** No `CreateFileMapping` exists in `src/`;
  the reader uses `ReadFile` with positional `OVERLAPPED`. A one-word documentation fix.

## Deferred, in the order they are worth doing

| # | item | why deferred |
|---|---|---|
| 1 | `docs/serving.md:473` reason paragraph, and the `ws2_32` link | cheap and stops re-derivation |
| 2 | README "memory-mapped" wording | one word |
| 3 | `TdrDelay` paragraph in the launcher header | documentation only |
| 4 | longest single launch measured and recorded | needs a profile run |
| 5 | `SO_EXCLUSIVEADDRUSE` decision, changed or documented | needs a transport change or a stated acceptance |
