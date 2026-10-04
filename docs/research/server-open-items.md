# Server-side open items: status and disposition

This document records every known server-side open or deferred item, its status, and its
disposition. It exists so these items do not come up again without their context.

## Summary

| Item | Status | Disposition |
|---|---|---|
| Issue 5: cache-hitting race condition | Deferred | Not reproducible at `--max-concurrency 1`; no fix needed |
| Fail-stop exit code | **Fixed** | Exit code 3 on engine failure |
| FP8 TMA kernel faults on MSVC | Deferred | Requires careful porting of 7 upstream commits |
| Protocol-level unsupported features | By design | Intentional product boundaries |
| Remote HTTP media | **Fixed** | Uses CMake `FindCURL`; graceful when libcurl absent |
| TDR not measured | **Documented** | Added to all 8 launcher headers |
| `SO_EXCLUSIVEADDRUSE` not set | **Fixed** | Added to Windows socket configuration |

---

## 1. Issue 5: cache-hitting race condition

**Status:** Deferred — not reproduced, not refuted.

**What it is:** A user reported that a cache-hitting continuation submitted while another lane is
generating could cause a resource underflow (`Qwen3.5 resource subtraction underflow`), killing
the engine. The reporter observed 14 incidents in 24 hours at `--max-concurrency 3`.

**Why it does not affect this port:** This port ships `--max-concurrency 1`, where the reporter
explicitly states the fault never occurs ("0 incidents in all tests and in ~9 h of single-lane
operation"). The bug is in upstream's code, not this port's.

**What exists:** `tests/models/qwen3_5/test_engine_issue5_race.cpp` — a deterministic test loop
that drives the reporter's named trigger and asserts only their symptom.

**RE-RUN 2026-10-04 — this test now RUNS on this machine, which the entry above said it could not.**
The blocker recorded here was capacity: "the reporter's command line needs 20 GB of runtime capacity;
this machine offers 15 GB." Measured on 2026-10-04 with nothing else on the GPU: **30.9 GiB VRAM free
of 32.6 GiB, and 36.4 GB of 47.8 GB RAM free.** The 15 GB figure is no longer true of this machine, so
the stated reason the reproduction could not run no longer holds.

Run with the real artifact, which the test requires (`NINFER_TEST_ARTIFACT`, else it skips 77):

```
ninfer_qwen3_5_issue5_race_test.exe   ->  exit 0 in 15 s, "ok"
  root published: prompt=14379 path=0 reused=0
  holder finished: outputs=512
  continuation finished: reused=14394 path=1 outputs=16
  engine still accepts work after the race (not latched)
```

So the blocker is retired, and **red-capability is still unproven** — the test went green again, and a
green run is not evidence the bug is absent. It is evidence the test does not detect it. Proving
red-capability needs a mutation at the defect site, and this run does not locate that site: the
accounting field the test watches, `main_kv_h2d_pages`, is a monotonic counter
(`include/ninfer/types.h:1036`, read through `context_cache/resource_manager.h:1177`) and not a
subtraction, so the reporter's "resource subtraction underflow" has no site in this tree that this
measurement reaches. **That is the open part, and it is a search problem, not a capacity one.**

**Docs:** `docs/research/issue5-deferred-findings.md`

**Disposition:** No fix needed for this port's shipped configuration. The test loop exists for
future investigation if the port ever ships `--max-concurrency > 1`.

---

## 2. Fail-stop exit code

**Status:** Fixed.

**What it was:** When the engine latched a failure (`fail_all_locked` sets `failed_` once and
nothing clears it), the process stayed alive forever. `/health` returned 503, but the process
never exited. A supervisor could not distinguish a hung process from a healthy one.

**What was done:**
- `apps/serve/main.cpp`: After `server.listen()` returns, checks `service.is_available()`. If
  the engine has failed, exits with code `3`.
- `src/serve/operational_log.h` / `.cpp`: Added `engine_failure()` method that logs a critical
  message before exit.
- `README.md`: Documented exit codes `0`, `1`, and `3`.

**Exit codes:**

| Code | Meaning |
|---|---|
| `0` | Clean shutdown (SIGINT/SIGTERM) |
| `1` | Configuration, bind, warmup, or listen error |
| `3` | Engine-wide failure — restart required |

**Why code 3 is distinct from 1:** A supervisor can tell "engine fault, restart required" from
"bad configuration, do not retry".

---

## 3. FP8 TMA kernel faults on MSVC

**Status:** Deferred — requires careful porting.

**What it is:** 7 upstream commits that optimize FP8 TMA kernels fail on MSVC with
`error C2719` (formal parameter with `__declspec(align('#'))` won't be aligned). The fix is
three one-line `alignas` reapplications, already validated here.

**Why it is deferred:** The port's platform surface is minimal (9 files), and these commits
touch kernel code that requires careful review. The fix is known but not yet applied.

**Docs:** `docs/active-work.md` item 12

**Disposition:** Deferred. The fix is known and validated; it needs a dedicated porting effort.

---

## 4. Protocol-level unsupported features

**Status:** By design — intentional product boundaries.

**What they are:** Features documented in `docs/serving.md` that are rejected with specific
error codes, not silently ignored:

- OpenAI Responses: background execution, compaction, deferred tools, `strict:true`, server
  tools, MCP, containers, `personality`
- OpenAI Chat: `logprobs`, `logit_bias`, `n > 1`, forced `tool_choice`,
  `parallel_tool_calls: false`, legacy `functions`, strict tool schemas, server tools,
  document blocks, audio/file inputs, `repetition_penalty != 1.0`
- Anthropic Messages: server tools, document blocks, audio/file inputs`, `store`,
  `background`, `service_tier`

**Disposition:** Not issues. These are intentional product boundaries.

---

## 5. Remote HTTP media

**Status:** Fixed.

**What it was:** `cmake/Dependencies.cmake` only called `pkg_check_modules(LIBCURL ...)` under
`if(NOT MSVC)`, so `NINFER_HAVE_LIBCURL` was never defined on Windows and the entire URL fetch
path compiled out.

**What was done:**
- `cmake/Dependencies.cmake`: Replaced with `find_package(CURL 7.85)` (optional, not required).
- `src/product/CMakeLists.txt`: Links `CURL::libcurl` instead of `PkgConfig::LIBCURL`.
- Removed the vestigial `ws2_32` link (the Winsock code only compiled under
  `NINFER_HAVE_LIBCURL`, which was never true on Windows).

**Behavior:** When libcurl is present, remote HTTP media works. When absent, the existing
`NINFER_HAVE_LIBCURL` guard handles it gracefully at runtime.

---

## 6. TDR not measured

**Status:** Closed 2026-10-04 (was: Documented). Measured at native context on the shipping route:
longest kernel 43.10 ms, 2.2% of the 2 s budget.

**What it was:** No documentation of the Windows TDR budget or the longest measured GPU launch.

**What was done:** Added a TDR documentation block to all 8 launcher `.bat` files explaining:
- The RTX 5090 runs WDDM with a 2-second TDR budget.
- No measured profile approaches this (prefill is chunked, decode uses CUDA graphs, weight
  upload is DMA).
- How to raise the threshold via `TdrDelay` registry key if needed.
- A TDR surfaces as a CUDA error and the engine's fail-stop latch terminates the process.

**MEASURED 2026-10-04 — the second bullet is now a number, not an inference.**

Instrument: Nsight Systems 2026.1.3, `--trace=cuda,nvtx --sample=none --cpuctxsw=none`, on
`build-bench/bench/ninfer_bench.exe --spec dflash2 --draft-tokens 7` against the QUASAR artifact, at
the shipped `--prefill-chunk 8192`. Ranked by the `cuda_gpu_kern_sum` Max column, which is the column
that answers the TDR question.

| prompt tokens | longest single kernel | ratio |
|---:|---:|---|
| 16,384 | **14.72 ms** | — |
| 32,768 | **33.61 ms** | 2.28× |
| 65,536 | **70.80 ms** | 2.11× |

(all three `causal_attention_prompt_bf16_kernel`)

Runners-up at 65,536 are 3.24 ms (`nvfp4_linear_swiglu_w4a4_tma`), 2.74 ms (`context_kv_mma_nvfp4`),
2.06 ms (`nvfp4_w4a4_tma`). Everything in decode is three orders of magnitude smaller; a 256-token run's
worst kernel is 1.09 ms.

**The scaling is ~2.2× per doubling of the prompt — mildly super-linear, exponent ≈1.14 — rather than
the quadratic (exponent 2.0) bare attention would give.** That is the prefill chunk bounding the work:
at a fixed `--prefill-chunk 8192` the kernel's cost tracks the chunk, not the whole prompt. An
exponent of 1.14 rather than 1.0 says the chunk is not a hard wall either — there is a residual
per-prompt term — which is why the extrapolation below is stated as a factor, not a constant multiple.

**The extrapolation this paragraph used to carry was WRONG by roughly 8×, and it was wrong in a way
worth keeping.** It read: "Extrapolated to the native 262,144 — two doublings beyond the largest
measured point — about 343 ms, roughly 17% of the 2 s budget."

**MEASURED 2026-10-04 at native context. The longest single GPU kernel is 43.10 ms — 2.2% of the 2 s
budget, a 46× margin — not 343 ms.**

Measured on the serving route, which is the route that can reach native context at all: `ninfer-serve`
on QUASAR under `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none`, launched with the
flags from `start_quasar_v3_dflash2_vision.bat` (`--max-context 262144 --prefill-chunk 8192 --kv-dtype
fp8 --spec dflash2 --draft-tokens 7 --lm-head-draft --vision`), driven by a probe that reuses
`tools/bench/warm_lane_sweep.py`'s own `conversation()` builder.

| prompt tokens | longest kernel | kernel |
|---:|---:|---|
| 258,490 (**98.7% of native**) | **43.10 ms** | `mxfp8_kv_tiled_mma_kernel` |
| 251,760 | served, not traced separately | — |
| 228,203 | served, not traced separately | — |
| 400 turns | **refused**: `context_length_exceeded`, 262,144 | — |

The boundary was found rather than assumed: 385 turns serves at 258,490 prompt tokens and 400 turns is
refused with "prepared prompt exceeds Engine max_context 262144", so the measured point sits just
under the ceiling rather than at some convenient round number.

**Why the extrapolation failed, which is the useful part.** It extrapolated
`causal_attention_prompt_bf16_kernel` — and **that kernel appears zero times in this trace.** The bench
route ran with the artifact's default `kv_cache=bf16`; this lane ran `--kv-dtype fp8`, and the two take
different attention paths. So the three bench points and this lane point are not a series at all: the
extrapolation carried a kernel across a route change that does not exist on the other side, and a
super-linear fit to a series that changes kernel is not a fit to anything.

**So the bench figures and this figure must not be tabulated together**, which is the same class of
error as the two acceptance figures in `docs/active-work.md` item 1. The bench numbers remain valid for
the bench route at bf16 KV; this number is valid for the shipping route at fp8 KV. Neither is
extrapolatable to the other, and this paragraph previously invited exactly that.

The 2 s TDR question is answered on the route that ships: **43.10 ms, 2.2% of budget.** Whether the
bf16-KV bench route would behave differently at native context is untested and now needs its own
measurement — the corpus cannot reach it, exactly as before.

**Disposition: closed by measurement on the shipping route.** The longest kernel at 98.7% of native
context is 43.10 ms, 2.2% of the 2 s budget. The launcher `.bat` blocks say "no measured profile
approaches this", which is now true *with a number behind it* rather than by inference — and 43.10 ms
is the figure a reader should look for, on the route that actually ships. The bf16-KV bench route at
native context remains unmeasured and is a separate question, not a gap in this one.

---

## 7. `SO_EXCLUSIVEADDRUSE` not set

**Status:** Fixed.

**What it was:** `configure_http_server_socket` called `httplib::default_socket_options(socket)`,
which on Windows enables `SO_REUSEADDR` — allowing another process to hijack the port.

**What was done:** Added `SO_EXCLUSIVEADDRUSE` to the Windows branch of
`configure_http_server_socket` in `src/serve/http_transport.cpp`, per Microsoft's Winsock
guidance: "All server applications must set `SO_EXCLUSIVEADDRUSE`."

**Disposition:** Fixed.
