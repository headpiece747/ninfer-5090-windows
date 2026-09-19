# FlashQLA Gated DeltaNet: gate-driven intra-card CP and the fused forward

Status: research note, 2026-09-19. Read from `QwenLM/FlashQLA@main` raw sources. No product code changed; this note is the only file written.

## The question

Exactly how does QwenLM/FlashQLA implement the GDN chunked forward pass -- in particular its gate-driven intra-card context parallelism (CP) and its algebraic reformulation -- so the algorithm can be ported into NInfer's hand-written C++/CUDA GDN kernel?

## Primary sources read

All fetched from `https://raw.githubusercontent.com/QwenLM/FlashQLA/main/<path>` (branch `main`, fetched 2026-09-19). Line citations below refer to those files.

| Path | Role |
|---|---|
| `flash_qla/ops/gated_delta_rule/chunk/cp_context.py` | CP decision + split + boundary correction orchestration |
| `flash_qla/ops/gated_delta_rule/chunk/blackwell_sm120/cp_fwd.py` | sm_120 warmup scan and `correct_initial_states` |
| `flash_qla/ops/gated_delta_rule/chunk/blackwell_sm120/fused_fwd.py` | sm_120 fused forward (our hardware) |
| `flash_qla/ops/gated_delta_rule/chunk/blackwell_sm120/prepare_h.py` | sm_120 state/warmup pass (`fused_gdr_h`), produces `ht` and `mt` |
| `flash_qla/ops/gated_delta_rule/chunk/blackwell_sm120/kkt_solve.py` | KKT + triangular inverse |
| `flash_qla/ops/gated_delta_rule/chunk/{blackwell,hopper}/fused_fwd.py` | non-sm120 contrast |
| `flash_qla/ops/gated_delta_rule/chunk/__init__.py` | pipeline entry, `CHUNK_SIZE` per arch |
| `flash_qla/ops/utils/group_reduce.py`, `cumsum.py` | helpers |
| `benchmark/benchmark_results_5090.txt`, `benchmark/bench_gated_delta_rule.py` | 5090 numbers and harness |
| `tests/ref_gdr.py`, `tests/test_gdr_unit.py` | independent FP64 oracle + CP equivalence test |
| `README.md` | stated design claims |

Labels used below: **[FACT]** = quoted/derived directly from source at the cited line; **[INFERENCE]** = my reading; **[UNVERIFIED]** = could not confirm from source.

---

## 1. The gate-driven split predicate

There are **two separate decisions**, and the design note conflates them.

**(a) Where the sequence is cut into CP segments — chunk-count only, not gate-driven. [FACT]**

`cp_context.py:63-72`:

```python
# Latency model: T = a·L_cp + b·(B·H·Lc/P) / L_cp + c
# Minimizing T yields the theoretical optimum: L_cp* ∝ √(B·H·Lc / P), where P = MULTI_PROCESSOR_COUNT, L_cp = max_local_chunks
# Scaled by empirical factor (3) and aligned to the nearest power of 2 for optimal SM scheduling & memory alignment.
max_local_chunks = 2 ** round(
    math.log2(math.sqrt(H * sum(num_chunks) / MULTI_PROCESSOR_COUNT) * 3)
)
max_local_chunks = max(max_local_chunks, 4)
```

`H` is `num_v_heads`, `MULTI_PROCESSOR_COUNT` is the device SM count (`cp_context.py:32`), and `sum(num_chunks)` is total chunks over the batch. The split loop (`cp_context.py:84-99`) cuts each raw sequence every `max_local_tokens = max_local_chunks * chunk_size` (`cp_context.py:80`), so **segments begin only on chunk boundaries** (`s += max_local_tokens`, `cp_context.py:92`). No gate value is consulted.

Whether CP is used at all is a heuristic occupancy gate (`cp_context.py:102-109`), for SM120:

```python
if ARCH == "SM90" or ARCH == "SM120":
    use_cp = Be * H <= 40 or (Be * H <= 56 and max(num_chunks) >= 128)
```

with `Be = sum(num_chunks) / max(num_chunks)` (`cp_context.py:106`). For a single long sequence `Be=1`, so `H_v=48` needs `max(num_chunks) >= 128` (>= 4096 tokens at chunk 32) to enable CP.

**(b) The gate-driven predicate is the *warmup window*: how many trailing chunks of a segment must be recomputed so the segment's zero-start state is a good approximation. [FACT]**

The predicate is a backward scan of `g` at chunk boundaries with a fixed log-decay threshold, `flash_qla/.../blackwell_sm120/cp_fwd.py:52-76`:

```python
for i_s in T.serial(num_iters):
    for i_h in T.Parallel(num_heads):
        g_fragment[i_h] = g[
            0,
            seq_end_idx - i_s * chunk_size - 1,
            i_h,
        ]
    for i_h in T.Parallel(num_heads):
        g_cumsum[i_h] += g_fragment[i_h]
    for i_h in T.Parallel(num_heads):
        if g_cumsum[i_h] < threshold and n_fragment[i_h] == num_iters:
            n_fragment[i_h] = i_s + 1
            f_fragment[i_h] = False
```

- **Threshold constant: `-10.0`**, default of `warmup_threshold` (`cp_context.py:156`) and of `threshold` (`cp_fwd.py:86`, `:230`, `:95-104`).
- **Units:** `g` is already the **chunk-local cumulative log-decay** (the pipeline applies `chunk_local_cumsum` first, `chunk/__init__.py:47-51`, and passes the result into CP, `:60-66`). Reading `g` at the *last* token of each chunk gives that chunk's total log-decay, so `g_cumsum` is the log of the fraction retained across the trailing `n` chunks. `g_cumsum < -10` means the retained fraction is below `e^-10 ≈ 4.5e-5`. **[FACT for the constant and the sum; INFERENCE for "retained fraction" as the physical reading]**
- `n_fragment` (= `num_warmup_chunks`) is the number of trailing chunks to process; `f_fragment` (= `fallback_mask`) is `False` when the decay took over, `True` when it never did within the segment.
- `get_warmup_chunks_bidi` (`cp_fwd.py:119-264`) runs the scan in both directions and takes `num_warmup_h = max(n_fwd, n_bwd)` (`cp_fwd.py:210-219`), plus separate forward/backward fallback masks.

So: **cut = chunk-count heuristic; correctness window = gate threshold −10.0.** The two are independent.

---

## 2. Boundary state correction

Scheme: **two passes — a parallel per-segment state pass, then a small sequential scan over segment boundaries** (not an associative scan, not a lookback). **[FACT]**

`cp_context.py:200-227` calls the warmup pass with `initial_state=None` and `output_final_state=True`, then the correction:

```python
_, ht, mt = fused_gdr_h(
    k=k, v=v, a=a, g=g, b=b,
    initial_state=None,
    output_final_state=True,
    output_h=False,
    cu_seqlens=cp_cu_seqlens,
    num_warmup_chunks=num_warmup_chunks,
    ...
)  # [cp_batch_size, num_v_heads, k_head_dim, v_head_dim]

cp_h0 = correct_initial_states(
    raw_h0=raw_h0, ht_buffer=ht, mt_buffer=mt,
    fallback_mask=fallback_mask, seq_map_r2c=seq_map_r2c, ...
)
```

The scan walks the CP segments of one raw sequence in order, carrying `h_fragment`. Per segment it writes the carried state into `cp_h0[idx]`, then updates the carry from `ht_buffer[idx]`, `cp_fwd.py:324-361`:

```python
for i_s in T.Pipelined(num_iters - 1, num_stages=1):
    idx = seq_start_idx + i_s
    T.copy(h_fragment, cp_h0[idx, bh, ...])          # cp_h0[segment] = incoming state
    T.copy(ht_buffer[idx, bh, ...], h_shared)        # this segment's end-state from zero-start
    T.copy(mt_buffer[idx, bh, 0:DK, 0:DK], m_shared) # segment transition matrix M
    if fallback_mask[idx, bh]:
        T.copy(h_fragment, hd_shared)
    T.copy(h_shared, h_fragment)
    if fallback_mask[idx, bh]:
        # h_fragment = ht + M @ h_prev   (variants select the operand order)
        T.gemm(hd_shared, m_shared, h_fragment, transpose_B=True, clear_accum=False)
```

**The correction is a matrix operation, not an elementwise add.** When `fallback_mask=True` (the gate did not decay enough inside the segment), the true end state is affine, `S_end = M · S_start + C`, so the correction is `h = ht + M @ h` with `M ∈ R^{DK×DK}` (`cp_fwd.py:348-361`). When `fallback_mask=False`, the previous state's contribution is below `e^-10` and is dropped: `h = ht` only. The `M=0` case is produced by `prepare_h.py:323-326` (`else` branch), so the branch is uniform. **[FACT]**

The `M` matrix is built in the same warmup kernel (`prepare_h.py`). For each warmup chunk it accumulates `M += X^T @ (K @ M)` with `X = -β ⊙ (A^T K)` (`prepare_h.py:272-314`), starting from `M_L = I`, `M_R = shift-I` (`prepare_h.py:254-259`), then scales by `exp2(Σ g_last)` (`prepare_h.py:318-322`). `calc_mt` is true only when the warmup covers the whole segment (`prepare_h.py:115-121`), i.e. exactly the fallback case. In CP mode `ht` and `mt` are stored in the input dtype (bf16/fp16), `ht_dtype = k.dtype if is_cp else torch.float32` (`prepare_h.py:630`).

The forward test `test_fwd_auto_cp` (`tests/test_gdr_unit.py:533-575`) checks CP and non-CP both against the FP64 oracle at **`RTOL = 0.02`** (`:23`, `:192-203`), on a 16384-token sequence (both fixed and varlen). So the CP result is validated as equivalent to the exact recurrence to a 2%-relative tolerance, not bit-exact.

---

## 3. What is fused in `fused_fwd.py`

FLA's reference decomposition (`tests/ref_gdr.py`) is the standard four-kernel chain: `torch_kkt_fwd` (`:77`) → `torch_solve` (`:116`) → `torch_w_u_fwd` (`:143`) → `torch_chunk_gdr_fwd` (`:186`) → `torch_chunk_o_fwd` (`:248`), i.e. `prepare_wy_wu -> fwd_h -> chunk_o`.

FlashQLA fuses this into a small number of kernels (`chunk/__init__.py:47-87`):

1. `chunk_local_cumsum` (standalone).
2. `kkt_solve` — **fuses** `torch_kkt_fwd` + `torch_solve` into one kernel (`kkt_solve.py`, returning `A = inv(I + StrictLower(β ⊙ K Kᵀ))`). It builds `A = K Kᵀ`, scales rows by `β`, sets `A = I + StrictLower(A)`, then inverts a 32×32 lower-triangular matrix via two 16×16 diagonal inversions plus one 16×16 sub-diagonal coupling block (`kkt_solve.py:104-183`), `threads=128` (`:207`), `async_copy` for K (`:87`).
3. `fused_gdr_h` (`prepare_h.py`) — **fuses** the w/u reconstruction with `fwd_h`: computes `X = -β ⊙ (Aᵀ K)` and `Y = g_last ⊙ (K S) - (g_last/g) ⊙ V`, updates `S = g_last·S + Xᵀ Y`, and emits `h`, `ht`, and `M` (`prepare_h.py:190-326`). It does not materialize `w`/`u` to global memory.
4. `fused_gdr_fwd` (`fused_fwd.py`) — **fuses** `fwd_h` and `chunk_o` in one kernel: it reconstructs `U = K S`, `W = V - exp(g)⊙U`, `Vd = Ag @ W`, `V' = (g_last/g)⊙Vd`, updates `S`, and simultaneously computes the output `P = Q Kᵀ`, `Ag = G ⊙ A ⊙ β`, `O = scale·exp(g)⊙(Q S) + (scale·G⊙P) @ Vd` (`fused_fwd.py:214-440`). `U = K S` is shared between the state update and the output.

TileLang structure / warp specialization (sm120 `fused_fwd.py`): one `T.Kernel` per `(DV-block, CP-segment, head)` with `threads=512` (`:107`), split into four 128-thread groups plus a 32-thread epilogue. **[FACT]**

| group | threads | role | nreg |
|---|---|---|---|
| S/state | `tx < 128` | init state; `S *= exp(g_last)`; `S += Kᵀ V'`; store final `ht` | `CONSUMER_S_NREG = 128` (`:192`) |
| V | `128..255` | precompute `exp(g)`, `exp(g_last-g)`; `U=K S`; `W=V-exp(g)U`; `Vd=Ag@W`; `V'` | `CONSUMER_V_NREG = 128` (`:191`) |
| O | `256..383` | `P=QKᵀ`; build `G`,`Ag`; `O=Q S`; `O=scale·exp(g)O + scale·G⊙P @ Vd`; store `O` | `CONSUMER_O_NREG = 128` (`:193`) |
| producer ×3 + epilogue ×1 | `>=384` | TMA loads Q,K / V,β / A,γ; final warp stores O/h | `PRODUCER_NREG = 32` (`:190`) |

The pipeline is manual barrier + double buffering: `q,k,v,a` are `alloc_shared((2, block_S, ...))` and `g,b` are `(2, block_S)` (`fused_fwd.py:143-148`), with named barriers (`data_is_ready` arrive 96/stage, `data_is_free` arrive 384/stage, plus `bar_0..bar_5`; `:175-184`) and `T.set_max_nreg` register reallocation (`:195,272,354,442`). Data movement is TMA (`T.tma_copy`, `:453-525`), not `cp.async`.

---

## 4. The algebraic reformulation

**[FACT]** The README states the claim without detail (`README.md:20`):

> "**Hardware-friendly algebraic reformulation**. We reformulate the forward and backward flows of GDN Chunked Prefill to a certain extent, effectively reducing Tensor Core, CUDA Core, and SFU overhead without sacrificing numerical precision."

The observable code-level changes versus FLA's reference chain:

- **`exp2` everywhere, never `exp`.** Every decay factor is `T.exp2(x * 1.442695)` (log₂e), e.g. `fused_fwd.py:285,289,382`; `prepare_h.py:210,319,352,356,418`. Kernels compile with `TL_ENABLE_FAST_MATH: True` (`fused_fwd.py:18`, `prepare_h.py:13`, `kkt_solve.py:17`). The ratio `g_last/g_j` is computed as `exp2((g_last - g_j)·log₂e)`, i.e. a subtraction in log space rather than a division (`fused_fwd.py:289`, `prepare_h.py:352`). **[FACT]**
- **The triangular inverse is separated from the decay.** FLA bakes the decay mask into the KKT matrix before the solve (`ref_gdr.py:103-108`: `attn = (β⊙k)kᵀ * decay_mask`). FlashQLA's `kkt_solve` inverts the *undecayed* `I + StrictLower(β⊙KKᵀ)` (`kkt_solve.py:104-119`), and the decay `G` is applied afterwards, once, as an elementwise factor: `Ag = G * A * β` with `G = lower(exp2(g_s - g_t))` (`fused_fwd.py:376-395`). **[FACT]**
- **The A-matrix product is moved early.** FLA keeps `w = A@(β g.exp() k)`, `u = A@(β v)` and forms `v_new = u - w@S` (`ref_gdr.py:225`), using raw `v` in the output attention (`ref_gdr.py:294`). FlashQLA forms `W = V - exp(g)⊙(K S)` and folds `A` into `Vd = Ag @ W` (`fused_fwd.py:321-339`), then uses `Vd` in *both* the state update and the output attention (`:344-347`, `:430`). This removes the separate `w`/`u` materialization and the raw-`v` GEMM path. **[FACT that the code does this; the algebraic equivalence to FLA is INFERENCE — it is the standard WY/delta-rule identity, and the tests assert only 2%-relative agreement with the FP64 oracle.]**
- **`fwd_h` and `chunk_o` share one `K S` product** in one kernel (`fused_fwd.py:302-317` and `:399-414`), instead of FLA's two kernels reading a materialized `h`.

The README attributes the savings to Tensor Core / CUDA Core / SFU. The exact per-op accounting is **not stated in-tree [UNVERIFIED]**; from the code the plausible mapping is: SFU ← uniform `exp2` (`g.exp()` appears many times per element in `ref_gdr.py:166,227,230,293`); CUDA Core ← log-space subtraction replacing division and fewer elementwise passes; Tensor Core / memory ← no `w`/`u` materialization and one shared `K S`.

---

## 5. sm_120 specifics vs `blackwell/` and `hopper/`

`chunk/__init__.py` selects the arch and the chunk size (`:10-26`): hopper `CHUNK_SIZE = 64` (`:14`), blackwell (sm100/sm103) `64` (`:19`), **blackwell_sm120 `32`** (`:24`). The sm120 kernels hard-assert this: `fused_fwd.py:662`, `prepare_h.py:584`, `kkt_solve.py:287`, `cp_fwd.py:93` all `assert chunk_size == 32`.

| property | `blackwell_sm120` | `blackwell` (sm100/103) | `hopper` (sm90) |
|---|---|---|---|
| chunk size | **32** | 64 | 64 |
| `block_DV` | **hardcoded 64** (`fused_fwd.py:734`) | `128` if `grid>=TARGET_NUM_CTAS` else `64` (`:930-933`) | `128` / `64` / `32` three tiers (`:731-736`) |
| Tensor Core path | MMA only, **no TMEM** (`alloc_tmem` count 0) | **tcgen05/TMEM** (`alloc_tmem` ×6, e.g. `v_tmem`, `o_tmem` `:194,196`) | MMA only, no TMEM |
| data movement | TMA + manual barriers | TMA | TMA |
| threads / warpgroups | 512 = 3 consumer WGs + 3 producer warps + 1 epilogue warp | 512, same split | 512, same split |
| register hints (fwd) | S 128, V 128, O 128, producer 32 (`fused_fwd.py:190-193`) | (WG-specific, TMEM accumulators) | S 160, V 128, O 128, producer 32 (`hopper/fused_fwd.py:190-193`) |
| `TARGET_NUM_CTAS` | defined (`:11-12`) but **unused** in the wrapper | used for `block_DV` | used for `block_DV` |
| `prepare_h` | `threads=512`, `num_stages=2`, nreg S 168 / X 160 / Y 160 / producer 24 (`prepare_h.py:36,92,175-178`), TMA | TMEM | — |
| `kkt_solve` | `threads=128` (`kkt_solve.py:207`), `async_copy` + `ptx_wait_group` (`:87,106`), 16×16 block inversion for chunk 32 | chunk 64 variant | chunk 64 variant |
| numerics | q/k/v/a/o bf16|fp16; `g`,`β` fp32; accum fp32; `ht`/`mt` **bf16 in CP mode** (`prepare_h.py:630`); `mt` consumed at fp32 accum (`cp_fwd.py:511`) | same family | same family |

**[FACT]** all rows above are read from the cited lines. The reason sm120 drops to chunk 32 and block_DV 64 while dropping TMEM is not stated in-tree **[UNVERIFIED]**; the sm120 kernel is a plain-MMA port (Blackwell consumer-GPU MMA, no tcgen05) and the smaller tile keeps register/shared footprint within the 128 KiB/SM limit of sm120.

---

## 6. Measured numbers on RTX 5090

Harness: `benchmark/bench_gated_delta_rule.py`; config `Warmup=10, Repeats=100, Backend=cudagraph`; libraries torch 2.12.1+cu130, fla 0.5.2, flashinfer 0.6.17, tilelang 0.1.13 (`benchmark_results_5090.txt:1-5`). Inputs: head dim 128; `g = logsigmoid(randn)/16` with 25% of heads forced to `g=0` (SWA-hybrid simulation, `bench_gated_delta_rule.py:211-239`). The NInfer-relevant head config is **27B TP1, `h_qk=16, h_v=48`** (`:79`), matching NInfer's `H_v=48, state 128x128`.

Forward rows for that config, quoted verbatim (`benchmark_results_5090.txt:135-153`):

```
27B TP1          1x32768              16    48       2.195ms     2.895ms     4.366ms     1.99x    1.32x
27B TP1          1x16384              16    48       1.157ms     1.468ms     2.147ms     1.86x    1.27x
27B TP1          1x8192               16    48       0.607ms     0.814ms     1.068ms     1.76x    1.34x
27B TP1          1x4096               16    48       0.339ms     0.430ms     0.528ms     1.56x    1.27x
27B TP1          1x2048               16    48       0.143ms     0.284ms     0.239ms     1.67x    1.98x
27B TP1          28672+4096           16    48       2.171ms     3.605ms     4.492ms     2.07x    1.66x
27B TP1          24576+8192           16    48       1.773ms     3.091ms     4.513ms     2.54x    1.74x
27B TP1          16384+16384          16    48       2.247ms     2.069ms     4.828ms     2.15x    0.92x
```

Column order (header `:9`): `flash_qla [fwd]`, `FI [fwd]`, `FLA [fwd]`, `vs FLA`, `vs FI`.

For `h_v=64` (`397B/122B TP1`, `:73-91`) the single-sequence speedups are larger and steadier: `1x32768` = `2.337ms` vs FLA `5.701ms` (**2.44x**) and vs FI `4.139ms` (**1.77x**); `1x16384` = `1.174ms` vs `2.803ms` (**2.39x**). At `h_v=48` the low-head/short cases lose to FlashInfer (`2048x4`: `0.98x`, `16384+16384`: `0.92x` vs FI), consistent with the CP overhead the `_calc_cp_seqs` comments describe (`cp_context.py:102-104`). The headline claim is **2–3× forward over FLA and ~1.3–1.8× over FlashInfer** at long sequence / high head count. **[FACT from the file; the "headline" framing is the README's `:14`.]**

The backward table (`:198-238`) is FLA-only, speedups ~0.6×–3.2× depending on head count, not needed for the forward port.

---

## 7. Portability to hand-written CUDA

**[INFERENCE]**, grounded in the fact/inference split above.

**Portable without TileLang (algorithmic):**

- The **CP segmentation** (`cp_context.py:63-99`) is host-side arithmetic plus a `cu_seqlens`-style array. Trivially portable; recompute `max_local_chunks` from `H`, total chunks, and SM count.
- The **gate warmup scan** (`cp_fwd.py:52-76`) is a tiny kernel: one block per CP segment (or one warp), `num_heads` way, serial over chunks reading one `g` per chunk. Hand CUDA, no warp specialization needed.
- The **boundary correction scan** (`cp_fwd.py:324-373`) is a small kernel: grid over `(raw_seq, head, DV-block)`, serial over the sequence's segments carrying a `DK×DV` state, writing `cp_h0[segment]` and applying `h = ht + M @ h` when the fallback mask is set. Existing MMA helpers suffice; no TMA/barrier choreography required.
- Treating each CP segment as a pseudo-batch row so the main kernel's grid grows by the segment count (`fused_fwd.py:107` uses `batch_size`, which is the number of `cu_seqlens` entries = segment count) is a launch-shape change, not a codegen dependency.

**Depends on TileLang / warp-specialization codegen:**

- The fused `fwd_h + chunk_o` single kernel with 3 consumer warpgroups + 3 TMA producer warps + epilogue warp and hand-placed named barriers (`fused_fwd.py:175-635`) is the class of thing TileLang exists to generate. A hand-written kernel can reproduce it, but that is original kernel engineering, not a port of the CP scheme.
- `alloc_barrier`, `set_max_nreg`, `use_swizzle`, `tma_copy` have direct CUDA counterparts (`cuda::barrier`, `setmaxnreg`, `cp.async.bulk.tensor`, block swizzle) so nothing is *conceptually* blocked; the cost is implementation.

**Minimal viable gate-driven CP on a kernel that already keeps the running state in registers and materializes `h_chunk`:**

1. **Segment**: host-side cut at chunk boundaries using the `_calc_cp_seqs` formula and the `Be*H` gate.
2. **Warmup pass**: run the existing state-passing stage per segment, but **starting from zero at the last `n` chunks** (`n` from the gate scan), producing `ht[segment]`. This is exactly `fused_gdr_h(initial_state=None, num_warmup_chunks=n)`.
3. **Transition matrix `M`**: when `fallback_mask=True`, accumulate the segment's affine transition `M` alongside `ht`. This is the one genuinely new piece of math. FlashQLA's recipe is explicit (`prepare_h.py:272-322`): with `X = -β ⊙ (AᵀK)` and `Y` the v-side feed, `M += Xᵀ @ (K @ M)`, then `M *= exp(Σ g_last)`. If NInfer can obtain `A` (its `prepare_wy_wu` already produces the equivalent), `M` is two extra GEMMs per warmup chunk plus a `DK×DK` accumulator.
4. **Correction**: a kernel that carries the segment-boundary state and applies `h = ht + M @ h` (fallback) or `h = ht` (no fallback), writing `cp_h0[segment]`.
5. **Main pass**: existing `prepare_wy_wu -> state_passing -> output` run per segment with `state_in = cp_h0[segment]`; only the grid/lookup changes (segment, not sequence, indexes the row).

**Judgement:** the split + warmup + lookback-correction **without** the exact `M` path is **days** of work — the pieces are small and reuse existing stages. The exact `M`-matrix correction is the **weeks** part: it needs a new `DK×DK` transition accumulation, a fallback matmul, and a precision decision (FlashQLA uses bf16 `M` at fp32 accumulation, `prepare_h.py:630` / `cp_fwd.py:511`). If NInfer instead **recomputes fallback segments from their true predecessor boundary** (correct but serializing those segments), the `M` path can be deferred and the whole split still lands in days — at the cost of losing parallelism exactly when the gate is weak.

---

## Fact vs inference (explicit separation)

**Quoted fact** (source-cited above): the `max_local_chunks` formula and chunk-boundary cut; the `Be*H` CP-enable gate; the `-10.0` threshold and the backward chunk-boundary `g` accumulation; the two-pass warmup-then-scan boundary scheme; the `M @ h` matrix correction gated on `fallback_mask`; the `fused_gdr_h`/`fused_gdr_fwd` fusion of w/u + fwd_h + chunk_o; the `exp2`/log-space-subtraction idiom; the undecayed KKT inverse plus post-hoc `G`; the sm120 chunk 32 / no-TMEM / `block_DV=64` / `threads=512` / 4-group configuration; the 5090 benchmark rows; `RTOL=0.02` CP equivalence.

**Inference** (my reading, not stated in-tree): the `e^-10` retained-fraction interpretation; the equivalence of the `Vd = Ag@W` reformulation to FLA's `v_new`/raw-`v` split; the Tensor Core / CUDA Core / SFU attribution of the README's savings; the portability/effort estimate and the minimal-port recipe.

**Unverified:** the reason sm120 forgoes TMEM and halves chunk size; the precise per-op overhead accounting behind the README claim; whether `fused_gdr_h`'s `M` is accumulated over the whole warmup window or only the skipped prefix (the code accumulates `M += Xᵀ(K M)` for every warmup chunk, so it is the transition over the warmup window — this is consistent but not commented); whether NInfer's `prepare_wy_wu` exposes an `A` directly reusable for `M`.

---

## What this means for NInfer's kernel

NInfer stages: `prepare_wy_wu -> state_passing -> output`; chunk 64; state 128×128; `H_v=48`; grid 384; register-resident state; `h_chunk` materialized. Grounding from the tree: `src/ops/linear_attention/gated_delta_net/chunked/state_passing.cuh:21-22` (`static_assert(kChunkSize == 64)`, `kStateDim == 128`), state-passing geometries NStrip 16 (8 warps) / 32 (16 warps) `:24-47`, `h_chunk` written to global `:177,358-372`, and `output.cuh:11` documents `out = scale * (exp(g) * q @ h_chunk^T + A @ v_new)` — i.e. NInfer is the **FLA-shaped** pipeline, not FlashQLA's fused `Vd`-in-output reformulation.

- **CP is additive, not a rewrite.** The cut/warmup/correction layers sit *around* the three existing stages. `state_passing` already produces per-chunk `h_chunk` and a final `state_out`, so the warmup pass is `state_passing` over a trailing sub-range starting from zero.
- **NInfer already has the hard part FlashQLA lacks on sm120:** a materialized `h_chunk` gives the chunk states for free, so a **simpler** boundary correction is available — the segment initial state is the predecessor's final state, and the affine transition `M` can be read off chunk-wise from the materialized states rather than only over a warmup window. FlashQLA only needs `M` over the warmup tail; NInfer can compute the equivalent over the full segment.
- **`A` is already produced** by `prepare_wy_wu`; reuse it for `X = -β ⊙ (AᵀK)` instead of re-deriving (this part is FlashQLA-specific and must be checked against NInfer's `A` convention — `kkt_solve` inverts the *undecayed* matrix while FLA bakes decay in, and NInfer's `output.cuh` formula suggests it follows FLA).
- **Grid impact:** NInfer's grid 384 = `H_v(48) × D_STRIPS(8)` for the value-strip-16 geometry. With CP, the row count becomes `segments × 48` and the grid scales toward the 170-SM RTX 5090's occupancy; this is the same mechanism as FlashQLA's segment-as-batch trick.
- **Chunk-size mismatch:** FlashQLA sm120 uses chunk 32 and its warmup threshold is tuned to that (`assert chunk_size == 32`, `cp_fwd.py:93`). NInfer at chunk 64 must re-derive the relationship between `num_warmup_chunks` and `threshold`; the scan reads one `g` per *chunk*, so the same −10.0 log-threshold yields fewer chunks of margin at chunk 64. It is directly reusable but the constant should be re-validated.
- **Precision:** plan for `M` (and `ht`) in bf16 with fp32 accumulation to match FlashQLA; NInfer's state is 128×128 fp32/bfloat16, so the `M` accumulator is the same shape class.

---

## Recommendation / open questions

**Recommendation.** Port the scheme in three independently verifiable units, in this order:

1. **Host segmentation + warmup scan** (threshold −10.0, chunk-boundary `g`), validated by asserting `use_cp` decisions and `num_warmup_chunks` against a reimplementation of `_calc_cp_seqs`/`get_warmup_chunks`.
2. **Boundary correction with a serial fallback**, not the `M` path first: run each segment from zero, then fix boundaries by having fallback segments start from the true predecessor's final state (recomputing if needed). This lands the split in days and is exact.
3. **Exact `M`-matrix correction** only if profiling shows fallback segments dominate. This is the weeks-scale piece.

**Open questions.**

1. Does NInfer's `prepare_wy_wu` yield the undecayed `A` or the decay-baked `A`? The `M` recipe depends on which (`kkt_solve.py:104-119` vs `ref_gdr.py:103-108`).
2. Is the `M` in `fused_gdr_h` the transition over the **warmup window** or the **skipped prefix**? The code accumulates over every warmup chunk, implying the former, but no comment states it and the correction identity only needs the prefix transition. **[UNVERIFIED]**
3. At chunk 64, does the −10.0 threshold still give the same boundary accuracy as the sm120 chunk-32 path, or does the coarser chunk require a tighter threshold?
4. Can NInfer's existing materialized `h_chunk` supply the affine transition without a dedicated `M` kernel (chunk-wise products of the per-chunk transition), making the "weeks" piece unnecessary?
5. Does the 5090 benchmark's `g = logsigmoid/16` with 25% zeroed heads exercise the fallback path enough to trust the measured speedups for NInfer's real `g` distribution?
