# FlashQLA fused GDN forward: implementation spec (C=64 target)

Status: implementation spec, 2026-09-19. Extracted from `QwenLM/FlashQLA@main` raw sources; line
numbers are theirs. Companion to `docs/research/gdn-flashqla-context-parallelism.md` (the survey).
Nothing here is in upstream Neroued/ninfer, and FlashQLA is TileLang/Python - this is a port target,
not a dependency.

Premise: the sm120 package is chunk-size **32**; the C=64 math and the 64x64 inversion exist in
`hopper/` and `blackwell/`. A C=64 sm120 port takes the sm120 pipeline shape (no TMEM, TMA) with
hopper's C=64 inversion and register hints.

## 1. Per-chunk mathematics

Symbols (B=1): `q,k [Heads,128]` bf16; `v [H_v,128]` bf16; `g` = **inclusive per-chunk cumsum** of
the raw log-decay, fp32; `beta` fp32; `S [128,128]` fp32 registers (bf16 mirror `S_b` for GEMMs);
`A`(=Ã) bf16; `scale = 128^-1/2`. `e_s = exp(g_s)`, `e_last = exp(g_{C-1})`.

```
Ã  = (I + strictLower(diag(beta) K K^T))^-1        # kkt_solve; NO decay inside
U  = K S
W  = V - diag(e) U
Ag = Ã ⊙ (e_s e_t^-1) ⊙ beta_t                     # = A_ref diag(beta)
Vd = Ag W                                          # = v_new (the delta)
V' = diag(e_last e_t^-1) Vd
S' = diag(e_last) S + K^T V'                       # state recurrence
O  = scale ( E (Q S) + (G ⊙ Q K^T) Vd ),  G[s,t]=e_s/e_t for s>=t else 0
```

`Vd` is computed once and used twice: decayed as `V'` in the state update, and directly in the
intra-chunk causal attention. This is what lets the kernel drop FLA's `w`/`u` materialization:
`u - w S = A_ref diag(beta)(v - E K S) = Ag W`.

## 2. Pipeline replacement map

| FLA (`ref_gdr.py`) | FlashQLA |
|---|---|
| `torch_kkt_fwd` + `torch_solve` | `kkt_solve` (fused hierarchical inversion) |
| `torch_w_u_fwd` | V-role: `U=KS`, `W=V-g*U`, `Vd=Ag@W`, `V'=g_rev*Vd` |
| `torch_chunk_gdr_fwd` | S-role: `S <- e_last S + K^T V'` |
| `torch_chunk_o_fwd` | O-role: `P=QK^T`, `G`, `Ag`, `Pg=scale*G*P`, `O=scale*E*(Q S)+Pg@Vd` |

Our engine is FLA-shaped (`prepare_wy_wu -> state_passing -> output`) and materializes W/U/v_new and
h_chunk; FlashQLA materializes none of them. That materialization is the target.

## 3. Kernel structure (sm120, C=32 source; C=64 deltas below)

- Grid `ceildiv(DV, block_DV) * B * H`, **threads=512** (`fused_fwd.py:107`); sm120 pins
  `block_DV=64` -> 96 CTAs at B=1,H=48; `T.use_swizzle(10)`.
- Four 128-thread groups: **S** (`tx<128`, state), **V** (`128..256`, U/W/Vd/V'), **O**
  (`256..384`, P/G/Ag/Pg/O), **producer** (`384..512`, 3x32 TMA groups for Q,K / V,beta / A,gamma,
  plus an O,S store group).
- `nreg`: S/V/O = 128 (C=32 sm120; hopper C=64 uses S=160), producer = 32; `T.set_max_nreg`.
- Shared (C=32): q,k `(2,C,128)` 16 KB each; v `(2,C,64)`; a `(2,C,C)`; g,b `(2,C)` fp32;
  h_shared `(128,64)` 16 KB; vd/vn `(C,64)` each; p `(C,C)`; g_exp/g_rev_exp `(C)` fp32.
- Fragments fp32: `h_fragment(128,block_DV)` (64 regs/thread on 128 S-threads), `o/v/u(C,block_DV)`,
  `p/a/g(C,C)`.
- Named barriers (`:175-184`): `data_is_ready` (3x32 arrival), `data_is_free` (3x128), plus
  `bar_0..bar_5` rendezvous (S/V/O handoffs). Stage parity `(i_s//2)%2`; intra-iteration `i_s%2`.
- Order per chunk: S copies state->h_shared and V computes g_exp/g_rev (bar_1); V does `U=K S`, S
  scales `S<-e_last S`; V does `W`, O does `G,Ag`, O does `O=Q S`; O computes `Pg`, scales `O`; V
  does `Vd=Ag@W` -> vd_shared -> `V'` -> vn_shared; S does `S+=K^T V'`; O does `O+=Pg@Vd`, stores O.
- TMA loads for q,k,v,a with OOB masked scalar fallback; g,b scalar-loaded to fp32 shared.

## 4. C=64 deltas (required for our chunk size)

1. **`kkt_solve` inversion becomes two-level**: 4x16x16 diagonal inversions + 2 couplings + two
   32x32 assemblies + one extra 32x32 coupling (`hopper/kkt_solve.py`). Buffers `a16i_shared(4,17,16)`,
   `a16o_shared(2,17,16)`; the sm120 C=32 code's `(2,17,16)`/`(1,17,16)` is not enough.
2. `fused_fwd`: tiles scale with `block_S`; set `CONSUMER_S_NREG=160`, restore hopper's
   `T.fence_proxy_async()` TMA fences (6 sites), adaptive `block_DV` (128/64/32 by CTA count).
3. `prepare_h`: split `m_shared_L/R (128,64)` with `bar_2=384` (hopper) vs sm120's merged scheme.
4. `cp_fwd`: both `assert chunk_size == 32` -> 64; `max_local_tokens` scales.
5. Register budget: at C=64 the O role holds `p,a,g` each `(64,64)` fp32 + `o(64,block_DV)`;
   keep 128 regs or drop `block_DV` to 32.

## 5. CP

`M = I` initially; per warmup chunk `Z = K M`, `M += X^T Z` with `X = -diag(beta) Ã^T K`, so
`M <- (I - K^T Ã diag(beta) K) M`; after the loop `M *= exp(sum_chunks g_last)`. Correction scan:
`x = ht` when the fallback mask is clear, `x = ht + M @ x` when set. Warmup window: backward scan of
`g` at chunk ends, stop at `threshold = -10.0`. `ht`/`mt` are bf16 under CP.

## 6. Numerics

q,k,v,a bf16; g,beta -> fp32 shared; all accum fp32; `S_b`, vd, vn, p, a_shared bf16; `A` stored
bf16; per-chunk `h` bf16; `ht` fp32 (bf16 under CP). Reference `tests/ref_gdr.py` is fp64;
FlashQLA's own equivalence tolerance is `RTOL = 0.02` on global L2 norms.

## 7. What this means for the port (our shapes: C=64, Hg=16, H=48, K=V=128, sm_120)

- This is **one new fused kernel** replacing `state_passing` + `output` (76% of our pipeline), plus
  the C=64 `kkt_solve` upgrade inside `prepare_wy_wu`.
- It removes the `h_chunk` DRAM round-trip (~201 MB each way at T=8192), which ncu shows is what
  makes our `output` DRAM-bound at 80.65%.
- It is a large hand-CUDA job (warp specialization, TMA, barriers, 512 threads). Build it as an
  `op-development.md` 7.1 route transaction: oracle case first, one candidate matrix, qualify before
  timing, measure through the public Op.
- MISSING from the sources (agent-flagged): exact TileLang lowering of `set_max_nreg`, per-barrier
  arrive ownership, and the sm120-vs-sm90 MMA/TMA descriptor details. These are implementation
  choices for the port, not spec gaps.
