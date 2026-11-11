# The ungated FP8 A8 TMA bands, and the measurement behind them

This is the authority for the median tile figures quoted in the nine shape files under
`src/ops/linear/**/fp8/shapes/`. Those comments carry numbers — `1.107`, `1.270`, `1.288`,
`1.270` — whose *meaning* (what was measured, on what, with what metric) lives here and nowhere
else. The harness that took them was withdrawn with the PORT-DISPATCH gates it had flipped, so this
document is the only surviving record of their configuration. Read it before changing a band ladder.

The band values themselves are gated: `tools/release/check_fp8_band_ladders.py` fails if a ladder
stops selecting what it selected at its recorded base revision. This document explains *why* the
bands are what they are; that gate is what stops them moving silently.

## What was measured

`NINFER_FP8_TMA_TILE` pins one tile for **both** arms, so each is measured at a matched tile rather
than at whatever the K=6144 ladder would otherwise have chosen. Seven interleaved repeats, oracle
GREEN on both arms of all five tiles.

Measured at **K=6144 with the linear_add residual epilogue**. The gates that use these tiles run at
K=5120 with different epilogues and different partials capacities, so transport-at-matched-tile is
the same *question* but not a measurement at those gates' own shapes. Ungating was decided on the
oracle first and the transport second; a gate is a correctness decision before it is a performance one.

## The result

| tile | median | min | max | bands where MMA wins |
|---|---|---|---|---|
| 64x256x128 (2,4,2,1) | 1.270 | 1.000 | 1.414 | 0 of 8 |
| 96x256x128 (3,4,2,1) | 1.288 | 1.056 | 1.370 | 0 of 8 |
| 192x128x128 (3,4,2,1) | 1.241 | 1.058 | 1.284 | 0 of 8 |
| 64x128x128 (2,4,**2**,1) | 1.107 | **0.943** | 1.313 | **1 of 9** (768) |
| 128x128x128 (2,4,**2**,1) split | 1.047 | **0.952** | 1.284 | **2 of 8** (448, 512) |

Ratios are MMA/TMA, so above 1.0 is TMA ahead. **TMA is ahead on the median at every tile, but it does
not win every band**: MMA is faster at 448 and 512 on the split 128x128 tile (by 2.7 % and 4.8 %) and
at 768 on 64x128 (by 5.7 %).

An earlier reading of this — "TMA wins every band" — was true only of the three tiles measured first
and was wrong as a general result. The two 128-row tiles are the ones that do not hold it, and they are
marked in the table.

## The control that makes it admissible

A forced-tile override that silently did nothing would still pass the oracle on the default ladder,
because the oracle would be running the tiles it always runs. The three tiles therefore have to
produce three *clearly distinct* timing profiles, and they do: at 64 tokens TMA reads
44.4 / 60.8 / 62.8 µs. Identical profiles would have meant the override never fired and the GREEN was
vacuous.

This is why the table can be read as evidence about the transport at all.

## Two structural limits, both measured

1. **The split 128x128 tile cannot be measured below 193 tokens.** `allocate_fp8_a8_workspace` sizes
   partials from the *current* invocation's token count (`src/ops/linear/fp8/fp8_linear_add_a8.cu`),
   so a measurement at a token count that selects a split tile allocates a buffer that does not fit
   it.

2. **Forwarding `scratch.partials` through `launch_fp8_a8` was a prerequisite of ungating, not a
   follow-up to it.** Every Windows-gated branch instantiates non-split schedules, so
   `if constexpr (kSplitWaveCtas > 0)` discarded the split block and the null `partials` was never
   dereferenced — harmless at the time. Four of those ladders carry split-K tiles in their `#else`
   arms, and `launch_fp8_a8` did not forward `scratch.partials` the way its TMA sibling
   `launch_fp8_a8_tma` does. Ungating a shape file whose ladder includes `Bulk`/`MidBulk` would have
   reached `FP8 split-K requires aligned caller partials` at runtime on first use. It fails loudly
   rather than corrupting, but it presents as a mysterious per-shape failure partway through ungating.

## Why ungating is per route

The SwiGLU ladder was PORT-DISPATCH-gated on Windows because upstream's TMA route faulted there with
`cudaErrorIllegalInstruction`. That fault was a descriptor ABI defect — `alignas(128)` on a by-value
kernel parameter, which MSVC's host stub rejected with `C2719` — and is resolved without downgrading
via `common::TmaKernelParam<Fp8TmaDescriptors>`, keeping `alignas(128)` and `.param .align 128` in
PTX. PyGPUkit #107 reports the same misalignment defect on this GPU and OS.

The ungating decision rests on three pieces of evidence per route, not on the transport being faster
in general:

- every tile a ladder selects is oracle-checked on both arms, at the token counts it serves;
- TMA leads on the median at the tiles that ladder selects, and the bands where MMA wins on the
  64x128 tile are at large token counts, outside the split-tile region these ladders use;
- `scratch.partials` is allocated for the whole selection band, so no rung can select a split tile
  with a null buffer.

A static assertion in `fp8_a8_tma_mma.cuh` binds the descriptor alignment comment to the compiler, so
that a change to either fails the build rather than the documentation.

## What is recorded here, and what is not

Figures, configuration, the control and the two structural limits — that is the whole set the shape
comments depend on. The rest of the ungating work (the per-site breakdown, the nine files, the
withdrawn harness) was working-out, and its conclusions are in the code and in the gate.