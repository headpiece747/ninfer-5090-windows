# Artifact conventions: matching the shipped models

A new conversion joins the shipped lanes only when it matches the properties below. They are
recorded here because two of them are invisible in any single recipe — they are visible only by
comparing an artifact against the two that ship — and because the third time a checkpoint with FP8
attention arrived, the choice had to be re-derived from scratch.

Everything here is a property of the shipped artifacts, measured or read from their directories.
[`qwen3.8-27b-artifact.md`](qwen3.8-27b-artifact.md) owns what those artifacts *contain*; this file
owns what a new one must *match*.

## 1. Text is all-NVFP4; a source's FP8 is re-encoded, not imported

| artifact | nvfp4 | fp8 | reach at fp8 KV |
|---|---:|---:|---:|
| `nvfp4qat.v3.ninfer` (QUASAR) | 13.75 GiB | **0.00 GiB** | 262,144 |
| `nvfp4full.v3.ninfer` | 13.47 GiB | **0.00 GiB** | 262,144 |
| `nvfp4swift.v3.ninfer` | 12.74 GiB | **0.00 GiB** | 262,144 |
| `nvfp4.v3.ninfer` (the port's earlier image, not shipped) | 7.84 GiB | 11.08 GiB | below the full context |

None of the three ships a single FP8 tensor. Their attention, GDN and MLP projections are all
NVFP4, and that is what keeps resident bytes low enough for the full context — see section 3.

Two routes reach it, and both are legitimate:

- **The source is already NVFP4 for every text linear.** QUASAR's source is
  `nvfp4-pack-quantized` with all 496 text sites quantized, so importing its packed-code and scale
  words bit-exactly yields an all-NVFP4 artifact. "No local encoder run and no calibration corpus
  are involved."
- **The source keeps some text projections in FP8.** NVFP4-full's source (`unsloth/Qwen3.8-27B-nvfp4`)
  carries 233 row-scaled FP8 matrices; Swift's ModelOpt checkpoint keeps its attention and GDN in
  E4M3. Both import the NVFP4 codes the source does have and **locally encode the rest to NVFP4**
  from the BF16 source, with the site input divisors either calibrated (NVFP4-full) or recovered
  from the checkpoint's own `input_scale` (Swift; the derivation is in the next subsection).

Re-encoding is not a concession. Measured on the fixed 1M corpus, Swift's re-encoded artifact scores
**4.68429** against the same recipe's FP8-importing build at **4.84938**: a per-tensor FP8 scale is
coarser than NVFP4's one scale per 16-element block, so the block scales more than pay for the
narrower codes. "Import every code word unchanged" preserves the producer's representation, which is
not the same thing as preserving the model's accuracy.

The rule follows: **import a source's NVFP4 codes where they exist and are structurally compatible;
locally encode to NVFP4 everywhere else. Do not import FP8 codes into a shipped artifact.** An
imported FP8 matrix is 8-bit where the shipped artifacts are 4-bit, on a memory-bound decode, and
it costs ladder steps at the context ceiling.

### The encoder, and how the activation divisor is obtained

`tools/convert/quantization/nvfp4.py` is the encoder (`NVFP4_MAXABS_DIVISOR_RNE_V1`), ported from
the fork that produced the all-NVFP4 artifacts. A recipe passes it directly —
`recipe.assign(..., method=nvfp4_maxabs)` — rather than through `METHODS`, because the module
imports `AuxiliaryValue` from `methods` and registering it there would be an import cycle.

It is not `grouped_absmax`. `Nvfp4Format` is a separate class from `QuantFormat`, and the block
scale is E4M3FN over 16-column groups under a global divisor, so the row-split quantizer cannot
produce `block_scale_k16_m128x4_v1`. Its two scales follow Transformer Engine's NVFP4 recipe:

```text
s_global = global_amax / (448 * 6)        s_block = (block_amax / 6) / s_global
```

**The activation divisor is derived, not calibrated.** The fork measures it with a forward pass over
a fixed corpus, which needs `transformers` and `accelerate`. A ModelOpt checkpoint already carries
the activation amax its own producer calibrated with, in the per-site `input_scale`:

| site | stored | divisor to bind |
|---|---|---|
| already NVFP4 | `input_scale = amax / (6 * 448)` | `d_x = 1 / input_scale` |
| FP8, being re-encoded | `input_scale = amax / 448` | `d_x = 6 / input_scale` |

The two differ only by NVFP4's factor of 6 in the global scale, and a ModelOpt checkpoint proves
both maxima are 448: its scaled codes saturate exactly 448.0 in the FP8 and the NVFP4 matrices
alike. The `1 / input_scale` form is already the divisor this port binds for imported NVFP4 sites,
which the published Swift artifact exercises end to end.

Prefer encoding from the checkpoint's **BF16** source when one exists — `ukisai/Swift-Qwen3.8-27b`
is ungated and 18 shards — rather than dequantizing FP8 and re-quantizing it, which carries the FP8
rounding into the NVFP4 result.

## 2. Imported words are copied, local words are proven

- Imported payloads are verified **word-for-word** against the source, including the fused-parent
  shared-divisor equality checks and the row transforms (attention q/gate per-head interleaving).
- Locally encoded payloads are verified against the **documented encoder profile** *and* an
  **independent decode oracle**, per [op-development.md](op-development.md).
- Both shipped artifacts ship a `verify_*` entry point that revalidates the complete ordered
  directory, both W8 endpoints against base rows, and the input divisors.

## 3. Keep device weights inside the envelope that reaches the full context

The engine refuses a profile whose minimum runtime reservation plus its 1 GiB automatic headroom
does not fit in what remains after weights, and it reports the byte counts when it refuses.

| device weights | reach at fp8 KV |
|---|---|
| 16.1 GiB (QUASAR) | 262,144 |
| 17.0 GiB (NVFP4-full) | 262,144 |
| 18.90 GiB (Swift, FP8 imported) | 240,000 |
| 19.7 GiB (the port's earlier image) | below the full context |
| 20.50 GiB (Swift, FP8 imported, DFlash2) | 180,224 |
| 15.3 GiB (Swift re-encoded) | 262,144 |

So **~17 GiB of device weights is the measured envelope** for the full native context at `fp8` KV;
this is an empirical boundary from four points, not a formula. A conversion that lands above it has
three options: re-encode FP8 text to NVFP4 (section 1), ship a documented lower ceiling, or raise
the ceiling with `--kv-dtype nvfp4` and pay the quality cost in section 5.

## 4. The lane shape

Each artifact ships **two lanes over one artifact**, selected at runtime by `--spec` — not two
artifacts. The artifact carries text, vision, MTP and the DFlash2 companion; the launcher picks one
draft.

| per profile | value |
|---|---|
| lanes | `dflash2` (draft 7) and `mtp` (draft 4 or 5) |
| vision | on |
| `--lm-head-draft` | on |
| `--device-state-slots` | 1 |
| `--max-context` | 262,144 |
| model id | `qwen3.8-27b-<artifact>-v3-<spec>-vision` |
| port | one per profile, from `profiles.py` |

Shared by every lane, in `INVARIANT_FLAGS`: `--kv-capacity auto`, `--kv-dtype fp8`,
`--prefill-chunk 8192`, `--max-concurrency 1`, `--host-state-slots 16`, `--host-kv-mib 8192`,
`--max-shared-prefixes 7`, `--max-private-continuations 8`,
`--max-long-anchors-per-continuation 4`, `--context-cache-policy rolling`, `--preserve-thinking`,
`--default-thinking-budget 4096`, `--pending-timeout-ms 600000`, and the explicit `--chat-template`.

`--kv-dtype` is currently a shared invariant, so no lane can differ on it without a new per-profile
field. ADR-0005 permits that: route, depth and per-profile flags are measurements, never convention.

## 5. Measure the flags; do not inherit them

`profiles.PROFILES` is the single source of truth, and every value in it is measured on the
**published** artifact through `v3_profile_matrix.py`:

| what | mode | note |
|---|---|---|
| context ceiling | `ceiling` | probes the descending ladder per spec × vision; refuses to guess |
| draft depth | `sweep` | every MTP depth at the measured ceiling plus a DFlash2 baseline |
| shipped values | `profile` | the only mode whose runtime/free pair describes a configuration a launcher starts |
| digests | `correct` | greedy, no request sampling. ADR-0002: speculation is *not* bit-identical, so "match control: none" is the expected result |
| KV dtype | `verify --kv-dtype` | `nvfp4` buys the full context at a measured quality cost |

Quality is measured with `ninfer-perplexity` on the fixed 1M corpus, per KV dtype. Recorded here:
bf16 against fp8 is 0.1% apart; nvfp4 against fp8 on Swift is **+0.74%** (4.84938 → 4.88517),
concentrated in one domain (+2.10% on zhwiki). That is the number to beat or accept before choosing
`nvfp4` KV to buy context back.

Then `tools/release/check_profile_consistency.py` must pass. It is the checklist: launchers
byte-identical to the table's render, the README and RELEASE_NOTES tables quoting the measured
tok/s and acceptance, the opencode entries, the packager's list, and the harnesses' coverage.

## 6. Checklist for a new artifact

1. Convert with an official recipe; the plan must be equivalent to the source recipe's.
2. Text all-NVFP4 (section 1). If the source keeps FP8, encode locally — do not import it.
3. Device weights inside the envelope (section 3), or a ceiling you can document.
4. Register the artifact in `v3_profile_matrix.ARTS`; `ceiling`, then `sweep`, then `profile`.
5. `correct` and record the digests; perplexity per KV dtype you intend to ship.
6. Add the two profiles, regenerate launchers, update the README/RELEASE_NOTES/opencode/packager.
7. `check_profile_consistency.py` green, then publish in order with the pin.
