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

**Re-encoding buys resident bytes, not accuracy — and the two changes were measured apart.** Swift's
artifact was first credited with a 4.68429-against-4.84938 `--quick` win over the build it replaced,
and the block scales were credited with it. That was wrong. The two builds differ in two ways, and
holding one fixed reverses the sign of the other. Three builds isolate them, each adjacent pair
differing in exactly one change — verified by hashing every binding, which leaves the endpoints as two
objects (`text/token_embedding`, `text/output_head`) and the text stack as 320:

| attention and GDN | endpoints | `--quick` | full |
|---|---|---:|---:|
| FP8, imported | FP8 | 4.84938 | 4.93874 |
| NVFP4, re-encoded | FP8 | 4.85155 | 4.96342 |
| NVFP4, re-encoded | **Q8** | **4.68429** | **4.92432** |

The endpoints are worth **−3.45% / −0.79%** (Q8 against FP8) and the attention and GDN re-encode is
worth **+0.05% / +0.50%**: it costs a little and it does not pay. The net, −3.40% / −0.29%, is the
figure that was misattributed to the block scales.

So the rule above rests on section 3's measurement and not on accuracy: a re-encoded text stack is
~3 GiB smaller, which is 83,584 more resident KV tokens and the full context instead of a documented
lower ceiling. State that as the trade it is. The endpoints are the part that pays, and for the
reason this file's original sentence gave — a Q8 group scale is finer than one scale per row, so
preserving the producer's W8 code word was not the same thing as preserving the model's accuracy.

**A source's bf16 exceptions are not a rule either.** NVFP4-full keeps 27 projections bf16, which is
its own source's mixed-precision allocation transplanted by the fork, with no reason recorded
anywhere. Applying that pattern to Swift's weights measures *worse* than encoding everything to
NVFP4: 4.7701 and 4.93254 against 4.68429 and 4.92432 on the subset and the corpus, for 0.77 GB more
file and about 0.5 GiB more resident. Follow a source's allocation when importing its codes; do not
transplant a pattern between checkpoints without measuring it.

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

### A file hash is not the artifact's identity

`tools/artifact/writer.py` seeds the 16-byte `artifact_id` from `uuid4()`, so two builds of one recipe
differ in every byte of the file hash while being the same model. Measured on the Swift re-encode: a
rebuild from the current recipe **on the other device** — CPU, against the original CUDA build —
produced identical index JSON, identical `file_bytes`, and an identical payload digest
(`7e9a3bebc65526c9b2aaf6502dafc82c32477ad309f8c306ff9195b2ee272bee`); only `artifact_id` differed. So
a file-level `sha256` answers a different question and reports a false mismatch. Compare with
`python3 tools/release/compare_artifacts.py A.ninfer B.ninfer`, which diffs the index and digests the
payload region separately and states why the id differs. Publish the file hash for download
verification and the payload digest as the artifact's identity.

## 3. Keep device weights inside the envelope that reaches the full context

The engine refuses a profile whose minimum runtime reservation plus its 1 GiB automatic headroom
does not fit in what remains after weights, and it reports the byte counts when it refuses.

| device weights | reach at `fp8` KV, vision on | vision off |
|---|---:|---:|
| 16.1 GiB (QUASAR) | 262,144 | 262,144 |
| 17.0 GiB (NVFP4-full) | 262,144 | 262,144 |
| 18.90 GiB (Swift, FP8 imported) | 240,000 | 262,144 |
| 19.7 GiB (the port's earlier image) | below the full context | below the full context |
| 20.50 GiB (Swift, FP8 imported, DFlash2) | 180,224 | 180,224 |
| 15.3 GiB (Swift re-encoded) | 262,144 | 262,144 |

Vision is a column because it decides a row: the FP8-imported Swift build serves the full context
with vision off and stops at 240,000 with it on. Both Swift lanes ship vision on, so 240,000 is their
ceiling for that build. Measured by `ceiling` mode, which renders the launcher's own configuration —
`--device-state-slots 1`, `--prefill-chunk 8192` and the context-cache set included, which together
reserve about 1.4 GiB more than the bare flags and are what put that build over the line.

So the envelope for the full native context at `fp8` KV **with vision on** sits between 17.0 GiB
(in) and 18.90 GiB (out) of device weights — an empirical boundary from five builds, not a formula.
Vision off moves it above 18.90 GiB, which is why the column above exists. A conversion that lands above it has
three options: re-encode FP8 text to NVFP4 (section 1), ship a documented lower ceiling, or raise
the ceiling with `--kv-dtype nvfp4` and pay the quality cost in section 5.

The same 3 GiB also shows up as resident KV, at different serving flags — so it is a second
measurement, not a second expression of the row above. Started identically with
`--kv-dtype int8 --kv-capacity auto --max-context 252928 --max-concurrency 2`, the re-encoded Swift
artifact reports **367,296 tokens** of KV capacity at 13.2 GiB runtime and the FP8-importing build it
replaced **283,712 tokens** at 10.4 GiB: **83,584 more resident tokens** for a download 3 GB smaller.
The flags were the same, so the difference is the artifact.

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

1. Convert with an official recipe; the plan must be equivalent to the source recipe's. Record the
   exact invocation next to the artifact — sources, `--components`, `--resource` overrides, `--name`
   and the recipe revision — because the artifact's own provenance stores only local source *paths*,
   and the recipe id is not versioned: two different behaviours have shipped under one id.
2. Text all-NVFP4 (section 1). If the source keeps FP8, encode locally — do not import it.
3. Device weights inside the envelope (section 3), or a ceiling you can document.
4. Register the artifact in `v3_profile_matrix.ARTS`; `ceiling`, then `sweep`, then `profile`.
5. `correct` and record the digests; perplexity per KV dtype you intend to ship.
6. Add the two profiles, regenerate launchers, update the README/RELEASE_NOTES/opencode/packager.
7. `check_profile_consistency.py` green, then publish in order with the pin.
