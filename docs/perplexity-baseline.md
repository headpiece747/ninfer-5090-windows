# Perplexity baseline (Windows port)

The protocol, so a number is comparable: `ninfer-perplexity.exe <artifact> --corpus
eval/corpora/perplexity-1m/manifest.json --kv-dtype <dtype>`. That is upstream's fixed corpus
(`ninfer-ppl-1m-v1`: 1,044,876 scored tokens, 496 windows, 4096/2048 context and stride). Measured
2026-09-22 on this machine, which is a different box and clock from any published figure.

| artifact | KV | PPL | notes |
|---|---|---|---|
| `qwen3_8_27b_nvfp4.v3.ninfer` (official stock) | fp8 | **4.90295** | full corpus, 4.21k tok/s |
| `qwen3_8_27b_nvfp4.v3.ninfer` (official stock) | bf16 | **4.89838** | full corpus, 4.20k tok/s |
| `qwen3_8_27b_nvfp4qat.v3.ninfer` (QUASAR QAT) | fp8 | **4.89741** | `--quick` (4 streams, 261,223 tokens) |
| `qwen3_8_27b_nvfp4qat.v3.ninfer` (QUASAR QAT) | fp8 | **5.88829** | custom corpus, 177,400 tokens |
| `qwen3_8_27b_nvfp4full.v3.ninfer` (NVFP4-full) | fp8 | **5.92007** | custom corpus, same text |
| `qwen3_8_27b_nvfp4swift.v3.ninfer` (Swift, re-encoded) | fp8 | **4.68429** | `--quick`, 2026-09-24; the same recipe importing the checkpoint's FP8 scored **4.84938** |
| `qwen3_8_27b_nvfp4swift.v3.ninfer` (Swift, re-encoded) | fp8 | **4.92432** | full corpus, 6.03k tok/s |
| Swift, FP8 imported (superseded) | fp8 | **4.93874** | full corpus, 4.72k tok/s; the build the re-encode replaced |
| Swift, re-encoded with NVFP4-full's bf16 exceptions | fp8 | **4.7701** | `--quick`; 27 projections kept bf16 |
| Swift, re-encoded with NVFP4-full's bf16 exceptions | fp8 | **4.93254** | full corpus |
| Swift, re-encoded, endpoints as FP8 | fp8 | **4.85155** | `--quick`; isolates the endpoints, which are Q8 in the rows above |
| Swift, re-encoded, endpoints as FP8 | fp8 | **4.96342** | full corpus |

The custom corpus is this repo's `docs/` and `tests/` markdown, concatenated in sorted order
(177,400 tokens). On it the two shipped artifacts sit 0.5% apart with QUASAR marginally better, which
is the like-for-like check: same text, same protocol, two artifacts. The full-corpus rows are not
comparable with the `--quick` row, since the corpus subset differs.

## Measuring a template

`--chat-template` was added to this tool and then removed: perplexity scores **raw text**, so it never
renders a chat and the flag could not change a score. Measured before removing it, three templates --
the artifact's embedded one, `tools/chat_templates/qwen3_8.jinja`, and
froggeric/Qwen-Fixed-Chat-Templates v22.5 -- produced byte-identical perplexities on every artifact.

What the run did establish is the artifact ranking this file was missing:

| artifact | PPL (--quick, fp8) |
|---|---|
| `qwen3_8_27b_nvfp4swift` (Swift, re-encoded) | **4.68429** |
| `qwen3_8_27b_nvfp4full` | **4.77136** |
| `qwen3_8_27b_nvfp4` (official) | 4.82676 |
| `qwen3_8_27b_nvfp4qat` (QUASAR) | 4.89741 |

On `--quick`, Swift's re-encoded artifact is lowest, NVFP4-full 1.9% above it, the official artifact
3.0% above that and QUASAR 4.6% above that. Swift is also the one artifact here whose text weights
are not imported: its attention and GDN are encoded to NVFP4 from the finetune's BF16 source,
because the same recipe importing ModelOpt's per-tensor FP8 scored 4.84938 -- a per-tensor FP8 scale
is coarser than NVFP4's one scale per 16-element block, so the block scales more than pay for the
narrower codes. That comparison is like-for-like: one recipe, one checkpoint, only the attention and
GDN encoding differs.

**The `--quick` ranking does not generalize, and this is the row that shows it.** On the full corpus
the official stock is 4.90295, the re-encoded Swift 4.92432 (0.44% above it) and the superseded Swift
4.93874 (0.73% above). So Swift is lowest on the four-stream subset and *not* lowest on the 496-window
corpus: a finetune can win one subset and lose the corpus, and `--quick` selects one stream per
domain while `full` scores every window. Quote a `--quick` ranking as a `--quick` ranking. The
re-encode's own gain is the part that holds on both protocols: 3.4% on the subset, 0.29% on the
corpus, same direction, one recipe and one checkpoint.

**The bf16 exception pattern is a source allocation, not a rule, and it measures worse here.** The
third build is the same re-encode with NVFP4-full's pattern applied -- 27 projections kept bf16, 247
NVFP4 parents exactly as that artifact has. It scores 4.7701 and 4.93254 against the all-NVFP4
build's 4.68429 and 4.92432, on both protocols, while costing 0.77 GB more file and about 0.5 GiB
more resident. So the ordering on both is all-NVFP4, then exceptions, then the FP8 import. That
pattern came from a third party's mixed-precision checkpoint, was transplanted by the fork to a
different one, and nothing recorded why; on Swift's weights it does not pay for itself.

**The endpoints, not the block scales, are where the re-encode's win comes from.** The re-encoded
build differs from the FP8-importing one in two ways, and three builds isolate them, because each
adjacent pair differs in exactly one change — verified by hashing every binding, which leaves the two
endpoints as two objects and the text stack as 320. The endpoints alone are worth **−3.45% / −0.79%**
(Q8 against FP8, two rows above); re-encoding attention and GDN alone is worth **+0.05% / +0.50%**, a
small cost. The net **−3.40% / −0.29%** is the figure an earlier reading credited to NVFP4's block
scales being finer than a per-tensor FP8 scale, and that reading was wrong. Re-encoding the text stack
is bought for resident bytes and context, not for accuracy — the artifact conventions now state it as
that trade.

**Every variant row is reproducible without editing the recipe.** Each is the shipped recipe plus a
`--override` file that reassigns one thing, from the same sources, components and resources. The
bf16-exception rows reassign the 27 projections NVFP4-full keeps bf16 to `bf16`/`cast_direct` from
`swift_bf16` and drop the activation divisors the recipe had recorded for them; the FP8-endpoint rows
reassign `text/token_embedding` and `text/output_head` to `fp8`/`fp8_row_maxabs` from `swift_bf16`.
The published artifact's own invocation is in Section 16 of the artifact reference.

**A chat template needs a different instrument.** Perplexity cannot see one. What can are the rendered
prompt -- the token counts the CLI reports, which is how the reasoning-effort alias gap was caught -- and
any chat-shaped scoring route. Recorded here so the flag is not added again.

## The accurate-activation change (2026-09-23)

`nvfp4_linear_swiglu_w4a4_tma.cuh` was the only one of twelve SwiGLU activation sites using the
approximate `silu_approx`; every other site (fp8 ×2, nvfp4 decode, nvfp4 small-t, nvfp4 non-TMA W4A4,
q4 ×4, q8 ×3) already called the accurate `silu`. Upstream took the approximation in PR #250 and
Neroued/ninfer#285 measured a model-level cost for it, so the TMA epilogue was restored to `silu` and
`silu_approx` deleted. All four `ninfer_linear_swiglu_*_test` cases pass against their FP64 oracle,
including the TMA route, which `kA4Cases` reaches at 256, 512 and 1024 tokens (`kNvfp4TmaBlockM` is
256).

The change is engine-level, so it moves every artifact. Measured fp8 on this machine:

| artifact | protocol | before | after | Δ |
|---|---|---|---|---|
| official stock | full corpus | 4.90295 | **4.901690** | −0.03% |
| official stock | `--quick` | 4.82676 | **4.805574** | −0.44% |
| NVFP4-full | full corpus | not measured | **4.987682** | — |
| NVFP4-full | `--quick` | 4.77136 | **4.824524** | +1.11% |
| QUASAR QAT | full corpus | not measured | **5.002337** | — |
| QUASAR QAT | `--quick` | 4.89741 | **4.948789** | +1.05% |

Two things follow, and neither is the naive reading of #285.

**The accurate form is not a strict perplexity improvement.** It lowers the official stock artifact
and raises the other two. Perplexity is not monotone in numerical accuracy: a perturbation moves a
checkpoint toward or away from its training distribution depending on the checkpoint. The accurate
form is still the contract-correct one -- it is the FP64 oracle's own definition, eleven of twelve
sites already used it, and upstream's published artifacts were produced with it -- but the quality
argument is "matches the oracle and upstream", not "always scores better". The full-corpus effect on
the two artifacts whose baseline was not measured is therefore unknown, not zero: only the official
artifact has a full-corpus before-and-after.

**`--quick` is not a subset of the full corpus.** The manifest's two modes name different text:
`quick` is stream 00 of each domain (four documents, 261,223 tokens), `full` is all sixteen
(1,044,876). Their magnitudes differ by more than an order of magnitude for the same change (0.44%
against 0.03% on the official artifact), so a `--quick` delta is not an effect size for the full
corpus.

The ranking recorded in the section above (`--quick`: NVFP4-full 4.77136 < official 4.82676 < QUASAR
4.89741) was measured before this change. After it, on both protocols, the official stock artifact is
lowest: full corpus 4.9017 < 4.9877 < 5.0023, `--quick` 4.8056 < 4.8245 < 4.9488.

## The one open discrepancy

A third-party conversion recipe publishes PPL **4.617** for the official stock artifact on this same
corpus. Measured here it is **4.898** with bf16 KV and **4.903** with fp8, so the 6.1% gap is not the
KV dtype -- and the bf16-versus-fp8 agreement of 0.1% says the KV path is sound in general.

Two candidates remain and neither is checked: the **corpus revision**, since the recipe may have been
run against an earlier `perplexity-1m`; and the **artifact revision**, since theirs may be a newer
export of the stock model than the one this port carries. The scoring path itself is upstream's, not
this port's, so a port-specific defect is not the leading explanation -- but it is not excluded either,
and the protocol above is what would settle it.

Checked and excluded since: the accurate-activation change recorded above moves the full corpus by
−0.03%, so the engine's SwiGLU activation is not the cause of the gap. That leaves the two candidates
named here.
