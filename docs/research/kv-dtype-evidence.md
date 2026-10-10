# Evidence on `k8v4` KV cache — is "compare k8v4 against the shipped fp8" still warranted?

Written 2026-10-01 for the open item in [`docs/active-work.md`](../active-work.md) §7 ("k8v4 KV against
the shipped fp8"). Primary sources only. Every figure is labelled **MEASURED** (someone ran it and
published the number), **DERIVED** (arithmetic I performed from a primary source, shown so it can be
checked), **VENDOR-CLAIMED** (a project asserting a figure without publishing the measurement),
**NOT FOUND**, **THIRD-PARTY ACADEMIC**, or **USER-REPORTED** (asserted in a first-party tracker by
someone who is not the project). No recommendation is made here; triage is a separate decision.

---

## Answer

The item's *original* justification is dead and stays dead — the withdrawn "within 0.08 % of BF16"
figure is not recoverable by any reading of any source I reached. But the item no longer rests on
nothing external: `k8v4` is a real, named, still-maintained KV format, and external evidence about it
now exists in a form that was not available when the note was written. That evidence, however,
**sharpens the reason the item is open rather than closing it.** Three things are now established.
First, the specific gap the item names is confirmed real and still unfilled: the only first-party
quality measurement of any `k8v4` in existence uses an **unquantized baseline, not FP8**, so **no
published k8v4-vs-FP8-KV quality number exists anywhere** — the same structural defect that
invalidated the withdrawn figure. Second, the one real measurement that does exist
(**GSM8K 0.860 vs a 0.900 unquantized baseline on Qwen3-4B**, vLLM PR #38479) is a *loss*, not a
parity claim, and on 200 questions it is 8/200 — about 1.9σ, unreplicated, no confidence interval —
so it neither establishes nor excludes a real effect. Third, and most consequential for this product:
**the format that vLLM measures is not the format this tree ships.** This repo's `k8v4` is
FP8-E4M3FN-row256 K + **NVFP4-G16** V; vLLM's `turboquant_k8v4` is unrotated FP8-E4M3 K +
**per-vector uniform 4-bit** V with no group scaling and no K scale plane. So even a sound external
k8v4 number would not transfer, and vLLM's own maintainer tracker shows the backend has open
reliability failures specifically on modern hybrid models and in combination with speculative
decoding — which is this product's shipping configuration. In short: the comparison the item asks for
is still unmade, the external evidence that has appeared does not make it look attractive, and the
prior justification should not be reinstated in any form.

---

## Findings

### 1. What `k8v4` actually is

| # | Finding | Label | Source |
|---|---|---|---|
| 1.1 | `turboquant_k8v4` is a named preset: `key_quant_bits=8`, `value_quant_bits=4`, `norm_correction=False`. The comment above the table states `key_quant_bits: 8 = FP8 keys, 3-4 = MSE (Lloyd-Max) quantized keys` and `value_quant_bits: 3-4 = uniform quantized values` | MEASURED (read in source) | `vllm/model_executor/layers/quantization/turboquant/config.py`, `TQ_PRESETS`, `main` — [raw](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/quantization/turboquant/config.py) |
| 1.2 | K side for k8v4 is **plain FP8, one byte per element, no rotation, no MSE, no stored scale**. `key_packed_size` returns `self.head_dim` when `key_fp8`; `key_mse_bits` returns 0 | MEASURED (read in source) | same file, `key_fp8` / `key_mse_bits` / `key_packed_size` |
| 1.3 | V side is 4-bit **uniform (16 levels) with one fp16 scale + one fp16 zero per whole head-vector** — i.e. `ceil(head_dim*4/8) + 4` bytes. There is **no group size**; the quantization group is the entire head-vector | MEASURED (read in source) | same file, `value_packed_size` |
| 1.4 | Cache layout: tensor is `(num_blocks, block_size, num_kv_heads, slot_size)`, `slot_size = key_packed_size + value_packed_size`; `block_size` supported values are `{16, 32, 64, 128}` | MEASURED (read in source) | `vllm/v1/attention/backends/turboquant_attn.py`, module docstring + `get_supported_kernel_block_sizes` — [raw](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/attention/backends/turboquant_attn.py) |
| 1.5 | Arithmetic check against the published slot size: at `head_dim=128`, k8v4 slot = 128 (K) + 64+4 (V) = **196 B**, matching PR #38479's stated 196 B exactly. Compression vs BF16 (512 B) = 2.61×, matching the PR's "2.6x" | DERIVED (shown) | arithmetic from 1.2–1.3 and #38479 |
| 1.6 | The FP8 key format is SM-dependent: `tl.float8e4nv` / `tl.float8e4b15`, auto-detected via `_use_fp8_e4b15`. PR #38479 labels the key "FP8 (E4M3)"; the docstring in `config.py` calls it "FP8 keys (no rotation/MSE)" | MEASURED (read in source) | `turboquant_attn.py` (`_use_fp8_e4b15`); PR #38479 "In-kernel FP8 cast" |
| 1.7 | **No QJL residual in any vLLM preset**, including k8v4: *"QJL is intentionally omitted: community consensus (5+ independent groups) found it hurts attention quality by amplifying variance through softmax."* The WHT rotation likewise applies only to the 3/4-bit-key presets | MEASURED (read in source) | `turboquant/config.py` docstring |
| 1.8 | **This tree's `k8v4` is a different codec.** `--kv-dtype k8v4` maps to `KvCacheStorage::Fp8KeyNvfp4Value`, documented as *"K 固定为 FP8-E4M3FN-row256，V 固定为 NVFP4-G16"* (K fixed FP8-E4M3FN-row256, V fixed NVFP4-G16). So: K carries a 2 B row scale and a Hadamard preparation that vLLM's k8v4 has no equivalent of, and V is NVFP4 group-of-16, not per-vector uniform | MEASURED (read in source) | `apps/cli/options.cpp:60`, `src/serve/serve_options.cpp:54`, `docs/maintainer/paged-kv-cache.md` L82–85 |
| 1.9 | Per token per head at D256, this tree: `fp8` = K 258 B + V 258 B = **516 B**; `k8v4` = K 258 B + V 144 B = **402 B**; `nvfp4` = 288 B; `bf16` = 1024 B | MEASURED (in-tree, documented geometry) | `docs/maintainer/paged-kv-cache.md` L275–281; `bench/README.md` L601 |

### 2. Does any first-party source publish quality measurements for k8v4?

| # | Finding | Label | Source |
|---|---|---|---|
| 2.1 | **The one first-party quality measurement.** Qwen/Qwen3-4B, `head_dim=128`, 5-shot GSM8K (200 questions) and NIAH (512→32K, 77 probes): baseline **GSM8K 0.900**, NIAH 100 %; `turboquant_k8v4` GSM8K **0.860**, NIAH 100 %; `4bit_nc` 0.840; `k3v4_nc` 0.780; `3bit_nc` 0.720 | MEASURED | vLLM PR **#38479**, merged 2026-04-15, commit `f4b42df04847dcbd3247f7f4c56dff45e40bdf0d`, co-signed-off by `mgoin64` — [PR](https://github.com/vllm-project/vllm/pull/38479) |
| 2.2 | **The baseline in 2.1 is unquantized, not FP8.** No FP8-KV control arm is reported. Therefore 2.1 does **not** answer "k8v4 vs FP8" — it answers "k8v4 vs nothing" | MEASURED (absence, read in source) | PR #38479, "Baseline: GSM8K 0.900, NIAH 100%" |
| 2.3 | The −4.0 pp GSM8K delta in 2.1 is **8 questions out of 200**, from one unreplicated run with no confidence interval and no seed control. At p≈0.9, n=200 the per-run sd is ≈4.24 questions, so 8 is ≈1.9σ — not significant at conventional levels. It neither establishes nor excludes a real effect | DERIVED (binomial arithmetic, shown) | from 2.1 |
| 2.4 | NIAH at long context: Qwen3-4B on H20, 4 context lengths (4K/8K/16K/32K) × 5 depths × 3 trials = 60 probes per config. `k8v4` **60/60 (100 %)**, identical to the FP16 baseline 60/60. Capped at 32K — "the model's max position embeddings"; follow-ups for 64K/128K/256K/1M stated as not yet added | MEASURED — but the PR is **OPEN / unmerged** | vLLM PR **#40122**, `mergedAt: null` — [PR](https://github.com/vllm-project/vllm/pull/40122) |
| 2.5 | A WikiText-2 PPL table in the maintainer tracker shows `auto`, `k8v4`, `4bit_nc`, `k3v4nc`, `3bit_nc` **all at exactly 6.3554** (Δ +0.00 %, 204,700 tokens). **This is not a credible discriminating measurement** — four materially different bit budgets cannot produce identical PPL to four decimals — and the same author separately reports that "exact token-level match against `auto` drops under TurboQuant". Recorded here as an internal inconsistency, **not** as k8v4 quality evidence | USER-REPORTED (internally inconsistent; excluded as evidence) | vLLM issue **#40069**, comment by `shanyulu` (DeepSeek-V2-Lite, TP=4, 4× RTX 4090) |
| 2.6 | **Throughput, from the same merged PR.** Qwen3-4B, 4× RTX PRO 6000 Blackwell: k8v4 is **79–100 % of baseline** output tok/s. short-decode 8977 → 7113 (79 %); TPOT short-decode 11.9 → 15.0 ms; very-long-prefill 233 → 234 (100 %). So in vLLM the memory saving did **not** buy decode throughput — k8v4 decode is slower than unquantized | MEASURED | PR #38479, perf tables |
| 2.7 | **Capacity, first-party, in the tracker.** DeepSeek-V2-Lite, TP=4, 4× RTX 4090; GPU KV cache tokens at batch 1: `auto` 382,576 → `k8v4` **686,576** (**1.79×**) → `4bit_nc` 1,137,952 → `k3v4_nc` 1,364,144. Decode tok/s batch 1: 82.15 → 180.38 (2.20×) | USER-REPORTED (first-party tracker, third-party prototype) | issue #40069, comment by `shanyulu` |
| 2.8 | **A memory figure in the tracker**: *"For Qwen3-32B + TQ k8v4 on H200: model load drops from 123 GiB → 62 GiB, baseline OOMs → 588K KV tokens at 52.7 tok/s."* This is about a per-layer decode-buffer bug (#40655), and the 588K-token figure is a real measured capacity on H200 | USER-REPORTED | issue #40069, comment by `bhoomit` |
| 2.9 | **The docstring's PPL table is unattributed.** `TurboQuantConfig` documents *"turboquant_k8v4: FP8 keys + 4-bit values, 2.6x, **+1.17% PPL**"*, and +2.71 % / +10.63 % / +20.59 % for the others. **No model, no corpus, no baseline is named.** It appears only in vLLM's auto-generated API reference | VENDOR-CLAIMED (provenance not located) | `turboquant/config.py` docstring; <https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/turboquant> |
| 2.10 | That docstring table **contradicts the merged PR** it should agree with: it gives `k3v4_nc` as "~3.5x" where PR #38479 measured **4.3x**. So the docstring table is not a controlled, reproducible figure set | DERIVED (cross-source inconsistency) | `config.py` docstring vs PR #38479 |
| 2.11 | **vLLM's own accuracy tracker says the results were never published.** Issue #40069 (author `mgoin`, member; OPEN, labelled `stale`) has under "### Accuracy": `[x] Long-context evals across presets (k8v4, t4nc, k3v4nc, t3nc): RULER, NIAH at 32K–1M, LongBench` marked **done**, but `[ ] Publish recommended config table (quality vs. compression vs. throughput) based on eval results` **unchecked**, and `[ ] Per-layer sensitivity sweep` unchecked | MEASURED (read in source) | vLLM issue **#40069** — <https://github.com/vllm-project/vllm/issues/40069> |
| 2.12 | **`k8v4` is absent from vLLM's user-facing feature documentation.** `docs/features/quantization/quantized_kvcache.md` on `main` documents **FP8 only** (per-tensor and per-attention-head, with llm-compressor calibration). TurboQuant appears in no feature doc while being a supported `CacheDType` | MEASURED (read in source) | <https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/quantized_kvcache.md> |
| 2.13 | Nothing in-tree measures k8v4 **quality**. It passes the FP64 oracle alongside the other four KV types, which is numerical correctness against an oracle, not a model-quality comparison | MEASURED (in-tree) | `docs/research/swift15-lane-measurement.md` L241–245 |

### 3. Actual VRAM saving of k8v4 vs fp8, and where it matters

| # | Finding | Label | Source |
|---|---|---|---|
| 3.1 | vLLM: at `head_dim=128`, FP8 KV slot = 128 + 128 = **256 B**; k8v4 slot = **196 B**. So **1.31×, −23.4 % of KV bytes**. The "2.6x" vLLM publishes is against **BF16** (512 B), not against FP8 | DERIVED (shown; 256/196 = 1.306) | arithmetic from 1.2–1.3; PR #38479 slot table |
| 3.2 | This tree, at native 262,144-token capacity: `fp8` pool **8.06 GiB** (33,024 B/token) → `k8v4` pool **6.28 GiB** (25,728 B/token). **1.78 GiB freed, −22.1 %**, consistent with the 516 B → 402 B geometry (1.284×) | MEASURED (in-tree capacity figures) | `model-cards/Qwen3.8-27B-nvfp4qat-NInfer/README.md` L189–198 |
| 3.3 | The saving scales linearly with context: **1.78 GiB at 262,144 tokens ≈ 0.22 GiB per 32,768 tokens**. It only becomes material near native context. Against 15.31–16.60 GiB of device weights, the whole `fp8` pool is 8.06 GiB | DERIVED (linear scaling; weights/bf16/int8/fp8 rows measured in-tree) | model card L183–198 |
| 3.4 | The freed memory **cannot buy context** past this model's native cap — it buys concurrency headroom. Per this project's own tracker: *"It does not change max context: upstream caps Qwen3.8-27B at `kNativeContext` = 262,144 and this proposal leaves that alone. The win is memory, not length."* | MEASURED (first-party, this project) | Neroued/ninfer issue **#123** |

### 4. Deprecated, discouraged, or absent from current upstream?

| # | Finding | Label | Source |
|---|---|---|---|
| 4.1 | **TensorRT-LLM: absent.** GitHub code search on `NVIDIA/TensorRT-LLM` returns **0** hits for `k8v4` and **0** for `turboquant` (control: `kv_cache_dtype` returns 105, so the query form works) | MEASURED (search, 2026-10-01) | GitHub code search API, `repo:NVIDIA/TensorRT-LLM` |
| 4.2 | TensorRT-LLM's quantization doc lists exactly two KV-cache recipes — **"FP8 KV Cache"** and **"NVFP4 KV Cache"** — and its model/hardware support matrices have no other KV dtype column | MEASURED (read in source) | `docs/source/features/quantization.md`, `main` |
| 4.3 | TensorRT-LLM's KV Cache Compression framework supports exactly two methods: **NVFP4 cold-page quantization** and **TriAttention**. No 4-bit KV, no k8v4 | MEASURED (read in source) | `docs/source/features/kv-cache-compression.md`, `main` |
| 4.4 | The only trace of the name in any NVIDIA tracker is a **user** in TRT-LLM issue #10241 writing *"in the meantime testing `--kv-cache-dtype turboquant_{k8v4, 4bit_nc}`"* — that thread is about SM120 NVFP4 support and the name appears only in that comment; no NVIDIA responder acknowledges it | USER-REPORTED | TRT-LLM issue #10241, comment by `montvid`, 2026-06-05 |
| 4.5 | **NVIDIA ModelOpt: absent.** 0 code hits for `k8v4` and 0 for `turboquant` | MEASURED (search, 2026-10-01) | GitHub code search API, `repo:NVIDIA/Model-Optimizer` |
| 4.6 | **SGLang: absent.** 0 code hits for `k8v4` | MEASURED (search, 2026-10-01) | GitHub code search API, `repo:sgl-project/sglang` |
| 4.7 | **FlashInfer: absent.** 0 code hits for `k8v4` | MEASURED (search, 2026-10-01) | GitHub code search API, `repo:flashinfer-ai/flashinfer` |
| 4.8 | **vLLM: present, not deprecated, actively developed.** `turboquant_k8v4` is in the `CacheDType` `Literal` on `main` and in `TurboQuantAttentionBackend.supported_kv_cache_dtypes`. Landed 2026-04-15 (`f4b42df`), last touched 2026-09-16 | MEASURED (read in source) | `vllm/config/cache.py` (`CacheDType`), `turboquant_attn.py` |
| 4.9 | **The name originates in vLLM, not NVIDIA.** The underlying algorithm is TurboQuant / HIGGS (arXiv:2504.19874, Zandieh, Daliri, Hadian, Mirrokni). Note the arXiv page lists **no venue** — the "ICLR 2026" attribution in vLLM's docstring is not corroborated by the arXiv record | MEASURED (read in source) | <https://arxiv.org/abs/2504.19874>; `turboquant/config.py` |
| 4.10 | **Summary: not deprecated anywhere. Absent from every NVIDIA first-party surface. Live and actively developed in vLLM only.** | DERIVED from 4.1–4.8 | — |

### 5. Documented accuracy risk with 4-bit KV

| # | Finding | Label | Source |
|---|---|---|---|
| 5.1 | **First-party, measured:** k8v4 costs **−4.0 pp GSM8K** vs unquantized on Qwen3-4B (2.1). Read alongside 2.3, this is a small unreplicated delta, not a demonstrated large one | MEASURED | PR #38479 |
| 5.2 | **First-party, measured, and large — for the *other* 4-bit presets.** Boundary layer protection is *"Empirically required for aggressive presets (k3v4_nc, 3bit_nc) — without it GSM8K drops ~30 points on Qwen3-4B."* So the 4-bit-**key** variants need FP16 first/last 2 layers to hold up at all | MEASURED (read in source) | `turboquant/config.py`, `get_boundary_skip_layers` docstring |
| 5.3 | **First-party, read in source:** norm correction is *"Re-normaliz[ing] centroid vectors to unit norm before inverse rotation during dequant… improving PPL by ~0.8% at 4-bit"* — and k8v4 has `norm_correction=False`. So k8v4 is the preset that forgoes the one mitigation vLLM documents | MEASURED (read in source) | `turboquant/config.py` |
| 5.4 | **Academic, measured:** *"low-bit quantization can silently destroy safety alignment: Mistral-7B loses 15.2 % of its refusals at only 1.03x perplexity, and no universal safe bit-width exists, with sharp model-specific phase transitions invisible to standard metrics."* 11 instruction-tuned models (3.8B–72B), 5 benchmarks, 1,894 prompts. **Caveat that cuts both ways:** the paper's own abstract says the vulnerabilities are "confirmed in production vLLM serving with **FP8** KV cache" — so this is not specific to 4-bit, and it does not by itself indict FP8 | THIRD-PARTY ACADEMIC (abstract read, not the full PDF) | arXiv:2606.09864 — <https://arxiv.org/abs/2606.09864> |
| 5.5 | **Academic, abstract:** QJL on K "inflates inner product variance by π/2, which softmax amplifies nonlinearly via Jensen's inequality"; "The K–V asymmetry is unconditional." This is the stated mechanism behind vLLM omitting QJL | THIRD-PARTY ACADEMIC (abstract only) | arXiv:2605.08114 — <https://arxiv.org/abs/2605.08114> |
| 5.6 | **Reliability risk specific to this product's configuration** (MTP/DFlash2 speculative decoding + modern model), user-reported in vLLM's own tracker: *"MTP × TurboQuant on Qwen3-Next hybrid produces degenerate token loops on tool calls, long-context recall, and streaming, while either feature alone works correctly."* TurboQuant + spec-decode + chunked-prefill also crashes CUDA-graph capture at `turboquant_attn.py:570` (issue #40807 / #40831) | USER-REPORTED (first-party tracker, not maintainer-confirmed) | vLLM issues **#40807**, **#40831**; comments by `noonghunna` on #40069 |
| 5.7 | User-reported breadth of that: *"The TurboQuant implementation does not current work with modern models, only legacy ones from 2024/early 2025."* / *"Any modern model like kimi-k, qwen3.5, qwen3.6 results in engine crashes or OOM because of the prefill bugs."* / *"The TurboQuant implementation does not work even on v0.22.0."* (≈38 open TurboQuant PRs at the time). Separately, a user reported in v0.20.0: *"it is not documented which models are supported. Every model I have tried so far fails to start."* | USER-REPORTED | comments by `gaby` on vLLM issue #40069 |
| 5.8 | **Mixed in vLLM's favour**, for balance: the merged PR that added hybrid support notes TurboQuant *"was failing with a `NotImplementedError` as soon as it encountered Mamba layers"* and that enabling it *"makes TurboQuant work reliably on hybrid architectures while preserving the current behavior and baselines for dense models"* | MEASURED (read in source) | vLLM PR **#39931**, merged 2026-05-05, commit `4f2af1a7` |
| 5.9 | vLLM's maintainer is not currently servicing the backlog: *"I unfortunately have a lot of work to do that is unrelated to this workstream at the moment"*, with `#sig-quantization` offered as the discussion venue | USER-REPORTED | comment by `mgoin` (member) on issue #40069 |

---

## What I searched and did not find

**Searched, not found — the load-bearing negative results:**

1. **No published k8v4-vs-FP8-KV quality measurement exists in any source.** Not in vLLM's PRs, issues,
   docs, or its API reference; not in TensorRT-LLM; not in ModelOpt; not in SGLang; not in FlashInfer;
   not in the TurboQuant/HIGGS paper's abstract; not in this repo. The single first-party quality
   measurement of anything named `k8v4` (PR #38479) uses an **unquantized baseline**, so it cannot be
   converted into a k8v4-vs-FP8 number without inventing the FP8 arm. This is the same structural
   defect that made the withdrawn "0.08 %" figure unusable, and it is still open today.
2. **No quality measurement of `k8v4` on any Qwen3.5 / 3.6 / 3.8-class model.** The only NIAH work
   (PR #40122) is explicitly capped at Qwen3-4B's 32K position limit and is unmerged.
3. **No published k8v4 result above 32K context.** vLLM's tracker marks 32K–1M RULER/NIAH/LongBench
   evals as done (#40069) but the "publish recommended config table" box is unticked and I could not
   find the results published anywhere.
4. **No provenance for the "+1.17 % PPL" docstring figure.** It is not in PR #38479, not in PR #39931,
   not in PR #39890 (which carries no measurements at all), and names no model, corpus or baseline.
5. **No published perplexity for *this tree's* k8v4 codec** (FP8-E4M3FN-row256 K + NVFP4-G16 V). vLLM's
   numbers are for a different V codec and a different K plane layout (§1.8), so they do not transfer.
6. **Zero occurrences of `k8v4` or `turboquant` in any NVIDIA first-party repository** — TensorRT-LLM,
   ModelOpt — and zero in SGLang and FlashInfer. The format is a vLLM-first-party name.
7. **No NVIDIA-published accuracy or quality figure for any 4-bit KV cache**, in either direction.

**Sources I read but deliberately excluded as evidence:**

- The DeepSeek-V2-Lite WikiText-2 PPL table in vLLM #40069 (all five presets identical to 4 dp) —
  excluded because it cannot be a discriminating measurement and contradicts its own author's
  token-match observation. Recorded as an inconsistency, not as a result.
- The `dfirlab/turboquant`, `turboquantcpu` (PyPI) and `varjoranta/turboquant-vllm` write-ups that
  surfaced in search — **secondary sources**, third-party reimplementations with their own marketing
  claims ("zero accuracy loss", "0.00% quality change"). Not used for any figure in this document.
- `vllm-metal`'s TurboQuant page and the arXiv survey 2607.05399 — secondary/tertiary; used only to
  locate the primary sources above.

**Method notes and limits of this search:**

- Code-search counts are GitHub's, on `main`, on 2026-10-01, unauthenticated-token quota permitting;
  a control query (`kv_cache_dtype`, 105 hits) confirmed the query form was live when `k8v4` returned 0.
- `docs/active-work.md` §7 was read in full (L132–138) and `docs/research/nvfp4-conversion-recipe-review.md`
  §6 (L629–657) was read and agrees with this document's central negative finding.
- I did not build, run or measure anything on this machine, and I did not run any k8v4 perplexity or
  benchmark. Every in-tree number quoted here is copied from an existing in-tree document, not
  re-measured by me.
- I did not read the full PDFs of arXiv:2606.09864 or arXiv:2605.08114 — only their abstracts, as noted.

---

## MEASURED-HERE 2026-10-08: this port's own FP8 KV costs 0.19% at 32K, so calibrated scales are not worth an engine change

The note above answers whether `k8v4` should replace the shipped `fp8` — a *format* question, closed on
external grounds with no measurement of this port's own cost. A different lever was raised by a survey of
the wider ecosystem: calibrated per-layer FP8-KV scales, measured elsewhere to recover 83% of FP8 KV's
long-context cost on this model family, performance-free, and shipped by one quantizer. That figure implies
the FP8 KV being calibrated costs roughly **3.3%** in the first place (10.84 against a 10.50 recovered value
at 32K). Nothing in this tree had measured the equivalent cost here, so the lever could not be priced.

It is now measured. `ninfer-perplexity` on the shipped `qwen3_8_27b_nvfp4nvidia`, quick corpus, 261,167
scored tokens, context/stride {4096/2048, 32768/16384}, `--kv-dtype` bf16 against fp8:

| context | bf16 | fp8 | fp8 cost |
|---|---|---|---|
| 4,096 | 4.386170 | 4.391526 | **+0.122%** |
| 32,768 | 4.193074 | 4.201042 | **+0.190%** |

Per domain at 32K the difference is not systematic: fp8 is slightly *better* on `chinese_reference`
(4.89271 against 4.89659) and `english_long_form` (6.66017 against 6.66971), slightly worse on
`english_reference` (6.09174 against 6.03504) and `ninfer_code` (1.55704 against 1.55631). The overall
+0.19% is a token-weighted mean over four domains that do not agree.

**Why this is an order of magnitude below the third-party figure, and it is not a measurement artifact.**
This tree's FP8 KV is `fp8-e4m3-r256`: `kKvFp8QuantGroup = 256` (`src/models/qwen3_5/state/decoder_state.h`)
means every 256 elements of a row carry their own E4M3 scale, so the quantization error is bounded per group
rather than per tensor. The format the external figure was measured on carries no comparable scale plane.
A calibrated per-layer scale has less to recover here precisely because the stored format already resolves
what a per-tensor scale would have thrown away.

**Verdict against the predicate set before the run**: at 32K, 1% or more would make an engine change to
accept calibrated scales worth pricing, and under 0.3% would mean the lever is spent. **0.190% is under the
threshold, so the lever is spent** — at best a fraction of a fifth of a percent, against an engine change
touching the KV quantization path. This does not reopen §7 of `active-work.md`; it *supports* that closure
from a second direction, and it means the capacity a quality recipe gives up is a capacity cost, not one
that a KV-precision change can buy back.

---

## MEASURED-HERE 2026-10-09: the comparison performed — refused as an invariant, adopted on one lane

The 2026-10-01 findings above said the item's comparison was unmade and that no k8v4-vs-FP8 quality number
existed anywhere. It has now been made, on this tree's own codec, with a real fp8 control arm on the same
binary and the same day. The answer is neither of the two the item anticipated: **the effect is
artifact-dependent**, so `k8v4` was refused as a shared setting and adopted as a per-lane one on the lane
measured to gain.

### Quality: five artifacts, worst domain +0.277%

`ninfer-perplexity <artifact> --corpus <1M-token manifest> --kv-dtype {fp8,k8v4}`, same binary, same day,
deterministic — the fp8 arm reproduces the published nvidia baseline to fourteen significant figures:

| artifact | fp8 | k8v4 | change | worst domain |
|---|---|---|---|---|
| nvidia | 4.686758 | 4.693563 | +0.145% | english_reference +0.157% |
| qat | 4.684860 | 4.691619 | +0.144% | english_reference +0.202% |
| full | 4.724219 | 4.730581 | +0.135% | ninfer_code +0.193% |
| noex | 4.727636 | 4.729300 | +0.035% | ninfer_code +0.277% |
| swift15 | 4.755739 | 4.760419 | +0.098% | ninfer_code +0.257% |

Every artifact pays, and the largest single-domain cost is a quarter of a percent — small on the scale the
2026-10-08 section above established for an FP8 change (+0.122% at 4K, +0.190% at 32K against bf16).

### Speed: interleaved, and why a global change was refused

The first pair (verify mode, NVIDIA lane, draft 7, fp8 then k8v4) read 249.4 -> 291.7 tok/s, +17%, with +9.8
acceptance points. A fingerprint across all eight lanes then produced deltas that cannot be one format's
effect — +35% on quasar's long Chinese against -36% on nvidia's, acceptance swinging +9 to -19 points. Two
facts separate the signal from the card:

* **Acceptance is deterministic per configuration.** It repeats to four decimal places inside one arm of the
  interleaved check below, and the verify path and the fingerprint agree on it exactly (48.50% and 58.0% for
  the same two configurations). Every acceptance delta in the fingerprint is therefore a real format effect,
  and the deltas point in both directions.
* **Decode needs interleaving.** fp8 and k8v4 alternate within one session on the same cell, each arm twice:

| lane, published cell | fp8 | k8v4 | change | acceptance |
|---|---|---|---|---|
| quasar dflash2 | 318.7 / 318.0 | 359.7 / 365.1 | **+13.8%** | 48.50% -> 57.99% |
| nvidia dflash2 | 307.6 / 304.0 | 302.7 / 308.2 | **-0.1%** | 45.92% -> 46.80% |

The +17% did not reproduce on the lane it was measured on at the shipped depth: it was a draft-7 measurement,
and at draft 9 that lane reads flat. What does reproduce is per-artifact: quasar gains 13.8% decode and 9.5
acceptance points on this cell, and its other two cells gain 27.3% and 35.2% with 8.7 and 9.0 acceptance
points; nvidia is flat; ninfer's DFlash2 lane loses on two cells (acceptance -5.9 points on code, -13.4 on
long Chinese). A change that helps one lane, is neutral on a second and costs a third is not an invariant.

### The acceptance axis filters all eight lanes

Acceptance is deterministic per configuration, so it needs no interleaving and it can be read for every
lane. The criterion is the one the depth decisions use -- the worst cell decides, with the mean breaking a
tie -- and it is applied per cell of the lane, because the format applies to every cell at once:

| lane | code | long-code | long-Chinese | worst cell fp8 -> k8v4 | verdict |
|---|---|---|---|---|---|
| quasar dflash2 | 48.5 -> 58.0 | 20.7 -> 29.4 | 16.8 -> 25.8 | 16.8 -> 25.8 **+9.0** | adopted |
| swift mtp4 | 57.7 -> 60.2 | 30.6 -> 35.5 | 41.2 -> 35.0 | 30.6 -> 35.0 **+4.4** | candidate |
| quasar mtp4 | 63.5 -> 64.0 | 31.0 -> 51.0 | 38.2 -> 31.4 | 31.0 -> 31.4 **+0.4** | candidate |
| nvidia mtp4 | 57.3 -> 54.4 | 33.2 -> 30.5 | 37.3 -> 36.1 | 30.5 -> 30.5 **0.0** | tie, mean -2.3 |
| ninfer mtp4 | 46.7 -> 52.7 | 34.1 -> 33.9 | 34.1 -> 41.1 | 34.1 -> 33.9 **-0.2** | refused |
| ninfer dflash2 | 56.7 -> 50.8 | 23.4 -> 24.8 | 36.5 -> 23.1 | 23.4 -> 23.1 **-0.3** | refused |
| swift dflash2 | 58.3 -> 51.8 | 31.1 -> 42.7 | 30.0 -> 27.5 | 30.0 -> 27.5 **-2.5** | refused |
| nvidia dflash2 | 45.9 -> 46.8 | 35.7 -> 36.7 | 45.1 -> 26.1 | 35.7 -> 26.1 **-9.6** | refused |

Two readings are possible and both are recorded. Under the worst-cell rule this repository ships depth
decisions by, the two candidates pass and are measured further (the interleaved A/B above); under a
stricter reading -- no served cell may regress at all -- only quasar's DFlash2 lane qualifies, because
both candidates regress certainly on long Chinese (quasar mtp4 -6.8 points, swift mtp4 -6.2). The depth
precedent is the same shape as the looser rule: nvidia's move to depth 9 shipped with prose -3.5% and
dialogue -5.5% disclosed. So the candidates are decided on their interleaved decode and their measured
quality cost, and any regression they carry lands in the row rather than in a footnote.

**And the deciding set changed while this was being written.** On 2026-10-09 the product owner stated the
workload is coding, and `CONTEXT.md` now records that the deciding scenarios are the code ones -- the
225-character code prompt and 36,000 characters of real code -- with every other domain measured, recorded
and disclosed rather than vetoing. For the format this is decisive: the candidates' code cells are positive
(quasar mtp4 +0.5 acceptance points on the published cell and +20.0 on long code; swift mtp4 +2.5 and
+4.9), so their long-Chinese regressions are disclosed costs, not blocks. It also reopens every depth whose
decision rested on Chinese, which is why the lanes are being re-read on served code rather than
grandfathered.

**Outcomes, measured the same day.** Five lanes ship `k8v4` and three keep `fp8` — **corrected
2026-10-10: this sentence said "four and four", which its own table below already contradicted;
`tools/release/profiles.py` carries five `kv_dtype="k8v4"` rows and three `"fp8"` ones, and the README's
count gate checks the same number.** The table itself stands.

| lane | deciding cells, interleaved | decision |
|---|---|---|
| quasar dflash2 | +13.8% published cell; +27.3/+35.2% served | adopted |
| quasar mtp4 | +1.0%, +35.6% decode; acceptance +0.5, +20.0 | adopted |
| swift mtp4 | +2.4%, +7.6% decode; acceptance +2.5, +4.9 | adopted |
| ninfer mtp4 | +12.1%, +2.1% decode; acceptance +6.0, -0.2 | adopted via the tie-break: its worst deciding cell is 0.5% lower, inside the arms' spreads, so the mean over the deciding cells decides at +4.3% |
| swift dflash2 | worst deciding cell 196.2 -> 244.4 (+24.6%), acceptance 31.06 -> 42.69; published cell -7.9% and -6.5 points disclosed | adopted |
| nvidia dflash2 | +0.4%, +3.7% decode; acceptance +0.9, +1.0 | refused: a tie and a few percent on code against -31.5% long Chinese and -19.1 acceptance points |
| ninfer dflash2 | worst deciding cell 184.9 -> 194.1 (+4.9%), acceptance 23.42 -> 24.84 | refused: +4.9% against -25.4% long Chinese and -13.4 points |
| nvidia mtp4 | worst deciding cell falls 149.2 -> 133.9 (-10.2%) | refused |

**The bound, stated once and applied to all eight.** The code-weighted rule decides on the code cells and
discloses the rest; a disclosure with no bound is not a rule, so the bound is: **a change is adopted when
the deciding cells' worst case improves by at least as much as the worst disclosed non-code cell gives
up.** It is what separates swift dflash2 (+24.6% against -4.2%) from ninfer dflash2 (+4.9% against
-25.4%) and nvidia dflash2 (+3.8% against -31.5%) -- three lanes whose code cells are all positive, and
only one of which is a trade worth making.

**Every depth stands, and each now rests on the format it ships.** The rule change forced five re-reads;
none moved a lane, but three overturned a *reading*:

* swift15 keeps depth 7: at served length depth 7 wins long code 208.2 against 182.0 (+14.4%) and long
  Chinese 171.9 against 165.6. The "+19.3% for depth 9 on long code" carried in this session's own notes
  is **withdrawn** -- no record carries it and the measurement is the other sign.
* swift mtp4 moves to depth 5, and the arc is kept rather than tidied: moved on fp8 evidence (long code
  158.2 against 140.2), reverted at the shipped format when the *published* cell fell 7.2% decode and 9.3
  acceptance points, then re-applied once the comparison matched the rule -- **maximin compares each
  option's worst deciding cell**, and depth 5's worst (long code 168.9) beats depth 4's (146.2) by 15.5%,
  with acceptance's worst rising too (42.8 against 35.5). The per-cell comparison was the error, not the
  measurement.
* quasar mtp4, ninfer mtp4 and nvidia mtp4 keep depth 4, each re-confirmed at the format it ships
  (long code: 183.8 against 133.0, 139.3 against 125.2, 141.9 against 128.0).

### The decision and its shape

- **Global**: refused. `("--kv-dtype", "k8v4")` sat in `INVARIANT_FLAGS` during the investigation and was
  withdrawn; the worst single cell loses 36% decode and 19 acceptance points.
- **Per-lane**: `kv-dtype` is now a per-profile field (`varying_flags`, rendered from each row), because the
  per-lane column is where an artifact-dependent setting belongs. **At that stage one row used it**:
  `start_quasar_v3_dflash2_vision.bat` ships `k8v4`, on +13.8% decode and +9.5 acceptance points on its
  published cell, +27.3/+35.2% on its served cells, for +0.144% corpus perplexity (+0.202% worst domain).
  **Corrected 2026-10-10: five rows use it now and three keep `fp8` — the outcomes table above is the count
  that stands; this sentence read as current while describing the first decision.**
- **The depth was re-swept under the new format** rather than inherited: on long Chinese the two depths tie
  under k8v4 (188.9 against 188.5 tok/s) where fp8 had depth 9 ahead by 8.8%, and on long code depth 9 leads
  by 15.1% (212.4 against 184.6, spreads 0.4/0.3%) where fp8 had depth 7 ahead by 3.3%. Maximin takes depth 9
  by 2.3% on the worst case and the served mean by 7.6%, so the shipped depth is unchanged and now rests on
  the format it ships.
- The other seven lanes keep `fp8`. That is why the flag left the invariant list: the list is for settings
  that should not vary, and the measurement says this one does.

### A defect found under this work

Moving `kv-dtype` out of `INVARIANT_FLAGS` silently broke the matrix's *probe* path: `build_args` had a
special case that substituted the requested format **only while the flag appeared in that list**, so probe
starts stopped passing a format at all, ran on the engine's default (bf16-sized KV), and were refused for
memory — while the new record field, falling back to its parameter, claimed the format that had never been
passed. Both are fixed (the probe emits the format it varies; the record reads it from the argument list with
null when absent), and `check_profile_consistency.py` now asserts the probe emits every flag it varies — seen
failing on `--kv-dtype` and passing the other six flags before the fix.

### Why the quality axis stops at perplexity

The corpus perplexity resolves the cost exactly — deterministic, 1M tokens, four domains, and the fp8 arm
reproduces the published baseline to fourteen significant figures. A generation-accuracy arm cannot resolve
an effect of this size, and the arithmetic is why (derived, not measured): for a question whose answer is
right about half the time the per-question standard deviation is 0.5, so a two-sigma resolution of a
difference of δ points needs about `1/δ²` questions, unpaired. A 1-point difference needs roughly 10,000
questions; a tenth of that needs a hundred times as many. Pairing narrows the variance of the *difference*
and so improves on that arithmetic — the harness's own 258-question, 7-11 GPU-hour design is priced for the
accuracy effects it was built for — but the order of magnitude stands: resolving fractions of a point takes
thousands to tens of thousands of questions and hours of card per arm, where the corpus perplexity resolves
the same change exactly in minutes. That is why the quality axis is measured there, and why the paired run
stays the instrument for a change whose quality signature is points rather than fractions of a point.
