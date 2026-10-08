# What the published `.ninfer` field ships, and the trade it states

**Research note. Written 2026-10-08.** Sources read that day: the model cards of the publishers
named below (fetched from the Hub in full), upstream `Neroued/ninfer` at `81c8ce09` (`upstream/dev`:
README, `docs/cli.md`, `docs/serving.md`, `docs/performance/qwen3.8-27b.md`),
`z-lab/Qwen3.8-27B-DFlash2`, the fork READMEs of `Wallawalla47/ninfer-custom` and
`iamwavecut/ninfer-all` (including its September 2026 reference measurements), and the upstream
tracker via `gh`. **Nothing here was measured by this session.**

Labels: **[read in source]** — read in a card, README, doc or config; **[measured]** — a figure the
named party reports having measured; **[third party]** — a community source, not upstream and not
this port. Every number carries its configuration.

## 1. The default topology per publisher

Every artifact below is a Qwen3.8-27B derivative in a v3 `.ninfer` container unless noted. "4-bit"
means NVFP4 sites; the groupwise artifacts use Q4/Q5/Q6/Q8 instead.

| Artifact (file size) | 4-bit sites | 8-bit / BF16 sites | Card's serving profile |
|---|---|---|---|
| `neroued/Qwen3.8-27B-nvfp4-NInfer` (23,719,715,844 B, 22.09 GiB) | MLP 0–55, 112 tensors | attention in/out, GDN qkv/z/out, embed, head, MLP 56–63 = row-FP8 (146); controls BF16 | `--spec mtp --draft-tokens 3 --lm-head-draft`, fp8 KV, 240,000; DFlash2: K7 |
| `neroued/Qwen3.8-27B-NInfer` (20,437,521,664 B, 19.03 GiB) | none | Q4/Q5 projections; Q8 embed/head | same MTP3 / fp8 / 240,000 |
| `d0xin/…LocalHessian-ActivationHeadroom-NInfer` (23,719,715,844 B) | MLP 0–55 W4A4 | attention + GDN + MLP 56–63 + head + embed row-FP8; GDN a/b BF16 | 5090: DFlash2 K5, fp8 KV, 250,000, thinking budget 2048; RTX PRO 6000: 262,144, C8 |
| `WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3` (15,017,456,896 B, 13.99 GiB) | none (GGUF IQ3_S/IQ4_XS/Q4_K…, 3.5 bpw) | MTP head Q6_K; GDN a/b BF16 | MTP3, rk8v4, 176,128 (24 GB) / 262,144 (5090); DFlash2 K5 as the alternative |
| `WaveCut/Huihui-Qwen3.8-27B-abliterated-NInfer-v3` (20,437,521,664 B) | none | official Q4/Q5 + Q8 vocab; DFlash2 adapter Q8 | MTP3 + lm-head, rk8v4, 198,400 |
| `WaveCut/…HauhauCS-Aggressive-DFlash2-NInfer-v3` (20,437,571,568 B) | none (groupwise Q4/Q5) | BF16 627 / FP32 96 tensors | DFlash2 K7 + lm-head, 32,768 auto; MTP3 as the alternative |
| `WaveCut/…MXFP8-CRACK-NInfer-v3` (18,210,749,936 B) | none (groupwise from MXFP8) | — | MTP3 + lm-head, 32,768 auto; carries no DFlash2 |
| `wallawalla47/…NVIDIA-NVFP4-NInferV3` (22,783,240,452 B, 21.2 GiB) | 128 MLP parents, imported bit-exact | 130 attention/GDN FP8 imported bit-exact; head + embed re-quantised FP8; GDN a/b BF16 | DFlash2 K7 + ngram 15/12, int8 KV, 240,000, thinking budget 16384 |
| `jgamboa/Qwen3.8-27B-NInfer-4090` (20,437,521,664 B) | none — official groupwise weights byte-identical | 512 projection inputs moved `A16Only`→`AllowA8` | MTP3 + ngram chain, rk4v4-e8, 100,000 (4090) |
| `MirkoCovizzi/…QUASAR-NVFP4-NInfer` (19,782,432,752 B, 18.42 GiB) | 256 parents: attention 32 / GDN 96 / MLP 128 | GDN a/b BF16; Q8 head/embed; DFlash2 Q8/BF16 | MTP3 + kvarn; server example DFlash2 K7 int8; validated to K15 |
| `kaushikvira/…swift15…v3` (+ 3 siblings) (19,782,447,364 B, 18.42 GiB) | 256 fused text parents, all-NVFP4 | W8G32 embed/head; MTP + DFlash2 BF16 | DFlash2 K7 + lm-head, k8v4, 262,144, C4, thinking budget 16384 |
| `cometkim/…nvfp4full-NInfer` (19,407,229,188 B, 18.07 GiB) | 247 parents + NVFP4 DFlash2 module | 9 BF16 exception parents; W8G32 embed/head | DFlash2 K7, int8, 262,144 |
| `cometkim/…nvfp4qat-NInfer` (18,638,510,576 B, 17.35 GiB) | 256 parents, 0 exceptions | W8G32 embed/head | DFlash2 K7, int8, 262,144 |
| `CaptainArni/Swift-1.5-Qwen3.8-27B-NInfer` (22,783,241,220 B, 21.2 GiB) | all 64 MLP layers | 144 GDN + 64 attention FP8; head/embed FP8; small GDN + draft BF16 | DFlash2 K7 + lm-head, nvfp4 KV, 262,144; MTP3 as the cheaper alternative |
| `Ostfralla/…NVFP4-NInfer` (18,324,067,840 B, 17.07 GiB) | NVFP4 (DeltaNet projections named; card does not itemise the rest) | GDN a/b BF16 | MTP4 + lm-head, int8, 262,144 auto, C6 |
| `kybrcore/…QUASAR-NVFP4-NInfer` (19,782,448,132 B) | 496 text projections NVFP4 | a/b BF16; Q8 endpoints | MTP3 + lm-head, nvfp4 KV, 262,144 |
| `ninfer-5080/Qwen3.8-27B-RTX5080` (16,461,267,456 B) | none (Q3/Q4/Q5, 3.953 bpw) | — | MTP3, q4 KV, 131,072, thinking budget 2048 |
| `igorls/…OrcaRouter…` (26,268,683,012 B, 24.46 GiB) | source mixed NVFP4/FP8 | BF16 embed/head preserved | DFlash2 K7 + lm-head, or MTP5; fp8 KV, 32,768 auto, C4 |
| `lyf/…Huihui…NVFP4` (21,492,695,040 B) | MLP 0–55 NVFP4 | attention + GDN FP8; head FP8 | MTP3, int8, 204,800 |
| `kvnxiao/…orcarouter-dflash2…` (22,783,240,452 B) | ModelOpt NVFP4 MLP | attention/GDN FP8; head/embed FP8 | MTP3, fp8, 32,768, vision; DFlash2 K7 alt |
| `hamixdd/…W4A16-RTX3090…` (~20.4 GB) | none (W4A16 groupwise) | — | DFlash2 K7 + lm-head, 172,032 (3090) |
| `2beng2/…OrcaRouter-GSQ-RCO-IQ3_XXS…` (12.13 GiB) | none (IQ3_XXS blocks) | — | DFlash2 K4 + lm-head, rk8v4, 262,144 |

The official card states its split directly: "Text layers 0–55 use NVFP4 MLP weights, while the
token embedding, attention input/output projections, GDN Q/K/V/Z and output projections, full
output head, and Text layers 56–63 MLP weights use row-scaled FP8. Control weights use BF16, with
separate MTP, Vision and DFlash2 weights." [read in source, `neroued/Qwen3.8-27B-nvfp4-NInfer`]

## 2. The trade the publishers state

### 2.1 Two topology families, and why each says it chose one

**"Keep the source's FP8 attention/GDN."** The official nvfp4 artifact, `d0xin`, `wallawalla47`,
`CaptainArni`, `lyf` and `kvnxiao` all import the source checkpoint's mixed split rather than
re-encoding it. `wallawalla47` states the reason: "No MLP or attention weight is dequantised and
then re-quantised: the 128 NVFP4 MLP and 128 FP8 attention/GDN objects are direct imports of the
checkpoint's stored weights, so the base model's quality is preserved by construction." [read in
source] `d0xin` calls its split "a conservative mixed NVFP4/FP8 topology" and sets the release goal
as "increase inference throughput and reduce VRAM usage while preserving FP8-level measured quality
and avoiding longer reasoning traces under the recommended B2048 serving profile". [read in source]

**"Quantize nearly everything to NVFP4."** `cometkim` (`nvfp4full`: "nearly the whole Text
backbone"; `nvfp4qat`: "every one of the 496 text linear layers is NVFP4 … with no high-precision
exceptions"), `kaushikvira` ("NVFP4 W4A4 group-16 on every text projection"), `MirkoCovizzi`
(QUASAR, 256 parents) and `kybrcore` go the other way. The closest published A/B is `cometkim`'s:
the NVIDIA-sourced re-quantization improved perplexity but lost speed — "Measurements using the
ninfer fixed corpus show that while PPL does indeed improve, the MTP/DFlash2 acceptance rate drops
significantly", with end-to-end MTP-3 −8.90 %/−9.67 % and DFlash2-K7 −4.06 %/−4.24 % on one RTX
5090. [third party, upstream #214 comments] The same card records the drafter-encoding trade: "the
NVFP4 module drafted 5.50 tokens/round at 64.3 % acceptance, against 5.75 at 67.9 % for the
`W8G32_F16S` encoding of the same drafter — the ≈1 GiB saving trades a few points of drafter
acceptance." [measured, `cometkim/…nvfp4full-NInfer`]

**Nobody publishes the same-checkpoint FP8-vs-NVFP4 attention A/B.** The port's own
`docs/research/attention-topology-2026-10-07.md` is the only measurement found that isolates it
(1.41 % PPL, +29 % relative DFlash2 acceptance, −14.2 % prefill for the full split; attention alone:
−0.81 % PPL, +77 % acceptance, prefill inside the noise, +0.73 GB) [measured here]. The published
field's nearest evidence is `cometkim`'s acceptance-drop measurement, in the same direction.

### 2.2 Quality they publish

| Claim | Value | Configuration |
|---|---|---|
| Official nvfp4 vs groupwise-int | GPQA-D 90.40 vs 87.37; IFBench 77.00 vs 77.67 | EvalScope 1.9.0, thinking, MTP3, INT8 KV [measured, upstream] |
| `d0xin` vs its FP8 build | MMLU-Pro subset 225/280 vs 224/280, McNemar p=1.0; FP8+B2048 224/280 | 280 questions, temp 0, effort xhigh, RTX PRO 6000 [measured] |
| `cometkim` nvfp4full vs nvfp4qat | GPQA-D 87.88 ± 2.62 vs 89.22 ± 2.49; AIME26 93.33 ± 3.34 vs 91.11 ± 3.85; LBv2 short 67.41 vs 66.30 | 3-round mean, 5090, INT8 KV, MTP3 [measured] |
| `jgamboa` int8 prefill vs official | perplexity 4.794439 vs 4.800742; 45/45 tasks both | 4090, byte-identical weights, activation permission only [measured] |
| `WaveCut` IQ3_S vs official | WikiText-2 PPL 7.071 vs 7.286; IFBench 80.33 vs 77.67 | 5090, own protocol [measured] |
| `Ostfralla` NVFP4 vs official groupwise | HumanEval+ 152/164 both; AIME25+26 55/60 both | greedy, 224 paired problems [measured] |

### 2.3 Prefill/TTFT is the number the agentic cards organise around

Upstream's own context profile (RTX 5090, FP8 KV, MTP disabled, no prefix reuse) makes prefill the
dominant phase at depth: nvfp4 12,819.1 tok/s at 7,680 → 4,016.4 at 260,096, with server TTFT
602.7 ms → 64,910.0 ms; groupwise-int 3,331.9 → 2,139.4 tok/s and 2,308.5 ms → 121,723.1 ms.
[measured, upstream] `kaushikvira` labels its prefill benchmark "the agentic-coding profile (long,
growing context; small incremental prompts)". [read in source] `wallawalla47`'s agentic A/B
(130 requests, 25K–135K prompts, DFlash2 K7, int8 KV, RTX 5090/Windows) is the strongest first-party
statement of what prefill is worth: its fork "serves 90.5 % of prompt tokens from cache against
85.8 %, prefills 33 % fewer prompt tokens, answers 39 % sooner on average (48 % at the median) and
finishes the workload 26 % sooner" than upstream. [measured] `MirkoCovizzi`'s laptop measurements
report the same lever from the other side: prefill chunk 256→1024 moved an 18-request agentic loop
from 1030 to 1472 tok/s and TTFT from 55.6 to 40.4 s. [third party, fork #17]

### 2.4 Context capacity is what gets given up

- `d0xin` (RTX 5090): "the full `--kv-capacity 262144` profile with Vision enabled does not fit
  within 32 GB"; its 5090 profile is text-only at 250,000. [read in source]
- `cometkim`'s HyperQuant KV codec "exists because of a hard memory constraint: on the 32 GB RTX
  5090 the ≈16 GiB of device weights leave room for at most ~262k tokens of INT8 group-64 KV, so
  the 524k, 786k envelopes only fit at hq's ~9× smaller per-key footprint." [read in source]
- `CaptainArni`: DFlash2 costs "~1.6 GiB more weights plus ~0.9 GiB of runtime" than MTP3; its
  262,144-token + vision row runs DFlash2 at 20.8 GiB weights / 7.07 GiB runtime / 2.80 GiB free.
  [measured]
- `iamwavecut/ninfer-all`'s largest-window table (RTX 5090, one request, the upstream artifact, as
  reproduced in WaveCut's card) is the clearest published capacity price: rk8v4 598,016 tokens
  without speculation, 544,768 with MTP3, 491,520 with DFlash2-5; rk4v4 872,448 / 794,624 / 716,800.
  WaveCut's Huihui card states the KV side of the trade: "`rk8v4` is a lossy
  KV format (about +0.08 % perplexity against bf16 on the vanilla checkpoint) chosen for the
  context window; `--kv-dtype int8` or `bf16` trade window for fidelity." [measured / read in source]
- Upstream's own evaluation could not run its 262,144-token text suite on the nvfp4 artifact: "Text
  evaluation used 262,144 tokens except Qwen3.8-27B NVFP4, which used 252,928 tokens to fit the RTX
  5090 after weights." [read in source, upstream README]

### 2.5 Thinking-budget tiers

`d0xin`'s B2048 tier is the one with a stated reason: without a budget, 36/280 generations hit the
4096-token ceiling; with it, reasoning median 379.5, p90 2072, 0/280 truncated, accuracy 225/280.
[measured] `wallawalla47` and `kaushikvira` run `--default-thinking-budget 16384` (the former with a
custom stop message), `ninfer-5080` runs 2048, and upstream ships the template default "thinking on;
effort `xhigh`" with no budget. [read in source]

## 3. Speculative-mode guidance: DFlash2 vs MTP

**Upstream recommends DFlash2 K7 for the DFlash2 path and demonstrates MTP3 everywhere else.** The
CLI guide: "For Qwen3.8-27B artifacts containing the DFlash2 companion weights, select `--spec
dflash2 --draft-tokens 7` … DFlash2 accepts every draft count from 1 through 15; **seven is the
checkpoint recommendation**." The README and every model-card example use MTP3; the CLI guide says
the published results use "MTP with three draft tokens and DFlash with seven draft tokens (block
length eight)". [read in source] The DFlash2 announcement (#188) measured K7, int8 KV, one RTX
5090: 321.1 tok/s AIME26#1, 265.5 code, 356.8 structured, 121.3 story (nvfp4). [measured]

**The drafter's own publisher** compares at 7 draft tokens on SGLang/H200: acceptance length 5.46
vs MTP 5.02 (GSM8K), 4.39 vs 3.91 (HumanEval); C1 throughput GSM8K 236.1 vs 178.5 tok/s. DFlash2
wins every task in that table. [measured, `z-lab/Qwen3.8-27B-DFlash2`]

**Third-party defaults split, and the depth is not uniformly 7.** Most DFlash2 cards pick K7
(`cometkim` — "the DFlash2 drafter (single parallel draft pass; the recommended lane)" — plus
`kaushikvira`, `wallawalla47`, `CaptainArni`, `kvnxiao`, `hamixdd`, `2beng2` K4, `igorls`).
`d0xin` picks K5 for the 5090. `MirkoCovizzi` states "MTP accepts 1–5 draft tokens; DFlash2 accepts
1–15. The best depth depends on the workload." MTP3 is the default on WaveCut's groupwise cards
(GSQ-RCO, Huihui, MXFP8-CRACK; HauhauCS defaults to DFlash2 K7), `lyf`, `kybrcore`, `Ostfralla`
(MTP4), `ninfer-5080` and `jgamboa`'s 4090 (MTP3 + ngram). [read in source]

**The measured depth curves** (`iamwavecut/ninfer-all`, Qwen3.8-27B, one RTX 5090, rk8v4, one
request, decode tok/s with acceptance in brackets) show why there is no single right depth:
short chat DFlash2 peaks at K7 (236.2, 0.36) over MTP K3 (184.2, 0.59); at 131,072 tokens DFlash2
K4 (164.0) beats K7 (145.0) and MTP K4 (134.2); at C=8 MTP3 (690.0) beats DFlash2-5 (469.4).
[third party, measured] `CaptainArni` prices the choice: DFlash2 "is faster on code and structured
output" for +1.6 GiB weights and +0.9 GiB runtime. [measured]

**Two caveats the field states.** The DFlash2 companion is model-specific: a finetune conversion
"accepted 3.3–5.1 % of drafts and decode fell to 67–75 tok/s, versus `--spec mtp --draft-tokens 3`
at 175–206 tok/s with 67–85 % acceptance". [third party, upstream #298] And greedy output can
depend on draft width: on a laptop 5090, MTP off vs K3 diverged at token 3 of 512, while the
proposal head was parity-safe. [third party, fork #17]

## 4. What users report as the bottleneck

**Time-to-first-token and prefix reuse, not quality or decode speed.** Every long-context agentic
report found on the tracker is about a cache miss turning into a full re-prefill:

- `qwen3_8_27b_nvfp4.ninfer`, Cline agentic coding, C1: "`reuse=append_frontier` 353 requests ttft
  ~270–300 ms … `reuse=restore_turn_checkpoint` 71 requests ttft 30,000–36,000 ms"; prompts growing
  136,133 → 149,705 tokens with the cache stuck at 6,887. [#77]
- 3090 Ti fork, 23k-token chat: every web-UI turn paid "a full re-prefill of an unchanged ~23k-token
  prompt (~23 s median TTFT)" after a 53-token utility request evicted the checkpoint, against
  0.03–0.16 s when the hit path ran. [#181]
- RTX 5090, Qwen3.8-27B NVFP4, C2: "16k shared system, no breakpoint, two sibling users — both
  `root`, 0/16484, 2363/2363 ms"; with a breakpoint, `shared_stable_prefix` at 91 ms. [#142]
- 5-agent fleet, C1, official nvfp4, int8 KV: "~54 s ttft on a 35k-token prompt that was mostly
  waiting for the slot; its actual prefill was ~5 s". [#40]
- RTX 6000, c=8 across agents: "any speed gains were completely eaten by constant large-context
  prefill". [#62]
- "Prompt caching is way less efficient compared to llamacpp … llamacpp reuses way more prompt cache
  and in the end is way faster with agentic coding", both given 50 GB of RAM. [#269]
- Production traffic: "98.5 % of served prompt tokens had been computed before … yet the engine's
  hit rate was 72 %", and "the first long request of every session pays a full 20–60 s prefill
  again". [#350]
- Planner stalls and reuse death: TTFT ≈ 142 s from a bounded planner search [#229]; 26k prompts
  that reused at 166 ms then stopped reusing for the engine's life, 14 s TTFT, until restart [#251].

Upstream's own rewrite announcement acknowledges the pattern and reports the fix at C=8: "total
runtime for 75 requests went from about 1,443 seconds to 956 seconds, and aggregate decode
throughput went from 491 to 713 tok/s" (#366, Qwen3.8-27B NVFP4). [measured, upstream]

**Quality is not the complaint.** No tracker report found frames the quantized artifacts as
user-visibly degraded; quality appears only as acceptance (above) or as a plan for the next
profile (#214, #70). The one long-context use report frames NInfer as the *only* option that runs
the whole job: "this was the only engine we found that runs the full 262,144 context with a real
4-bit KV cache and DFlash2 at the same time — SGLang's NVFP4 KV path hit `KeyError:
'float4_e2m1fn_x2'` … and `trtllm_mha` is SM100-only. NInfer just worked." Its measured run:
k8v4, DFlash2 K3, thinking off — ~120–180 tok/s decode, ~3,470 tok/s prefill on a 163,909-token
prompt, accept ~49 %, needle at 164k passed. [#192]

**Engine reputation, on that evidence.** NInfer's known strength here is single-GPU long-context
capacity (262k + 4-bit KV + speculation) and raw decode; its known weakness is context-cache reuse
against llama.cpp for agentic loops, which upstream has been actively rewriting (#366), and the
forks sell themselves on prefix caching and prefill speed (`Wallawalla47/ninfer-custom` measured
above; `iamwavecut/ninfer-all` benchmarks prefill and TTFT per card).

## What the published defaults imply, and where they disagree

**The field's default, read across the table:** MLP-only NVFP4 with FP8 attention/GDN and an FP8
(or Q8) embed/head, DFlash2 K7 where a DFlash2 companion exists (MTP3 where one does not),
int8/fp8 KV, a context target of 240,000–262,144, and a thinking budget between 2,048 and 16,384.
The official nvfp4 artifact keeps attention/GDN at 8-bit, and so do `d0xin`, `wallawalla47`,
`CaptainArni`, `lyf` and `kvnxiao` among the NVFP4 artifacts (the WaveCut and `jgamboa` lanes use
groupwise Q4/Q5 instead). A port adopting the FP8-attention recipe would move toward the field's
default, not away from it.

**Where the published defaults disagree:**

1. **Attention/GDN at FP8 vs NVFP4.** Six NVFP4 artifacts keep FP8 (upstream, `d0xin`,
   `wallawalla47`, `CaptainArni`, `lyf`, `kvnxiao`); six quantize it (`cometkim` ×2,
   `kaushikvira`, `MirkoCovizzi`, `kybrcore`, `Ostfralla`). No one publishes a same-checkpoint A/B;
   the one measured move to NVFP4 lost acceptance, which is the axis the port measured.
2. **DFlash2 vs MTP as the default.** Upstream's quick-start and most groupwise cards use MTP3;
   upstream's own CLI calls K7 "the checkpoint recommendation" and every DFlash2 card ships K7. The
   port shipping both is more choice than any single publisher offers.
3. **Depth.** 7 is the published default; the measured optimum is workload-shaped (K4 at 131k
   tokens, K5 for `d0xin`'s 5090, K3–4 at higher concurrency).
4. **KV dtype.** int8, fp8, k8v4, nvfp4, rk8v4 and q4 all appear as defaults; each is chosen to make
   a context target fit, and only `WaveCut` states the quality price of its codec.

**What the published field does not answer.** No publisher reports the prefill/TTFT cost of keeping
FP8 attention (only `cometkim` reports the acceptance cost of the opposite move); none offers an
attention-only FP8 variant; none publishes a port-style choice among lanes on a fixed 32 GB card;
and no card recommends a default for hardware other than the one its measurements came from.
Those gaps are exactly where the port's own measurements
(`docs/research/attention-topology-2026-10-07.md`, `lane-regression-2026-10-07.md`) stand alone.
