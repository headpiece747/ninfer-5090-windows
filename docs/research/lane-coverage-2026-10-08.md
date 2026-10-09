# Lane coverage: what the eight shipped lanes actually do, measured 2026-10-08

This is the record of a coverage sweep across the eight shipped Qwen3.8-27B lanes. It answers a
question the records could not: which parts of the product had never been measured, and what the
measurements say now that they have. Everything here is MEASURED; where a number is an estimate or a
projection it says so. Raw records live under `profiles/bench/` (gitignored, as `bench/README.md`
documents) and the decision trail is `.audit/models-rebuild.tsv`.

## Why this sweep existed

The shipped lanes were covered on decode throughput, draft acceptance and depth sweeps for
speculation, but four things had never been measured per lane: **TTFT**, **cache behaviour**, **the
shipped 262,144-token window**, and **preemption/agent-recovery**. A fifth, **vision**, had been
measured on one lane only. A sixth, **quality at depth**, had never been measured at all.

## 1. TTFT and cache, per lane

Five cases per lane at each lane's own shipped serve arguments, parsed from its launcher rather than
reconstructed — including the launcher's `NINFER_CUDA_SYNC=blocking` and its explicit
`--chat-template`, because the launcher says why: *"A harness that starts Serve has to start it this
way too, or what it measures is not what ships."* 40 records, zero failures.

| lane | cold-short (30 tok) | 64K prefill | tok/s | warm continuation | reuse | shared-prefix revisit |
|---|---|---|---|---|---|---|
| quasar dflash2 | 71.6 ms | 6,993 ms | 9,224 | 58.8 ms | 7,695 · checkpoint | 112.4 ms |
| quasar mtp4 | 60.6 | 6,983 | 9,238 | 59.9 | 7,695 · checkpoint | 81.3 |
| ninfer dflash2 | 59.3 | 7,023 | 9,186 | 59.2 | 7,695 · checkpoint | 89.5 |
| ninfer mtp4 | 54.5 | 7,245 | 8,904 | 68.3 | 7,695 · checkpoint | 90.0 |
| swift dflash2 | 76.4 | 7,196 | 8,965 | 58.8 | 7,695 · checkpoint | 89.5 |
| swift mtp4 | 49.0 | 7,025 | 9,183 | 65.8 | 7,695 · checkpoint | 94.1 |
| nvidia dflash2 | 46.9 | 7,225 | 8,928 | 69.0 | 7,695 · checkpoint | 100.7 |
| nvidia mtp4 | 49.4 | 7,054 | 9,145 | 73.4 | 7,695 · checkpoint | 105.3 |

**The cache engages on all eight lanes**: 7,695 of 7,718 tokens reused through the `checkpoint` path
on the continuation, and 4,151 of 4,169 on the shared-prefix revisit. Cold 64K spans 7,024-7,289 ms
(3.8 %), so **cold prefill TTFT is not where these lanes differ**. Cold-short spans 47-76 ms on single
samples, which is not safe to call a lane property.

An unexplained 26 % gap remains between a lane's own configuration (7.04 s at 64K) and the campaign's
built-in cold profile (9.47 s) on the same artifact and case. The two differ in `--prefill-chunk`,
capacity, prefix reuse, host context, vision, spec route and chat template, and the tree elsewhere says
capacity changes which prefill plan is chosen. It is recorded as measured-but-unattributed.

## 2. The shipped window, exercised

`cold-long-256k` on every lane: prompt 260,081, `success/200`, cache 0. **TTFT 66.92-68.41 s**,
prefill 66.73-68.19 s (99.6 % of TTFT), 3,814-3,898 tok/s. The 262,144 ceiling is served on every lane,
and the spread is 2.2 %.

The depth curve now has five points: **11,868 tok/s at 4k, 10,482 at 45k, 9,184 at 64.5k, 6,462 at
131k, 3,850 at 260k** — a 2.4x decay from 64K to 256K. No TTFT figure may be quoted without its depth.

Measured VRAM at the shipped configuration reads `runtime 10.6 GiB | free 2.64-2.71 GiB` on every
DFlash2 lane and `9.96 GiB | 2.85-3.28 GiB` on every MTP lane, against the recorded 11.6 GiB and
~1.50 GiB free. The ~1 GiB difference matches the tiled saturation guard returning ~1.35 GiB, so **the
published figures are stale in the direction that understates headroom**. Reported, not edited:
`tools/release/profiles.py` is the authority and a published figure needs its own measurement.

## 3. Preemption, replay and snapshot recovery

The cases' premises require their own tight KV capacity — `preemption-replay` runs at
`--max-context 512 --kv-capacity 512 --no-prefix-reuse --host-context-mib 0`. A lane-faithful config
would hand them 262,144 tokens and let every case report success while measuring nothing, so each run
merges the tool's profile arguments with only the lane's spec route, vision, template and thinking
flags. **40 case-runs, and `report.py`'s own `required_mechanism_status` reads `observed` for all 40.**
`failed_conditions` never covered this; the tool says so itself — *"Required mechanisms do not change
whether measured latency is retained."*

| case | replayed | time | host traffic |
|---|---|---|---|
| `preemption-replay` | 238-255 tokens | 65.5-75.7 ms | **zero** |
| `agent-continuation-replay` | 424-448 tokens | 130.0-150.6 ms | **zero** |
| `*-snapshot` cases | — | — | state 147-560 MiB d2h + 147-374 MiB h2d, KV 8-36 MiB |

**A replay moves zero bytes in either direction** — it recomputes from a device-resident checkpoint,
while a snapshot restore is the path that round-trips through host memory. Figures are near-identical
across all eight lanes, so **recovery does not depend on the artifact or the spec route**. The
tracker's symptom did not reproduce: no 23 s post-eviction TTFT, no 30-36 s agent stall.

## 4. Vision

| case | prompt | reuse | vision phase | TTFT |
|---|---|---|---|---|
| `media-cold-image` | 428 | 0 · root | 16.1-19.8 ms | 128.2-149.7 ms |
| `media-prefix-continuation`, second turn | 464 | 441 · checkpoint | **0.0 ms** | 46.7-62.9 ms |
| `media-prefix-append`, new image | 1,490 | 441 · checkpoint | 43.3-45.1 ms | 189.6-218.9 ms |

**A zero vision phase is the proof the media cache engages** — preprocessing is skipped, not hidden.
24 runs, all `success`, no error codes, and again near-identical across lanes.

## 5. Quality at depth, for the candidate decision

`ninfer-perplexity` over `eval/corpora/perplexity-1m/manifest.json`, identical protocol per depth:

| arm | PPL @4K (496 windows) | vs shipped | PPL @64K (16 windows) | vs shipped | per-64K-window |
|---|---|---|---|---|---|
| shipped nvidia | 4.68676 | — | 4.43339 | — | 10.3 s |
| attn8 (official build) | 4.64891 | −0.81 % | 4.39967 | −0.76 % | 10.4-10.5 s |
| a8policy (official build) | 4.66132 | −0.54 % | 4.42387 | −0.21 % | 15.0-15.1 s |

Today's figures reproduce the earlier prototype measurements to six significant figures (a8policy
4.661317 recorded, attn8 4.648907 recorded), so the landed recipes reproduce their prototypes.

## The candidate A/B, and a correction it forced

TTFT A/B on the nvidia lane's serve args, arms interleaved and repeated, 64,509-token prompt:

| arm | prefill | rate | vs shipped |
|---|---|---|---|
| shipped nvidia | 7,021.9 / 7,176.8 ms | 8,989 / 9,187 tok/s | — |
| a8policy_official | **11,609.4 / 11,628.8 ms** | 5,547 / 5,557 tok/s | **+62 / +65 %** |
| attn8_official | 7,176.4 / 7,205.5 ms | 8,953 / 8,989 tok/s | +0.4 / +2.2 % (neutral) |

**This contradicts a recommendation made earlier in the same session**, on the strength of a −47.6 %
prefill figure from `attention-topology-2026-10-07.md`. That figure was measured with `ninfer_bench`
at **pp2048**; the serving route at 64,509 tokens reverses it. The same session made the same error
twice — the first application of a pp2048 rate at 64K produced a wrong "46 % of TTFT is prepare"
claim. **A rate measured at one point on the depth axis does not transfer to another, and here it
inverts.** The serving route's number is the one that describes the product.

Decision shape from everything measured today: **attn8 dominates a8policy** — better quality at both
depths, no 64K prefill penalty, TTFT-neutral, at +3.9 % size. a8policy retains its claim on short-turn
decode and acceptance from an earlier measurement that has **not** been re-verified on the serving
route, and its quality advantage halves at depth while its throughput cost grows.

## Depth choice: the maximin answer for the four DFlash2 lanes

The gap noted above (`nvfp4fullnoex` had depth 7 only) was closed with the full 1-15 sweep, and then the
question the siblings' original sweeps could not answer was asked of all four lanes: what does the
**worst** domain say, since the lanes ship under a maximin rule and code is this port's most favourable
domain. Depths 7 and 9, five domains each, one invocation per domain so every comparison comes from a
single rotated run:

| lane | code | prose | chinese | dialogue | repetition | worst domain d7 → d9 | choice |
|---|---|---|---|---|---|---|---|
| nvfp4fullnoex | 300.9 → 368.9 | 161.3 → **154.9** | 141.0 → 143.1 | 197.3 → 259.5 | 211.6 → 237.2 | 141.0 → **143.1** | **d9 (+1.5 %)** |
| quasar | **384.3** → 337.4 | **157.2** → 145.5 | 146.8 → 149.8 | **232.6** → 230.7 | 194.3 → 237.1 | **146.8** → 145.5 | d7 (−0.9 %) |
| nvidia | 265.5 → 325.4 | **157.3** → 151.8 | 139.2 → 139.6 | **220.9** → 208.7 | 217.0 → 237.2 | 139.2 → 139.6 | d9 (+0.3 %) |
| swift15 | 327.1 → 361.7 | **150.4** → 143.3 | **148.3** → 143.9 | 201.2 → 239.6 | 204.9 → 252.7 | **148.3** → 143.3 | d7 (−3.4 %) |

All four lanes shipped depth 7 when this table was taken; commit 8306f79d then moved nvfp4fullnoex to 9 on
its deciding domain, so `profiles.py` is authoritative and carries 7/7/9/7 -- see the re-measurement below.
The maximin answer **confirms 7 for quasar and swift15**, makes it a **tie on
nvidia** (+0.3 %, well inside a single invocation's spread), and favours **9 on nvfp4fullnoex by 1.5 %**.
So the shipped policy is defensible and the only candidate change would be one lane, by a margin that
wants a repeat before it is shipped.

**The finding that matters more than the ranking**: under maximin the depth choice is nearly degenerate.
Whichever depth is chosen, a lane's worst-domain throughput lands within 3 % — while individual domains
swing 10-31 % between the two depths. The deciding domain is chinese or prose in every lane, which are
exactly the domains where speculation helps least. A rule that decides on the worst domain is therefore
also a rule that can be moved by a 0.3 % margin, which is worth knowing the next time a shipped value is
changed on it.

Two instrument facts this required, both worth carrying forward. **`--domain` accepts repeats and honours
only the first**, so a multi-domain invocation silently runs one domain. And **`DOCUMENTED_SAMPLING` is
temperature 1.0**, so text, digest and acceptance all vary between runs by construction: within one
invocation the three sub-runs agree to 2-3 %, across invocations the same configuration can differ by
tens of percent. Only within-invocation comparisons are usable, which is why every figure above is one
invocation. What discriminates a different **engine build** is `runtime_gib` (11.6 GiB before the
saturation guard, 10.6 after) — not the digest, which the sampling makes uninformative.

## Re-measured 2026-10-09: the depth choice under a traffic-like prompt, and what is still unsettled

The table above was measured at `DOCUMENTED_SAMPLING` (temperature 1.0) on the matrix's 90-260 character
domain prompts. It was re-taken 2026-10-09 with `tools/bench/realtext_acceptance.py`, which now takes
`--draft-tokens`, on long real prompts, the same artifact, two rounds per cell, every cell identical
across rounds because this instrument is greedy:

| treatment | depth 7 | depth 9 | change |
|---|---|---|---|
| the instrument's default corpus file, 3 x 12,000 chars (this repository's own C++ source) | 229.7 / 229.2 tok/s | 233.9 / 233.7 | **+1.9 %** |
| real code, 3 x 12,000 chars from three files | 220.2 / 214.6 | 214.5 / 214.0 | -1.4 % (2.6 % round spread) |
| the matrix's 225-character code prompt | 408.3 / 409.0 | 347.4 / 346.7 | **-15 %** |

**The last row contradicts the table's +22.6 %** on the same artifact, prompt and depth pair, and the
artifact is unchanged (mtime 2026-10-07 12:17), so the difference is protocol rather than build. The
likely variable is the one this document already flags above: the table was measured at temperature 1.0
and this instrument is greedy, so the two are measuring the depth's economics on differently-sampled text.

**Neither protocol settles the served case, and that is the finding.** The lanes serve at the model's own
sampling configuration, which is temperature 1.0 -- so the table's protocol matches what is served and its
prompts do not, while this instrument's prompts match and its greedy sampling does not. What the served
decision needs is the third combination: traffic-like prompts, at the served sampling, with the two depths
interleaved in one session rather than measured in separate invocations, because at temperature 1.0 the
across-invocation spread is tens of percent while within one invocation it is 2-3 %. Until that is run,
**depth 9 stands as shipped** for nvfp4fullnoex: under greedy it is +1.9 % on the instrument's default corpus
file and -1.4 % (inside
the round spread) on long code, so it is neutral to slightly better on both regimes the lane serves.

### Settled 2026-10-09: the decision is degenerate at this card's noise level

The third protocol was run: traffic-like prompts, the served sampling (temperature 1.0 with the model's own
top_p/top_k), and the two depths **interleaved in one session** with the order rotated pair by pair, six
pairs per treatment, through `tools/bench/paired_depth_sweep.py`, and re-derived per request from the serve
logs with each run's first request dropped -- because the matrix's own note documents that one as a
transient (its MTP5 lane reads 261.7 tok/s on request 1 against 169.8 for requests 2..n).

| treatment | mean | median | sd | worst pair | best pair |
|---|---|---|---|---|---|
| the corpus's C++ source (this repository) | -2.4 % | -3.9 % | 9.7 % | -15.4 % | +11.9 % |
| three mixed code files | -2.0 % | +0.5 % | 7.9 % | -16.8 % | +4.3 % |
| long Chinese (`zhwiki/01.txt`, 67 % CJK) | -5.7 % | -11.8 % | 20.9 % | -22.5 % | +30.6 % |

**Every mean is negative for depth 9 and every one is inside its own noise**: effects of 2-6 % against
standard deviations of 8-21 % over six pairs, so each mean's standard error is 3-9 % and no sign is
established. The two protocols also disagree in sign on the same treatment -- greedy reads +1.9 % on the
corpus's C++ where the served sampling reads -2.4 % -- and both readings are inside their spreads.

**That is the settlement: the choice is degenerate, so depth 9 stands and further measurement is not
warranted.** §146 said this before any of these runs: under maximin a lane's worst-domain throughput lands
within 3 % of either depth, and the rule's own deciding margin was 1.5 %. Resolving a 2-6 % effect at this
spread would need tens of pairs per treatment, for a decision worth at most 6 % of one lane. What improves
instead is the rule's input: `v3_profile_matrix.py --domain-from-file NAME=PATH` registers a prompt from a
file and carries its name into every record, verified by a run whose records read
`domain=probe-cpp sampling=default`, so the next maximin re-run can use prompts resembling what the lanes
serve instead of the 90-260 character constants.

**Why the incumbent stays rather than the smaller mean being taken as the answer.** A negative mean whose
standard error is larger than itself is not evidence that depth 9 is slower; it is the absence of a
resolvable difference, and a shipped value is not changed on that -- the burden of proof sits with the
change, which would move one lane by a few percent of 220 tok/s while invalidating every figure that
depends on it. Three of three treatments reading negative is mildly suggestive on its own (one in eight,
if the three were independent coin flips), but they share one lane, one card and one session, so they are
not independent and no sign is claimed. What would settle it is a difference larger than the spread; the
table's own +22.6 % would have done, if it had reproduced.

## Open items

- The **fixed ~45 ms inside the `prefill` phase** on near-zero-work requests, seen three independent
  ways (cold-short 43.6-51.6 ms for 30 tokens; a text continuation 43.8 ms for 23 new tokens; a
  vision continuation 43.5-47.9 ms for 23 new tokens). **Measured 2026-10-09: it is not KV page
  allocation.** The same 200-character cold request reads a floor of 61.2 ms at `--kv-capacity auto`
  (`KV 262,144`, 4,096 pages) and 63.4 ms at `--max-context 32768` (`KV 32,768`, 512 pages) -- eight
  times fewer reserved pages, the same cost. Combined with the original observation that it appears on
  continuations as well as cold shorts, that makes it a **per-request** cost in the request path rather
  than a page-allocation or a per-conversation restore. Which request-path step it is remains open.
- The published lane figures in `tools/release/profiles.py` are stale for VRAM (~1 GiB) and were
  already noted stale for throughput.
- **item 1-5, 8 of the coverage list are now closed.** Still open and instrument-ready: soak, the
  cancellation and queue cases, spec correctness at temperature > 0, and cross-path determinism.
- **Needing build work, and the most promising for making the models better**: the precision-assignment
  search is not exhausted (no arm isolates q/k from v/o, or moves MLP activations from A8 to A16), and
  **the drafter's own precision has never been varied while the target is held fixed** — a lever
  directly on acceptance.

## Raw records

| what | where |
|---|---|
| TTFT/cache, eight lanes | `profiles/bench/ttft-2026-10-08-lanes/` |
| 256K per lane | `profiles/bench/ttft-2026-10-08-256k/` |
| preemption and agent replay | `profiles/bench/ttft-2026-10-08-preempt-agents/` |
| vision | `profiles/bench/ttft-2026-10-08-vision/` |
| candidate A/B | `profiles/bench/ttft-2026-10-08-candidates/` |
| quality at depth | `profiles/perplexity/depth-2026-10-08/` |
| decision trail | `.audit/models-rebuild.tsv` |
