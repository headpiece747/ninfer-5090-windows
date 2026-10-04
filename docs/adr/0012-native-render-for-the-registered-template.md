# ADR-0012: the registered template is rendered in C++, gated by its digest

Status: accepted and implemented (2026-09-23). The native renderer is landing; the
[verification](#verification) section states what was measured and what is still projected.

## Context

On a cached prefix, host preparation is the whole of time-to-first-token. ADR-0010 removed the
tokenize half by splicing at a verified seam, and ADR-0011 removed one of the render's two probe
renders. What remains, measured at 229 messages:

| term | cost | source |
|---|--:|---|
| the frontend's own work (context build, layout, copies) | **3.47 ms** | the bench's `{{ messages\|length }}` arm |
| the interpreter's execution of the template | **~24 ms** | the bench's `both probes off` arm |
| the next-turn probe | **~24 ms** | ADR-0011's measurement |
| **total render** | **48.3 ms** | the bench's shipped arm |

The field log's own split puts that at `prepared` ≈ 55 ms and warm TTFT ≈ 70 ms, for a request whose
prefill is served from cache and whose tokenize is 0.5 ms.

Neither remaining term is addressable in place. The ~24 ms is the vendored interpreter's architecture:
`third_party/llama-jinja` is llama.cpp's `common/jinja` engine, and it is an AST walk by design —
*"each statement or expression recursively calls `execute(ctx)`"*. Its known quadratic trap
(PR #27034) is already fixed in this copy, and the remaining cost is per-value work over a piece-table
string model, which is a library redesign rather than a port's fix. The next-turn probe cannot be
removed either: it answers a content test (the template computes `last_query_index` by testing whether
a user message's content is wrapped in `<tool_response>`), so no option and no shape signature
determines it.

What the field does for a *known* template is the shape this decision adopts. llama.cpp ships a
hand-written renderer beside its Jinja engine — *"a small built-in dispatcher in `src/llama-chat.cpp`
for known formats… renders messages directly in C++ … without needing Jinja"* — with Jinja as the
generic fallback for templates it does not know.

## Decision

Render the **registered** template in C++, with the Jinja path kept as the generic route and as the
oracle that proves the C++ one.

- A native renderer is registered for one template digest. At `CompiledChatTemplate::resolve`, the
  template's `sha256` is compared against the registered digest; on a match the native renderer is
  available, on any other source it is not and Jinja renders as it does today. The digest is computed
  with the existing `sha256` in `frontend/digest.h`; no allowlist exists in this tree today, so this
  adds the first one.
- The native renderer takes the same inputs and returns the same `RenderedChat` the Jinja path does,
  so it is a drop-in at the one call site and the rest of the frontend does not change.
- It produces the boundary surface **by construction** rather than by derivation: it knows which bytes
  it took from which message, where each message's serialization ends, and where the generation prompt
  begins. That retires the two probes and the marker parsing that `inspect_prompt_layout` performs,
  which is the second half of this decision and the part with the larger correctness claim.
- The template is not edited. It is the artifact's, it is already the thing the model was trained
  against, and the Jinja route must keep rendering it exactly as it does now.

## The two semantics the transcription has to reproduce exactly

Both were read out of the vendored interpreter rather than assumed, because both are places a
difference would be silent:

- **`|trim` is `strip(true, true)` over *codepoints*, not bytes.** `runtime.cpp:274` aliases `trim` to
  `strip`; `value.cpp:545` calls `string::strip(true, true)`; and that walks `unicode::characters`
  testing `unicode::whitespace(cp)` = `(0x1c..0x1f) || 0x85 || utf8proc_category in {ZS, ZL, ZP}`.
  The native renderer therefore trims by Unicode whitespace, not by `isspace`.
- **`tojson` uses `json.dumps`' defaults**: `indent = -1`, item separator `", "`, key separator `": "`,
  `ensure_ascii` false (`value.cpp:136-158`).

## Why C++ is the right form here and not a generic template engine

`AGENTS.md` asks for explicit implementations for supported architectures and forbids string-driven
execution. A native renderer for a digest-pinned template is the explicit implementation; it is not a
second template engine, because it renders one template and refuses every other.

The template's own shape makes the transcription tractable rather than a guess: its output is the
concatenation of its `{{- '…' }}` literals, because the `-` modifiers strip the source's own
whitespace. The renderer is therefore a faithful transcription of 194 lines of control flow with those
literals inlined — not a reinterpretation.

## Verification

The whole build is gated on one instrument, and it exists before the renderer does: a host-only
differential loop that renders one conversation both ways, reports the **first differing byte** and
the differing boundaries, and times both. It runs over the frontend test corpus and the bench's
synthesized conversations, including the arms that vary tools, thinking, continuation, tail shape and
media.

- Byte-identity is the acceptance criterion, not similarity: the rendered text and every field of
  `RenderedChat` must be equal.
- The Jinja path stays compiled in, so the oracle is always available and the differential test is a
  permanent gate rather than a one-off. **Obtaining the oracle is the part that goes wrong.**
  `CompiledChatTemplate::render` dispatches to the native renderer whenever the template digest is
  registered, so resolving the same source twice and calling `render` on both compares the native
  renderer with itself. The loop resolves its oracle from the source plus a trailing template
  comment: the digest changes, the comment emits nothing, and the interpreter renders. Its control
  compares the oracle against a source carrying a literal the template emits, so a comparison
  reporting "identical" reads as a broken instrument rather than a clean result -- a control that
  compares a render with itself cannot fail and validates nothing.
- A corpus without media cannot see a media divergence. The synthesized conversation had no media
  parts, and the fixture-template media tests resolve `qwen3_6.jinja`, whose digest is unregistered
  and therefore take the Jinja path.
- The renderer is not selected until the loop is green across the corpus.

### What the loop did not catch

It compared the native renderer with itself from the day the renderer landed until the oracle was
fixed, so it could not detect any native defect. Two shipped under it: `render_native` emitted the
media placeholder text and never recorded `MediaPlaceholderByteSpec`, and it appended the template's
`<|vision_start|><|image_pad|><|vision_end|>` wrapper inside the message's input span, so on
tokenization the wrapper was literal text rather than control tokens. Both stayed latent until the
launchers began passing the registered template, which selected the native path for every lane; from
that commit on, every image and video request failed with `invalid_prompt`, and the launcher verifier
recorded ninety passing runs followed by none.

## Risks

- **Template drift.** The template is the artifact's, and a new artifact may embed a different one.
  The digest gate is the mitigation: an unrecognised digest takes the Jinja path, so drift costs
  speed and never correctness. A future template is a new digest and a new transcription.
- **Two implementations of one serialization.** That is the price. It is bounded by the digest, and it
  is the same price llama.cpp pays for its built-in formats.
- **Trim and `tojson` equivalence.** The template trims and serializes through the interpreter's own
  piece-table `string`, whose exact behaviour (which bytes `trim` removes, how `tojson` escapes and
  orders keys) the native renderer must reproduce. This is the most likely place for a silent
  difference, which is why byte-identity is checked rather than reasoned about.

## Consequences

- **Measured**: the render drops from 48.3 ms to **5.99 ms** on the same 229-message conversation --
  8.1x, against a 3.47 ms floor for the frontend's own work. The interpreter path renders the same
  conversation in 48.33 ms through an unregistered template, so the comparison holds the input fixed
  and varies only the path.
- **Projected, not measured**: `prepared` ≈ 8 ms and warm TTFT ≈ 22 ms. Those follow from the render
  figure and the field log's split, and the end-to-end path has not been timed with the new renderer.
  **Both superseded — see "The end-to-end projection, measured 2026-10-02" immediately below, and
  `:243`, where the `prepared` ≈ 8 ms half is separately refuted. Do not quote this bullet.**

### The end-to-end projection, measured 2026-10-02, and warm TTFT now exists

Getting to a warm measurement required finding that **`/v1/chat/completions` could not declare a shared
prefix at all.** Its parser can only set a cache boundary on a content **part**
(`openai_chat_request.cpp:352`), which becomes a `MessagePartBoundary` marker — and the renderer
leaves that location, and `LeadingInstructionBoundary`, unresolved (`native_render.cpp`, the cache
marker switch). No frontier, so no shared prefix was ever written on that protocol. The
`/v1/messages` route sets the boundary on the **message** (`anthropic_messages_request.cpp:266`),
which resolves to a `MessageBoundary` and is served.
`tools/bench/check_shared_prefix_reuse.py` is the instrument, and it exits 0 when the warm path is
reached and 2 when its own control cannot see a hit.

**CORRECTED 2026-10-03 — the two unresolved locations were a divergence from the Jinja path, not a
design choice, and the sentence above originally said "by design".** That justification quoted this
ADR's own code comment, which asserted the native answer was *"the same answer the Jinja path gives
when the layout cannot place them"*. Jinja resolves both locations: `MessagePartBoundary` from
`origin.part_ends` / `part_tags` / `part_media` and `LeadingInstructionBoundary` from
`source_boundary` (`chat_template.cpp:440-472`). The native renderer recorded only per-message
offsets, had no per-part data to resolve against, and so left both unset. This is the ADR's stated
risk — *"two implementations of one serialization"* — landing as a silent, protocol-visible
difference, on a boundary this ADR's own warm measurement depended on.

The window is bounded by this ADR's own dates: the native renderer landed 2026-09-23 (`06d7843e`), and
ADR-0009's declared-boundary measurement (`301` of `344` tokens reused as `shared prefix`) was
2026-09-22, on Jinja. So the renderer diverged from the path ADR-0009 had measured, and the
divergence is closed below.

**What closing it did *not* do is restore ADR-0009's end-to-end result** — see the measurement below.
The fix is renderer parity, not a serving change.

Fixed 2026-10-03: `native_render.cpp` records each content part's end offset while emitting and
resolves both locations from it. Verified as a **renderer-level** result — the differential oracle
(`bench/models/qwen3_5/chat_render_bench.cpp`) now carries a marker corpus covering all four
locations, so `cache_boundaries` is non-empty in both arms for the first time, and the native and
Jinja renders agree on every marker including the media part's; `test_frontend.cpp`'s
`test_registered_template_cache_markers` pins the same comparison through the registered template. The
oracle previously compared an **empty** `cache_boundaries` in both arms and reported "identical",
which is why a renderer resolving no marker at all passed it.

**Measured 2026-10-03, and the answer is that the fix is renderer-correct but end-to-end inert.**
`tools/bench/check_shared_prefix_reuse.py`, against `qwen3_8_27b_nvfp4qat.v3.ninfer` with the app tree
rebuilt to carry the change, read:

| arm | prompt tokens | hit | path |
|---|---:|---:|---|
| `repeat` (control) | 3282 | 3275 | `private_response_replay` |
| `grow` (unmarked) | 3824 | 0 | `root` |
| `shuffle` (negative) | 3282 | 0 | `root` |
| `marked grow` | 3824 | 3817 | `private_response_replay` |
| `marked diverge` | 3284 | 0 | `root` |
| `anthropic B` (warm) | 8079 | 8054 | `shared_stable_prefix` |

**The same instrument was then run with this change reverted and both trees rebuilt. Every row is
byte-identical.** So the 3817 reuse on the OpenAI marked arm is `private_response_replay`, which was
already available and does not depend on the cache-boundary marker — a `prompt_cache_breakpoint`
governs *shared publication*, not replay. The instrument therefore cannot distinguish "a shared
candidate was published" from "none was", because replay dominates that arm.

Two consequences, both recorded because the first one corrects this ADR and the second corrects the
correction:

- **The OpenAI declared-boundary arm still does not reach `shared_stable_prefix`.** It reuses through
  replay. That is ADR-0009's second mechanism — the shared candidate is available and loses the
  valuation — and not the marker location. The `/v1/messages` arm is the only one that reaches
  `shared_stable_prefix`, through `MessageBoundary`, which the native renderer already resolved before
  this change.
- **ADR-0009's 2026-09-22 measurement of `shared_stable_prefix 1` on a declared OpenAI boundary is
  therefore not reproduced today, and this fix did not restore it.** What was restored is renderer
  parity with the Jinja path. Whether ADR-0009's result depended on the run's cache state — ADR-0009
  records that admission rides on surplus capacity or repetition, both of which are history-dependent
  — is **unresolved**, and is a separate question from the renderer divergence fixed here.

The three `shared_reuse_*` counters would decide it, and this instrument cannot read them: the lane
is killed rather than retired, so the engine's shutdown-tail throughput record is never written and
the last record on disk predates the final request. Measured at both `--stats-interval-ms 500` and
`100`. Read carelessly that stale record reports `shared_stable_prefix=1`, which belongs to a
*previous* run's Anthropic arm.

**The oracle's reasoning-effort arms compared the native renderer with itself until 2026-10-03.** The
Jinja side called `compiled.render(...)`, where `compiled` is the *registered* source, so `render`
dispatched to `render_native` (`chat_template.cpp:164`) — the same function the other side called
directly. All seven arms printed `identical` regardless of what either implementation did, and "all
seven effort arms identical" was reported as evidence before this was found. The arms now use the
Jinja `oracle`; all seven still agree, and the corrected comparison was falsified per-arm (corrupting
the `minimal` alias alone moves only that arm, reporting a difference at byte 46). **The verdict
survived; the evidence for it had not existed.**

**The oracle is not a gate.** `chat_render_bench` appears in no script under `tools/`, no
`.githooks/` entry, and no ctest registration — it is run by hand with `--native`. What the suite
gates is `test_registered_template_cache_markers`, which compares **2 of the 9 `RenderedChat`
fields** (`text` and `cache_boundaries`). There is no `operator==` on `RenderedChat` and no field
count, so **a tenth field would compile clean and fail nothing** — the bench's field list is a
hand-written sequence of `report(...)` calls and the test names two fields explicitly. Two fields are
compared at reduced granularity even in the bench: `media_placeholders` by `size()` only, and
`rewrite_checkpoint` by `offset` only, so a divergent `kind` passes everywhere. Closing that needs a
structural change (a field-count assertion, or a defaulted `operator==`), not another careful pass.

| | prompt tokens | prefix reused | `prepare` | TTFT | path |
|---|---:|---:|---:|---:|---|
| cold | 8,080 | 0 | 3.52 ms | 710.92 ms | `root` |
| warm | 8,079 | 8,054 | 3.89 ms | 159.71 ms | `shared_stable_prefix` |

Read those for what they are. `prepare` at **3.89 ms** is the native renderer preparing 8k tokens: the
same order as this ADR's projected ~8 ms, so **consistent with, not a confirmation of**, a projection
written for a ~172k-token conversation this artifact cannot reach. The 4.5x TTFT gap between the two
rows is the reusable prefix working, and it is the first measurement here of a warm request whose
prefill came from cache with a real decode behind it — every earlier "warm" figure on this ADR's
subject was `private_response_replay`, which returns a stored response and never decodes.

**The `prepared` ≈ 8 ms projection is refuted**, separately and by a different measurement.

`tools/bench/warm_lane_sweep.py`, one lane, the native path active, the engine's own
`request-log-jsonl` fields (`timings_seconds.prepare`, `result.prompt_tokens`):

| prompt tokens | `prepare` (ms) |
|---:|---:|
| 18 | 0.10 |
| 4,727 | 1.48 |
| 20,879 | 7.71 |
| 42,415 | 12.21 |

**The projection is wrong, not merely untested.** It puts `prepared` at ≈8 ms for a ~172k-token
conversation; measurement has it at 12.21 ms for 42,415 tokens, already above the projected value at
a quarter of the projected size, and still rising. Extrapolating the measured near-linear trend
gives ≈49 ms at ~172k tokens, against a projected ≈8 ms -- roughly 6x. The derivation failed
because it treated the render saving as the substance of `prepared`: at 42k tokens the render is
roughly 1.4 ms of the 12.21 ms, so tokenize, layout and copies dominate, and those scale with the
prompt rather than shrinking with it.

**SUPERSEDED 2026-10-02 by the table above — warm TTFT is now measured.** Kept because each sentence
cost a wrong conclusion. A growing conversation on `/v1/chat/completions` returned
`prefix_cache_hit_tokens = 0` and path `root` at every size tried (4,727 / 20,879 / 42,415 tokens), and
the shared read was reported as "refused at the shortlist key". Both were harness and protocol facts,
not cache behaviour: the OpenAI route cannot write a shared prefix (see above), and the shortlist-key
guess was made before the counters were readable. The only rows that hit here took
`private_response_replay`, which returns a stored response and never decodes — so every earlier "warm"
figure on this ADR's subject was replay latency.

At 8k tokens warm TTFT measures 159.71 ms. The projection's ~22 ms is for a ~172k-token conversation,
which this artifact cannot hold (about 117k), so that half remains untested at the size it was written
for rather than confirmed or refuted. The `prepared` ≈ 8 ms half is separately refuted, below.

Also measured here, re-deriving the precondition above after the renderer's digest was corrected:
`CompiledChatTemplate::render` over a 229-message, 689,748-byte conversation takes **5.59 ms**
(`ninfer_qwen3_5_chat_render_bench --sweep 32,64,128,229`), against the 5.99 ms recorded on
2026-09-23 for the same shape. The render half of this ADR stands; only the end-to-end projection
fails.
- The boundary machinery simplifies: no `prefix()` probe and no marker parsing to establish what the
  renderer knows, which is where the second half of the win comes from.
- `--chat-template` overrides and every unrecognised artifact continue through Jinja unchanged. Both
  directions were checked: the registered digest takes the native path, and `qwen3_6.jinja`'s digest
  is not registered and still renders at the interpreter's rate.
- The renderer is written for one digest and one template; it is not a family, a plugin or a
  discovery mechanism, and adding a second template means a second transcription.

## Why this needs recording

It duplicates a serialization that the artifact owns, which is a deliberate trade against a measured
cost, and it retires machinery (the probes and the layout parse) that three earlier records —
ADR-0007, ADR-0009 and ADR-0011 — reason about. A reader who finds the probes gone needs to know what
replaced them and on what evidence. It also commits the port to updating the renderer whenever the
registered template changes, and the digest gate is the boundary of that obligation.
