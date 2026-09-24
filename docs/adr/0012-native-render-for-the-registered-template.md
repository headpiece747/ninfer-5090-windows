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
