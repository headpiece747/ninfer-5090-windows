# ADR-0011: the generation boundary is the layout's, not a second render's

Status: accepted (2026-09-23)

## Context

Host prompt preparation is `render + tokenize`. ADR-0010 removed the tokenize half by splicing at a
verified seam; what is left is the render, and the render is not one render.

`CompiledChatTemplate::render` renders the conversation once for its output and then, for boundaries
it cannot otherwise place, *proves* each one by rendering a message subset again and accepting the
result only if it is a prefix of the full render. The bench's own probe arms price that discipline on
a 229-message conversation:

| arm | render | what it adds |
|---|--:|---|
| both probes on | **72.2 ms** | everything |
| `--generation-prompt off` | 48.7 ms | `prefix(messages.size())` — the generation boundary |
| `--tail-assistant off` | 47.0 ms | the next-turn probe — whether the template retains an open turn |
| both off | **24.3 ms** | the one render the answer actually needs |

`prefix(messages.size())` re-renders the whole conversation to learn one byte offset: the frontier
before the generation prompt, which becomes `rewrite_checkpoint.offset` and the comparison length for
the next-turn probe. It is the most expensive probe there is, because its input is the whole
conversation.

## What the probe was for, and what it was stricter than

The probe establishes the frontier by rendering `messages[0..n]` with `add_generation_prompt=false`
and accepting its length only if that render is a prefix of the output.

That is stricter than the trust the same function already places in a layout-derived boundary. A
cache marker at a message boundary uses `result.message_boundaries[count]` — set by
`inspect_prompt_layout`, which parses `<|im_start|>`/`<|im_end|>` markers that the input marking
proves are template bytes — and falls back to the probe only when the layout could not place it.
`generation_begin` was the one boundary that always paid for a render.

## Measurement

A temporary, environment-guarded probe in `prefix()` reported its own answer next to the boundary the
layout had already established, over the frontend test corpus and a 229-message bench conversation:

- 62 `generation_begin` calls. The layout had placed the boundary in **56**, and the probe returned
  the **identical** offset in all 56 — zero disagreements.
- The 6 without a layout value are small conversations (n=1, 2, 3), where the probe remains the only
  source. They are the cases the layout cannot place: a leading instruction folded into the preamble,
  or a template that leaves the final turn open.
- The field-like case: n=229, probe `714987`, layout `714987`, and the trailing block's `begin`
  `714987`.
- The other candidate was refuted rather than adopted: `layout.messages.back().begin` disagreed with
  the probe in 4 cases — all of them cases the layout had *not* placed, where the last parsed block
  is a real message rather than the generation block. It is not a substitute.

After the change: the render at 229 messages goes **72.2 → 48.3 ms**, the rendered bytes are
identical (`715028`), and the frontend test passes.

## Decision

`generation_begin` takes `result.message_boundaries[messages.size()]` when the layout places it, and
falls back to `prefix(messages.size())` only when it does not. The continuation case is unchanged: it
already used the layout's final block.

The narrowing is deliberate and recorded. The layout's value is a frontier of the render actually
being sent — the end of a closed message block, in template bytes — which is the same class of
boundary the cache-marker path already accepts without a probe. What is given up is the probe's
additional statement that a *truncated* render reproduces that prefix. It was never observed to
differ, in 56 corpus cases or in the 229-message conversation, and the probe still guards every case
the layout cannot place.

The next-turn probe is **not** changed, and this was measured rather than assumed. It answers a
different question — whether appending a turn changes the history's rendering — and the obvious
shortcut, deriving it from the typed `preserve_thinking` option, is refuted: instrumented over the
frontend corpus and the bench, the probe disagreed with the hint in **3 of 12** calls, and the same
typed value produced both answers. The reason is in the template: whether the tail assistant's
reasoning is emitted turns on `last_query_index`, which the template computes by testing whether a
user message's **content** is wrapped in `<tool_response>`. So neither the typed option nor any shape
signature determines the answer — it is a content test, and only executing the template observes it.
The request is also already at two renders, so the probe cannot be merged away.

## Consequences

- The render drops by the measured ~24 ms on every request whose last message has a placeable
  boundary. Applied to the field log's own split (`render 77.7 ms` of `prepared 79.5 ms`, warm TTFT
  93.8 ms), that is `prepared` ≈ 55 ms and warm TTFT ≈ 70 ms — derived from the log, not re-measured
  end to end.
- No template loses its fallback: the probe runs exactly where it did before, minus the cases the
  layout covers.
- Verification: `tests/models/qwen3_5/test_frontend.cpp` asserts `rewrite_checkpoint->offset` against
  the position of the generation prompt's `<|im_start|>assistant` and covers this path
  (`test_rewrite_checkpoint_trace`); the bench's rendered-byte count is the byte-identity check; the
  release gate covers the rest.
- What remains of the render's proof work is the next-turn probe, ~24 ms, and the vendored
  interpreter's own per-render cost, ~24 ms, which
  [`docs/research/prompt-preparation-cost.md`](../research/prompt-preparation-cost.md) records as the
  engine's architecture rather than this port's.

## Why this needs recording

It removes a proof, and a reader who finds `generation_begin` without a probe needs to know what
replaced it, on what evidence, and which cases still take the probe. The measurement is also the
argument: the probe and the layout never disagreed where both could answer, and the one candidate
that would have been wrong — the trailing block's `begin` — was refuted by the same instrument rather
than by reasoning about templates.
