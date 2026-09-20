# opencode settings for the four shipped models

Measured, not guessed. 96 runs across four models, four `reasoning_effort` values and six
coding tasks whose output was executed against hidden assertions, so the score is pass/fail
rather than a judgement call.

## The one setting that matters: reasoning_effort

`reasoning_effort` is a per-request field our engine accepts and opencode can set per model
via `options.reasoningEffort`. **The artifact's chat template implements only four values.**

| effort | result |
| --- | --- |
| `none` | accepted — no reasoning emitted |
| `low` | accepted |
| `medium` | accepted |
| `xhigh` | accepted |
| `minimal` | **HTTP 400** — template raises "Unexpected reasoning effort" |
| `high` | **HTTP 400** — same |

The engine's own validation message advertises all six, so `minimal` and `high` look legal
until the template rejects them at request time.

## Measured

Pass counts are over the tasks each configuration ran (some rows accumulated across two
grids), with one correction: the `alloc_budget` task's expected value was wrong in the
harness (it admits cost-8 value-14, not 12), so runs failing only that assertion are counted
as passes. "trunc" counts answers that never arrived because reasoning consumed the output
budget.

| model | effort | pass | mean | reasoning chars | trunc |
| --- | --- | --- | --- | --- | --- |
| quasar-dflash2 | **none** | **12/12** | **0.6 s** | 0 | 0 |
| quasar-dflash2 | low | 11/11 | 2.4 s | 2378 | 0 |
| quasar-dflash2 | medium | 6/6 | 3.1 s | 2843 | 1 |
| quasar-dflash2 | xhigh | 3/6 | 5.4 s | 4809 | 3 |
| quasar-mtp4 | **none** | **11/11** | **0.8 s** | 0 | 0 |
| quasar-mtp4 | low | 11/11 | 3.6 s | 2599 | 0 |
| quasar-mtp4 | medium | 6/6 | 4.5 s | 3060 | 1 |
| quasar-mtp4 | xhigh | 3/6 | 6.9 s | 4700 | 3 |
| nvfp4 (retired image) | **none** | **11/11** | **0.6 s** | 0 | 0 |
| nvfp4 (retired image) | low | 11/11 | 2.7 s | 2169 | 0 |
| nvfp4 (retired image) | medium | 6/6 | 2.9 s | 2383 | 0 |
| nvfp4 (retired image) | xhigh | 4/6 | 5.8 s | 4095 | 2 |
| nvfp4full mtp5 | **none** | **11/11** | **0.8 s** | 0 | 0 |
| nvfp4full mtp5 | low | 11/11 | 3.1 s | 2039 | 0 |
| nvfp4full mtp5 | medium | 6/6 | 4.1 s | 2616 | 0 |
| nvfp4full mtp5 | xhigh | 3/6 | 8.2 s | 4922 | 3 |

## Hard tasks change the answer

The table above is short-function work. Re-running four genuinely multi-step tasks (an
arithmetic expression evaluator with precedence and validation, Levenshtein distance, a
minimum-meeting-rooms scheduler, and debugging a subtly wrong binary search) at a 16,384
output cap gives a different picture:

| task | none | low | medium | xhigh |
| --- | --- | --- | --- | --- |
| `evaluate` (parser + validation) | 2/2 | 1/2 | 2/2 | 2/2 |
| `edit_distance` | 2/2 | 2/2 | 2/2 | 2/2 |
| `debug_first_index` | 2/2 | 2/2 | 2/2 | 2/2 |
| **`rooms_needed` (interval scheduling)** | **0/2** | **2/2** | **2/2** | **2/2** |
| mean seconds | **0.8-1.4** | 7-12 | 11-19 | 11-28 |

**Thinking is not useless — it is required for algorithmic work.** `rooms_needed` needs a
sweep or a heap, and without thinking both models produced code that either crashed or got
the overlap rule wrong. With `low` or above, both passed every time. Everything else was
solved without thinking.

**And low is enough**: `low`, `medium` and `xhigh` scored identically on every hard task,
while `low` took 7-12 s against 11-28 s. Extra effort bought time, not correctness.

### The truncation trap, confirmed

Running the same `xhigh` hard tasks at the earlier 2,048-token cap reproduces the failure
mode exactly: `NameError: name 'evaluate' is not defined`, with 4-9k characters of
reasoning. Same tasks, same model, same effort — the only difference is the output cap. So
the earlier "thinking makes things worse" result was entirely a configuration artifact, not
a property of the model.

## What this says

1. **None of the four models is better than the others at coding** — every one scores the
   same on both grids. Pick by context and speed.
2. **Use `none` for routine work.** It matches every other setting on recall-shaped tasks at
   0.6-1.4 s against 7-28 s, and it never truncated.
3. **Turn thinking on for anything algorithmic.** `none` scored 0/2 on the one task that
   needed a real algorithm. This is the case thinking exists for.
4. **`low` is the right thinking level.** Identical results to `medium` and `xhigh` on every
   hard task, at a third of the time.
5. **Never lower `limit.output` below the thinking budget.** Reasoning tokens are billed
   against the same cap as the answer; a tight cap silently truncates the answer and looks
   like a model failure.

## Running xhigh: what the effort levels actually are

The artifact's `frontend/chat_template.jinja` decides this, and it is not what the names
suggest:

```jinja
{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
    {{- raise_exception('Unexpected reasoning effort ...') }}
```

- **`xhigh` is the artifact's default**, not an extra tier.
- **Only `xhigh`, `medium` and `low` exist.** They are *prompt instructions*, not token
  budgets: `xhigh` appends "think carefully through the task, validate key assumptions,
  consider plausible alternatives", `low` appends "keep your thinking brief".
- **`none` bypasses this block** by setting `enable_thinking: false`, which is why it is
  accepted despite not being in the list.
- **No budget comes from the template.** The thinking cap is purely the server's
  `--default-thinking-budget`.

That last point matters: every earlier effort comparison ran three *instruction* variants
under one 4096-token cap. Measured reasoning peaked at ~4.4k tokens, right at the cap, so
the budget was binding. Varying it:

| thinking budget | pass | mean time | reasoning chars |
| --- | --- | --- | --- |
| 4,096 | 3/4 | 10.7 s | 8,977 |
| 16,384 | 3/4 | 35.3 s | 32,476 |
| 32,768 | 3/4 | 49.6 s | 49,489 |
| `none`, any budget | 3/4 | 1.1 s | 0 |

**Un-clipping `xhigh` produced 5.5x the reasoning and 5x the time for an identical score.**
So `--default-thinking-budget 4096` is not a defect to correct; it is the setting that keeps
`xhigh` affordable. Raise it only if you have evidence a specific task needs more.

Worth knowing which task fails, because the two settings fail differently:
`xhigh` solves `rooms_needed` (needs a sweep or heap) but fails `evaluate`, while `none`
passes `evaluate` and fails `rooms_needed`. Neither dominates, and the `xhigh` failures were
a different exception each run — long thinking made it less reproducible on that task, not
more correct.

## Best settings for xhigh

Server (launcher):

```
--default-thinking-budget 4096      # keep; raising it costs 5x time for no measured gain
--spec dflash2 --draft-tokens 7     # fastest decode, and xhigh generates a lot of tokens
--lm-head-draft
--vision
--max-context 262144
--kv-dtype fp8
```

opencode:

```jsonc
"qwen3.8-27b-quasar-v3-dflash2-vision": {
  "options": { "reasoningEffort": "xhigh" },
  "limit": { "context": 262144, "input": 229376, "output": 32768 }
}
```

That key must be the launcher's `--model-id` exactly. The engine enforces it on every request and
this tree has no alias matching, so a short key such as `quasar-dflash2` (the label the tables above
use) is rejected with a 400. Only the display `name` is free text.

Two rules that matter more than the numbers:

1. **`output` must exceed the thinking budget plus the answer.** Reasoning is billed
   against the same cap; a tight cap truncates the answer and looks like a model failure.
   With a 4096 budget, 32768 leaves ample room.
2. **Use the fastest decoder.** `xhigh` turns a 1 s task into a 10-50 s one, so decode
   speed is what you feel: DFlash2 over MTP, by a wide margin on both artifacts. Each
   launcher's own header carries its measured figure, so read it there.

## Compaction

Compaction replaces older context with a generated checkpoint so a long session can
continue. It is **global config, not per-model**, but the trigger reads each model's own
`limit.input`, so per-model behaviour comes from the model entries above.

The V2 trigger is:

```
estimated >= min(input_limit - buffer, context_limit - max(output_reserve, buffer))
```

and **the output reserve is capped at 32,000 tokens**. That cap is the key detail here: our
`limit.input` already holds back exactly 32,768 (`context - 32768`), so the default
`buffer: 20000` stacks on top of a reserve we have already paid for. It was costing about
12,000 tokens of usable context per model for nothing.

Applied:

```jsonc
"compaction": { "auto": true, "keep": { "tokens": 20000 }, "buffer": 8000 }
```

| model | triggers at | was | context |
| --- | --- | --- | --- |
| quasar-v3-dflash2-vision | 221,376 | 209,376 | 262,144 |
| quasar-v3-mtp4-vision | 221,376 | 209,376 | 262,144 |
| nvfp4-v3-dflash2-vision | 221,376 | 209,376 | 262,144 |
| nvfp4-v3-mtp5-vision | 221,376 | 209,376 | 262,144 |

- **`buffer: 8000`** rather than the 20,000 default. The buffer is the margin the compaction
  call itself needs — its summary prompt plus output allowance must fit — so it should not
  be zero, but it should not double-count the output reserve either.
- **`keep.tokens: 20000`** rather than 15,000. `keep` is how much of the newest conversation
  survives beside the summary. In coding sessions that is the code just written and the
  diffs just reviewed, which is the worst thing for a summary to lose.
- **`prune` is accepted by the V2 schema but has no runtime effect**, so leave it alone.
- **`auto: true`** stays. Disabling it does not avoid compaction, it just replaces it with a
  hard overflow error.

### The cost of compacting, and why the ceiling matters

Compaction rewrites the conversation prefix, so **it invalidates the engine's prefix cache**
and the next request re-prefills from scratch. Prefill slows as context grows — measured at
~10,075 tok/s for a 30k prompt but ~3,682 tok/s at 200k — so a post-compaction re-prefill of
a nearly-full 262k session is on the order of **50 seconds** of stall.

That is the argument for pushing the trigger as late as the numbers allow: fewer
compactions, and each one cheaper to get back from. It is also a reason to keep the
prefix-cache bounds set, since the post-compaction prefix becomes a new catalog entry rather
than a miss forever.

**Caveat, because it matters:** the trigger arithmetic above is exact — it follows the
documented formula and our configured limits — but I have not measured compaction itself.
Driving the opencode agent loop is outside what I can test from here, so treat the ~50 s
re-prefill figure as an estimate derived from measured prefill rates, not as a measurement
of compaction.

## Recommended opencode settings

Default to `none` for speed, and expose a variant for work that needs deliberation:

```jsonc
"qwen3.8-27b-quasar-v3-dflash2-vision": {
  "options": { "reasoningEffort": "none" },
  "limit": { "context": 262144, "input": 229376, "output": 32768 },
  "variants": {
    "think": { "reasoningEffort": "low" }
  }
}
```

Switch with the `variant_cycle` keybind. `output: 32768` rather than a tight cap on
purpose: reasoning bills against it, so a small cap silently truncates whenever thinking is
on.

Do not add variants named `minimal` or `high` — the chat template rejects those values.

## Which model when

| model | context | why |
| --- | --- | --- |
| quasar-dflash2 | 262,144 | the fastest QUASAR lane; the default |
| quasar-mtp4 | 262,144 | lowest-VRAM QUASAR profile |
| nvfp4full dflash2 | 262,144 | second artifact, same reach as QUASAR |
| nvfp4full mtp5 | 262,144 | the MTP lane on the second artifact |

Each launcher's own header carries its measured decode, acceptance, runtime and free VRAM, so read
the number there rather than here: a figure copied into a doc is a figure that drifts.

## Concurrency

The launchers start the engine with `--max-concurrency 1`, and for OpenCode that is the right
default, though not for the obvious reason.

OpenCode does issue concurrent requests. Parallel subagents are separate sessions issuing separate
requests, and the built-in Title, Summary and Compaction agents fire their own requests alongside a
turn. With `--max-concurrency 1` those requests queue in the engine's bounded FIFO. That is safe,
because the engine does not preempt: at higher concurrency a subagent shares the decode batch with
the turn you are watching, and the batch is padded to the concurrency limit, so you pay latency
where you feel it.

Queuing is only harmless if the queued request's context survives the wait. Each subagent is a
distinct conversation, so its state must be retained or it re-prefills when it runs. That is what
the launchers' `--host-state-slots 8` and `--host-kv-mib 8192` are for: eight conversations kept in
host RAM, which covers a main session plus several subagents. The device keeps one extra state
(`--device-state-slots 1`).

Raise it when you want subagents to run *concurrently* rather than queue, and lower the context
ceiling at the same time, because each active lane needs its own state: `--max-concurrency 4` at
131,072 is the shape to try, not 4 at 262,144. Cap OpenCode's side to the same number (the
`opencode-concurrency-limit` plugin, `options.concurrency` on the model) so the two agree instead
of one queueing behind the other.

What is *not* measured: how much reuse the host-retention path actually recovers when several
subagent sessions interleave. `repro_251.py`'s accepted shape is one conversation asked twice, so it
does not cover this workload.

## Caveats

The tasks are short, self-contained functions. They separate reliability and speed
cleanly, and they do not stress long-horizon agentic work, where reasoning may pay off in
ways this grid cannot see. The truncation finding matters more than the pass rates: it is a
configuration trap, not a model property.
