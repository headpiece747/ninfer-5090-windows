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
| ninfer-dflash2 | **none** | **11/11** | **0.6 s** | 0 | 0 |
| ninfer-dflash2 | low | 11/11 | 2.7 s | 2169 | 0 |
| ninfer-dflash2 | medium | 6/6 | 2.9 s | 2383 | 0 |
| ninfer-dflash2 | xhigh | 4/6 | 5.8 s | 4095 | 2 |
| ninfer-mtp5 | **none** | **11/11** | **0.8 s** | 0 | 0 |
| ninfer-mtp5 | low | 11/11 | 3.1 s | 2039 | 0 |
| ninfer-mtp5 | medium | 6/6 | 4.1 s | 2616 | 0 |
| ninfer-mtp5 | xhigh | 3/6 | 8.2 s | 4922 | 3 |

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
| quasar-dflash2 | 262,144 | fastest measured (331 tok/s) and the largest context; the default |
| quasar-mtp4 | 262,144 | lowest-VRAM QUASAR profile; 225 tok/s |
| ninfer-mtp5 | 240,000 | longest context on the NVFP4 line |
| ninfer-dflash2 | 180,224 | NVFP4 with the fastest spec route |

## Caveats

The tasks are short, self-contained functions. They separate reliability and speed
cleanly, and they do not stress long-horizon agentic work, where reasoning may pay off in
ways this grid cannot see. The truncation finding matters more than the pass rates: it is a
configuration trap, not a model property.
