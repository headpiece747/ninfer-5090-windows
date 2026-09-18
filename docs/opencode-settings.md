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

## What this says

1. **None of the four models is better than the others at coding.** Every one scores the
   same. Pick by context and speed, not by expectation of quality.
2. **`none` is the best default.** Full pass rate at 0.6-0.8 s, versus 2.4-3.6 s for `low`
   and 5.4-8.2 s for `xhigh` — a 7-10x cost for thinking, with no correctness gained.
3. **More thinking made things worse, and the mechanism is truncation, not reasoning.**
   Every `xhigh` failure is `NameError: name '<fn>' is not defined` — the code block never
   arrived. 4-8k characters of reasoning against `max_tokens: 2048` consumed the entire
   output budget. **If you raise `reasoningEffort`, raise `limit.output` with it**, or you
   will trade a correct short answer for a truncated long one.
4. **`medium` is the sweet spot if you want thinking at all**, and it only stayed clean on
   the two ninfer profiles.

## Recommended opencode settings

```jsonc
"qwen3.8-27b-quasar-v3-dflash2-vision": {
  "options": { "reasoningEffort": "none" },
  "limit": { "context": 262144, "input": 229376, "output": 32768 }
}
```

Rationale for `output: 32768` rather than a tight cap: reasoning tokens are billed to the
same budget as the answer, so a small cap silently truncates whenever thinking is on. 32k
leaves room either way.

If you want a thinking ladder, opencode variants are the right mechanism — one model entry,
selectable profiles, no duplicate model keys.

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
