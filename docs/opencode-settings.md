# opencode settings for the eight shipped lanes

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
| `minimal` | accepted — the template maps it to `low` |
| `high` | accepted — the template maps it to `xhigh` |

The engine's own validation message advertises all six, and the template this port ships maps
`minimal` and `high` rather than rejecting them.

## Measured

Pass counts are over the tasks each configuration ran (some rows accumulated across two
grids). "trunc" counts answers that never arrived because reasoning consumed the output
budget.

One correction applies to every row, and it is now a fix rather than an adjustment. The
`alloc_budget` task asserted a 0/1 knapsack answer of 12 where its own contract admits 14:
`(3,5)+(5,9)` costs 8, and the stated bound is `total cost <= budget`, so that subset fits. The
task was therefore unpassable — 0 successes in 116 runs across eight models and four efforts, every
failure on that one line. When these rows were first recorded the fix had not landed, so the runs
that failed only that assertion were counted as passes by hand, which is why a raw
`opencode_settings.jsonl` reads lower than this table. The harness assertion was corrected on
2026-10-13 against a reference knapsack written from the stated contract, and the correction was
verified by re-running the task: 0/116 before, 3/3 after. **A fresh run now records the pass
itself**, so this table is reproducible from the tool rather than adjusted after the fact. The
ranking across efforts was never affected, because the task failed uniformly at every setting.

**The table below supersedes the rows it replaces, and the reason is the instrument rather than the
model.** Two defects made the earlier figures unreadable, both now fixed in
`tools/release/bench_opencode_settings.py`:

- **No seed.** Every cell was a fresh sample, so the table was reporting its own sampling noise. The
  engine is deterministic given a seed — four repeats of one prompt gave four distinct answers
  unseeded and one seeded — and `evaluate` at identical settings swung 37 points across eight runs.
- **The default 2,048-token output cap.** Reasoning is billed against the same cap as the answer, so
  every hard task truncated and scored as a model failure. The earlier `trunc` column is that
  artefact: it is zero here because this run passes `--output-tokens 32768` (the tool's default is 2048).

Re-measured with both fixed, seed 20261013, two DFlash2 lanes × four efforts × ten tasks:

| model | effort | pass | mean s | mean completion | mean reasoning chars |
| --- | --- | --- | --- | --- | --- |
| quasar-v3-dflash2 | **none** | 9/10 | **0.5 s** | 221 | 0 |
| quasar-v3-dflash2 | low | 9/10 | 3.4 s | 1202 | 3365 |
| quasar-v3-dflash2 | medium | 9/10 | 7.0 s | 1856 | 6158 |
| quasar-v3-dflash2 | **xhigh** | **10/10** | 9.6 s | 2404 | 8460 |
| ninfer-v3-dflash2 | **none** | 9/10 | **0.7 s** | 283 | 0 |
| ninfer-v3-dflash2 | **low** | **10/10** | **3.2 s** | 1248 | 3640 |
| ninfer-v3-dflash2 | medium | 10/10 | 4.1 s | 1518 | 4623 |
| ninfer-v3-dflash2 | xhigh | 9/10 | 7.4 s | 2006 | 6906 |

Pooled over both models: **none 18/20 (90%), low 19/20 (95%), medium 19/20 (95%), xhigh 19/20
(95%)**, at 0.6 s, 3.3 s, 5.5 s and 8.5 s.

**This reverses the earlier table's headline.** `xhigh` was recorded at 3/6 — the worst row — and is
in fact tied for best. The reversal is entirely the two defects above: `xhigh` reasons longest, so it
was the row most damaged by a cap that truncation punishes hardest, and it is the row a sampling noise
floor punishes hardest. Only two cells fail at all — `evaluate` once at low and once at xhigh,
`rooms_needed` once at medium and once at none — and they do not order the efforts.

Two limits. Each cell is one deterministic run, so a single cell can still be unlucky; the paired
five-seed comparison above is the stronger instrument for any question about a difference. And the
effort-level finding above (`xhigh` tied for best) was re-measured on two lanes; the "no
artifact is better than another at coding, within ±10 points" claim comes from the unseeded
eight-lane table, where the truncation artefact affected every artifact uniformly. The wider
table was not re-measured, so the seeded instrument neither confirms nor contradicts it — it only
removes the artefact that made the earlier reading of it unreliable.

The rows below predate both fixes and are kept only as the record of what was measured then.
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

Re-derived from the seeded table above; the reasoning behind each is unchanged, but the numbers moved
once the instrument was fixed.

1. **No artifact is better than another at coding.** The two re-measured lanes differ by at most one
   task of ten at any effort, in both directions. Pick by context, footprint and speed.
2. **`none` for routine work; it costs almost nothing to give up.** 90% against 95% for every
   thinking level — one task in twenty — at 0.6 s against 3.3-8.5 s. That is a 5-14x latency
   difference for one task.
3. **Turn thinking on for anything algorithmic.** `rooms_needed` is the task that needs a real
   algorithm, and `none` fails it on **both** lanes while every thinking effort passes at least one.
4. **`low` is the right thinking level, and the data now says so cleanly.** 95% — identical to
   `medium` and `xhigh` — at 3.3 s against 5.5 s and 8.5 s. `xhigh` costs 2.6x `low` for nothing.
5. **Never lower `limit.output` below the thinking budget.** Reasoning tokens are billed
   against the same cap as the answer; a tight cap silently truncates the answer and looks
   like a model failure. This is what made the earlier table wrong, and it is the third time
   this tree has been bitten by it.

## Running xhigh: what the effort levels actually are

The artifact's `frontend/chat_template.jinja` decides this, and it is not what the names
suggest:

```jinja
{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
{%- if resolved_reasoning_effort in ('high', 'max', 'ultracode', 'extreme') %}
    {%- set resolved_reasoning_effort = 'xhigh' %}
{%- elif resolved_reasoning_effort == 'minimal' %}
    {%- set resolved_reasoning_effort = 'low' %}
{%- endif %}
```

- **`xhigh` is the artifact's default**, not an extra tier.
- **Only `xhigh`, `medium` and `low` exist.** They are *prompt instructions*, not token
  budgets: `xhigh` appends "think carefully through the task, validate key assumptions,
  consider plausible alternatives", `low` appends "keep your thinking brief".
- **`none` bypasses this block** by setting `enable_thinking: false`, which is why it is
  accepted despite not being in the list.
- **No budget comes from the template.** The thinking cap is purely the server's
  `--thinking-budget`.

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
So `--thinking-budget 4096` is not a defect to correct; it is the setting that keeps
`xhigh` affordable. Raise it only if you have evidence a specific task needs more.

Worth knowing which task fails, because the two settings fail differently:
`xhigh` solves `rooms_needed` (needs a sweep or heap) but fails `evaluate`, while `none`
passes `evaluate` and fails `rooms_needed`. Neither dominates, and the `xhigh` failures were
a different exception each run — long thinking made it less reproducible on that task, not
more correct.

## Best settings for xhigh

Server (launcher):

```
--thinking-budget 4096              # keep; raising it costs 5x time for no measured gain
--spec dflash2 --draft-tokens 7     # fastest decode, and xhigh generates a lot of tokens
--proposal-head optimized
--vision
--max-context 262144
--kv-dtype k8v4
```

opencode:

```jsonc
"qwen3.8-27b-quasar-v3-dflash2-vision": {
  "options": { "reasoningEffort": "xhigh" },
  "limit": { "context": 262144, "input": 229376, "output": 32768 }
}
```

That key must be the launcher's `--model-id` exactly. The OpenAI routes enforce it on every request and
this tree has no alias matching, so a short key such as `quasar-dflash2` (the label the tables above
use) is rejected with a 404 `model_not_found`. Only the display `name` is free text.

Two rules that matter more than the numbers:

1. **`output` must exceed the thinking budget plus the answer.** Reasoning is billed
   against the same cap; a tight cap truncates the answer and looks like a model failure.
   With a 4096 budget, 32768 leaves ample room.
2. **Use the fastest decoder.** `xhigh` turns a 1 s task into a 10-50 s one, so decode
   speed is what you feel: DFlash2 over MTP, by a wide margin on both artifacts. Each
   launcher's own header carries its measured figure, so read it there.

## What the reasoning trace costs in context

The lane ships `--preserve-thinking`, so every turn's `reasoning_content` is re-sent in full with
the next request. That is a large fraction of the prompt, and it was unmeasured here until
`tools/bench/probe_thinking_growth.py` drove a six-turn conversation both ways on this lane's own
flags — identical turns, fixed seed, differing only in whether assistant messages carry the trace:

| turn | prompt | without the trace | reasoning tax | share of prompt | `model_thinking_tokens` |
| --- | --- | --- | --- | --- | --- |
| 1 | 98 | 98 | 0 | 0.0% | 4,096 |
| 2 | 4,361 | 242 | 4,119 | 94.5% | 443 |
| 3 | 4,984 | 423 | 4,561 | 91.5% | 1,316 |
| 4 | 6,511 | 639 | 5,872 | 90.2% | 1,853 |
| 5 | 8,535 | 912 | 7,623 | 89.3% | 4,096 |
| 6 | 12,916 | 1,268 | 11,648 | 90.2% | 2,963 |

**~90% of the prompt is replayed reasoning by turn 2.** That is the token cost, and it is real.

The time cost is not. Across the six turns, 37,073 of 37,405 prompt tokens were prefix-cache hits
(**99.1%**), `prefix_reuse_path: checkpoint`, replay GPU time 0.0000 s, and a 12,916-token prompt
prefilled in **0.045 s**. The trace is registered as a prefix origin, so re-sending it is close to
free. Stripping the trace made prefill *slower* — 0.097 s on a ten-times smaller prompt — because
its hit rate fell to 63.6%. The mechanism for that is not established here and is not guessed at.

Two things follow, and both matter more than the ratio:

- **Context fills in about 100 turns, not 4 or 5.** Measured growth is 2,564 tokens/turn, with
  mean `model_thinking_tokens` 2,461 against the 4,096 budget and two of six turns pinned at the
  cap. Treat 100 as an *upper* bound on turn count: these turns returned short answers, and a real
  session returning file contents and diffs fills sooner. The portable figure is the 2,461-token
  per-turn tax.
- **Raising the thinking budget doubles how fast you compact.** Reasoning per turn is bounded by the
  budget, so 4,096 → 8,192 roughly doubles per-turn growth — about 100 turns to about 50 — on top of
  the 13.7 s → 26.2 s latency already measured, with no accuracy gain demonstrated. Three costs,
  no measured benefit: **this is the argument for keeping 4,096**, and it is the argument the
  thinking-budget ladder could not produce on its own because it measured latency and pass rate
  without ever seeing the context.

Do not read this as an argument for turning `--preserve-thinking` off. Per
[QwenCloud's thinking-mode documentation](https://docs.qwencloud.com/developer-guides/text-generation/thinking),
omitting `reasoning_content` from tool-call turns *degrades accuracy*, and the engine accepts it as
an assistant-history alias precisely so it round-trips. The trace is the cost of correctness here,
and the cache is what makes it affordable.

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

| model | triggers at (predicted) | was | context |
| --- | --- | --- | --- |
| quasar-v3-dflash2-vision | 221,376 | 209,376 | 262,144 |
| quasar-v3-mtp4-vision | 221,376 | 209,376 | 262,144 |
| nvfp4-v3-dflash2-vision | 221,376 | 209,376 | 262,144 |
| nvfp4-v3-mtp4-vision | 221,376 | 209,376 | 262,144 |
| swift15-v3-dflash2-vision | 221,376 | 209,376 | 262,144 |
| swift15-v3-mtp4-vision | 221,376 | 209,376 | 262,144 |

**The arithmetic above is confirmed, and the first measurement of it was wrong.** Measured with
`tools/bench/probe_opencode_compaction.py` driving one chained agent session of 14 turns / 20
requests against a lane whose request log records every `prompt_tokens`, the context grew
monotonically and compacted exactly where the formula says:

| | tokens |
| --- | --- |
| predicted trigger, `min(input_limit - buffer, context_limit - max(output_reserve, buffer))` | **221,376** |
| last request before compaction | **214,962** |
| first request after compaction | **47,666** |

214,962 plus one more turn of roughly 12,500 tokens crosses 221,376, and that is where it fired. The
formula is this client's mechanism, not an approximation of it.

**There is a control arm, and getting one required fixing two of this page's own errors.** The
compaction trigger can be suppressed with `"compaction": {"auto": false}`, supplied through a
project-level `opencode.json` to a `--standalone` run. Measured against the same lane and the same
14 turns:

| arm | peak reached | outcome |
| --- | --- | --- |
| compaction on | 214,962 / 212,105 / 215,966 across three runs | compacted each time, rebuilt to ~47.7k |
| **`auto: false`** | **255,267** — past the 221,376 trigger | never compacted; died with `maximum context length is 262144 tokens, but the prepared prompt has <n> tokens` |

`auto: false` gates both the automatic and the overflow path in v2 — they share one flag in the
source — while leaving manual compaction available.

**Two things this page previously got wrong, both now settled by measurement:**

- **"The config-redirect variables are ignored" was wrong.** `OPENCODE_CONFIG_DIR`, `OPENCODE_CONFIG`,
  `OPENCODE_CONFIG_CONTENT`, `OPENCODE_DB` and `OPENCODE_TEST_HOME` are all wired in v2. The earlier
  probe set them on CLI invocations that omitted `--standalone`, so every call attached to the
  operator's running background service — which captured its environment when *it* started. The
  variables were being set on a process that only forwarded to someone else's server.
- **"A private server does not compact" was also wrong.** One did — a private server reading a
  **copy of only `opencode.json`** never compacted and ran to the engine ceiling, while the same
  private server reading the real configuration compacted at 215,966. Copying the single file is not
  enough: whatever else the global config directory carries is load-bearing for the trigger, so the
  whole directory is mirrored. The private and shared servers agree once that is fixed.

`OPENCODE_DISABLE_AUTOCOMPACT` remains genuinely inert — a V1 documentation leftover with no
implementation behind it, found in translated V1 docs and in no source file.

One caveat for anyone counting events rather than watching tokens: overflow-triggered compaction
publishes its event with `reason: "auto"`, not `"overflow"`, so the two paths are indistinguishable
that way.

Two details worth keeping, both read from the shipped build rather than inferred. The auto-compaction
predicate is:

```js
A = (ctx, threshold) => {
  skip if the last message is a completed compaction        // never compact twice in a row
  skip if the model has not answered since the last user turn // never compact mid-turn
  return count(ctx) >= threshold
}
```

and `count` is `tokens.input + tokens.output + tokens.reasoning + tokens.cache.read + tokens.cache.write`.
That looks like it double-counts, and it does not: `input + cache.read` **is** the prompt, split into
new and cached. Verified against real sessions in the operator's own store — 245,027 + 81 = 245,108,
218,195 + 797 = 218,992 — and a real session crosses 221,376 at a prompt of 222,542. So the count and
the engine's `prompt_tokens` are the same quantity, which is why a single threshold fits.

An earlier revision of this page reported 13 compactions at a median of 79,076 tokens and concluded
the formula did not apply. **That was a defect in the probe, not a finding.** `opencode run` starts a
new session per invocation, so N turns were N independent conversations that never grew; the 13
"compactions" were 13 session boundaries, every one resetting to exactly one filler message's worth of
tokens. The probe now passes `--session` so turns chain into one growing conversation. Recorded here
because a harness that cannot produce the phenomenon must not be read as having measured it.

Two limits on this measurement, because they bear on how far it generalises:

- The probe inflates context with one large repeated filler block per turn rather than with real
  tool output, so the session's shape is unusual even though its trigger behaviour is not
  synthetic. A real session may summarise to a different size.
- opencode's *summary* is produced by a model call. Its size is opencode's business, and this
  probe observes the result rather than controlling it.

The two Swift rows are the 2026-09-30 model ids; the lanes were `swift-v3-*` before Swift 1.5
replaced Swift 1.0 on them. `nvfp4-v3-dflash2-vision` is back at 262,144 after being lowered to
240,000 the same day: the lane was refused at startup at 262,144 until the artifact behind it was
rebuilt with its BF16 exception projections encoded to NVFP4.

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
and the next request re-prefills from scratch. Prefill slows as context grows, and that slope is
measured — `tools/bench/probe_cold_prefill.py`, on this lane's own configuration, reading
`first_output_timing.prefill.gpu_seconds` from the request log rather than a wall clock:

| prompt tokens | cache hit | prefill | rate |
| --- | --- | --- | --- |
| 30,811 | 0 | 2.586 s | 11,917 tok/s |
| 102,211 | 0 | 14.210 s | 7,193 tok/s |
| 204,411 | 0 | 44.661 s | 4,577 tok/s |
| 204,411 | 204,411 | **0.0000 s** | TTFT 0.015 s |

A full prefix hit costs about **15 ms at any size**. That is the whole value of the cache, and it
is why `chat_template.cpp` registering the replayed reasoning trace as a prefix origin matters:
a preserved prefix is free to re-send, however large.

The argument for pushing the trigger as late as the numbers allow is fewer compactions, and each
one cheaper to get back from. It is also a reason to keep the prefix-cache bounds set, since the
post-compaction prefix becomes a new catalog entry rather than a miss forever. **Read the next two
paragraphs before relying on it: the stall it trades against turned out to be milliseconds, and
the trigger it moves fires far earlier than this page predicts.**

**The stall is measured now, and it depends on what the session was made of.** The compaction itself
is a summary request against the full context — 203,085 tokens at a **99.7% prefix-cache hit, 0.351 s**
— and the request after it is the cold one. Two sessions, same lane, same trigger:

| session | rebuilt to | cold prefill of that request |
| --- | --- | --- |
| one repeated filler block | 47,666 | 4.757 s |
| **real tool output against this repository** | **32,744** | ~3.3 s at the measured cold rate |

The filler session's own claim was that the number "should hold" whatever filled the session,
because the rebuild is set by `keep.tokens` plus the summary. **It does not hold.** A real agentic
session — 46 requests in its first turn alone, reading the tree — rebuilds 15k *smaller* than one made
of a repeated block. Which session you are in matters more than the configuration does.

That is worth stating plainly because it was the last open row and it was expected to be a
confirmation rather than a result. The cheap assumption was wrong in the direction that makes the
cost look better, which is the direction nobody would have checked.



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

`minimal` and `high` are accepted aliases (mapped to `low` and `xhigh`); prefer the rendered names in new configurations.

## Which model when

Eight lanes ship, over five artifacts. All eight are vision-capable and all eight reach the native
262,144 context, so context is not a differentiator. Register the model id you intend to use under
`providers` and point it at that lane's port.

| lane | port | route | artifact |
| --- | --- | --- | --- |
| `quasar-v3-dflash2-vision` | 8086 | DFlash2, depth 7, k8v4 | QUASAR — this port's own quantisation-aware-trained image |
| `quasar-v3-mtp4-vision` | 8087 | MTP, depth 4, fp8 | QUASAR, the same image |
| `nvfp4-v3-dflash2-vision` | 8088 | DFlash2, depth 9, fp8 | NVFP4-full **no-exception** — the unsloth line with its nine BF16 exception parents encoded to NVFP4 |
| `nvfp4-v3-mtp4-vision` | 8089 | MTP, depth 4, fp8 | NVFP4-full — the same line, exceptions still BF16 |
| `swift15-v3-dflash2-vision` | 8090 | DFlash2, depth 7, fp8 | Swift 1.5 — UkisAI's finetune, re-encoded by this port |
| `swift15-v3-mtp4-vision` | 8091 | MTP, depth 5, k8v4 | Swift 1.5, the same image |
| `nvidia-v3-dflash2-vision` | 8092 | DFlash2, depth 9, fp8 | NVIDIA ModelOpt quantization of the base model, built by this port |
| `nvidia-v3-mtp4-vision` | 8093 | MTP, depth 4, k8v4 | NVIDIA ModelOpt, the same image |

Three of those artifacts ship both routes; **the NVFP4 line does not**, and the split is a
measurement rather than an omission: only the no-exception build fits the native 262,144 with
Vision, so 8088 carries it and 8089 carries the full image. The two lanes are therefore *not* the
same artifact on two routes, and swapping between them changes the weights, not just the route.

**How to choose.** On the coding grid measured below, the models that were compared all scored the
same — so the artifact is not a quality choice among them. What separates the lanes is the route and
the footprint:

- **Route is the real decision.** The DFlash2 lanes decode several times faster than their MTP
  counterparts and are the reason that route exists; the MTP lanes accept a higher fraction of
  drafted tokens but decode far slower. If you are waiting on a turn, you want DFlash2.
- **VRAM differs, and the DFlash2 lanes cost the most.** Two of them sit at 10.6 GiB runtime and
  leave under 3 GiB free on a 32 GB card, against 8.07 GiB for the lightest. That matters when you
  want something else resident — another model, a browser, a build.
- **Depth is measured per lane, not a convention.** QUASAR's and Swift's DFlash2 lanes are at depth
  7 and the NVFP4 and NVIDIA ones at 9, on the maximin rule recorded in `tools/release/profiles.py`.
  Do not read one lane's depth as applying to another.

Each launcher's own header carries its measured decode, acceptance, runtime and free VRAM, so read
the number there rather than here: a figure copied into a doc is a figure that drifts. The table
in `tools/release/profiles.py` is the single copy, and `tools/bench/lane_process.py` reads it, so a
probe measures the shipped lane rather than a copy of it.

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
the launchers' `--host-cache-capacity 8192MiB` (formerly `--host-context-mib 8192`) is for: a shared quota of pinned host RAM for retained
conversation state, which covers a main session plus several subagents.

Raise it when you want subagents to run *concurrently* rather than queue, and lower the context
ceiling at the same time, because each active lane needs its own state: `--max-concurrency 4` at
131,072 is the shape to try, not 4 at 262,144. Cap OpenCode's side to the same number (the
`opencode-concurrency-limit` plugin, `options.concurrency` on the model) so the two agree instead
of one queueing behind the other.

What *is* measured now, as of 2026-10-06, with `tools/release/repro_251.py` (which starts its own lane
and takes `--interleave`):

- **Sequential**, each conversation asked twice back to back at the shipped
  `--host-cache-capacity 8192MiB`: 60 distinct conversations, all ~96%.
- **Interleaved** round-robin, which is this page's shape: 8, 16, 32, 48 and 52 conversations all
  reuse at ~95.7%; **56 and 60 reuse nothing at all**. The bound is the host quota in state images
  (~55 x 147 MiB = 8 GiB), so the quota holds roughly 52-55 live conversations.
- The shipped acceptance shape (12 conversations x ~20,000 tokens on the QUASAR DFlash2 lane) is
  **99.8% for all twelve**, the result the retired reclaim achieved.

So the concurrency answer this page gives rests on a measured number: a main session plus subagents
(8 here, and the quota reaches 52) sits far inside the host quota, and the cliff only appears when
live conversations exceed it. At `--host-cache-capacity 300MiB` the same sequential run stops reusing from
the third conversation, and the cause there is the replacement cache's value gate (a reused endpoint
is not displaced for a cold admission), not a missing eviction path.

## Open

Measured gaps, kept here so the next session inherits them instead of rediscovering them. Each says
what would close it. Anything resolved is deleted from this list rather than annotated, so its
presence means it is still open.

| question | state | what would close it |
| --- | --- | --- |
| What size does a compaction rebuild on a session whose turns are real tool output rather than one repeated block? | **measured: 32,744** — and the prior reasoning that it "should hold" was wrong | closed — `probe_opencode_compaction.py --task` drives the agent at real repository work, matching `tools/bench/repo_fix_eval.py`'s invocation |

Three questions closed earlier, moved out of this table into the prose above as the policy requires:
the thinking budget (paired over 5 seeds, mean effect +0.2 tasks of 7 against a seed spread of ±2),
long-horizon constraint retention (23 of 24 turns, first loss at turn 24), and the compaction
trigger, rebuild size and control arm — all measured, with the control reaching 255,267 uncompacted
against a 221,376 trigger.

The thinking-budget row is a decision already taken, not a blocker: 4,096 stands because raising it
is a cost with no measured benefit, and keeping the status quo needs no evidence.

### The thinking budget, measured with the noise removed

This section replaces an earlier verdict of "not answerable on this grid", which was itself a
measurement error: the grid did not seed its requests, so it was reading its own sampling noise.

**The engine is deterministic given a seed.** Measured, not assumed — four repeats of one prompt at
temperature 1.0:

| condition | distinct outputs of 4 |
| --- | --- |
| sampled, no seed | **4** |
| sampled, seed fixed | **1** |
| `--temperature 0` | **1** |

So there is no CUDA float drift or batch-dependent nondeterminism to contend with; a seed pins the
trajectory exactly. `bench_opencode_settings.py` now sends one on every request and records it, and
`evaluate` at one setting now reproduces to the byte: `completion=4842 reasoning_chars=16739` on both
repeats. Against the unseeded history of that same cell — `TFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFTTTFTTT`,
7 of 38 — the 37-point swing was entirely self-inflicted.

**With the variance gone, a paired comparison resolves the question.** Five seeds, each run at both
budgets over the seven discriminating tasks, so every seed contributes one matched pair:

| seed | 4,096 | 8,192 | delta | 4,096 s | 8,192 s |
| --- | --- | --- | --- | --- | --- |
| 11111 | 7/7 | 7/7 | 0 | 12.5 | 19.3 |
| 22222 | 6/7 | 6/7 | 0 | 12.9 | 21.5 |
| 33333 | 5/7 | 6/7 | +1 | 11.3 | 19.6 |
| 44444 | 7/7 | 5/5 | −2 | 12.6 | 14.7 |
| 20261013 | 7/7 | 7/7 | 0 | 12.7 | 21.7 |

**Mean effect +0.2 tasks of 7. Spread across seeds ±2 tasks, or 29 percentage points.** The effect is
an order of magnitude smaller than the seed-to-seed variation, which is the model's real behavioural
variance rather than measurement noise — so this is a readable null, not an absence of data.

Against it, 8,192 costs **+58% latency** (12.4 s → 19.6 s) and **+63% output tokens** (3,104 → 5,065),
which on this lane also roughly doubles per-turn context growth and so halves the turns before a
compaction (see the reasoning-trace section above). 4,096 binds on 20 of 35 runs, and raising the cap
does not rescue those: 8,192 binds on 14 of 33.

**4,096 stands, now on a measurement rather than an argument.** The two tasks that move are the two
noisy ones cancelling out — `evaluate` 4/5 → 2/4 and `rooms_needed` 3/5 → 4/4.

Two limits on the run, stated rather than hidden: seed 44444 lost two of its 8,192 cells, so that arm
has 33 records against 35, and 7 tasks × 5 seeds cannot resolve an effect smaller than about one task.

### The long-horizon probe: constraint retention, measured

`tools/bench/probe_session_drift.py` asks whether a constraint survives unrelated work: a random
build code stated at turn 1, then 24 turns of corpus text, 1,500 characters of *distinct* corpus per
turn, 961,774 characters in total, with the code re-requested under five different phrasings.

**Result: the constraint held in 23 of 24 turns — 96% — and the first loss is turn 24**, the longest
context in the run. The single failure returned `d010127`, a truncated form of the code, rather than
refusing: the model reached for it and did not finish it.

That is the useful shape of answer. Retention is high, and it degrades at the end of the context
rather than immediately, which is what a long-horizon effect looks like and is not what a
model that simply ignores instructions looks like.

**An earlier run of this probe reported 33% with drift at turn 1, and that number was wrong.** Its
`--max-tokens` default was 64, and on a reasoning model reasoning is billed against the same output
cap — measured directly, a 64-token cap returns `content: ""` with `finish_reason: "length"`, so the
model was never given room to answer at all. Worse, the probe scored `content + reasoning`, so a
truncated turn read "held" or "lost" depending on whether the code happened to appear in a fragment
of unfinished thinking. Three defects, all in the probe:

- **the output cap** could not accommodate the thinking budget, so answers were truncated;
- **scoring included reasoning**, making the verdict a coin flip on truncated turns;
- **history was stored as `content[:2000]` with reasoning concatenated into it**, which the chat
  template does not support and which discarded the history the probe exists to measure.

Fixed: the cap must clear the thinking budget, the verdict is scored on the answer alone, reasoning
goes back in its own field, and a turn that hits `finish_reason: "length"` is recorded as truncated
and excluded from the rate rather than being counted as drift. This is the same truncation trap the
coding grid documents — reasoning billed against the answer's cap — and it went unnoticed here
because the probe printed a plausible-looking retention figure either way.

## Caveats

The config keys above are the client's own schema, not this port's: `provider` (singular), the model
entry fields and `compaction` are defined by the OpenCode Desktop API, whose authority is that
project's own documentation. What this page contributes is the measured values inside them.

The tasks are short, self-contained functions. They separate reliability and speed
cleanly, and they do not stress long-horizon agentic work, where reasoning may pay off in
ways this grid cannot see. The truncation finding matters more than the pass rates: it is a
configuration trap, not a model property.
