# Host prompt preparation: the cost that grows with the conversation

This note records two things, and keeps them apart. The first is what the upstream tracker and the
wider ecosystem already know about the cost of preparing a prompt on a prefix-cache hit. The second
is where our own preparation time goes, read out of the code. Neither is a measurement of our
server; the instrument for that is named at the end, and its first reading is still outstanding.

## The field reading this starts from

A real agent session (229 messages, 170,625 prompt tokens at its end) served by this port, with the
engine's prefix cache at 97.9-99.8% on every turn. The `started` record reports preparation, and the
`done` record reports time to first token (TTFT):

| request | messages | prompt | `prepared` | `tokenize` | TTFT |
|---|---|---|---|---|---|
| #1 | 62 | 66,979 | 1.0 s | 422 ms | 11.8 s (cold) |
| #53 | 139 | 100,052 | 2.6 s | 628 ms | 3.2 s |
| #70 | 177 | 132,330 | 3.4 s | 835 ms | 4.3 s |
| #86 | 209 | 160,430 | 4.2 s | 1.0 s | 5.4 s |
| #96 | 229 | 170,625 | 4.5 s | 1.1 s | 5.4 s |

Preparation is 78-83% of TTFT at the end of the session, and it grows with the message count. The
remainder is the uncached tail (~1,700 tokens at ~2.0k tok/s), a ~60 ms queue wait, and the first
decode. So the growth in ask-to-first-response is not in the engine: the cache is already at 99%.

**`prepared - tokenize` is not "the render", and an earlier reading in this session said it was.**
`prepared` is all of `Frontend::prepare`, which also covers `convert_messages`,
`prepare_context_cache` and `assign_text_positions`. The split is now measurable instead of derived:
`PromptPreparationStats` carries `render_seconds` and `context_cache_seconds` alongside
`tokenize_seconds`, and the `started` record prints all three.

## What the code does per request

Read from the tree at `ba087d5b`, not measured.

`CompiledChatTemplate::render` (`src/models/qwen3_5/frontend/chat_template.cpp:153`) renders the
template more than once per request. Each additional render is a full pass over every input message:

- `:252` — the output render itself.
- `:353` — `prefix(1)`, when the first message is an instruction and the first boundary did not come
  out of the layout.
- `:357` — `prefix(messages.size())` whenever `add_generation_prompt` is set, which is every
  ordinary generating turn. This is a full render of the whole conversation with
  `add_generation_prompt` cleared.
- `:388-396` — the checkpoint probe, taken when the last real user turn is followed by a closed
  assistant message. That is the ordinary agent-loop shape. It renders the whole conversation **plus
  a synthesized empty user message**.
- `:419` — `prefix(marker.after_message_count)` once per `MessageBoundary` cache marker.

The `prefix` lambda (`:285`) is where the cost concentrates: it deep-copies the entire message
context as `Json` (`:289`) and then calls `compiled_.render` again (`:294`) to compare the result
against the output with `same_prefix`. The comparison is the point - *"Equal text alone does not
prove equal control-token interpretation"* (`:109`) - so the probes are not redundant work; they are
the proof that a byte frontier is a safe prefix-cache boundary.

So an ordinary turn on an agent workload does **one output render plus two full probe renders**, and
three deep copies of the message context, where a naive reading of "render the chat template once"
would predict one.

The template's own cost is the constant factor on each of those renders: `tools/chat_templates/qwen3_8.jinja`,
which is what the launcher passes, walks the message list twice per render: once in reverse to find
`last_query_index` (`:105-117`, including a `messages[::-1]` copy) and once forward to emit the
turns (`:123-186`). Each pass calls the `render_content` macro and applies `|trim` once per message,
and `:138` makes the emitted `<think>` block depend on `last_query_index` - which is why the render
of a given prefix can legitimately change between turns, and why the probes exist. How large that
constant factor is, is measured below.

## What the render actually costs, and what the bench got wrong

An earlier reading in this session inferred the render from `prepared - tokenize` and named it the
dominant term. A bench was then built to check that inference, it measured **~80 ms** for a
229-message conversation, and the note originally concluded from it that the render was ruled out.

**That conclusion was wrong, and the way it was wrong is the most useful thing here.** The bench is
real and it is in this tree - `bench/models/qwen3_5/chat_render_bench.cpp`, host-only, no artifact
and no GPU - and it does measure what it says. What it does not do is reproduce the state of the
process that serves requests, and for this measurement that turned out to be the whole question.

The server's own phase timers, on a real 229-message / 130,869-token request:

```
prepared 3.8s, contract 1.0 ms, convert 0.6 ms, render 2.4s, tokenize 1.4s, positions 0.3 ms, cache prep 0.02 ms
```

The phases sum to 3.803 s against 3.8 s reported, so preparation is fully accounted for and **the
render is 2.4 s of it** - the dominant term after all. `prepare_context_cache` (16-21 us),
`convert_messages` (0.4-0.6 ms), `build_tool_call_output_contract` (~0.8 ms) and
`assign_text_positions` (~0.3 ms) are all negligible on the real path, as reading had suggested.

### The render's cost is per message, not per byte

Two requests against the same server, same total bytes, different message counts:

| shape | messages | prompt tokens | out bytes | render |
|---|--:|--:|--:|--:|
| long conversation | 229 | 130,869 | 711,925 | **2,386 ms** |
| same bytes, few messages | 2 | 126,410 | 694,305 | **82 ms** |

Same bytes, 29x apart. And within the 229-message family the cost is flat across shapes - 229
messages of tools, no tools, tool calls, no tool calls, reasoning, no reasoning all land between
956 ms and 2,656 ms - so it tracks the **number of messages**, not the content, not the tools, and
not the bytes.

That is the signature of a per-**allocation** cost rather than a per-byte one: each message costs a
few Json objects and strings regardless of how big they are.

### The same body, same code, in two processes

The bench gained `--from <body.json>`, so it can render the conversation a server rendered instead
of one that merely looks like it. Same body, same code, same three render calls, same 300
checkpoints, same output size:

| process | messages | calls | checkpoints | render_ms |
|---|--:|--:|--:|--:|
| server (`ninfer-serve`) | 229 | 3 | 300 | **2,430-2,708** |
| bench, same body | 229 | 3 | 300 | **76.4** |

A 32x difference with the input held identical, so the cause is **process-level, not in the
frontend**. A neutral calibration loop - 20,000 short string allocations, identical checksum
1,368,890 in both - makes it visible:

| process | calibration_ms |
|---|--:|
| bench | **0.56 - 3.10** |
| server | **23.9 - 34.6** |

The server process runs ordinary host allocation ~40x slower than a fresh process on the same
machine, and the render, which is allocation-dense, inherits it.

### What was ruled out, each by a measurement

- **Build tree and flags**: the same bench measures 77.7 ms linked against `build/` and 77.8 ms
  against `build-test/`; both trees are Release with identical `/O2 /Ob2 /DNDEBUG`.
- **Memory pressure and commit**: the server commits 40.3 GiB on a 47.8 GiB machine with a 12.5 GiB
  working set, so the trimmed-heap theory looked strong. Restarting with `--host-state-slots 4
  --host-kv-mib 1024` dropped the commit to 30.3 GiB and the working set to 3.1 GiB, and the render
  stayed at 2,418 ms with calibration at 34.6 ms.
- **Holding a large allocation**: reserving 16 GiB committed-but-untouched in the bench changed
  calibration from 0.568 to 0.591 ms.
- **CUDA in the process**: initialising the CUDA runtime in the bench changed calibration from
  0.564 to 0.562 ms.
- **CPU starvation**: the server's wall and CPU times track each other (4.4 s wall, 4.17 s process
  CPU over a warm request), so the thread is running, not descheduled.
- **Priority and affinity**: `Normal`, all 32 logical processors.
- **The cancellation socket probe**: the interpreter's checkpoint fires 300 times per request and
  costs 6.5 ms in total, even though each call ends in an httplib `select()`.
- **Cache markers, template identity, special tokens, thinking default, tool-call argument size,
  tool schemas**: each varied, none moved the reading.

So the render is where the time is spent, and the render's code is not what is slow: **the serving
process executes it about 30x slower than a fresh process does.** What makes that process slow is
the open question, and it is not a prompt-preparation question.


## What upstream already knows

`docs/performance.md` in the upstream tree publishes
`server_ttft_ms = 1000 * (prepare_seconds + vision_seconds + prefill_seconds)`, so preparation is a
first-class TTFT term there too. Beyond that:

- **`Neroued/ninfer` PR #205** (merged) states the regime outright: *"Tokenisation is 79 to 81% of
  TTFT"* on a prefix-cache hit, and *"it is worth 5 to 15% of TTFT on cached-prefix traffic - agents
  and chat UIs that resend a growing conversation, which is exactly the traffic the prefix cache
  exists for."* On the cold long prompt the same report measures preparation at 0.9-1.3% of the run
  and calls it *"no measurable change"*.
- **PR #193** (merged) replaced the BPE merge map with a flat open-addressed table: `Tokenizer::encode`
  at 0.746-0.804 of base.
- **PR #205** (merged) skips `utf8proc` for pure-ASCII text: `encode` at -13 to -22%.
- Both of those are **already in this port**: the ASCII fast path at `src/text/unicode.cpp:19-29`
  (commit `641ef3e7`) and `BpeMergeTable` at `src/models/qwen3_5/frontend/tokenizer.h:60-70`.

Neither merged change is about the template render or the probe renders.

**Issue #269**, *"Prompt caching is way less efficient compared to llamacpp"*, is the user-facing
version of the same complaint and is closed. The reporter's own middleware was appending a hidden
tail message per turn, so the reusable frontier froze at the start of the open turn and every turn
re-prefilled ~60K tokens. After removing it: 96-98% cached and **TTFT 0.6-1.5 s**. The engine side
of that complaint is settled; a client that rewrites the tail still defeats any prefix cache.

No issue or pull request on the tracker describes preparation itself as the dominant TTFT term after
those two fixes. Repository-wide searches for `prepare_seconds`, `tokeniz` and `preparation` return
nothing about render cost, and the `jinja` hits are feature requests (#245, #182, #78) - the chat
template is a feature area upstream, not a performance one.

## What the wider ecosystem does about it

- **vLLM PR #47583**, *"[Perf][Frontend] Add opt-in incremental prompt-encoding cache for multi-turn
  chat"*: *"every request re-sends the whole conversation ... yet `_tokenize_prompt` re-tokenizes the
  full prompt every turn. That cost is linear in history length (~0.5 s per turn at ~200K tokens) ...
  With automatic prefix caching, tokenization is frequently the dominant TTFT term for long
  conversations."* Result: median multi-turn TTFT 0.92 s to 0.47 s; per-turn encode at 128K,
  151.82 ms to 2.77 ms (54.9x); at 512K, 782.94 ms to 10.95 ms (71.5x). The mechanism is a bounded LRU
  of `(text, token_ids)`, longest-prefix match, a 1024-character back-off to a pre-token boundary,
  re-encoding only from the seam, **verification** of the overlap against the cached encoding, and a
  full re-encode for anything unprovable. SGLang's SMG L1 prefix tokenization cache is cited as
  precedent, and vLLM #45514 is a separate, composable mechanism for cold prompts.
- **vLLM PR #20065** caches chat-template *resolution*, worth *"6-8 seconds"* over 1000 requests
  because `_detect_content_format` loads and iterates the Jinja template per call.
- **vLLM PR #30700** warms the template eagerly at startup, moving an ~850 ms first-request
  compilation cost off the first user request.

Two consequences for us. First, tokenizing the whole conversation each turn is a recognised defect
with a shipped design, so the tokenize share is recoverable. Second, none of that work is about
rendering: vLLM renders the full conversation text every turn too, and caches the *encoding*. A
template is a whole-conversation loop, so the render is not incremental by construction - for us the
recoverable render cost is the redundant probe renders, not the output render.

## A number that does not agree

Our `tokenize` runs at ~6.4 us/token (170,625 tokens in 1.1 s, ~155k tokens/s). Upstream's own
server row in PR #205 is 14.18 ms for an 8,475-token prompt, ~600k tokens/s. That is ~4x apart, on
different hosts, and this port carries both merged encode changes. It is recorded here as an open
question, not a finding: no A/B on one host has been run.

## What this note does not establish

- **What makes the serving process slow is not known.** The 32x is measured with the input held
  identical, and the calibration shows the process itself is ~40x slower at ordinary host
  allocation, but the cause is open. The bench's diagnostic arms for the hypotheses that were tested
  and refuted - a held allocation, CUDA initialisation, a disabled low-fragmentation heap - were
  reverted; the readings are in the table above. A sharper next experiment would be to take the same
  calibration *inside the server at startup*, before the artifact is loaded, and again after, to
  separate a process that starts slow from one that becomes slow.
- **The render's own shape is understood but not optimised.** It makes three full passes per request
  and the `prefix(messages.size())` probe is ~28 ms of the 80 ms a fresh process needs - worth having
  in a process that is not already 30x off, and not the current bottleneck.
- **No fix is proposed here.** The obvious candidates (fewer full renders per request, a verified
  encoding splice across turns like vLLM's) are dwarfed by the process-level factor, so fixing them
  first would be optimising the wrong thing.
- **Preparation runs concurrently.** It runs on `httplib::ThreadPool` workers
  (`src/serve/http_server.cpp:238-240`), so anything shared across requests needs real
  synchronization.
