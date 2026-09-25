# Field log: a real agent session, 2026-09-22 -- the provenance of the two prefix defects

Both defects this port has been chasing are cited from one log, and until now the log itself was
nowhere in the tree: ADR-0007 line 44 and ADR-0009 line 12 both say "a live 51-request agent log",
and `tools/release/profiles.py` repeats the figures. This file is that log's record, so the two
claims can be traced to their evidence instead of to each other.

## Provenance

Captured 2026-09-22, 14:07:23 to 14:20:19, from a real agent session -- not a synthetic workload and
not a test rig -- against the QUASAR DFlash2 Vision launcher on the production port 8086. 51
requests, `thinking off`, 12 tools, 2 media items per request, streaming OpenAI chat completions.

The session owner reports the launcher as `start_quasar_v3_dflash2_vision.bat`, run from the source
tree. That file ships in both release bundles too, and the copies differ, so the tree's copy is the
one that matters -- the dates below show which revision of it produced this log. Its cache-relevant
flags at that revision, from the tree at `d845dbb7`:

```
--host-state-slots 16 --max-shared-prefixes 7 --max-private-continuations 8
--max-long-anchors-per-continuation 4 --context-cache-policy rolling
```

The remaining flags are the v1.1.0 bundle's, since the two bundle copies differ only in the three
lines below:

```
--vision --spec dflash2 --draft-tokens 7 --lm-head-draft --host 127.0.0.1 --port 8086
--model-id qwen3.8-27b-quasar-v3-dflash2-vision --max-context 262144 --device-state-slots 1
--kv-capacity auto --kv-dtype fp8 --prefill-chunk 8192 --max-concurrency 1
--host-kv-mib 8192 --preserve-thinking --default-thinking-budget 4096
--pending-timeout-ms 600000
```

**This is a tree build of 2026-09-22 14:07, not the v1.1.0 release.** The session owner's "v1.1.0"
is the release that was current that morning, but three pieces of evidence place the run on the
tree, with v1.2.0 in progress:

- The tree launcher resolves `%~dp0ninfer-serve.exe` first and `build\apps\ninfer-serve.exe` second.
  No exe sits beside the tree's copy, so it ran the tree build.
- The bat passes `--context-cache-policy rolling`, and the v1.1.0 bundle's binary **rejects that
  flag** -- so that binary could not have started at all.
- The tree launcher gained `--host-state-slots 16` and `--context-cache-policy rolling` together in
  `1a5dd7cf` on 2026-09-21 22:01 (before the log) and `--chat-template` in `c97ced1c` on 2026-09-22
  17:09 (after it). The last commit before the log's 14:07 is `d845dbb7` at 13:51, whose launcher
  carries exactly the five flags above and no template -- the log's startup block, field for field.
  The v1.2.0 bundle's binary is timestamped 15:17, an hour after the log, so that bundle did not
  exist yet either.

So the run has the rolling cache policy and the 16-slot host pool, and predates only the pinned chat
template.

The engine's own startup block, verbatim, because a figure without its configuration is how a wrong
number gets committed:

```
INFO  engine ready | qwen3.8-27b | total 11.6s | weights 16.9 GiB
INFO  capacity | KV 262,144 tokens, fp8, auto | pages 4,096/4,096 | runtime 10.7 GiB | free 2.66 GiB
INFO  context cache | 1 active + 1 cached device states | host 16 states, 8.00 GiB KV | private 8 | shared 7 | anchors 4
INFO  media | 16 preprocess workers | cache 1.00 GiB | live 2.00 GiB
INFO  listening on http://127.0.0.1:8086 | model qwen3.8-27b-quasar-v3-dflash2-vision | auth disabled
```

So: `--host-state-slots 16`, `--max-private-continuations 8`, `--max-shared-prefixes 7`,
`--max-long-anchors-per-continuation 4`, `--kv-capacity auto`, `--kv-dtype fp8`.

## The requests

Every completed request's prompt size, reported cache hit and reuse path, in order. The 5-second
throughput lines are omitted; the `started` and `done` lines they sit between are what this file is
about.

| req | messages | prompt | cache | path | TTFT |
|---|---|---|---|---|---|
| 1 | 373 | 195,955 | 0 (0.0%) | -- | 1m 13.0s |
| 2 | 375 | 196,365 | 196,139 (99.9%) | private endpoint | 7.4s |
| 3 | 377 | 196,817 | 196,497 (99.8%) | private endpoint | 7.8s |
| 4 | 379 | 197,366 | 197,134 (99.9%) | private endpoint | 7.8s |
| 5 | 381 | 198,955 | 197,599 (99.3%) | private endpoint | 8.5s |
| 6 | 383 | 199,331 | 199,188 (99.9%) | private endpoint | 7.9s |
| 7 | 385 | 199,472 | 199,451 (100.0%) | private endpoint | 7.7s |
| 8 | 387 | 199,577 | 199,556 (100.0%) | private endpoint | 7.8s |
| 9 | 389 | 199,918 | 199,849 (100.0%) | private endpoint | 8.0s |
| 10 | 391 | 209,861 | 200,062 (95.3%) | private endpoint | 12.9s |
| 11 | 393 | 210,064 | 210,035 (100.0%) | private endpoint | 8.1s |
| 12 | 395 | 210,504 | 210,248 (99.9%) | private endpoint | 8.2s |
| 13 | 397 | 210,733 | 210,680 (100.0%) | private endpoint | 8.2s |
| 14 | 399 | 210,981 | 210,928 (100.0%) | private endpoint | 8.2s |
| 15 | 401 | 213,548 | 211,170 (98.9%) | private endpoint | 9.5s |
| 16 | 403 | 214,135 | 214,102 (100.0%) | private endpoint | 8.3s |
| 17 | 405 | 214,560 | 214,527 (100.0%) | private endpoint | 8.3s |
| 18 | 407 | 214,889 | 214,660 (99.9%) | private endpoint | 8.4s |
| 19 | 409 | 215,121 | 215,088 (100.0%) | private endpoint | 8.4s |
| 20 | 411 | 215,725 | 215,232 (99.8%) | private endpoint | 8.7s |
| 21 | 413 | 218,495 | 218,462 (100.0%) | private endpoint | 8.5s |
| 22 | 415 | 219,609 | 218,620 (99.5%) | private endpoint | 9.0s |
| 23 | 417 | 220,115 | 220,082 (100.0%) | private endpoint | 8.5s |
| 24 | 419 | 220,500 | 220,223 (99.9%) | private endpoint | 8.6s |
| 25 | 421 | 220,867 | 220,575 (99.9%) | private endpoint | 8.6s |
| 26 | 423 | 221,041 | 221,008 (100.0%) | private endpoint | 8.6s |
| 27 | 337 | 171,953 | **0 (0.0%)** | -- | **59.6s** |
| 28 | 2 | 50,351 | **0 (0.0%)** | -- | 7.8s |
| 29 | 4 | 50,768 | 50,735 (99.9%) | private endpoint | 1.7s |
| 30 | 6 | 51,263 | 50,864 (99.2%) | private endpoint | 1.7s |
| 31 | 8 | 52,137 | 51,357 (98.5%) | private endpoint | 1.8s |
| 32 | 10 | 52,609 | 52,346 (99.5%) | private endpoint | 1.8s |
| 33 | 12 | 53,330 | 53,295 (99.9%) | private endpoint | 1.8s |
| 34 | 14 | 53,579 | 53,443 (99.7%) | private endpoint | 817ms |
| 35 | 16 | 53,981 | 53,656 (99.4%) | private endpoint | 1.9s |
| 36 | 18 | 54,338 | 54,058 (99.5%) | private endpoint | 1.9s |
| 37 | 20 | 54,652 | 54,617 (99.9%) | private endpoint | 2.0s |
| 38 | 22 | 55,168 | 55,133 (99.9%) | private endpoint | 2.0s |
| 39 | 24 | 55,656 | 55,280 (99.3%) | private endpoint | 2.1s |
| 40 | 26 | 56,063 | 56,027 (99.9%) | private endpoint | 2.1s |
| 41 | 28 | 58,200 | 56,169 (96.5%) | private endpoint | 2.5s |
| 42 | 30 | 59,494 | 59,459 (99.9%) | private endpoint | 2.2s |
| 43 | 32 | 59,697 | 59,662 (99.9%) | private endpoint | 2.2s |
| 44 | 34 | 60,042 | 60,007 (99.9%) | private endpoint | 2.2s |
| 45 | 36 | 60,459 | 60,424 (99.9%) | private endpoint | 2.3s |
| 46 | 38 | 60,730 | 60,584 (99.8%) | private endpoint | 2.3s |
| 47 | 40 | 60,932 | 60,844 (99.9%) | private endpoint | 2.2s |
| 48 | 42 | 61,382 | 61,048 (99.5%) | private endpoint | 2.4s |
| 49 | 44 | 61,532 | 61,476 (99.9%) | private endpoint | 2.3s |
| 50 | 46 | 61,815 | 61,641 (99.7%) | private endpoint | 2.3s |
| 51 | 48 | 61,945 | 61,915 (100.0%) | private endpoint | 2.5s |

## Defect 1: no prefix is shared across conversations

Requests 1-26 are one conversation (373 to 423 messages). Request 27 is a different one -- 337
messages, 171,953 tokens -- and it reports `cache 0 (0.0%)` and costs **59.6 s**. Request 28 starts a
third conversation with **2 messages and 50,351 tokens**, which is the system prompt plus the 12 tool
definitions, and it too reports `cache 0 (0.0%)`. Request 29, four messages into that same
conversation, immediately reports 99.9% against request 28's prefix.

So every conversation in this log begins with the same ~50k-token system-and-tools prefix, and reuse
works *inside* a conversation (95-100% from the second request onward, at every length from 50k to
221k) while it never happens *across* one. The two zero-cache requests are exactly the two
conversation switches, and request 27's 59.6 s is that prefix being prefilled from scratch at
3.11k tok/s.

This is ADR-0009's subject. Its finding: on the OpenAI protocol the serve layer clears
`allow_engine_automatic_shared_prefixes`, so the frontend's structural boundaries are never declared
and the only automatic boundary is the end of the last message -- the whole prompt. A shared prefix
therefore exists only for a request whose prompt matches at the very end, and a common leading
prefix nobody declared is never a candidate. A *declared* boundary is served, which is why the
defect is a missing declaration rather than a broken reuse path.

## Defect 2: the private turn closure never serves

Count the reuse paths in the table: **49 hits, all `private endpoint`**. `private turn closure`
appears zero times, and no shared path appears at all. Requests 2 through 26 reuse 196k-221k tokens
per turn and every one of them is an endpoint.

That is the same thing the engine's own test asserts and fails:
`ninfer_qwen3_5_prefix_real_test`, case `exercise_host_restore`, expects `path=2`
(`PrivateTurnClosure`) and gets `path=1` (`PrivateEndpoint`) with `state_h2d=0` and `degraded=1` --
the KV prefix is reused but no State is restored. The field log is what that test looks like in
production: the reuse happens, the checkpoint that would carry the continuation state does not
survive to serve it.

The failure is history rather than a live symptom, and the note above should be read that way. It was
fixed the same day, in `2e229b3e` ("reuse a demoted turn closure instead of its owner's session
endpoint"), after `1fa5b888` had recorded the attribution. Verified 2026-09-25 against this tree:
`ninfer_qwen3_5_prefix_real_test` passes -- exit 0, `ok`, all fifteen cases including
`exercise_host_restore` -- with the shipped `qwen3_8_27b_nvfp4qat.v3.ninfer`, with the published file
this port replaces, and with the official `qwen3_8_27b_nvfp4.v3.ninfer`. So the case is this log's
regression protection, and the log cannot date a defect that has since been fixed.

One reading to avoid: `private endpoint` is not by itself the symptom. The engine selects it when the
reuse source is the session's own endpoint checkpoint (`request_plan.cpp`,
`CheckpointKind::SessionEndpoint`), which is the normal path for a conversation whose state is still
resident, and selects `private turn closure` when the state has to come back from a demoted turn. This
log's defect was that the endpoint was stale while a closure that should have been preferred was
ignored -- not that the endpoint path was chosen. A live 115-request coding session on the shipped
`nvfp4qat` lane (2026-09-25, `out/agent_session_coding.log`) reused its own endpoint on 112 turns at
99.4-100.0% cache hit with 155-853 ms TTFT and no host restore at all: that is the healthy case, and it
did not reproduce this defect.

A third path name is worth recognising for the same reason: `private_response_replay`. Driving three
conversations round-robin on the same lane (2026-09-25, `out/agent_restore.log`) produced it on all nine
reusing turns at 94.6-95.2% cache hit and 111-147 ms TTFT, again with no host restore. That experiment
did not reach the demoted-turn path either -- one device state slot and sixteen host slots absorbed
three short conversations without eviction -- so `private turn closure` remains covered by the engine
test rather than by a live session here.

This is ADR-0007's subject, and the reason it matters is that the dropped checkpoint is the one
holding the speculative-decoding state: `private_turn_closure` is the path that restores MTP/DFlash
state.

The log does not attribute its `dflash2 accepted` rates, and an earlier draft of this file claimed it
did. They range from 24.2% to 89.9% and do not track the reuse path: request 21 reuses 100.0% and
accepts 25.7%, request 4 reuses 99.9% and accepts 36.4%, while request 37 reuses 99.9% and accepts
80.0%. Nor do they track the uncached tail -- request 21's is 33 tokens. Acceptance here is explained
by neither the missing closure nor the tail size, and this log cannot say what drives it.

## What this log does not establish

- It is one session on a tree build rather than a release, so it cannot date the defects or prove they
  are still present; the test and the request-log counters are the current instrument.
- It does not separate "the closure was dropped" from "the closure was never published". Both produce
  `private endpoint` on the next turn, and ADR-0007's counters (`pressure_private_owners_degraded`,
  `pressure_checkpoints_dropped`) are what split them.
- It says nothing about the shared path's *value*: with `shared 7` configured, `shared_stable_prefix`
  never appearing here is consistent both with "never declared" (ADR-0009's finding) and with
  "declared and declined".
- The cached path's ~7-8 s TTFT at 99.9% hit (req 2-26) is the host-State restore, not the prefill;
  this log does not attribute it further.
