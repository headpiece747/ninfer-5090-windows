# The resource-subtraction underflow: issue 5, investigated 2026-09-25

`headpiece747/ninfer-5090-windows` issue 5 reports that `"Qwen3.5 resource subtraction underflow"`
kills the engine: every in-flight request fails with HTTP 500 in one scheduler step, `/health` then
answers 503 `{"status":"unavailable"}`, and the process stays alive without recovering. Fourteen
incidents in 24 h, against the v1.1.0 release, with the two `cometkim` v3 artifacts and never with the
`neroued` one, at `--max-concurrency 3`.

This note records what the port did about it and what it measured, so the answer does not have to be
re-derived.

## What the guard is

`src/models/qwen3_5/program/context_work.cpp:222` throws when a resource release exceeds what is held,
across device active lanes, device state slots, device main and backend KV pages, host state slots and
host KV bytes. `src/runtime/engine/engine_core.h:1912` (`fail_all_locked`) is what turns that into the
reported shape: any worker exception sets `failed_` **once, at one site, and nothing ever clears it**,
resets the scheduler, and fails the in-flight requests. `docs/serving.md:70` documents the consequence
as the contract — engine-wide failure answers 503, and temporary queue saturation deliberately does
not. So the "stays 503 until restart" half is fail-stop by design, not a bug in itself; the reporter's
request there (recover, or exit non-zero so a supervisor can tell the difference) is a product decision.

## Why it does not reproduce here

The reporter supplied a deterministic case: their own command line, their artifact, two OpenAI chat
streams with 24 tools, `reasoning_effort: medium`, assistant turns echoed back with
`reasoning_content`, and default catalog bounds. They measured **FAIL after ~50 s**. On the current
tree that same run produced:

| run | requests | reuse path | cache hit | prompts | overlap | result |
|---|---:|---|---:|---:|---:|---|
| their deterministic case | 698 in 4 min | 696 `private_endpoint` | median 100% | median 119,882, max 142,887 | 339 of 698 | **no underflow**, `/health` ok |
| soak, same configuration | 6,070 in ~57 min | same shape | same | a 220k-token cold prefill concurrent with the other chain | continuous | **no underflow**, `/health` ok |

The trigger they isolated — a continuation that reuses cached state starting while a second request is
still generating — was therefore present continuously in both runs, and the failure did not occur. The
soak covered about the same span as their production mean interval (14 incidents in 24 h, so ~1.7 h),
which is suggestive rather than conclusive.

The likely reason is timing. **v1.1.0 is `c15791f2`, 2026-09-20**, and two prefix-cache fixes landed
after it:

- `feee7122` 2026-09-21 — "retain a superseded resident under rolling context-cache policy";
- `5076f445` 2026-09-22 — "size the default shared catalog for one request's candidates".

The second is the port's answer to upstream's #270, in the same family as #251 (prefix reuse stopping
until restart, fixed here 2026-09-19) and #229 (a fixed 5 ms planner budget causing root re-prefill in
multi-session states, fixed by `d4929686`). Upstream has **no** report of the underflow itself, so this
failure is v1.1.0's experience rather than a standing upstream defect: the family it belongs to is now
fixed in this tree.

## What this does not establish

- Absence in ~1 hour is not absence over a day. A longer soak, or more chains, would raise confidence;
  the harness for both runs is reproducible from the numbers above.
- The shipped lanes run `--max-concurrency 1`, where the reporter never saw an incident ("0 incidents
  in all tests and in ~9 h of single-lane operation"), so the trigger is further away than their
  production configuration — but that is a configuration fact, not a property of the artifact.
- Their production configuration also differs in `--host-kv-mib 24576`, `--kv-capacity 294912`,
  `--pending-timeout-ms 3600000` and `--max-pending-requests 32`. The runs above reproduced all of
  those.

## Evidence

`out/issue5_repro.log` and `out/issue5_repro.jsonl` hold the deterministic run; `out/issue5_soak.log`
holds the soak's own output. The artifact under test was `cometkim`'s own published file, verified
against the Hub's object id (`ac98cd39…`) before the run, reached through a hardlink because the engine
accepts only a `.ninfer` extension.
