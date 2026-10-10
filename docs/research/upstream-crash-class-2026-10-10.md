# Upstream's crash-class issues, read against this tree (2026-10-10)

Prompted by the soak in flight: it hunts the failure class upstream's tracker is full of, and the useful
question is which of those reports this tree has already fixed, which are stale, and which name a mechanism
we still carry. Every claim below was read on 2026-10-10 from the tracker
(`gh issue view <n> --repo Neroued/ninfer`) and from this tree's own history and source; the commands are
named so a reader can re-run them.

## Fixed, and the fix is in this tree

| issue | mechanism | fix |
|---|---|---|
| #210 (CLOSED) | `activate()` published the KV block table on the **legacy default stream** while prefill ran on a `cudaStreamNonBlocking` stream, so a stale block-table entry could be consumed. Reproduced 8 times in 4.2 days, **including at `--max-concurrency 1`** on ~99%-cache-hit turns. | upstream `064965c7` *refactor(kv): require explicit publication streams* and `75a89050` *fix(runtime): wait for compute before request cleanup*, both Neroued 2026-09-30, both on `upstream/dev` and in this tree |
| #329 (CLOSED) | `ProgramImpl::abort` released the lane without a device synchronization, so a cancelled prefill's queued table copies could read the next request's page indices. | the same `75a89050` |

Verified in the source rather than in a commit message:
`src/models/qwen3_5/program/storage/kv_address_space.h:271` declares
`commit_activation(KVActivationReservation&&, cudaStream_t stream)` with **no default**, `:230` forwards the
caller's stream, and both call sites pass `device.stream`
(`src/models/qwen3_5/program/transactions/binding.cpp:580`, `:585`). The reporter's own workaround — a
`cudaStreamSynchronize(nullptr)` after the publish — did not hold, and their analysis says why: it completes
the *write* but cannot undo an overlap that already happened. Ordering the publish on the compute stream is
the fix, and that is what upstream did.

This port had the defect documented before upstream fixed it: `1d0745b7` (2026-09-29, headpiece747) records
the unordered publish and ratchets its surface.

## Open, but stale

**#320** re-reports #210's mechanism and states it is "still present on both master (`bace20dc`, 2026-09-24)
and dev (`8eaed538`)". It was filed 2026-10-01, one day *after* `064965c7` removed it. Read today,
`upstream/dev`'s `kv_address_space.h:271` carries the same no-default signature as this tree, so the *named*
publish path is fixed on the ref the issue points at. Scope of this negative: the publish path #210 and #320
describe; both trees still default `cudaStream_t stream = nullptr` on unrelated helpers
(`src/core/linear_attention_state.h`, `src/core/paged_kv_cache.h`,
`src/models/qwen3_5/program/storage/state_store.h`), which is not the mechanism those issues name. The
issue stays open upstream; closing it is upstream's call.

## Open, and it touches a route this port now ships

**#333** (OPEN, no comments): `cudaErrorLaunchTimeout` inside `TextContext::prefill_impl<DFlashFeatureSink>`
at the **start of a prefill that immediately follows a ~98%-cache-replay prefill** of the same large
conversation. Their route is `--kv-dtype k8v4 --spec dflash2 --draft-tokens 7 --lm-head-draft` with prompts
around 138k tokens, two occurrences in two days. **Five of this port's eight shipped lanes now run k8v4**
(quasar DFlash2 and MTP4, ninfer MTP4, swift DFlash2 and MTP4), and this port's TDR figure — 43.10 ms
longest kernel, `server-open-items.md` item 6 — was measured on the **fp8** route. The route that ships is
the route without a longest-kernel measurement, and the upstream report against it is a watchdog timeout.
That is a named measurement gap, not a suspicion.

## Open, and it is this port's own open coverage item

**#184** (OPEN, no comments): a client that disconnects while the engine is in the *context-materialization*
phase keeps the only slot until the transaction finishes, and every later request queues and is rejected
after `--pending-timeout-ms`. Two mechanisms, both in upstream code this tree carries: the serve-side SSE
heartbeat cannot run while the engine blocks synchronously inside the request, and the engine skips
cancellation while a context transaction is open (`snapshot_cancellations` returns all-false under
`has_context_transaction()`). This port ships `--max-concurrency 1` and 262,144-token contexts, so the
window here is at least as long as the reporter's. It maps onto this tree's open coverage item "the
cancellation and queue cases" (`lane-coverage-2026-10-08.md`), and the report supplies the observables and
three suggested directions.

## Not ours

- **#357** (OPEN): `0xC0000409`/`0xC0000005` in a hybrid host-tier restore path, dying at
  `cudaStreamWaitEvent(ctx_.stream, layer_ready_[layer], 0)`. Neither `layer_ready` nor any flag of that
  profile (`--host-cache-mib`, `--adaptive-mtp`, `--use-alt-prefix-caching`, `--gdn-state-fp16`,
  `--prefill-cublas`) exists in this tree or on `upstream/dev`: it is a packaging fork's own code, and its
  reporter says removing their host tier stops it.
- **#391** (OPEN, 2026-10-10): a crash on the second GPU of a multi-GPU setup. This product is one GPU.
- **#208** (OPEN): the intermittent `cudaErrorIllegalAddress` the soak in flight targets. Its report is
  MTP4; the soak runs a DFlash2 lane, so it exercises the workload shape rather than that backend.

## What would close the two that matter

1. **#333's question for this route**: an nsys longest-kernel measurement on a **k8v4** lane at native
   context, the way item 6 measured fp8. Needs the card.
2. **#184's question for this product**: a lane at `--max-concurrency 1`, a cold ~150k-context request whose
   client disconnects during materialization, then observe whether the slot is held and later requests
   queue. That is the coverage item, with a reproduction recipe from the report. Needs the card.
