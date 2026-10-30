# A 512-token context with prefix reuse resolves 1.00 GiB and then hangs the first request

**Target:** `Neroued/ninfer`
**Severity:** medium — a resolved, startup-accepted configuration deadlocks the Engine instead of
serving or being refused. It needs a small `--max-context`, so the shipped 262,144-token lanes are
not on this path; a short-context lane or a caller that sets one is.

**Found:** 2026-10-29, integrating `upstream/dev` `458863be2` into this port (35 commits,
`996d92146..458863be2`). The port's suite hung for two hours on this; the port is not otherwise
affected, because every shipped launcher uses the full context.

## Summary

With prefix reuse enabled, `--max-context 512` resolves a context pool of **1.00 GiB** at startup
(`KV ~1,536 tokens`, the engine's own line) and then the first request never completes. The request
is admitted, the materialization that precedes prefill finishes, and the worker parks: no output, no
error, no timeout. Removing prefix reuse serves the same request in ~1 s, and the same flags at
`--max-context 768` resolve a 14.3 GiB pool and serve.

An explicit capacity also reproduces it: the capacity the Engine accepts at startup can be too small
for the first unit, and the reservation path (`reserve_resident_units`) then has nothing to reclaim
and no way to report, so it waits instead of raising. This second half is the part worth fixing on
its own: **an insufficient explicit capacity is a legal configuration error and should be refused at
startup, not deadlocked at the first request.** `--context-capacity 256MiB` at 512/256 is refused
cleanly ("startup failed | planning runtime"), so the check exists — the 1.00 GiB automatic value
passes it and is still not enough for a unit.

## Reproduction

Measured on an RTX 5090, MSVC build, against a Qwen3.8-27B `nvfp4qat` v3 artifact:

```bat
ninfer.exe <artifact>.ninfer --prompt=Say --max-tokens 4 --spec none ^
  --max-context 512 --prefill-chunk 256 --max-concurrency 1 --prefix-reuse
```

Observed: startup prints `context | device 1.00 GiB across 1 rank | KV ~1,536 tokens | ~3.0 x
512-token contexts`, then the process produces no answer and does not exit (killed after 180 s; a
180 s cap is 30x the 5.9 s the same request takes below).

Controls, same artifact and flags:

| Change | Result |
| --- | --- |
| `--no-prefix-reuse` | answers, `finish reason output-limit`, 5.9 s |
| `--max-context 768` (reuse on) | answers; startup resolves 14.3 GiB |
| `--max-context 1024` / `2048` / `4096` (reuse on) | answers; 14.3 GiB |
| `--max-context 512 --context-capacity 384MiB` | answers (HTTP path unaffected) |
| `--max-context 512 --context-capacity 256MiB` | refused: `startup failed | planning runtime` |

So the trigger is the combination of a small context and reuse, and the size of the resolved pool
decides the outcome: 256 MiB is refused, 1.00 GiB (what `--max-context 512` resolves on its own) is
accepted and hangs, 384 MiB explicit at the same context serves. That non-monotonicity is the
finding worth chasing: it is not simply "too small is bad".

## What the port does

`tests/models/qwen3_5/context_budget.h`'s `context_capacity_bytes` is upstream's, and after
`bea5a04ec` unified KV and State into one pool it counts a retained checkpoint's **State record
only**. A retained checkpoint also owns the KV pages of the history it resumes, so the helper
under-budgets by roughly `pages(max_context)` per retained checkpoint. The port adds those pages
(the fix is recorded in `port_delta_baseline.json` with its reason), which raises the same test's
explicit capacity from 1.078 GiB to 1.23 GiB and serves in ~1 s. The helper is test-only, so this
does not fix the Engine's own automatic resolution above — that one is upstream's.

## Not reproduced

- `--prefix-reuse` at 768 or above, any backend tested (`none`, `mtp`).
- Without reuse at 512 (that is the control that answers).
- Through the port's shipped launchers (262,144-token context, MTP/DFlash2).
