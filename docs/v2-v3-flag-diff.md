# v2 launcher flags vs v3: a systematic diff

v2 was tuned over a long period, so its `.bat` files are a record of accumulated knowledge.
Diffing every flag in both generations (15 v2 files across both repositories against the four v3
launchers plus `launcher_env.bat`) shows what v3
dropped and whether any of it was worth keeping.

## Only in v2

| flag | verdict |
| --- | --- |
| `--build build\|build_vision`, `--config Release` | build-script flags, not serve flags. N/A. |
| `--dest`, `--prompt`, `--messages` | from `download_model.py` and `test_prompt.bat`. N/A. |
| `--cors` | **removed on purpose.** It let any browser page drive the model and read completions, and opencode talks to the endpoint as a native HTTP client. Not a loss. |
| `--max-new 128\|256` | a server-side default output cap. opencode always sends its own `max_tokens`, so it only affects clients that do not — and a default of 128 would silently truncate those. v3 omitting it is safer. |

## Only in v3

`--max-shared-prefixes 7`, `--max-private-continuations 8`,
`--max-long-anchors-per-continuation 4` — the context-cache bounds. Without them the defaults
gave 1/5 round-2 hits at a 19.8% token-level rate; with them, 5/5 at 99.1%.

## Values that differ

| flag | v2 | v3 | why v3 |
| --- | --- | --- | --- |
| `--max-context` | 131072, 240000, 262144 | 262144 | each is the measured maximum that serves per profile |
| `--draft-tokens` | 5, 7 | 4, 5, 7 | per artifact: depth 4 is fastest on QUASAR, 5 on NVFP4 |
| `--prefill-chunk` | 1024, 4096, 8192 | 8192 | measured best |
| `--kv-capacity` | explicit values and `auto` | `auto` | sizes each pool from the VRAM left after weights |
| `--host-kv-mib` | **16384** | 8192 | measured: no effect on cache hits (below) |
| `--host-state-slots` | **16** | 8 | measured: no effect on cache hits (below) |
| `--port`, `--model-id` | one port, v2 naming | four ports, `-v3` suffixed | four profiles run side by side |

Identical in both: `--host`, `--kv-dtype fp8`, `--max-concurrency 1`,
`--device-state-slots 1`, `--spec`, `--vision`, `--lm-head-draft`, `--preserve-thinking`,
`--default-thinking-budget`, `--pending-timeout-ms`.

## The one candidate worth testing, and its result

`--host-kv-mib` and `--host-state-slots` were the only substantive difference: v2 gave the
pinned **host** KV pool twice the room and twice the cached state slots. Host KV is system
RAM rather than VRAM, and it is what retains cached conversation states — which is what a
prefix hit reuses — so it was a plausible source of extra cache hits.

Measured with the same five-distinct-prompt method that produced the 99.1% figure
(`tools/release/check_host_kv.py`):

| host-kv-mib | state slots | round-2 hits | token hit rate |
| --- | --- | --- | --- |
| 8192 (v3) | 8 (v3) | 5/5 | 99.1% |
| 16384 (v2) | 16 (v2) | 5/5 | 99.1% |
| 16384 | 8 | 5/5 | 99.1% |
| 8192 | 16 | 5/5 | 99.1% |

No difference in any combination. The host pool is capacity, not a constraint: what limits
cache retention is the shared/private/anchor bounds, which v3 sets explicitly. **v2's larger
host KV is not worth adopting**, and it would pin an extra 8 GB of the 48 GB of system RAM
for nothing.

## Conclusion

Nothing in v2 is worth carrying into v3. Every difference is either a build-script flag, a
deliberate removal, or a value v3 measured and improved. The two candidates that looked like
accumulated wisdom — host KV size and state slots — turned out to be inert once the cache
bounds are set.

One v2-era behaviour is worth knowing about rather than copying: the vision launchers used a
separate `ninfer-serve-vision.exe` binary, so v2 shipped two zips. v3 enables Vision with
`--vision` on one binary, which is why v1.1.0 ships a single archive.
