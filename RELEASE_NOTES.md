# NInfer Windows v1.2.0 (RTX 5090)

## What changed in 1.2.0

Three defects: two retention faults and the configuration that hid them, plus a stray byte in the
shipped chat template. All four launchers change, so an upgrade behaves differently without any
action on your part.

- **A growing conversation stopped reusing its own prefix.** Once the State pools filled, publishing
  the newest checkpoint meant replacing a resident, and the capture could not be valued against doing
  so, so the reusable frontier pinned and every later request re-prefilled its whole tail. Measured on
  one conversation at one Device checkpoint slot: the frontier froze at 52,723 tokens with time to
  first token rising from 2.6 s at 65k context to 15.1 s at 117k. The launchers now pass
  `--context-cache-policy rolling`, which lets the portfolio fold decide whether a capture is worth more
  than the resident it would displace; the frontier then tracks the conversation (104,283 of 117,180
  tokens cached, 4.1 s).
- **Five conversations did not all stay cached**, because the host State pool could not hold the
  prefixes plus the private continuations they leave behind, and the shortfall was silent: five prompts
  sent and then resent gave 3/5 hits at a 59.1% token-level rate. `--host-state-slots` is now 16, which
  covers every configuration the other two bounds permit (7 shared prefixes plus 8 private
  continuations). The same test gives 5/5 at 98.9%.
- **`--device-state-slots 8` is no longer needed** as a workaround for the first defect. The launchers
  stay at one Device slot, which saves 1.3 GiB and a measured decode cost.
- **A release check now refuses to package** when the cache bounds stop retaining the working set, so
  this class of silent shortfall cannot reach a release again.
- **Every prompt began with a stray zero-width character.** The chat template embedded in the shipped
  artifact carried a UTF-8 byte-order mark, and the renderer emitted it as the first character of the
  prompt: one wasted token on every request, and on the vision path it also broke prefix reuse, so a
  follow-up turn re-prefilled instead of reusing what it had. The template loader strips a leading BOM
  now, and both prompt goldens match the reference tokenizer's counts again (58 and 18).

The request log gained a `policy` field and a `captures` group (`offered`, `no_vacancy`,
`plan_refused`, `infeasible`) naming why a capture was refused. The log schema version is 22.

First Windows release on the **v3 artifact line**, with speculative decoding working on
upstream-shaped artifacts, four measured-optimal launchers, and two production bugs fixed
that earlier builds shipped.

## Requirements

- NVIDIA GeForce RTX 5090 (32 GB, `sm_120a`) and a recent driver.
- **No CUDA toolkit needed at runtime.** The CUDA runtime is linked statically; the engine
  loads from this folder.
- Windows 10/11 x64.

## What runs

| Launcher | Artifact | Spec | Vision | Context | Decode | Draft accept |
| --- | --- | --- | --- | --- | --- | --- |
| `start_quasar_v3_dflash2_vision.bat` | QUASAR QAT | DFlash2 (7) | yes | 262,144 | **343 tok/s** | 62.5% |
| `start_quasar_v3_mtp4_vision.bat` | QUASAR QAT | MTP (4) | yes | 262,144 | 220 tok/s | 58.3% |
| `start_ninfer_v3_dflash2_vision.bat` | NVFP4-full | DFlash2 (7) | yes | 262,144 | **345 tok/s** | 63.7% |
| `start_ninfer_v3_mtp5_vision.bat` | NVFP4-full | MTP (5) | yes | 262,144 | 254 tok/s | 64.2% |

Every number was measured on an RTX 5090 with the exact arguments the launcher passes, in one
interleaved pass; every context ceiling is the highest value the engine accepts for that
configuration -- the next step up is refused, not degraded. Decode varies by up to ~9% between
sessions on a card whose clocks are not pinned, so compare lanes to each other and expect your own
absolute figures to differ.

**QUASAR is the recommended profile**: our own artifact, at the full 262,144 context, with a DFlash2
lane that measures within noise of the other artifact's (343.4 against 344.6 tok/s, measured
interleaved). ADR-0004 records why the second lane rides a third-party repository with no in-house
fallback. Vision is free on both (the with/without comparison is in
[ADR-0004](docs/adr/0004-vision-only-and-third-party-artifact.md)).

## Getting a model

`download_model.bat` fetches the recommended QUASAR QAT artifact and verifies its SHA-256.
The QUASAR profile comes from `cometkim/Qwen3.8-27B-nvfp4qat-NInfer` and the NVFP4-full
profiles from `cometkim/Qwen3.8-27B-nvfp4full-NInfer`; both repositories ship a v3 container, so
either downloads and runs directly. Both are SHA-256 verified by `download_model.bat`, which
offers the choice. The pinned size and hash are the published artifact's: if a download is
refused on size, the repository republished and the pin needs updating from HuggingFace's blob
metadata (a republish is how the previous pin went stale).

Put the `.ninfer` file at `C:\AI\models\` (the path `launcher_env.bat` expects), then double
click the launcher you want. Each launcher checks the engine and the artifact exist before
starting, and leaves the failure on screen if they do not.

## A pre-existing v2 copy must be upgraded first

Neither published repository ships v2 any more, so a fresh download needs no upgrade. A copy
fetched before the republish does: the v3 engine **rejects v2 containers outright** and names
the tool in the error:

```
python3 tools/upgrade_ninfer_v2_to_v3.py INPUT.ninfer OUTPUT.ninfer
```

That upgrade is offline and preserves the weight bytes. On Windows this now works; in
earlier builds the script called POSIX-only `os.posix_fadvise`/`os.fdatasync` and died with
`AttributeError`. The tool ships in this archive, beside the `chat_templates/` data it reads.

## Speculative decoding is not bit-identical to plain decoding

Greedy output differs between no-spec, each MTP depth and DFlash2, deterministically. This
is documented engine behaviour rather than a defect: acceptance compares a proposal token
against the target argmax for its verify column, and the maintainer notes state that
speculation "does not impose token or logits equality between different quantization,
prefill or kernel paths" — the batched verify kernel is not the single-token decode path,
so a near-tie can flip. Speculation measured 3-4x faster (67-83 tok/s without it, 239-343
with it).

## Fixed in 1.1.0

- **DFlash2 on upstream-shaped artifacts.** Our loader demanded a fused
  `dflash2/layers/*/attention/query_key_value` parameter that upstream's converter never
  emits — it groups `attention/query`, `attention/key` and `attention/value` instead. Any
  artifact not produced by our own upgrade path was refused at startup with
  `FATAL missing logical parameter`. The runtime already assembles the fused parent itself,
  so the redundant requirement is gone. The official Qwen3.8-27B and community
  fuller-NVFP4 artifacts now serve with `--spec dflash2`.
- **A graph-capture fault in the NVFP4 A4 TMA route.** The Windows port kept the tensor-map
  descriptor in a device buffer filled from a caller stack frame; under CUDA Graph capture
  that copy becomes a node whose source is gone by replay, so the TMA unit read a dead
  frame and the kernel trapped with an illegal instruction. The descriptor now travels by
  value in parameter space, as upstream does, which is also what makes it replay-safe.
- **Artifact reads blocked writers.** `FILE_SHARE_READ` alone meant Windows refused any
  write to a file the engine had open, where POSIX permits it. Reads now share
  read/write/delete.
- **Python tooling could corrupt artifacts on Windows.** `os.open` without `O_BINARY` opens
  in text mode, which translates CRLF and stops at `0x1A` inside payload bytes, and
  `os.pread`/`os.pwrite` do not exist there. Both are now portable.
- **Context-cache bounds.** The launchers set `--max-shared-prefixes 7
  --max-private-continuations 8 --max-long-anchors-per-continuation 4`. With the engine
  defaults, five distinct prompts resent gave 1/5 round-2 hits at a 19.8% token-level hit
  rate, with four of five re-prefilling in full on every call and no signal; the bounds
  give 5/5 at 99.1% and cost nothing measurable.
- **CORS removed from every launcher.** They passed `--cors` with no `--api-key`, which
  lets any browser page on the machine drive the model and read completions. opencode talks
  to these endpoints as a native HTTP client, so CORS bought nothing.

## Known limitations

- All four profiles reach the full 262,144 context with Vision. Earlier builds capped the
  NVFP4 lane because that artifact carried 19.7 GiB of device weights; the one shipped now
  carries 17.0 GiB.

## Release numbering

Versions are published in order and never skipped. `v1.0.1` and `v1.0.2` were built on the
maintainer's machine and never released — their archives are still in `C:\AI\releases` — so the
published list reads `1.0.0, 1.0.3, ...`. Nothing was withdrawn, and `1.0.0` is unaffected.

Patch numbers carry fixes to the shipped profile set; a minor number carries a new artifact line,
which is what `1.1.0` is: the v3 container, the measured profile table, and the QUASAR lane.
