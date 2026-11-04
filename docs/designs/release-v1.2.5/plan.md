# v1.2.5 release plan

Status: ready for packaging
Owner: Tobias

## Why a release, not just commits

`v1.2.4`'s archive carries engine `00947874`. The tree builds `f3bc4e01ca39`, which incorporates 11 upstream
commits from `Neroued/ninfer@dev` (`5b834fc35`..`a754fcf3b`) and four native Windows port commits
(`58203f11f`..`280079df8`).

The new capabilities and fixes in this release:
- **Interactive Web Monitoring Dashboard**: Real-time web UI (`dashboard/`) providing live telemetry,
  per-rank memory tracking, responsive chart layouts, Paper theme curation, and Docker deployment support.
- **Engine Observability Endpoints**: Internal snapshot and event streaming API (`/v1/internal/monitor/...`)
  distinguishing active request slots, cached context pages, and system memory allocations.
- **Runtime Starvation Prevention** (`a754fcf3b`): Implemented request queue fairness and priority scheduling
  under context pressure, validated by `tests/models/qwen3_5/test_engine_fairness_real.cpp`.
- **Cache Retention & Reclaim Refinements** (`76b5370bd`): Refined LRU and context page eviction policies to
  maximize hit rates under tight GPU memory budgets.
- **MSVC 128-bit Integer Hardening**: Added native fallback `ninfer::Uint128` arithmetic (`core/uint128.h`) for
  MSVC, resolving cost-model overflow and ensuring accurate context eviction calculations.
- **Test Suite Restorations & Mutation Seam**: Restored CMake registrations for `ninfer_topk_logprobs_test`
  (re-enabling Op perturbation verification in `check_test_mutation.py`) and `ninfer_uint128_test`, bringing
  the verified test suite to 198 tests.
- **Clean Cross-Platform Process ID Handling**: Encapsulated platform process ID resolution in `current_process_id()`,
  eliminating orphaned `<unistd.h>` includes on Windows.

`RELEASE_NOTES.md` is titled v1.2.5 and its lane table quotes `f3bc4e01ca39`'s figures, which the v1.2.4
archive does not contain. Leaving it untagged means the notes describe a build users cannot download.

## Version

**v1.2.5**, a patch release. No artifact format or launcher argument changed. The engine incorporates upstream
observability APIs, context starvation and retention improvements, and Windows MSVC type hardening.

## What ships

- `ninfer-serve.exe`, `ninfer.exe`, `ninfer-perplexity.exe` built from HEAD, with `ninfer_ops.dll`,
  `ninfer_moe_cpu.dll`, the five FFmpeg DLLs and `jpeg62.dll`.
- Root files: README, RELEASE_NOTES, LICENSE, NOTICE.
- The eight generated launchers plus `launcher_env.bat`, `download_model.bat`, `download_model.py`,
  `build_model.py`.
- `tools/upgrade_ninfer_v2_to_v3.py`, `tools/chat_templates`, `tools/convert`, `tools/artifact`, and the
  proposal ranking data.

## Certified Engine Identity

The archive built from this tree carries `ninfer-serve.exe` at `f3bc4e01ca39229359f7447b43332b5561c2f2b654eb3b6c9340641ce31d39ad`,
identical to `MEASURED_ENGINE`, so the published figures and the shipped binary are the same engine.

## Release target and branch

- Releases are published on **`origin`** (`headpiece747/ninfer-5090-windows`), where v1.2.4 and its
  predecessors live. Because `upstream` (`Neroued/ninfer`) exists as a remote, every release command
  must pass `-R headpiece747/ninfer-5090-windows` explicitly.
- The work is on `dev`. `main` is the release line: fast-forward it to `dev`, tag `v1.2.5` there, push
  `main` and the tag, then create the release with the archive and `SHA256SUMS`.

## Gates before publishing

1. Invariant gates: `check_doc_links.py`, `check_doc_citations.py`, `check_dead_types.py`,
   `check_retired_names.py`, `check_local_paths.py`, `check_port_delta.py --check`,
   `check_lane_figures.py`, `check_test_mutation.py`, `check_test_baseline.py`.
2. `package_release.py` — runs doc links, cache capacity and the recorded test baseline; refuses a Debug
   build; writes the archive, `SHA256SUMS` and prints the digest.
3. `check_release_archive.py` — extracts the archive and runs live inference acceptance test against port 8086.

## Packaging and Acceptance Verification

- **Archive:** `ninfer-windows-v1.2.5-rtx5090.zip` (324.7 MB)
- **SHA-256:** `34e3d63e62a7152fa67ecbf94f49429aa16af6c1336b62e252bc6ca789ab8873`
- **SHA256SUMS:** covers all 85 staged files in `stage-v1.2.5/SHA256SUMS`
- **Engine Hash:** `f3bc4e01ca39229359f7447b43332b5561c2f2b654eb3b6c9340641ce31d39ad` (byte-for-byte identical to `MEASURED_ENGINE`)
- **Live Acceptance:** Verified by `check_release_archive.py` (served QUASAR DFlash2 on port 8086, HTTP 200, 16 completion tokens).
