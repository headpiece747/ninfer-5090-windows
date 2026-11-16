# v1.2.6 release plan

Status: packaged and locally verified; ready for branch cut
Owner: Tobias

## Why a release, not just commits

`v1.2.5`'s archive carries engine `f3bc4e01ca39`. `dev` now builds engine `3959d3d68b6e`,
integrating upstream `Neroued/ninfer@dev` updates through `d9ad9dea4` alongside Windows WDDM and
runtime correctness fixes:

- **Upstream Synchronization (`a754fcf3b`..`d9ad9dea4`)**: Concurrent execution capacity expansion
  (`54466d120`), unified context reclamation and retention cost accounting (`3b76b98c1`),
  multimodal prefix lookup and recovery (`1fdb37160`), grammar cache and token boundary fixes
  (`8e4648129`), Jinja capture lifetime hardening (`13ebb92f6`), malformed tool history rejection
  (`4461c9683`), and tuned RTX 5090 NVFP4 SwiGLU / FP8 / Q4 kernels (`7b87d8953`, `9dd11157b`).
- **Windows WDDM Headroom Guard**: All eight shipped launchers pass `--memory-headroom 1024MiB` in
  `tools/release/profiles.py` to avoid Windows WDDM Shared System Memory eviction while keeping
  `316,237-424,673` KV tokens above the `262,144` ceiling across every shipped artifact line
  (leaving C++ `kDefaultMemoryHeadroomBytes` at upstream's `512 MiB`).
- **Windows Portability & Sanitizer Fixes**: Restored `ConsoleOutputCodePage` on `ninfer` and
  `ninfer-perplexity`, async `GetAddrInfoExW` DNS in media acquire, `alignas(128)` `TmaKernelParam`
  preserving CUDA Graph capture without MSVC `C2719`, Direct I/O IOCP queue enablement, `selection.cuh`
  `cp.async` tail bounds guard (`compute-sanitizer --tool memcheck` clean), and MSVC interpose guard
  in `tests/cmake/NinferTests.cmake`.

## Version

**v1.2.6**, a patch release under `RELEASE_NOTES.md`'s versioning policy (no new artifact line or
container change since `v1.2.5`).

## What ships

- `ninfer-serve.exe`, `ninfer.exe`, `ninfer-perplexity.exe` built from `HEAD`, with `ninfer_ops.dll`,
  `ninfer_moe_cpu.dll`, the five FFmpeg DLLs, and `jpeg62.dll`.
- Root files: `README.md`, `RELEASE_NOTES.md`, `LICENSE`, `NOTICE`.
- The eight generated launchers plus `launcher_env.bat`, `download_model.bat`, `download_model.py`,
  `build_model.py`.
- `upgrade_ninfer_v2_to_v3.py`, `chat_templates/`, `tools/convert`, `tools/artifact`, and the proposal
  ranking data.

## Certified Engine Identity

The archive built from this tree carries `ninfer-serve.exe` at
`3959d3d68b6eb4c985b7513473cca5019d6f285429d40759567bccfb54b6ee89`, identical to `MEASURED_ENGINE` in
`tools/release/profiles.py`, so the published figures and the shipped binary match.

## Release target and branch

- Releases are published on **`origin`** (`headpiece747/ninfer-5090-windows`), where `v1.2.5` and its
  predecessors live. Because `upstream` (`Neroued/ninfer`) exists as a remote, release commands must
  pass `-R headpiece747/ninfer-5090-windows` explicitly.
- Work lands on `dev`. Per `AGENTS.md`, `main` is updated as a squashed release cut (`git merge --squash dev`), tagged `v1.2.6` on `main`, and pushed with the archive and `SHA256SUMS`.

## Gates before publishing

1. Fast & CI gates: `.githooks/pre-commit` (links, citations, encoding, profile consistency, port delta,
   converter/release pytest suite, `ruff`, `mypy`, `check_lane_figures.py`) and
   `tools/scripts/verify_as_ci.cmd`.
2. C++/CUDA suite & sanitizers: `tools/scripts/test_v3.cmd`,
   `tools/release/check_test_baseline.py --no-cache`, `tools/release/check_test_mutation.py`,
   `tools/scripts/test_v3_asan.cmd`, and `tools/scripts/test_v3_compute_sanitizer.cmd`.
3. Packaging & acceptance: `tools/release/package_release.py` (refuses Debug CRT, runs `pe_audit` and
   `check_archive_contents.py`) followed by `tools/release/check_release_archive.py`.

## Packaging and Acceptance Verification

- **Archive**: `ninfer-windows-v1.2.6-rtx5090.zip` (`438,507,862` bytes / `418.2 MB`, `92` entries;
  `SHA256SUMS` covers `91` files, all hashes verified by `check_archive_contents.py`).
- **Archive SHA-256**: `5360f480cdc3d60679faf38cd3e27fdbbbb0d76731a30dd961967998e15f4c92`
- **Shipped Engine SHA-256 (`./ninfer-serve.exe` inside zip)**:
  `3959d3d68b6eb4c985b7513473cca5019d6f285429d40759567bccfb54b6ee89`
- **Live Archive Acceptance (`tools/release/check_release_archive.py`)**:
  - Extracted to isolated temp directory and launched `start_quasar_v3_dflash2_vision.bat`
  - `engine beside the launcher: True`
  - `serves from the extracted copy : True  ['qwen3.8-27b-quasar-v3-dflash2-vision']`
  - `chat : HTTP 200, 16 tokens` (`exit 0`)
- **Runtime & Gate Summary on `3959d3d68b6e`**:
  - `tools/release/soak.py` (`qwen3.8-27b-quasar-v3-mtp4-vision`, `16.0 min`): `1,073` requests
    (`67.0 req/min`), p50 `0.64s`, p95 `2.16s`, max `3.42s`, `86,960` completion tokens, `0` failed requests.
  - `--max-concurrency 2` HTTP smoke (`qwen3.8-27b-quasar-v3-mtp4-vision`): `2/2` parallel
    `/v1/chat/completions` requests returned HTTP `200` (`64` completion tokens each) in `0.391s` total wall time.
  - `tools/scripts/test_v3.cmd` + `check_test_baseline.py --no-cache`: `200/203 passed` (`3` expected baselined failures, `4` optional skips; `16` `required_tests`).
  - `tools/scripts/test_v3_asan.cmd`: `8` host ASan tests passed (`FAILED=0`).
  - `tools/scripts/test_v3_compute_sanitizer.cmd`: `5`-test `compute-sanitizer --tool memcheck` subset clean.
