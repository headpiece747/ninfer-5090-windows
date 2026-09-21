# Upstream & fork survey — 2026-09-18

Repo: `C:\AI\ninfer-v3-windows`, branch `dev` at `fb9ae756`.
Our base: `5b4303c0` (= `neroued/master`, 2026-09-16 23:48:22 +0800). Our work: `5b4303c0..dev` = 76 commits (`git rev-list --count 5b4303c0..dev`).
Fresh fetches run this session:

```
git fetch upstream --tags
git fetch upstream '+refs/pull/*/head:refs/remotes/upstream/pr/*'   # moved pr/233, pr/271
git fetch cometkim                                                    # no ref updates
```

Authoritative remote tips (`git ls-remote upstream master dev`):

| ref | tip | date |
|---|---|---|
| upstream/master | `f76e19c0` | 2026-09-17 13:15:03 +0800 |
| upstream/dev | `8eaed538` | 2026-09-16 11:31:57 +0800 |

Both are already in `dev` (`git merge-base dev upstream/master` = `f76e19c0`; `git merge-base dev upstream/dev` = `8eaed538`). We merged the base at `e358a0cd` (parent 2 = `5b4303c0`) and master at `db51584e` (parent 2 = `f76e19c0`).

## 1. Upstream master/dev since base

**History between base and the current master tip — 4 commits, all already merged, all upstream-only:**

| commit | date | subject | paths touched |
|---|---|---|---|
| `cb30e070` | 2026-09-17 | bench(ops): time the public q5 a16 linear_add op through graphs and CSV | `bench/ops/`, `src/ops/linear_add/q5/` |
| `a9a0d10a` | 2026-09-17 | perf(ops): retune the q5 a16 linear_add routes at one token and above 512 columns | `src/ops/{attn_input_proj,gdn_input_proj,linear,linear_add}/…q4_q5|q5`, `tests/` |
| `b39de4d5` | 2026-09-17 | perf(ops): route the q5 a16 narrow-column linear shapes to narrow tiles | `src/ops/linear/q5/`, `tests/` |
| `f76e19c0` | 2026-09-17 | test(ops): close the q5 a16 route coverage gaps found in review | `bench/ops/`, `src/ops/`, `tests/` |

Proof of scope: `git log --name-only 5b4303c0..f76e19c0` — every path starts with `bench/`, `src/ops/`, or `tests/`. None touch `download_model.*`, `launcher_env.bat`, `start_*.bat`, `tools/release/`, or `docs/`, so nothing we ship is affected.

**New since our last sync:** none.
- `git log --oneline f76e19c0..upstream/master` → empty.
- `git log --oneline 5b4303c0..upstream/dev` → empty; `git merge-base --is-ancestor 8eaed538 5b4303c0` = true (the dev tip was already in our base).

## 2. cometkim branches since last fetch

Last-fetch baseline from reflogs: `fetch cometkim` at **2026-09-17 18:45:14 -0400**. Today's fetch produced no ref updates, so nothing changed.

- `cometkim/feat/nvfp4-dflash2` **has not moved**: tip is still `f9c333e8` (2026-09-16 17:37:44 +0900), the second parent of our merge `ebcfaa81`. `git log f9c333e8..cometkim/feat/nvfp4-dflash2` is empty; `git merge-base --is-ancestor f9c333e8 dev` = true.
- Every cometkim branch tip is dated 2026-09-16 (`git for-each-ref --sort=-committerdate refs/remotes/cometkim/`). Notable tips: `cometkim/dev`=`024f6a94`, `feat/build-speed-integration`=`2b5c2733`, `feat/kernel-perf`=`7e79acb0`, `feat/qwen3.8-nvfp4qat`=`c3d6aee7`, `feat/qwen3.8-nvfp4full`=`ac8e0b75`, `feat/windows-port`=`60b22bdc`.
- Merged into `dev`: only `feat/nvfp4-dflash2`. All other cometkim branches are unmerged (`git branch -r --no-merged dev`).

**Draft-binding conflict risk** — branches touching `src/models/qwen3_5/load/dflash2.cpp` (not `dflash.cpp`):

| branch | commit(s) relative to merge-base with dev (`1d8587bc`) |
|---|---|
| `cometkim/cometkim/dev` | `ee44f526` squash (through `246255e9`) |
| `cometkim/feat/dflash2` | `7ddded5f` feat(dflash): integrate NVFP4 routes and complete v3 conversion profiles |
| `cometkim/feat/kernel-perf` | `7ddded5f` |
| `cometkim/feat/build-speed-integration` | `7ddded5f` |

`git diff dev cometkim/feat/dflash2 -- src/models/qwen3_5/load/dflash2.cpp` shows they set the selector codebooks' input use to `{"dflash2/candidate_walk"}`, whereas our fix `9898ecb1` removed the input use entirely so absent/`candidate_walk`/`final_hidden` all load. Merging any of those branches will conflict in `dflash2.cpp`. `dflash.cpp` is byte-identical between `dev` and those branches (`git diff --stat` empty).

Also unmerged and relevant: card commits `c3d6aee7` (qat) and `ac8e0b75` (nvfp4full), both "docs(cards): make the v3 artifact canonical" (see §4).

## 3. Open PRs

`gh pr list -R Neroued/ninfer --state open` = **22 open** (state from GitHub; `git merge-base --is-ancestor <tip> upstream/master` is unreliable for squash-merges). 131 `upstream/pr/*` refs are fetched, **109 of which are stale** (closed/merged) — a ref's existence does not mean open. Only **PR 233** and **PR 271** moved since the last fetch (fetch output + reflog).

PRs whose tip touches nvfp4 / swiglu / tma / dflash / draft / Windows-MSVC:

| # | title | author | created | tip | gh mergeable | files | assessment |
|---|---|---|---|---|---|---|---|
| 264 | NVFP4 fused SwiGLU: take a partial last M tile… | MichaelDementii | 2026-09-16 | `fdd1dd1b` | MERGEABLE | `src/ops/linear_swiglu/nvfp4/*` (4), test | Adopt candidate: extends partial-M-tile to the fused route; same epilogue files as merged `05507ab0`, mild conflict. |
| 233 | Windows: native MSVC build with vcpkg-managed dependencies | troubadour-hell | 2026-09-12 | `672c0eb8` (new today) | MERGEABLE | ~28: `CMakeLists.txt`, `CMakePresets.json`, `vcpkg.json`, `apps/windows-utf8.manifest`, `src/artifact/file_io.*`, nvfp4 TMA files, `tools/upgrade_ninfer_v2_to_v3.py` | Strongest Windows candidate: `672c0eb8` merged current master into the branch; overlaps our own port. Adopt selectively. |
| 221 | MTP graph profiles carry no topology class… draft window past five fails at startup | MichaelDementii | 2026-09-09 | `f4bad56c` | CONFLICTING | `src/targets/qwen3_6_35b_a3b/impl/variant.cpp`, new test | Directly relevant (we ship mtp depth 4/5), but base is stale. |
| 235 | build: lower CUDA floor to 12.9, stage oversized GEMM smem dynamically | teo-mateo | 2026-09-13 | `527e7144` | CONFLICTING | ~28 nvfp4/w8 TMA + `CMakeLists.txt` | Broad build+kernel change; conflicts with our nvfp4/MSVC work. |
| 167 | The fp8 A8 GEMM stages its operands through TMA… | MichaelDementii | 2026-09-03 | `f0c5ed88` | CONFLICTING | new `fp8_a8_tma.cuh`/`fp8_a8_schedule.cuh`, 5 ops + tests | Perf only; our profiles use fp8 **KV**, not fp8 A8 GEMM. Low priority. |
| 107 | fix(qwen3.8): wire-format detect nvfp4 artifact profile | koloved | 2026-08-28 | `b03557e8` | CONFLICTING | `src/targets/qwen3_6_27b/*`, `qwen3_6_35b_a3b/*`, `registry.cpp` | Targets qwen3_6 packages, not our qwen3_8_27b artifacts. |
| 84 | feat(platform): native Windows (MSVC + CUDA) build | devan-carlin | 2026-08-22 | `bf29ea84` | CONFLICTING | ~38 files incl. nvfp4 TMA | Rival to 233; same author as closed 82. Less advanced than 233. |
| 59 | Windows support: build and run NInfer on Windows 11 (MSVC+CUDA+Ninja) | pelebel | 2026-08-19 | `ac5ab1f5` | CONFLICTING | ~40 incl. webui update, `serve_options.*` | Draft (`isDraft=true`); least advanced of the Windows trio. |

Previously evaluated PRs — current state:

| PR | prior verdict | now | note |
|---|---|---|---|
| 264 | adopted | **OPEN** | Tip unchanged; `git diff dev upstream/pr/264` non-empty, so this was a decision, not a git merge. |
| 115 | superseded | **MERGED** 2026-08-29 | Squash `5327d676` on master (`git log --grep`). |
| 82 | — | **CLOSED** (not merged) | Same author/title as 84; `git merge-base --is-ancestor` 82→84 = false. |
| 233 | — | **OPEN** | Moved today (see above). |
| 167 | — | **OPEN** | |
| 127 | — | **CLOSED** | Cold-pool v2; superseded/abandoned. |
| 235 | — | **OPEN** | |
| 131 | — | **CLOSED** | NVFP4 cold pool; superseded/abandoned. |

Supersession: 115 is done (merged). 82 was closed and 84 carries the same Windows/MSVC work (git-wise 82 is *not* an ancestor of 84, so not a strict supersession). 127/131 are closed. Among the three Windows PRs, **233 now leads** (MERGEABLE, merged current master). No new PR supersedes 264/167/235.

## 4. Artifacts / model cards

**Upstream: nothing new.**
- `git diff --stat 5b4303c0..upstream/master -- README.md docs/ model-cards/` → empty.
- Upstream artifact list (README.md:14-18): `qwen3_6_27b`, `qwen3_8_27b`, `qwen3_6_35b_a3b` (`groupwise-int` + `nvfp4`), each linked to `huggingface.co/neroued/…`. Our two shipped fork artifacts are absent.
- `model-cards/` on master is unchanged (the qwen3.8 nvfp4 card exists; no qat/full cards).

**cometkim: unmerged card commits re-canonicalize the Hub v3 files.**

| branch tip | commit | card | canonical filename | container | SHA-256 |
|---|---|---|---|---|---|
| `feat/qwen3.8-nvfp4qat` `c3d6aee7` | docs(cards): make the v3 artifact canonical | `Qwen3.8-27B-nvfp4qat-NInfer` | `qwen3_8_27b_nvfp4qat.ninfer` | 3 | `8b86901a…` |
| `feat/qwen3.8-nvfp4full` `ac8e0b75` | docs(cards): make the v3 artifact canonical | `Qwen3.8-27B-nvfp4full-NInfer` | `qwen3_8_27b_nvfp4full.ninfer` | 3 | `ac98cd39…` |

Both demote v2 to "superseded" (qat v2 SHA `3bd37e03…`; full v2 SHA `abb1e120…`) and drop the `.v3` suffix from the canonical Hub name.

Our side (`dev`):
- `model-cards/Qwen3.8-27B-nvfp4qat-NInfer/README.md` still says Filename `qwen3_8_27b_nvfp4qat.ninfer`, container **2**, SHA `3bd37e03…` — i.e. the pre-`c3d6aee7` card. **There is no `nvfp4full` card in `dev` at all** (`git ls-tree -r --name-only dev -- model-cards/`).
- `download_model.py:26-46`: quasar `source` `qwen3_8_27b_nvfp4qat.ninfer`, `target` `…nvfp4qat.v2.ninfer`, `upgrade_to` `…nvfp4qat.v3.ninfer`, pinned sha256 `df3c9c3a…` size `18_638_209_796` (v2); nvfp4full pinned sha256 `ac98cd39…` (matches the branch's v3 card) target `…nvfp4full.v3.ninfer`.
- Launchers use the `.v3.ninfer` names (`launcher_env.bat:7`, `start_{quasar,ninfer}_v3_*.bat`, `tools/release/profiles.py`), while cometkim's new canonical Hub name is the plain base name.
- The quasar pin `df3c9c3a…` matches **neither** card's v2 SHA (`3bd37e03…`) nor v3 (`8b86901a…`); it cannot be reconciled from these sources.

## 5. Artifact format / binding contract / serve CLI flags

- `git diff 5b4303c0..upstream/master -- src/serve/serve_options.cpp docs/cli.md docs/serving.md` → **empty**.
- `git diff --stat 5b4303c0..upstream/master -- src/serve src/artifact src/runtime/contract include/ninfer` → **empty**.

→ **No flag added, removed, or renamed upstream since base; no artifact-format or binding-contract change.** The pinned flags in `tools/release/profiles.py` (`INVARIANT_FLAGS` + per-profile) all still exist in `upstream/master:src/serve/serve_options.cpp`:

```
--kv-capacity --kv-dtype --prefill-chunk --max-concurrency --device-state-slots
--host-state-slots --host-kv-mib --max-shared-prefixes --max-private-continuations
--max-long-anchors-per-continuation --preserve-thinking --default-thinking-budget
--pending-timeout-ms --vision --spec --draft-tokens --lm-head-draft --host --port
--model-id --max-context
```

`docs/cli.md` exists on master (`ls-tree upstream/master docs/cli.md`) and is unchanged.

Open-PR caveat (future risk, not merged): PRs 173, 152, 61 (and draft 59) modify `src/serve/serve_options.cpp`/`.h`. They are CONFLICTING against current master, so their `serve_options` diffs are entangled with base drift and no flag change reaches us unless they are merged/rebased.
