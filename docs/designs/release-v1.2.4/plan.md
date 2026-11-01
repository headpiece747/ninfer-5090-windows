# v1.2.4 release plan

Status: draft for review
Owner: Tobias

## Why a release, not just commits

`v1.2.3`'s archive carries engine `874fce8a`. The tree builds `00947874`, which includes `7eb03a266`
(integer multiplies widened after a code-scanning finding). The tree's documentation also changed
substantially: flags the parser rejects, a route that does not exist and a refusal table built from codes
the server never emitted were corrected, and every lane figure was re-measured.

`RELEASE_NOTES.md` is titled v1.2.3 and its lane table now quotes `00947874`'s figures, which the published
archive does not contain. The tree is already a release candidate; leaving it untagged means the notes
describe a build nobody has.

## Version

**v1.2.4**, a patch. No artifact format, protocol or launcher argument changed. The engine changed in one
respect (integer widening in geometry multiplies); the rest is documentation and measurement.

## What ships

- `ninfer-serve.exe`, `ninfer.exe`, `ninfer-perplexity.exe` built from HEAD, with `ninfer_ops.dll`,
  `ninfer_moe_cpu.dll`, the five FFmpeg DLLs and `jpeg62.dll`.
- Root files: README, RELEASE_NOTES, LICENSE, NOTICE.
- The eight generated launchers plus `launcher_env.bat`, `download_model.bat`, `download_model.py`,
  `build_model.py`.
- `tools/upgrade_ninfer_v2_to_v3.py`, `tools/chat_templates`, `tools/convert`, `tools/artifact`, and the
  proposal ranking data.

## The one claim that needs care

The folded decode column is the 2026-11-01 window's reading: **+7.8 %** median against the previous table,
range +3.4 % to +10.3 %. The notes must present it as that window's reading and say so plainly, exactly as
`profiles.py` does. They must not say the engine became faster — the same binary reads 6-8 % apart between
sessions, and the previous release note already warns about it.

Acceptance moved on five lanes by 0.1-1.2 points, which is inside the instrument's session spread.

## Release target and branch

- Releases are published on **`origin`** (`headpiece747/ninfer-5090-windows`), where v1.2.3 and its seven
  predecessors live. `gh` resolves the working copy to `Neroued/ninfer` because the `upstream` remote
  exists, so every release command must pass `-R headpiece747/ninfer-5090-windows` explicitly.
- The work is on `dev`. `main` is the release line: fast-forward it to `dev`, tag `v1.2.4` there, push
  `main` and the tag, then create the release with the archive and `SHA256SUMS`.

## Gates before publishing

1. `package_release.py` — runs doc links, cache capacity and the recorded test baseline; refuses a Debug
   build; writes the archive, `SHA256SUMS` and prints the digest.
2. `check_release_archive.py` — verifies the archive's contents against the packager's own file list.
3. The archive should contain the retitled notes and the re-rendered launchers.

## Review dispositions (2026-11-01)

Three fresh-context reviewers attacked this plan. What survived, and what changed:

| Finding | Disposition |
|---|---|
| The archive is not tied to the certified binary; the binary might predate `7eb03a266` | **Fixed.** The concern came from the commit *date* (15:06 UTC) being later than the build (14:44 UTC). The files were on disk at 14:42, and an incremental build of both trees reports `ninja: no work to do`, so `build/apps` is current with HEAD and the hash stays `00947874`. The gate is now explicit: hash `build/apps/ninfer-serve.exe`, compare with `MEASURED_ENGINE`, and hash the same exe from the extracted archive. |
| The +7.8 % must not be published as an engine figure | **Fixed.** The notes say it is the instrument's session condition, quote 6-10 % between sessions, and state that the engine is unchanged. |
| `RELEASE_NOTES.md` and `lanes.md` prose contradict the folded table | **Fixed.** The v1.2.3 paragraph now says "in the v1.2.3 table … read 314 against 262"; `lanes.md` quotes the current cells (336.3 / 329.2 / 314.5 / 271.2). |
| `README.md` still announced v1.2.3 | **Fixed.** It now announces v1.2.4. |
| `dev` is not pushed | **Fixed.** The sequence pushes `dev`, then fast-forwards `main`, then tags. |
| `main` fast-forward versus AGENTS.md rule 33 ("squashed release cut") | **Accepted as a recorded discrepancy.** `main`'s tip is a commit shared with `dev`, so the practice is a fast-forward. The rule is not edited during a release; the precondition `git merge-base --is-ancestor main dev` is checked and exits 0. |
| The test gate needs `NINFER_TEST_ARTIFACT` and a current `build-test` | **Fixed.** The variable is the QUASAR artifact the GPU workflow names — `qwen3_8_27b_nvfp4qat.v3.ninfer`, under the models directory — and `build-test` reports `ninja: no work to do`. |
| `check_release_archive.py` is not `check_archive_contents.py` | **Accepted.** The packager already runs the content check; `check_release_archive.py` extracts the archive and serves it, so it is run after packaging as the runtime proof. |

## Risks

- The suite gate is the long pole: a full ctest run, and it must match `test_baseline.json`.
- The v1.2.3 archive stays published and is not amended. A reader comparing releases sees a 7.8 % decode
  step that is mostly the window; the notes are the only place that can say so.
- `origin/main` currently sits at the v1.2.3 tag commit; a fast-forward to `dev` publishes every commit
  since, including the documentation audit and the tooling change.
