# Upstream open and deferred items — research, 2026-10-07

One section per item. Every claim carries the command or URL that produced it. `gh` runs used the
authenticated account `headpiece747` (`gh auth status`: "Logged in to github.com account
headpiece747"). The upstream repository is `Neroued/ninfer` (`gh api repos/Neroued/ninfer --jq
'{fork,default_branch,archived,disabled,visibility}'` → `{"fork":false,"default_branch":"master",
"archived":false,"disabled":false,"visibility":"public"}`); `upstream/dev` below means the fetched
ref at `81c8ce09` (`git log --oneline -1 upstream/dev`). Where a source does not answer a question,
this says so rather than inferring.

---

## 1. Issue 5 / upstream #339 — resource subtraction underflow

- **State: OPEN, zero comments, no fix acknowledged.** `gh api repos/Neroued/ninfer/issues/339
  --jq '{number,title,state,created_at,updated_at,comments,labels:[.labels[].name]}'` →
  `{"comments":0,"created_at":"2026-09-30T15:29:50Z","labels":[],"number":339,"state":"open",
  "title":"`serve` wedges with 503 for all requests after `Qwen3.5 resource subtraction underflow`
  (checked_resource_difference, context_work.cpp:222) under `--max-concurrency 2`","updated_at":
  "2026-09-30T15:29:50Z"}`. The thread contains no comment at all; the only events are three
  cross-references: `gh api repos/Neroued/ninfer/issues/339/timeline --jq '[.[] | {event,
  actor:.actor.login, created_at, source:.source.issue.number}]'` →
  `qzshch 2026-10-03 → 362`, `qzshch 2026-10-03 → 345` (both unrelated feature proposals:
  `gh api repos/Neroued/ninfer/issues/345 --jq .title` → "feat(qwen): add per-lane retrieval KV
  windows with media replay"), and `Neroued 2026-10-04 → 366` (the rewrite announcement).
- **The issue body, decisive sentences** (`gh api repos/Neroued/ninfer/issues/339 --jq .body`):
  - "With `--max-concurrency 2`, when two large-context requests are in flight at the same time
    (60k–70k input tokens each, both with `--preserve-thinking`; one incident additionally had a
    vision image request), the engine throws: `openai.APIError: Qwen3.5 resource subtraction
    underflow`".
  - "9 total occurrences that day, all with `--max-concurrency 2` under heavy concurrent load."
  - Control: "`--max-concurrency 1`, continuous all-day usage: **0** occurrences of this exception."
  - "The Windows port did not modify the affected files: `src/models/qwen3_5/program/context_work.cpp`
    and `src/models/qwen3_5/program/planning/pressure.cpp` are byte-identical to upstream".
- **The named site no longer exists in upstream/dev, and no later commit touches that failure mode.**
  - `git log --oneline upstream/dev -S "checked_resource_difference"` → last (newest) commit is
    `b9114396 feat(runtime): replace context cache and add preemptive scheduling`.
  - `git show b9114396 -- src/models/qwen3_5/program/context_work.cpp` shows the deletion:
    `-detail::PhysicalResources checked_resource_difference(detail::PhysicalResources value,` and
    `- throw std::logic_error("Qwen3.5 resource subtraction underflow");`.
  - `git grep -n "checked_resource_difference\|resource subtraction underflow" upstream/dev` →
    exit 1, no matches. The same grep against the port's `HEAD` matches only the port's own docs
    (`docs/research/server-open-items.md`, `docs/research/issue5-deferred-findings.md`).
  - The file itself is 62 lines at the upstream tip (`git show
    upstream/dev:src/models/qwen3_5/program/context_work.cpp | Measure-Object -Line`).
  - `git log --oneline b9114396..upstream/dev -- src/models/qwen3_5/program/context_work.cpp` → one
    commit, `a8e212ac perf(engine): reduce preparation and context management overhead`, whose diff
    for that file contains neither string. So since the issue was filed, upstream's last ~30 commits
    (three constrained-decoding commits plus the bf16 Op family, `git log --oneline -30 upstream/dev`)
    changed that failure mode only by deleting it.
- **What the port's partial re-test covers** (`docs/research/server-open-items.md`, lines 131-137):
  `tools/bench/ttft/cases.py`'s `reporter-concurrency3-growth` ran "the reporter's *settings* —
  three lanes, `--max-concurrency 3`, host state and KV enabled, their pending values — against the
  new cache, and it completed **360/360 requests with no failure and no underflow**
  (`constructed=true`, `admission_fallback_reason: none`, `revoked_checkpoints: 0` on every request;
  98.9% median cache reuse)." The same paragraph states the limits: "It is not the reporter's case:
  the prompts are ~2,000 tokens rather than 60,000-70,000, and it declares no tools and no reasoning
  effort. Those three are what an actual re-test would have to match."
- **Is that enough to comment on the issue? What a comment would add:**
  - It would add the verified fact that `checked_resource_difference` and its throw were removed by
    `b9114396`, so the exact symptom (`Qwen3.5 resource subtraction underflow`) cannot be produced by
    that code path in current upstream — a fact not yet in the thread (0 comments).
  - It would add a re-test under the reporter's *settings* but not their *client shape*, with the
    limits named above.
  - It would **not** establish a fix for the reporter's workload, reproduce their trigger, or cover
    the replacement cache under 60k-70k concurrent prompts. The port's run is on the port's tree, not
    upstream. Nothing in the thread asks for or records a re-test result, so a comment is the only
    mechanism to put these facts there; it is not sufficient to close the issue.

## 2. FP8 TMA kernels and MSVC C2719

- **The seven commits item 12 names are the FP8 TMA tuning series of 2026-09-28** (the fourteen
  commits merged as `d44ab584` minus the seven non-FP8 ones; `git log --oneline -20 d44ab584`).
  For each, `git merge-base --is-ancestor <c> upstream/dev` → true, and the key files it touches
  still exist at `upstream/dev` (verified with `git cat-file -e upstream/dev:<path>`; the only
  `Revert` in `upstream/dev` is `915cd38d`, a q4 gemv revert, `git log --oneline upstream/dev
  --grep=Revert`):

  | commit | subject | what it changes | at upstream tip |
  |---|---|---|---|
  | `344d69b8` | perf(linear): tune fp8 14336x5120 for mxfp8 | `n14336_k5120.cu` ladder + `linear.h` + its test | file exists; ladder in place |
  | `7489500d` | perf(linear): tune fp8 16384x5120 decode and tma schedules | `n16384_k5120.cu` ladder | file exists |
  | `b3f018ab` | perf(linear): tune fp8 5120x6144 tma schedules | `n5120_k6144.cu` ladder | file exists |
  | `7ede9b44` | perf(linear): tune fp8 5120x17408 tma and split-k routes | `n5120_k17408.cu` ladder | file exists |
  | `40bfe7dc` | perf(ops): tune fp8 fused projections with tma split-k | `fp8_a8_tma_mma.cuh` (+108 lines), `fp8_a8_mma_common.cuh`, attn/gdn input proj, linear_add, linear_swiglu | file exists; later modified only by this commit |
  | `909fb087` | perf(linear): optimize fp8 34816x5120 with tma split-k | **created `src/ops/linear/fp8/fp8_a8_tma_mma.cuh`** and introduced `struct alignas(128) Fp8TmaDescriptors` (`git log --oneline --diff-filter=A upstream/dev -- src/ops/linear/fp8/fp8_a8_tma_mma.cuh`; `git log --oneline -S "struct alignas(128)" upstream/dev -- <same path>` → only this commit) | file exists |
  | `7f6aafed` | perf(attention): optimize fp8 and k8v4 prefill with split-kv | adds `mxfp8_tiled_mma.cuh`, `mxfp8_tiled_launch.cuh`, `mxfp8_tiled_plan.h`, `causal_tiled_merge.cuh`; removes the old `tiled_mma.cuh` | files exist |

- **All three descriptor structs upstream still carry `alignas(128)` today**:
  `git grep -n "struct alignas(128)" upstream/dev -- src/ops` →
  `src/ops/linear/bf16/bf16_a16_tma_mma.cuh:14`, `src/ops/linear/fp8/fp8_a8_tma_mma.cuh:19`,
  `src/ops/linear/nvfp4/nvfp4_a4_tma.cuh:24`. `fp8_a8_tma_mma.cuh` has not been touched since
  `40bfe7dc` (`git log --oneline upstream/dev -30 -- src/ops/linear/fp8/fp8_a8_tma_mma.cuh` → two
  commits).
- **Why MSVC rejects it (C2719)** — first-party:
  - Microsoft, [Compiler Error C2719](https://learn.microsoft.com/en-us/cpp/error-messages/compiler-errors-2/compiler-error-c2719):
    "The `align` `__declspec` modifier is not permitted on function parameters. **Function parameter
    alignment is controlled by the calling convention used.**"
  - Microsoft, [x64 Calling Convention](https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention),
    §Alignment: "Most structures are aligned to their natural alignment. … **Alignment above 16
    bytes must be done manually.**" §Parameter passing: structs "of other sizes are passed as a
    pointer to memory allocated by the caller. For these aggregate types passed as a pointer …
    the caller-allocated temporary memory must be 16-byte aligned."
  - **Documented acceptance ceiling: 64.** NVIDIA's own header lowers the value under MSVC —
    read locally in CUDA 13.3 `include/cuda.h` (the `CUtensorMap_st` block):
    `#if defined(_MSC_VER) #define TENSOR_MAP_ALIGN 64 #else #define TENSOR_MAP_ALIGN 128 #endif`,
    applied only inside `#if defined(__cplusplus) && (__cplusplus >= 201103L)`. The port's probe
    (`tools/scripts/probe_tma_align.cmd`, present in the tree; its numbers are quoted in
    `src/ops/linear/bf16/bf16_a16_tma_mma.cuh:24-28`) measures the same ceiling on this toolchain:
    "a by-value `__grid_constant__` descriptor block accepted at 8/16/32/64 and rejected at 128 and
    256 with four C2719 sites each".
  - **NVIDIA documents the recommended pattern**, and it is by-value `__grid_constant__`:
    CUDA Programming Guide 4.12.2, [Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html):
    "**The recommended approach is to pass the tensor map as a const `__grid_constant__` parameter
    to a kernel.** The other possibilities are copying the tensor map into device `__constant__`
    memory using `cudaMemcpyToSymbol` or accessing it via global memory. When passing the tensor map
    as a parameter, some versions of the GCC C++ compiler issue the warning 'the ABI for passing
    parameters with 64-byte alignment has changed in GCC 4.6'. This warning can be ignored." The
    same page states the transfer mechanism: "transferred from host to device as a `const` kernel
    parameter annotated with `__grid_constant__`".
- **Is the port's `alignas(64)` workaround consistent with that guidance? Yes.**
  - The recommendation is to remove the over-128 override and keep the by-value `__grid_constant__`
    parameter; 64 is within the measured acceptance ceiling and is what NVIDIA's header uses under
    `_MSC_VER`.
  - Port `src/ops/linear/fp8/fp8_a8_tma_mma.cuh:26` — `struct alignas(64) Fp8TmaDescriptors`, with a
    `static_assert` and the C2719 reason in the comment above it.
  - Port `src/ops/linear/bf16/bf16_a16_tma_mma.cuh:43` — `struct Bf16TmaDescriptors` with **no**
    override (comment at `:14-42`: the `CUtensorMap` attribute is not applied at all under this build
    because nvcc's MSVC host pass reports `__cplusplus = 199711L`), and the encode destination at
    `:56` is `alignas(64) CUtensorMap result{}`.
  - Port `src/ops/linear/nvfp4/nvfp4_a4_tma.cuh:30` — `struct Nvfp4A4TmaDescriptors` with no
    override, same reason.
  - The port's ratchet records these as Windows necessities with the measurement:
    `tools/release/port_delta_baseline.json`, `paths."src/ops/linear/bf16/bf16_a16_tma_mma.cuh"`
    ("CUtensorMap carries no alignment attribute under this build … so the by-value descriptor needs
    an explicit `alignas(64)`, and 128 is rejected with C2719"), and the fp8/nvfp4 entries as
    `"disposition": "windows"`.
  - The port's own mechanism research is `docs/research/grid-constant-tma-descriptor-msvc.md`
    (MSVC ABI, the 64 ceiling, the CUDA guide quote, CUTLASS's alias-without-override).
- **Has upstream added more commits in this family since? Yes — the bf16 family.**
  - `git log --oneline -S "struct alignas(128)" upstream/dev -- src/ops/linear/bf16/bf16_a16_tma_mma.cuh`
    → `ecbc3357 perf(ops): expand bf16 linear templates and unify epilogues` (2026-09-26) introduced
    `struct alignas(128) Bf16TmaDescriptors`; it is still at `upstream/dev` line 14.
  - The port's ten-commit merge is exactly `7971ff18^..81c8ce09` (`git log --oneline
    7971ff18^..81c8ce09` → ten commits: three constrained-decoding commits and **seven** "support and
    tune bf16 …" commits; `docs/active-work.md` item 24 calls it eight). Two of them touch the bf16
    TMA file — `a64b5eea` and `c02a07b4` (`git show --stat c02a07b4` adds a `#pragma unroll
    Schedule::kConsumerKUnroll`) — and upstream's kernel still takes
    `const __grid_constant__ Bf16TmaDescriptors descriptors` by value at line 93.
  - The port's bf16 file keeps no override and the by-value parameter
    (`src/ops/linear/bf16/bf16_a16_tma_mma.cuh:125`), which is the item-24 merge decision recorded in
    `docs/active-work.md:2688-2691`.

## 3. Issue 5's open question for the reporter

- **The question as the port recorded it** (`docs/research/issue5-deferred-findings.md:74-81`):
  "Their table tests 1 (never) and 3 (fails); the boundary between them is untested and upstream
  documents **2**. **Does `repro_dsh.py` underflow at `--max-concurrency 2`?**"
- **That premise does not match issue #339 as fetched, and the question is answered by the report
  itself.** The issue title, the launch flags, the timeline and the control all say **2**, not 3
  (`gh api repos/Neroued/ninfer/issues/339 --jq .body`, quoted in item 1 above): "With
  `--max-concurrency 2` … 9 total occurrences that day, all with `--max-concurrency 2`". Searched
  over the whole tracker, no underflow report at concurrency 3 exists: `gh issue list --repo
  Neroued/ninfer --state all --search "resource subtraction underflow"` → only #339, and `gh api
  "search/issues?q=%22max-concurrency+3%22+repo:Neroued/ninfer"` returns 24 hits that are not about
  the underflow (#229, #366, #270, #137 among them). So the
  reporter already ran 2 and it underflowed there; the untested side of the boundary is below 2, and
  1 is the only value below 2.
- **Has the port's question been asked or answered anywhere in the tracker? No.**
  `gh api "search/issues?q=repro_dsh+repo:Neroued/ninfer"` → `{"total_count":0,"items":[]}`;
  #339 has 0 comments (`gh api repos/Neroued/ninfer/issues/339 --jq .comments` → `0`). The port has
  not posted it (its own records say the tracker is not written to without the owner's say-so).
- **Upstream's documented default and range, and its examples:**
  - `git show upstream/dev:src/serve/serve_options.cpp` lines 362-363: `if
    (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) { throw
    std::invalid_argument("--max-concurrency must be in [1,8]"); }`.
  - `git show upstream/dev:docs/serving.md` line 884: "| `--max-concurrency N` | resident execution
    lanes; valid range `1..8` | `1` |".
  - Its own examples use **2**: `git show upstream/dev:README.md` → lines 71 and 212 both
    `--max-concurrency 2 \`; `git show upstream/dev:docs/serving.md` → line 16
    `--max-concurrency 2 \`.
  - The port's launchers all pass 1: `git grep -n "max-concurrency" HEAD -- '*.bat'` → every profile
    file, `--max-concurrency 1`.
- **Consequence for the recorded question:** it is moot as written (the value it asks about is the
  value the report is about). What remains genuinely untested is the reporter's workload against the
  replacement cache — which is the re-test #366 asks for, not a new question to the reporter.

## 4. Why the scheduled GitHub Actions workflow does not fire

**This measurement is now overtaken: one scheduled run exists, created ~6 h 56 min after its cron
slot.**

- `gh api "repos/headpiece747/ninfer-5090-windows/actions/runs?event=schedule&per_page=100"` →
  `total_count: 1`, run `37610654570`, `event":"schedule"`, `created_at":"2026-10-07T10:55:59Z"`,
  `conclusion":"success"`. `gh api
  repos/headpiece747/ninfer-5090-windows/actions/runs/37610654570 --jq '{name,event,path,
  head_branch,head_sha,run_started_at,updated_at,workflow_id}'` → `gpu-suite`, `.github/workflows/
  gpu.yml`, branch `dev`, head `36553aec`, started 10:55:59Z, completed 11:06:44Z; the single job ran
  on `5090-box` (`gh api .../runs/37610654570/jobs` → runner_name `5090-box`).
- The workflow's cron is `0 4 * * *` (`git show origin/dev:.github/workflows/gpu.yml`), so the run
  started **6 h 55 min 59 s after its scheduled time**. It ran on `36553aec`, the commit that
  *documents* the failure (`git show -s --format='%h %ad %s' 36553aec` → 2026-10-07 06:04:19 -0400,
  "docs(ci): the scheduled path does not fire, and that is measured rather than assumed").
- The earlier measurement in `docs/active-work.md` item 23 ("no scheduled run has ever been created",
  "the schedule missed four five-minute windows") was true when taken and is now superseded: the probe
  workflow itself never fired (`gh api "repos/.../actions/runs?per_page=100"` filtered to
  `schedule-probe` → only a push run at 09:39:42Z and a `workflow_dispatch` at 10:01:35Z), and the
  probe file is no longer on the default branch (`git ls-tree -r --name-only origin/dev --
  .github/workflows` → `ci.yml`, `gpu.yml` only).
- **Repository facts, re-checked:** `gh api repos/headpiece747/ninfer-5090-windows --jq
  '{fork,private,default_branch,pushed_at,owner_type:.owner.type}'` → `{"fork":false,"private":false,
  "default_branch":"dev","pushed_at":"2026-10-07T12:43:20Z","owner_type":"User"}`;
  `gh api repos/headpiece747/ninfer-5090-windows/actions/permissions` → `{"enabled":true,
  "allowed_actions":"all","sha_pinning_required":false}`; `gh api
  repos/headpiece747/ninfer-5090-windows/actions/workflows` → `gpu-suite`, state `active`.
- **Documented causes, verbatim:**
  - [Events that trigger workflows](https://docs.github.com/en/actions/reference/events-that-trigger-workflows#schedule):
    "The `schedule` event can be delayed during periods of high loads of GitHub Actions workflow
    runs. High load times include the start of every hour. **If the load is sufficiently high enough,
    some queued jobs may be dropped.**"; "This event will only trigger a workflow run if the
    workflow file exists on the default branch."; "Scheduled workflows will only run on the default
    branch."; "In a public repository, scheduled workflows are automatically disabled when no
    repository activity has occurred in 60 days."
  - [Disabling and enabling a workflow](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/disable-and-enable-workflows):
    "To prevent unnecessary workflow runs, scheduled workflows may be disabled automatically. When a
    public repository is forked, scheduled workflows are disabled by default. In a public repository,
    scheduled workflows are automatically disabled when no repository activity has occurred in
    60 days."
  - **No repository or account setting that disables schedules exists in the documented API
    surface.** The [REST permissions page](https://docs.github.com/en/rest/actions/permissions)
    enumerates every `/repos/{owner}/{repo}/actions/permissions*` endpoint: permissions (`enabled`,
    `allowed_actions`, `sha_pinning_required`), access (private repos), artifact-and-log-retention,
    fork-pr-contributor-approval, fork-pr-workflows-private-repos, selected-actions,
    self-hosted-runners, workflow (token permissions). None is schedule-specific. The org-level
    equivalent (`GET /orgs/{org}/actions/permissions`) governs `enabled_repositories` /
    `allowed_actions` / `sha_pinning_required` and does not apply here anyway — the owner is a
    `User`, not an organization (`gh api repos/... --jq .owner.type` → `User`).
- **Which documented cause fits, and which do not:**

  | documented cause | fits? | evidence |
  |---|---|---|
  | delay / dropped jobs under load | **yes** | a run appeared 6 h 56 min late; GitHub's own sentence above says runs can be delayed and dropped |
  | workflow file not on default branch | no | `git ls-tree origin/dev -- .github/workflows` shows it; the run fired from `dev` |
  | workflow deactivated | no | `gh api .../actions/workflows` → `state: active` |
  | fork default-disabled | no | `fork: false` |
  | public-repo 60-day inactivity | no | `pushed_at` 2026-10-07, same day |
  | a repository/account setting | no | no such setting in the documented permissions surface; `enabled: true`, `allowed_actions: all` |

- **A 2026 GitHub-side change is reported by users, not confirmed by GitHub.** There is one
  scheduled-workflow *feature* change in the changelog — [GitHub Actions: Late March 2026
  updates](https://github.blog/changelog/2026-03-19-github-actions-late-march-2026-updates/) added
  "Timezone support for scheduled workflows" — which does not explain a non-firing schedule. The
  mechanism-level evidence is GitHub's own postmortem of the 2026-08-26 Actions incident
  (`Invoke-RestMethod https://www.githubstatus.com/api/v2/incidents.json`, "Incident with Actions",
  2026-08-26): "This impact was triggered by **saturation of writes to the database primary used by
  the service processing triggers for Actions workflows**." From that date, community reports
  describe exactly this repository's symptom: [community discussion
  205984](https://github.com/orgs/community/discussions/205984) — "zero schedule-triggered runs exist
  anywhere on this account, on any repo … manual dispatch always works"; [community discussion
  206019](https://github.com/orgs/community/discussions/206019) — a daily-cron table showing runs
  created 39 min, 3 h 27 min and 10 h 41 min late and then "no run created", on an active public repo
  whose owner had ruled out the 60-day rule, and multiple confirmations of 3.5-10.7 h delays
  (Aug 27 – Oct 2, 2026). The status API shows Actions incidents on 2026-10-05 (critical, "14.3% of
  workflow runs … did not start within five minutes") and 2026-10-06 (several services), and
  `Invoke-RestMethod https://www.githubstatus.com/api/v2/status.json` today reads "All Systems
  Operational".
- **Plainly:** the documented cause that fits is the delay/drop behaviour, amplified by a
  GitHub-side trigger-service degradation reported since 2026-08-26; the other documented causes are
  excluded by measurement. The port's "the schedule does not fire" is now correctly "it fired once,
  ~7 h late"; whether the next slot fires on time is not established here.

## 5. Keeping the self-hosted runner alive without an administrator

- **GitHub's own documentation:**
  - [Adding self-hosted runners](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners):
    "On Windows, if you want to install the self-hosted runner application as a service, **you must
    open a shell with administrator privileges**. We also recommend that you use `C:\actions-runner`
    as the directory for the self-hosted runner application".
  - [Configuring the self-hosted runner application as a service](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/configure-the-application):
    "Configuring the self-hosted runner application as a service on Windows is part of the
    application configuration process. If you have already configured the self-hosted runner
    application but did not choose to configure it as a service, you must remove the runner from
    GitHub and re-configure the application."
  - The runner directory on this machine has no `svc.cmd` (`Get-ChildItem C:\AI\actions-runner
    -Force` → `config.cmd`, `run.cmd`, `run-helper.cmd`, `start-if-idle.cmd`; no `svc.cmd`), which is
    consistent with the Windows docs: the service form is `config.cmd`'s service option, run from an
    elevated shell, not a separate `svc.cmd install`.
- **Microsoft, first-party:** "To install a Windows service, **you must have administrator
  credentials on the computer where it's installed**" ([Tutorial: Create a Windows service app,
  .NET Framework](https://learn.microsoft.com/en-us/dotnet/framework/windows-services/walkthrough-creating-a-windows-service-application-in-the-component-designer)).
- **What this machine's token is.** `whoami /groups` → `Mandatory Label\Medium Mandatory Level
  S-1-16-8192`, `BUILTIN\Administrators S-1-5-32-544 … Group used for deny only`, `NT
  AUTHORITY\Local account and member of Administrators group … Group used for deny only`. So the
  shell is a **non-elevated administrator** (the Administrators SID is present but filtered), not a
  standard user.
- **Measured, non-elevated (all four probes run from the current shell; the two that created tasks
  were deleted in the same command, and both deletions and the post-checks succeeded):**
  - `schtasks /create /sc onlogon /tn oc-research-probe-20261007 /tr "cmd.exe /c exit"` →
    `ERROR: Access is denied.` exit 1; `schtasks /query` → "The system cannot find the file
    specified", i.e. nothing was created.
  - `schtasks /create /sc onlogon /tn oc-research-probe4-20261007 /tr "cmd.exe /c exit"
    /ru mancave\tobia /it` → `ERROR: Access is denied.` exit 1; nothing created.
  - **Control — `schtasks` itself is not the blocker:** `schtasks /create /sc daily /st 23:59
    /tn oc-research-probe5-20261007 /tr "cmd.exe /c exit"` → `SUCCESS: The scheduled task
    "oc-research-probe5-20261007" has successfully been created.` exit 0; queried (`Ready`), then
    deleted (`SUCCESS: … successfully deleted`).
  - **A per-user logon task is creatable without elevation via the ScheduledTasks module:**
    `Register-ScheduledTask -TaskName oc-research-probe3-20261007 -Trigger (New-ScheduledTaskTrigger
    -AtLogOn -User $env:USERNAME) -Action (New-ScheduledTaskAction -Execute cmd.exe -Argument
    '/c exit')` → **CREATED**; the registered task reported `Principal.UserId: tobia`,
    `LogonType: Interactive`, `RunLevel: Limited`, and the trigger's `UserId: MANCAVE\tobia`; then
    `Unregister-ScheduledTask` → DELETED (post-check found nothing).
- **What exactly is denied:** not task creation and not the per-user logon trigger — it is
  `schtasks`' **ONLOGON** registration. Microsoft's [schtasks
  reference](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/schtasks-create)
  defines `ONLOGON` as "Specifies that the task runs whenever a user (**any user**) logs on", and
  its own Important note says "**Only Administrators can schedule tasks, regardless of the value of
  the `/ru` parameter.**" A per-user `-AtLogOn -User <self>` trigger is a different registration and
  the ScheduledTasks cmdlet performed it as a medium-integrity token.
- **Recommended alternatives where the user has no administrator rights:**
  - **A per-user logon task via `Register-ScheduledTask`** (measured above). It starts the runner
    when *that user* logs on; it does not cover a reboot with nobody logged in.
  - **A Startup-folder shortcut** — Microsoft's own consumer guidance: "Right-click on Start and
    select Run. In the Run dialog box, type either `shell:startup` or `shell:common startup` and
    select Enter" ([Configure Startup applications in
    Windows](https://support.microsoft.com/en-us/windows/experience/startup-boot/configure-startup-applications-in-windows)).
    Same coverage limitation; no delay/idempotency controls.
  - **The runner service remains the only reboot-with-nobody-logged-in option and needs elevation**
    (GitHub's sentence above).
- **Caveats, stated rather than smoothed over:** the successful `Register-ScheduledTask` ran under a
  **non-elevated administrator** token; whether a true standard-user account can register the same
  per-user task is **not established** here (only one account exists on this machine). And this
  falsifies the port's recorded statement that "a scheduled task needs elevation on this machine —
  `schtasks /create` answered 'Access is denied' for a per-user logon task"
  (`docs/active-work.md` item 23; the same claim is in the comment of
  `C:\AI\actions-runner\start-if-idle.cmd`): it is true of the `schtasks` ONLOGON form and false of
  the ScheduledTasks cmdlet.

## 6. Upstream #251 — no LRU eviction observed

- **State: OPEN, 2 comments, last activity 2026-09-17.**
  `gh api repos/Neroued/ninfer/issues/251 --jq '{number,title,state,created_at,updated_at,comments}'`
  → `{"number":251,"state":"open","created_at":"2026-09-15T15:19:35Z","updated_at":
  "2026-09-17T17:42:05Z","comments":2}`. The two comments (`gh api
  repos/Neroued/ninfer/issues/251/comments`):
  - `giveen` cross-references #270: "the failure shape (fills once, no eviction, no log signal,
    permanent for that owner tier) looks like the same mechanism."
  - `albertov` posts the third-party LRU patch for the **shared** catalogue, whose diagnosis is
    quoted: "A candidate carries pressure standing only with explicit or requested evidence, or when
    two reuse domains already demand its key; otherwise it can only claim a vacant slot."
    (`docs/research/prefix-state-eviction.md` records that this port ported that patch, then lost it
    with the rewrite.)
- **No post-rewrite report of the same symptom exists in #251's thread**, and the closest #366
  reports are different mechanisms:
  - `CaptainArni` in #366 (`gh api repos/Neroued/ninfer/issues/366/comments`): "A shared system
    prompt is never reused between different requests … Two requests with the same system prompt and
    different user messages get `cached_tokens: 0` on the second one", reproduced on the official
    artifact (`abb7f14f`). **That symptom is measured fixed on a later commit:** `qzshch`'s comment
    tables `abb7f14f` vs `68c54356` and shows marked request B at **3,761 cached tokens** on
    `68c54356` (vs 0 on `abb7f14f`), with unmarked C/D also reusing. So the shared-prefix report is
    not the port's cliff, and on the port's own merge tip it does not reproduce.
  - `kaushikvira` in #366: "in the fits regime, 5/20 continuation turns selected root (0 cached)
    despite ~30 GiB host headroom … `request_done` no longer exposes a selection reason". The
    maintainer's dev agent asked for the probe and JSONL on 2026-10-05; no follow-up appears in the
    thread (last comment 2026-10-06). **Whether that is the port's value-gate mechanism is not
    established** — no data was provided.
  - `#371` (2026-10-05) is a different mechanism: "A cancelled follow-up discards the consumed
    continuation; a different follow-up (or an edited last message) re-prefills the whole
    conversation" (`gh api repos/Neroued/ninfer/issues/371 --jq .body`), attributed to `abort()`
    clearing the slot and to turn-closure replacement.
  - `#177` ("Private continuation cache can become permanently saturated across independent
    conversations", open, one patch comment by `splickz`) and `#270` are the pre-rewrite versions of
    this symptom; `#269` was closed after its reporter traced the drop to his own SSE proxy, not the
    engine (`gh api repos/Neroued/ninfer/issues/269/comments`).
- **What the port's 2026-10-06 re-measurement would add** (`docs/research/prefix-state-eviction.md`,
  "RE-MEASURED 2026-10-06" through "MEASURED 2026-10-06: both options"):
  - A post-rewrite reproduction of the issue's exact shape: `repro_251.py`, six conversations asked
    twice, three pool configurations; `--device-state-slots 1 --host-context-mib 0` → "95.9%, 0.0% x5
    — reproduced at conversation 2"; `--host-context-mib 300` → reproduced at 3;
    `--host-context-mib 8192` → not reproduced ("capacity, not reclamation").
  - The log signature that distinguishes it from a lost preferred source: post-cliff requests read
    `generation.admission` `preferred_reused_tokens: 0` with `fallback_reason: "none"` and
    `prefix_reuse_path: root`.
  - The located mechanism, which is a **policy gate, not a missing path**: the replacement's
    `admits` rejects a victim with `victim.reused && (!admission->priority.reused ||
    victim.last_demand >= admission->priority.last_demand)` — verified at
    `src/runtime/engine/context_cache/resource_manager.h:1256-1258` in the port's working tree (the
    note cites `:1282-1285`; the upstream tip has the same gate at lines 1259-1261). The port's
    pre-rewrite `d05ee90a` reclaimed the oldest unpinned continuation regardless, so the measured
    cliff is the consequence of a policy difference between the two caches.
  - The workload shapes that bound it: sequential 12 conversations of ~20,000 tokens asked twice
    back-to-back reuse **99.8% for all twelve** at the shipped QUASAR DFlash2 lane; interleaved
    round-robin reuses for 52 conversations and fails for 56 (host quota ≈ 8 GiB / 147 MiB ≈ 55
    state images), and that second failure is state lifetime, not the value gate.
  - Caveat to state with it: the measurement is on the port's tree (the merge of the replacement),
    not on `Neroued/ninfer`'s own build.

## 7. Upstream #229 — the materialization search grant pinned at 5 ms

- **State: OPEN, 2 comments, last activity 2026-09-30.**
  `gh api repos/Neroued/ninfer/issues/229 --jq '{number,title,state,created_at,updated_at,comments}'`
  → `{"number":229,"state":"open","created_at":"2026-09-10T17:54:01Z","updated_at":
  "2026-09-30T12:58:47Z","comments":2}`. The comments:
  - `Wallawalla47` (2026-09-21): "I implemented this longer 250ms window (or 50ms when queued
    requests are waiting) … it was necessary to also make sure to change the separate code which
    imposed an overall cap on the search window as well as the specific search budget."
  - `dnnspaul` (2026-09-30): "We are deploying the budget widening locally (250 ms idle / 50 ms busy
    allowance, grant cap raised accordingly) on top of current `master` and **will report the
    before/after counters here**." No such report appears; the issue's `updated_at` is that comment.
- **The earlier, closed duplicate:** `#176` "Materialization search budget is capped at 5 ms …"
  (`gh api repos/Neroued/ninfer/issues/176 --jq '{state,state_reason,closed_at}'` →
  `{"state":"closed","state_reason":"completed","closed_at":"2026-09-10T17:43:14Z"}`), closed by
  `d4929686`, whose author comment says: "It starts with a short budget and can extend it when the
  expected remaining benefit justifies the estimated completion cost."
- **Has anything landed upstream since? Yes, and it removed the mechanism.**
  - `b9114396` deleted both files: `git log --oneline upstream/dev -- src/runtime/engine/
    context_cache/materialization_budget.h` and `…/materialization_planner.h` → `b9114396` and
    `04350ba9`; `git ls-tree -r --name-only upstream/dev -- src/runtime/engine/context_cache` → six
    files, neither budget/planner among them.
  - `git grep -n "time_budget\|budget_ns\|search_budget\|search_grant\|grant_ns" upstream/dev --
    src/runtime src/models` → no match in code; the only `search_budget_ns` occurrences are in
    `eval/corpora/perplexity-1m/data/ninfer/02.txt` (a corpus text quoting the old code).
  - **The rewritten cache has no wall-clock search budget.** What remains is a **value** budget:
    `struct ReclaimCursor` with `std::optional<std::uint64_t> gain_limit; std::uint64_t sacrificed = 0;`
    (`upstream/dev:src/runtime/engine/context_cache/resource_manager.h` lines 63-64), set from
    `recovery_gain` in `begin_reclaim` (line 486) and enforced at the admission gate ("This outer
    budget also covers Host victims of a preservation transfer", line 1261). The old
    `plan_reclaim` name still exists, but it is the Program-side demotion/release planner
    (`src/models/qwen3_5/program/transactions/reclaim.cpp`), not a time-bounded search.
- **What the port's evidence would add to #229:**
  - The failure was **deterministic policy, not timing**: `950c87cb` ("test(runtime): let the planner
    take a caller-supplied time source") had already left the 5 ms outcome unchanged, so the failing
    test (`test_candidate_search_prefers_deep_reuse_without_eviction`) was not machine-dependent
    (`docs/research/prefix-state-eviction.md`, "The search grant, and the last failing test").
  - A measured fix at the grant ceiling: `2fcffaa9 fix(runtime): scale the materialization search
    grant instead of pinning it at 5 ms` (port commit, touches `materialization_budget.h` and its
    test), reverted by `7ae7bb40` with no recorded reason, then re-landed (`git show --stat
    2fcffaa9 7ae7bb40`; both are on `dev`).
  - And the fact that **both the fix and its file are now gone from this tree too**:
    `git cat-file -e HEAD:src/runtime/engine/context_cache/materialization_budget.h` → "does not
    exist"; `git grep -n kMaximumGrantNs HEAD` matches only `docs/`. So a comment on #229 would be a
    historical confirmation plus the statement that the mechanism the issue names no longer exists
    in the rewritten cache, and the symptom needs re-testing there (the same re-test #366 asks for).

## 8. Where the three unposted write-ups belong

- **(a) The constrained-tool guard (`docs/research/upstream-constrained-tools-guard.md`).**
  - **No existing issue covers it.** Searches run today, all returning zero or nothing on the
    defect: `gh api "search/issues?q=include_model_defaults+repo:Neroued/ninfer+in:comments"` →
    `total_count 0`; `gh api "search/issues?q=%22constraints+require+default+EOS%22+repo:Neroued/
    ninfer+in:comments"` → 0; `gh api "search/issues?q=%22one+output+language%22+…+in:comments"` →
    0; `gh api "search/issues?q=InvalidToolConstraint+repo:Neroued/ninfer"` → 0;
    `gh api "search/issues?q=include_model_defaults+repo:Neroued/ninfer"` (bodies too) → only #197,
    about `ignore_eos`; `gh api "search/issues?q=%22constrained+tool%22+repo:Neroued/ninfer"` → 5
    hits (#168, #294, #45, #366, #223), none about the guard. This matches the port's 2026-10-06
    search recorded in the write-up.
  - **The guard is still live at the upstream tip**, and was only re-ordered by the later
    constrained-decoding commit: `git show upstream/dev:src/models/qwen3_5/frontend/frontend.cpp`
    lines 928-930 still read `if (impl_->defaults.token_ids.empty() || !caller_stop.token_ids.empty()
    || !caller_stop.strings.empty() || !caller_stop.include_model_defaults || …) throw
    RequestError(tool_constraint ? RequestErrorKind::InvalidToolConstraint : …,
    "constraints require default EOS, text output, no custom stops, and one output language");`;
    `git show 2734a56e -- src/models/qwen3_5/frontend/frontend.cpp` shows the change is only
    `-!caller_stop.include_model_defaults || caller_stop.publish_stop_token` →
    `+!caller_stop.strings.empty() || !caller_stop.include_model_defaults ||`. The test file is still
    untouched since the guard landed: `git log --oneline 41e50d0d..upstream/dev --
    tests/models/qwen3_5/test_engine_prefix_real.cpp` → no commits, and the file still has 19
    `include_model_defaults` sites; the six scenarios and the control were re-verified against
    `upstream/dev` (tools at lines 536/813/816/1085/1191/1341; `include_model_defaults = false` at
    525, 620, 703, 1094, 1199, 1350; `exercise_shared_rewrite_materialization` declares tools and
    never disables model defaults).
  - **Venue: a new issue**, because no issue or comment covers it. A title in the port's own framing
    would be: "Constrained tool calling rejects `stop.include_model_defaults = false`, which six of
    upstream's own prefix scenarios set". The alternative venue is a comment on **#33** — the
    constrained-decoding umbrella, still open, whose last comment (2026-10-07, Neroued) says "finally,
    all features are supported now" (`gh api repos/Neroued/ninfer/issues/33/comments`) — but that
    thread is about unsupported fields, and the guard is a defect in the just-landed feature.
- **(b) The #251 re-test → comment on issue #251** (open; the venue the port's re-measurement is
  about). #251 is also named in #366's "Related reports" list, and #366's own text asks reporters to
  "update and try the cases that gave you trouble before" — so a post-rewrite result is the form of
  comment the maintainer asked for.
- **(c) The #229 evidence → comment on issue #229** (open; the 5 ms grant is its subject). #176 is
  closed as completed and is the wrong venue. The comment's substance is in item 7 above: deterministic
  policy failure, the 250 ms fix and its revert, and the fact that `materialization_budget.h` no
  longer exists upstream or in the port.
- **Related, but a different set:** `docs/upstream-reports/README.md` says it holds "Three issues
  found while working on this port … **None has been posted**" — the TMA-descriptor graph-capture
  report (aimed at upstream PRs #233/#82), the `codegraph` junction report, and the
  `reasoning_effort` mismatch. Those are not the three items above and are still unposted too.

## Sources and method

- `gh` commands run (each quoted beside its claim): `gh api repos/Neroued/ninfer/issues/{33,176,177,
  229,251,269,334,335,338,339,345,360,362,366,371,376}` and their `/comments` and `/timeline`;
  `gh issue list --repo Neroued/ninfer --state all --search <terms>`; `gh api
  "search/issues?q=<query>+repo:Neroued/ninfer[+in:comments]"`; `gh api repos/Neroued/ninfer`; and
  for the port's own repository `gh api repos/headpiece747/ninfer-5090-windows` plus
  `/actions/permissions`, `/actions/workflows`, `/actions/runs`, `/actions/runs/37610654570` and
  its `/jobs`.
- Git evidence is from `upstream/dev` at `81c8ce09` and the port's `HEAD`/`origin/dev`, named per
  claim (`git grep`, `git log -S`, `git show`, `git merge-base --is-ancestor`, `git cat-file -e`).
- Primary web sources: docs.github.com (events-that-trigger-workflows, disable-and-enable-workflows,
  rest/actions/permissions, add-runners, configure-the-application), github.blog changelog
  (2026-03-19), githubstatus.com API (incidents and status), learn.microsoft.com (schtasks-create,
  compiler-error-c2719, x64-calling-convention, the Windows-service walkthrough), support.microsoft.com
  (Startup applications), docs.nvidia.com (CUDA Programming Guide 4.12.2), the actions/runner tree
  (`gh api repos/actions/runner/git/trees/main?recursive=1`), and the GitHub Community discussions
  205984/206019 (user reports, labelled as such).
- **Not established:** whether the port's partial re-test would reproduce the reporter's #339 trigger
  at their client shape (it was not run at 60k-70k tokens with tools/reasoning); whether
  `kaushikvira`'s #366 root selections are the port's value-gate mechanism (no logs were provided);
  whether `Register-ScheduledTask` works for a true standard-user account (only a non-elevated
  administrator token exists here); why exactly the 2026-10-07 10:55:59Z run fired when it did
  (GitHub published no statement; only the documented delay/drop sentence and the community reports
  of hours-long delays since 2026-08-26); and whether upstream regards the constrained-tool guard as
  a defect (unreported).
