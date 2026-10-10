# The silent shell death, diagnosed (2026-10-10)

Three times in two days a background shell's whole process tree died at once with nothing recorded: two
verification chains on 2026-10-09, and a 220-cycle soak on 2026-10-10 that died at cycle 68 of 220
(11:24:25) with its harness shell, driver, client and lane all gone. The `AGENTS.md` rule written after
the first two named the mechanism as a Windows job close. This note records what a controlled experiment
replaced that with, and the escape it implies. Everything here is measured on this machine on 2026-10-10;
instruments are named so a reader can re-run them.

## What each death left behind

| | chains (10/9) | soak (10/10 11:24) |
|---|---|---|
| driver output | lost (buffered in the dead process) | the harness's `.out` for the shell **does not exist** |
| own log | ended cleanly at a step boundary | frozen mid-turn at `c68 k3`, client blocked in `urlopen` on `req#481` |
| engine log | n/a | frozen mid-prefill, **no CUDA error, no shutdown line** |
| WER dump | none | none |
| Application/System log | nothing | **nothing** (checked 11:20-11:30) |
| harness record | still "running" | still "running" |

## Mechanisms tested, with their predictions

| # | hypothesis | test | verdict |
|---|---|---|---|
| 1 | a job close on a job containing the tree | `IsProcessInJob` + `QueryInformationJobObject` on the live tree | **falsified**: the lane (41508) and the outer driver (32524) are **not in a job**; the middle driver is in one it will not describe (error 6), and a job member's children escape anyway (`SILENT_BREAKAWAY_OK`), so a close kills only the middle process -- never the tree |
| 2 | a console teardown | arm A: post `WM_CLOSE` to a two-level tree's console window | **confirmed**: both levels died in the same second, exit **`0xC000013A`** (STATUS_CONTROL_C_EXIT), **no WER, no event-log entry** -- the exact signature above |
| 3 | an external tree-killer | two detached monitors sampling every 5 s | **untested** (needs a recurrence; now instrumented) |
| 4 | the lane died first | client would have raised; its log ends cleanly mid-turn | **falsified** |
| 5 | a machine-level event | System log 11:20-11:30 | **falsified** (empty) |
| -- | a root/driver death cascading | arm C: terminate the root of a two-level tree | **falsified as a cascade**: the child kept heartbeating |

## The escape, verified rather than assumed

`DETACHED_PROCESS` alone is **not** enough, and the reason is the interpreter: the same sleeper script,
spawned three ways, reports its own `GetConsoleWindow()` --

| launch | console window |
|---|---|
| `vllm-env\Scripts\python.exe` + `DETACHED_PROCESS | CREATE_BREAKAWAY_FROM_JOB` | **non-NULL** -- the venv trampoline allocates a console |
| the uv-managed interpreter directly, same flags | **NULL** |
| `vllm-env\Scripts\python.exe`, no flags | NULL (it inherits this shell, which has none) |

So a long detached run must launch the **managed interpreter directly**, and console-freedom is checked
by observation (`GetConsoleWindow()` in the process, or a `conhost.exe` whose parent is in the tree), not
assumed. The monitors watching the running soak were relaunched that way and confirmed console-free.

## Instrument defects found on the way

- **`OpenProcess` is not a liveness test.** An exited process keeps its pid and stays openable while any
  handle exists, so it answers "up" for a corpse -- which is how arm A first read as
  `alive=True exit=3221225786`. Fixed with `GetExitCodeProcess` (`STILL_ACTIVE` = 259) in the monitors.
- **The console window's owner is the client, not `conhost`,** under these pseudo-consoles, so arm B
  ("kill the console host") aimed at the root again and was dropped rather than re-aimed: here the
  pseudo-console's host is the harness itself, which this diagnosis will not kill.
- **The first monitors had consoles of their own** -- created at the moment they were launched, visible in
  the `conhost.exe` list -- so they were exposed to the teardown they exist to observe. Relaunched
  console-free.

## The trigger, found in the harness's own log

The harness logs every event it handles, and the death second is in it. At **15:24:25Z** (local 11:24:25,
the second the soak's tree died) the service evicted and re-booted the location for this directory:

    15:24:25.015  "location services evicted" directory="C:\AI\ninfer-v3-windows"
    15:24:25.027  "watcher stopped" (a burst of them)
    15:24:25.188  "location services booted" ... durationMs=95
    15:24:25.197  cause="InterruptError: All fibers interrupted without error at ServerProcess.start"

A shell is hosted by that location's server process, so the eviction kills its whole tree at once -- arm A's
signature, produced by the harness itself. The cadence is **roughly hourly per location**: 2026-10-10 shows
12:14, 13:15, 14:16, 15:17, 15:24, 16:18, 17:19, 17:32, 18:20, 18:33, 19:21Z, and **2026-10-09 shows 34**
(00:38, 01:39, 02:40 ... 22:00, 23:01, 23:39Z). That covers the two chain deaths as well: their shells were
hosted the same way, and an hourly reaper ran throughout the window in which they died. Their own death
*times* were never recorded, so that attribution rests on the mechanism being available and firing, not on
a timestamp.

**The escape is confirmed in the field, not only in the lab.** The detached, console-free soak ran straight
through this location's later evictions (17:32Z and 18:33Z) and kept serving, while every hosted shell in
the same window died with its location. `tools/scripts/check_console_free.py` is the verification, and
`GetConsoleWindow()` is the in-process test.

## What remains unknown

**Why the service evicts a location about hourly** -- a policy, a leak, or a bug -- is opencode's business
and not something this diagnosis can settle from outside. It is worth reporting upstream; the log lines
above are the reproduction. The monitors stay in place for a recurrence, though the mechanism is now
understood well enough that a recurrence would add little.
