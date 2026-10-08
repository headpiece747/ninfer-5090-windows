# Known failures of the profiling recipe

Each entry is an incident, not a rule. The rules are in `SKILL.md`; these are the times they were read
and did not act, with what the failure looked like from the outside.

## 2026-10-08 — two faults in one nsys profile, and a table that looked complete

**What happened.** An nsys run was set up to attribute the prefill cost curve by comparing an 8K-token
prefill against a 64K one. Both traces were written, both parsed, and the summary printed a full kernel
table with shares. None of it was usable.

**Fault 1, and it is lesson 8 exactly.** The harness launched `nsys` as the child process and then
terminated *that*, intending to stop the serve. That kills the profiler, not the profiled process: no
report was written at the time, and both `ninfer-serve` processes were still running afterwards. The
traces only appeared much later, when the serves were finally stopped and the profilers that had
survived the terminate finished on their own.

**Fault 2.** Both arms used the same port. The first arm's serve survived, so the second arm's request
was answered by the **first** arm's process, and the second trace captured a serve that never received
its request. No error was raised anywhere.

**How the failure looked.** Like success. The tell was arithmetic:

- the 8K arm reported **39,058.9 ms of kernel time** for a request whose traced wall time was 3.4 s;
- the 64K arm reported **fewer** instances of the same kernels than the 8K arm (688 → 192) for 8.4x the
  tokens.

Neither number is possible, and both were printed by a parser that had read real files.

**The recipe that avoids all of it.**

1. One port per arm. A collision is invisible from the client side.
2. Stop the **serve**, and leave `nsys` to finish (or let it stop itself with `--duration`). Stopping
   the profiler discards the trace; stopping the subject lets the profiler finalise.
3. Pass `--force-export true`, or a stale `.sqlite` from an earlier run is reused and the new trace is
   never read.
4. `nsys stats` writes a **box-drawn** table, not whitespace-separated columns: rows start with `|` and
   fields sit between pipes. A whitespace split parses nothing and reports zero kernels, silently.
5. Before reading anything out of the trace, check that the kernel totals are **not larger than the
   request's own wall time**. That single comparison refused every impossible figure above.
