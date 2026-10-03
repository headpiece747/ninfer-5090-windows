#!/usr/bin/env python3
"""Report what one serving-lane run actually did, from the engine's own request log.

Built to answer two questions that a wall clock cannot:

  1. Does the prefix cache actually ENGAGE on this workload? `prefix_cache_hit_tokens` says so
     directly, and without a nonzero hit count there is no cache effect to explain a divergence.
  2. Where did the end-to-end time go? ADR-0012 projects `prepared` ~8 ms and warm TTFT ~22 ms
     with the native renderer and states the end-to-end path was never timed with it. These are
     the fields to time it with.

Field names are taken from `format_request_done_json` in src/serve/request_log.cpp, read rather
than assumed: `result.{prefix_cache_hit_tokens,prefix_reuse_path,prompt_tokens,completion_tokens}`
and `timings_seconds.{prepare,ttft,vision,prefill,decode,total}`, with the event name from
`event_base`.

Every figure printed here is MEASURED. The ADR figures it prints beside them are PROJECTED, and
the script says so on every run, because a projection left standing beside a measurement is
exactly how a projection becomes a fact unearned.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

# ADR-0012's own words, carried here so the comparison is explicit rather than remembered.
PROJECTED_PREPARED_MS = 8.0
PROJECTED_WARM_TTFT_MS = 22.0
MEASURED_RENDER_MS = 5.99
INTERPRETER_RENDER_MS = 48.3


def cache_selection_counters(path: Path) -> dict[str, Any]:
    """Return the context-cache selection counters from the LAST throughput record.

    These live on a `throughput` event, not on `request_done`, which is why an earlier version of
    this script reported that no shipped surface emitted them: it filtered the log to
    `request_done` and then reported the counters as absent. They were there the whole time, on a
    record this script was discarding.

    `format_throughput_json` (src/serve/request_log.cpp) writes them as deltas against the previous
    record, and the final one is written at shutdown. So the lane has to be stopped gracefully for
    them to exist at all -- TerminateProcess skips that write.

    Fails loudly if there is no throughput record, because reporting zero counters as "none were
    emitted" is the specific wrong answer this function exists to stop.
    """
    if not path.is_file():
        print(f"  FAIL: no request log at {path}")
        raise SystemExit(1)
    latest: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "throughput":
            latest = record
    if latest is None:
        print(f"  FAIL: {path.name} holds no throughput record, so the cache-selection counters are")
        print("  not in it. The lane was probably killed rather than stopped gracefully, or")
        print("  --log-stats-interval-ms is 0.")
        raise SystemExit(1)
    selections = (latest.get("context_cache") or {}).get("selections") or {}
    return selections


def request_done_records(path: Path) -> list[dict[str, Any]]:
    """Return the request_done records. Fails loudly rather than reporting over an empty set."""
    if not path.is_file():
        print(f"  FAIL: no request log at {path}")
        raise SystemExit(1)
    done = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "request_done":
            done.append(record)
    if not done:
        print(f"  FAIL: {path} has no request_done records, so there is nothing to report")
        raise SystemExit(1)
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="a request-log-jsonl written by ninfer-serve")
    parser.add_argument(
        "--warm-from",
        type=int,
        default=2,
        help="1-based request index from which the prefix counts as warm; default 2",
    )
    arguments = parser.parse_args()

    records = request_done_records(arguments.log)
    print(f"  {arguments.log.name}: {len(records)} completed request(s)")
    header = (
        f"  {'req':>3} {'prompt':>7} {'hit':>5} {'prep':>8} {'prefill':>8} "
        f"{'decode':>8} {'ttft':>8} {'total':>8}  path"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    warm: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        timing, result = record["timings_seconds"], record["result"]
        print(
            f"  {index:3d} {result['prompt_tokens']:7d} {result['prefix_cache_hit_tokens']:5d}"
            f" {timing['prepare'] * 1000:7.2f}m {timing['prefill'] * 1000:7.2f}m"
            f" {timing['decode'] * 1000:7.2f}m {timing['ttft'] * 1000:7.2f}m"
            f" {timing['total'] * 1000:7.2f}m  {result['prefix_reuse_path']}"
        )
        if index >= arguments.warm_from:
            warm.append(record)

    if not warm:
        print(f"  FAIL: no warm requests from index {arguments.warm_from}; send more")
        return 1

    def median(field: str) -> float:
        return statistics.median(r["timings_seconds"][field] for r in warm) * 1000

    prepare_ms, ttft_ms = median("prepare"), median("ttft")
    hits = [r["result"]["prefix_cache_hit_tokens"] for r in warm]
    tokens = {r["result"]["completion_tokens"] for r in warm}

    print()
    print(f"  warm requests          : {len(warm)} (from request {arguments.warm_from})")
    print(f"  prepare, median        : {prepare_ms:.2f} ms")
    print(f"  ttft, median           : {ttft_ms:.2f} ms")
    print(f"  prompt_tokens, warm    : {sorted({r['result']['prompt_tokens'] for r in warm})}")
    print(f"  completion_tokens      : {sorted(tokens)}")
    print(f"  prefix_cache_hit_tokens: warm min {min(hits)}, max {max(hits)}")
    print()

    # The selection counters, which say WHY a request took the path it took. ADR-0009's three:
    # a candidate that reached the plan, one the Program refused, one skipped at the shortlist key.
    selections = cache_selection_counters(arguments.log)
    watched = (
        "shared_stable_prefix",
        "shared_reuse_candidates",
        "shared_reuse_declined",
        "shared_reuse_key_mismatch",
    )
    print("  cache selection counters (deltas, from the last throughput record):")
    for name in watched:
        print(f"    {name:28s} {selections.get(name, '(absent)')}")
    print()
    if selections.get("shared_reuse_key_mismatch"):
        print("  A shared index entry was SKIPPED before planning because no shortlist key matched at")
        print("  its frontier -- the incoming prompt did not hash to the resident prefix there.")
    elif selections.get("shared_reuse_declined"):
        print("  A shared entry reached the plan and the Program REFUSED it.")
    elif selections.get("shared_reuse_candidates"):
        print("  A shared entry reached the plan and was not refused; another path won the valuation.")
    else:
        print("  No shared entry reached a reuse plan in this run.")
    print()

    # The cache question, answered by the field rather than inferred from a latency difference.
    if max(hits) == 0:
        print("  THE PREFIX CACHE NEVER ENGAGED on this workload. Every warm request recomputed the")
        print("  whole prompt. There is therefore no cache state to explain any difference between")
        print("  requests here, and no prefix-hit effect these figures can speak to.")
    else:
        print(f"  The prefix cache engaged: up to {max(hits)} hit tokens on a warm request. Any")
        print("  difference between request 1 and the rest has a cache state that could produce it.")

    print()
    print("  MEASURED HERE (this run, this artifact, this lane):")
    print(f"    prepare, warm median {prepare_ms:.2f} ms")
    print(f"    ttft,    warm median {ttft_ms:.2f} ms")
    print()
    print("  CARRIED FROM ADR-0012, in its own words:")
    print(f"    MEASURED   render, native path       {MEASURED_RENDER_MS:.2f} ms")
    print(f"    MEASURED   render, interpreter path  {INTERPRETER_RENDER_MS:.2f} ms")
    print(f"    PROJECTED  prepared                  ~{PROJECTED_PREPARED_MS:.0f} ms")
    print(f"    PROJECTED  warm ttft                 ~{PROJECTED_WARM_TTFT_MS:.0f} ms")
    print()
    print("  Promote by editing ADR-0012 with these numbers and this command, not from this report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
