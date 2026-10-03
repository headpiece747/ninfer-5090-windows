#!/usr/bin/env python3
"""Measure `prepared` and warm TTFT on a real serving lane, against conversation size.

ADR-0012 projects, from the render figure and the field log's split:

    prepared ~ 8 ms, warm TTFT ~ 22 ms

for a 229-message, ~690 KB conversation whose prefill is served from cache, and it says
plainly that "the end-to-end path has not been timed with the new renderer". This is the
instrument that would time it, and it reads the engine's own `request-log-jsonl` rather than a
wall clock, because the project rule is that when a component figure exists next to an end-to-end
one the engine's own fields decide where the time went.

WHAT THIS CANNOT DO, STATED UP FRONT. ADR-0012's exact conversation is ~690 KB of rendered text,
which is roughly 172k tokens. This artifact reaches about 117k context (a recorded 65k request
already costs 15.1 s of TTFT). The projection's stated conditions are therefore NOT reproducible
on this artifact, and this script measures a curve at sizes that fit instead of pretending to
reproduce the original. The largest size actually sent is printed, so the bound is never implicit.

Each size is sent TWICE in one lane: the first request is cold (path `root`), the second is the one
reported, because "warm TTFT" in ADR-0012 means a request whose prefill is served from cache. The
script reports the reuse path and the hit-token count alongside the timing, so a number labelled
warm can be checked against the field that makes it warm rather than taken on trust.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from first_request_lane import (  # noqa: E402
    REPO_ROOT,
    serve_command,
    stop_lane,
    wait_until_ready,
)

FILLER = "The quick brown fox jumps over the lazy dog. " * 4


def conversation(messages: int, chars: int) -> list[dict[str, str]]:
    """`messages - 1` alternating turns plus one short final user ask.

    The trailing short ask is what makes this usable: `conversation(n)` and `conversation(n + 1)`
    share their first `n - 1` messages EXACTLY, so sending them in order gives a genuinely shared
    prefix with one new turn at the end -- the shape a chat client produces, and the only shape in
    which "warm" means a cached prefix plus a real decode.

    Sending the same body twice does NOT produce that. It takes `private_response_replay`, which
    returns a stored response and never decodes, so its TTFT measures replay rather than the
    prefill-plus-decode path ADR-0012 projected for. The first version of this sweep did exactly
    that and every warm row came back on the replay path.
    """
    body = (FILLER * (chars // len(FILLER) + 1))[:chars]
    turns: list[dict[str, str]] = []
    for index in range(max(0, messages - 1)):
        role = "user" if index % 2 == 0 else "assistant"
        turns.append({"role": role, "content": body})
    turns.append({"role": "user", "content": "Reply with one short sentence."})
    return turns


def post(port: int, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--exe", type=Path, default=REPO_ROOT / "build" / "apps" / "ninfer-serve.exe")
    parser.add_argument("--sizes", default="1,8,32,64", help="message counts; default 1,8,32,64")
    parser.add_argument("--chars", type=int, default=3000, help="chars per message; default 3000")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--port", type=int, default=8260)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "profiles" / "warm_lane")
    arguments = parser.parse_args()

    sizes = [int(part) for part in arguments.sizes.split(",") if part.strip()]
    arguments.log_dir.mkdir(parents=True, exist_ok=True)
    request_log = arguments.log_dir / "warm_lane.requests.jsonl"
    server_log = arguments.log_dir / "warm_lane.log"
    if request_log.exists():
        request_log.unlink()

    command, dropped = serve_command(arguments.exe, arguments.artifact, arguments.port, False, request_log)
    if dropped:
        print(f"  dropped {len(dropped)} flag(s) with prefix reuse ON: {', '.join(dropped)}")

    print(f"  starting one lane on port {arguments.port}", flush=True)
    handle = server_log.open("wb")
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        # Its own console process group, so stop_lane's CTRL_BREAK reaches the lane alone.
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    try:
        if not wait_until_ready(arguments.port, process, arguments.timeout):
            print(f"  FAIL: lane never became ready; see {server_log}")
            return 1
        print("  lane ready", flush=True)

        for messages in sizes:
            # Seed the cache with the shorter conversation first, so the longer one that follows
            # reuses a real prefix and has to decode its new turn. Reported, not seeded.
            #
            # `max(1, messages - 1)` clamped to 1, which at messages == 1 sent the SAME body twice
            # and produced a private_response_replay row that had nothing to do with prefix reuse.
            # Below 2 there is no shorter conversation, so the seed is skipped rather than faked and
            # the request is sent once; such a row is then reported as unseeded instead of warm.
            seed_count = messages - 1
            plan = [(2, messages)] if seed_count < 1 else [(1, seed_count), (2, messages)]
            for attempt, count in plan:
                body = {
                    "model": "qwen3.8-27b-quasar-v3-dflash2-vision",
                    "messages": conversation(count, arguments.chars),
                    "max_tokens": arguments.max_tokens,
                    "temperature": arguments.temperature,
                    "seed": arguments.seed,
                    "enable_thinking": False,
                    "stream": False,
                }
                try:
                    post(arguments.port, body, arguments.timeout)
                except urllib.error.HTTPError as error:
                    detail = error.read().decode("utf-8", "replace")[:200]
                    print(f"  {messages:4d} message(s), request {attempt} REJECTED: {error.code} {detail}")
                    break
                except Exception as error:  # noqa: BLE001
                    print(f"  {messages:4d} message(s), request {attempt} FAILED: {type(error).__name__}: {error}")
                    break
            print(f"  {messages:4d} message(s): prefix seeded, then the reported request", flush=True)
    finally:
        # stop_lane, not terminate(): on Windows terminate() kills the process, so the shutdown
        # throughput record carrying the cache-selection counters is never flushed.
        stop_lane(process)
        handle.close()

    if not request_log.is_file():
        print(f"  FAIL: no request log at {request_log}")
        return 1
    done = [
        json.loads(line)
        for line in request_log.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("event") == "request_done"
    ]
    if not done:
        print(f"  FAIL: {request_log} has no request_done records")
        return 1

    header = f"  {'msgs':>5} {'warm?':>6} {'prompt_tok':>11} {'hit':>6} {'prep_ms':>9} {'ttft_ms':>9}  path"
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))
    warms: list[tuple[int, float, float, int, str]] = []
    replays: list[int] = []
    for index, record in enumerate(done, 1):
        timing, result = record["timings_seconds"], record["result"]
        path = result["prefix_reuse_path"]
        hit = result["prefix_cache_hit_tokens"]
        # A replay path returns a stored response and never decodes, so its TTFT is not a
        # prefill-plus-decode measurement. Such a row is reported and then excluded, because a
        # plausible number taken on the wrong path is the failure this whole script exists to avoid.
        valid = hit > 0 and path != "private_response_replay"
        print(
            f"  {'':>5} {('yes' if valid else ('REPLAY' if path == 'private_response_replay' else 'NO')):>6}"
            f" {result['prompt_tokens']:11d} {hit:6d} {timing['prepare'] * 1000:8.2f}"
            f" {timing['ttft'] * 1000:8.2f}  {path}"
        )
        if hit > 0:
            if valid:
                warms.append(
                    (index, timing["prepare"] * 1000, timing["ttft"] * 1000, result["prompt_tokens"], path)
                )
            else:
                replays.append(index)

    print()
    if replays:
        print(f"  EXCLUDED {len(replays)} row(s) on index {replays}: private_response_replay returns a")
        print("  stored response without decoding, so their TTFT is replay latency, not TTFT.")
    if not warms:
        print("  No request combined a prefix cache hit with a real decode, so there is no warm")
        print("  TTFT to report. That is a result about the cache on this workload.")
        return 1
    print(f"  usable warm requests: {len(warms)}")
    print(f"  largest conversation actually sent: {max(w for _, _, _, w, _ in warms)} prompt tokens")
    print(f"  prepare, median over warm: {statistics.median(p for _, p, _, _, _ in warms):.2f} ms")
    print(f"  ttft,    median over warm: {statistics.median(t for _, _, t, _, _ in warms):.2f} ms")
    print("  paths counted as warm: " + ", ".join(sorted({p for _, _, _, _, p in warms})))
    print()
    print("  prepare scales with the conversation (frontend render, tokenize, build) and is")
    print("  independent of the reuse path, so its trend is the comparable quantity.")
    print("  ADR-0012 projects prepared ~8 ms and warm TTFT ~22 ms for a ~172k-token conversation.")
    print("  The largest conversation sent here is the number above. Anything beyond it is an")
    print("  extrapolation and must be recorded as untested, not as confirmed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
