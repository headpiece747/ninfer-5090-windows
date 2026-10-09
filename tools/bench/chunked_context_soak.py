"""Drive a serving lane with a growing conversation, fed in chunks, and find out if it dies.

#208 reports intermittent `cudaErrorIllegalAddress` while running Qwen3.8-27B NVFP4 with MTP
speculative decoding under sustained agentic workloads. Reporter 2 supplies the shape: "reading a
large file in chunks to fill up my context. It failed at different points for both of my runs."

So this tool fills a conversation in chunks and then decodes at the filled window, which is what the
reports describe and what a large-prompt benchmark does not do. It ends by asking the server for
`/v1/models`: if the lane stops answering, the run reproduced a crash, and the engine's own log carries
the reason (open a lane with its output captured before diagnosing one, or the only evidence is a line
that says nothing).

Two traps, both met while writing this, are handled structurally:

- The decode turn appends a distinct user message, because an identical request body is served from the
  stored response and never decodes -- the run then reports a plausible 0.1 s with the previous
  completion. A turn whose completion count is far below `--decode` and whose prompt did not grow is
  the signature of that, and it is why every turn prints both.
- The model id is read from `/v1/models` rather than assumed. A wrong alias answers 404 and no request
  reaches the engine.

Usage:
    python tools/bench/chunked_context_soak.py --port 8188 --cycles 3 --chunks 6 \
        --chunk-chars 100000 --decode 2048
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

WORDS = ("the quick brown fox jumps over the lazy dog while the engine attends to every token "
         "in a long context window and the page directory tracks each key value pair carefully ")


def chunk(cycle: int, index: int, target_chars: int) -> str:
    """Deterministic filler, distinct per cycle so a later run is not a cache replay."""
    unit = f"[c{cycle} k{index}] " + WORDS
    return (unit * (target_chars // len(unit) + 1))[:target_chars]


def fetch_model(base: str) -> str:
    with urllib.request.urlopen(f"{base}/v1/models", timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["data"][0]["id"]


def turn(base: str, model: str, messages: list[dict[str, str]], max_tokens: int,
         label: str) -> dict:
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0}).encode()
    request = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=3600) as response:
        payload = json.loads(response.read().decode("utf-8"))
    usage = payload.get("usage", {})
    print(f"    {label}: prompt={usage.get('prompt_tokens')} "
          f"completion={usage.get('completion_tokens')} {time.time() - started:.1f}s", flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--cycles", type=int, default=3,
                        help="conversations; each one refills from empty")
    parser.add_argument("--chunks", type=int, default=6, help="chunks per conversation")
    parser.add_argument("--chunk-chars", type=int, default=100_000,
                        help="characters per chunk, about four per token")
    parser.add_argument("--decode", type=int, default=2048,
                        help="max tokens for the sustained-decode turn")
    parser.add_argument("--json", action="store_true", help="print each response object")
    args = parser.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    try:
        model = fetch_model(base)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
        print(f"no lane answering {base} ({type(error).__name__}): start ninfer-serve first, "
              "with its output captured", file=sys.stderr)
        return 1
    print(f"    model id: {model}", flush=True)

    for cycle in range(args.cycles):
        print(f"  cycle {cycle}: {args.chunks} chunks of {args.chunk_chars:,} chars", flush=True)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "You are a careful assistant."}]
        for index in range(args.chunks):
            messages.append({"role": "user", "content": chunk(cycle, index, args.chunk_chars)})
            payload = turn(base, model, messages, 32, f"c{cycle} k{index}")
            if args.json:
                print(f"      {json.dumps(payload)[:400]}")
        messages.append({"role": "user", "content":
                         f"Cycle {cycle}: summarise every chunk above in one long paragraph and "
                         "keep writing until you reach the length limit."})
        payload = turn(base, model, messages, args.decode, f"c{cycle} decode")
        if args.json:
            print(f"      {json.dumps(payload)[:400]}")

    try:
        fetch_model(base)
        print("lane still answering /v1/models: no crash in this run")
        return 0
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
        print(f"LANE STOPPED ANSWERING ({type(error).__name__}): read the engine log for the "
              "cudaError, this run reproduced a crash", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
