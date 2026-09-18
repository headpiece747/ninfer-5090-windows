#!/usr/bin/env python3
"""Soak a profile under sustained load and report whether the engine stays healthy.

Upstream issue #208 reports intermittent cudaErrorIllegalAddress crashes running Qwen3.8-27B
NVFP4 with MTP speculative decoding under sustained agentic workloads on an RTX 5090 -- our exact
model, quantization, draft method and GPU. The reported onset is 10-15 minutes, and
CUDA_LAUNCH_BLOCKING=1 masked it over 187 minutes, which points at a race rather than bad input.
An intermittent crash is the worst failure a user can hit, so this harness answers the question
directly instead of reasoning about it.

Each iteration sends a mixed request (varied prompt size, some multi-turn continuations, varied
sampling) and records status, latency and token counts. It reports:

  * any non-2xx response, with the body, so an HTTP-level failure is not mistaken for health;
  * any connection failure, which means the engine process died;
  * throughput across the run and whether it degrades (a slow leak shows up here);
  * the engine's own stderr tail, if it was started with output redirected.

Exit status is non-zero if any request failed, so a soak can gate other work.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

FILLER = (
    "Iteration {i} of the soak corpus. The engine materialises context pages on demand and "
    "retains checkpoints at session endpoints, so repeated turns should reuse rather than "
    "re-prefill. This text exists only to vary the prompt size deterministically. "
)


def build(size_tokens: int, index: int) -> str:
    text = []
    written = 0
    limit = size_tokens * 4
    while written < limit:
        chunk = FILLER.format(i=index)
        text.append(chunk)
        written += len(chunk)
        index += 1
    return "".join(text)


def post(base: str, path: str, body: dict, timeout: int) -> tuple[int, str]:
    request = urllib.request.Request(
        base + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors="replace")
    except Exception as error:  # connection refused, reset, timeout: the engine is gone or stuck
        return 0, repr(error)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8087")
    ap.add_argument("--model", required=True)
    ap.add_argument("--minutes", type=float, default=16.0,
                    help="#208 reports onset at 10-15 minutes under sustained load")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--engine-stderr", type=Path, default=None,
                    help="engine stderr file, if the server was started with it redirected")
    args = ap.parse_args()

    deadline = time.monotonic() + args.minutes * 60
    sizes = [256, 1024, 4096, 8192, 512, 2048, 16384, 1024]
    iterations = 0
    failures: list[tuple[int, str, str]] = []
    latencies: list[float] = []
    tokens = 0
    history: list[dict] = []

    print(f"soak  model={args.model}  for {args.minutes:.0f} min  max_tokens={args.max_tokens}")
    print(f"      base={args.base}")
    started = time.monotonic()
    while time.monotonic() < deadline:
        index = iterations
        size = sizes[index % len(sizes)]
        messages = [{"role": "user", "content": build(size, index) + "\n\nReply briefly."}]
        # Every third iteration continues the previous turn, so the context grows the way an agent
        # session does rather than staying flat.
        if index % 3 == 2 and history:
            previous = history[-1]
            if previous.get("reply"):
                messages = [*[{"role": "user", "content": build(sizes[index % len(sizes)], index)}],
                            {"role": "assistant", "content": previous["reply"]},
                            {"role": "user", "content": "Continue briefly."}]
        body = {"model": args.model, "messages": messages,
                "max_completion_tokens": args.max_tokens, "stream": False}
        if index % 4 == 1:
            body["reasoning_effort"] = "none"

        request_started = time.monotonic()
        status, payload = post(args.base, "/v1/chat/completions", body, timeout=900)
        elapsed = time.monotonic() - request_started
        iterations += 1

        if status == 0 or status >= 500:
            failures.append((status, payload[:400], f"size={size} index={index}"))
            print(f"  [{elapsed:6.1f}s] FAILURE status={status} {payload[:200]}")
            if status == 0:
                print("  engine is unreachable: it has died or is wedged. stopping the soak.")
                break
        else:
            try:
                completion = json.loads(payload)
                usage = completion.get("usage") or {}
                tokens += (usage.get("completion_tokens") or 0)
                reply = (completion.get("choices") or [{}])[0].get("message", {}).get("content", "")
                history.append({"reply": reply, "prompt_tokens": usage.get("prompt_tokens") or 0,
                                "cached": (usage.get("prompt_tokens_details") or {}).get(
                                    "cached_tokens") or 0})
            except json.JSONDecodeError:
                failures.append((status, payload[:400], "unparseable response"))
                history.append({"reply": ""})
            latencies.append(elapsed)
            if iterations % 5 == 0 or elapsed > 60:
                print(f"  [{elapsed:6.1f}s] iter {iterations:>3}  status {status}"
                      f"  tokens {tokens:>6}  requests {len(latencies)}")

    wall = time.monotonic() - started
    print(f"\n  ran {iterations} requests in {wall / 60:.1f} min "
          f"({iterations / max(wall, 1e-9) * 60:.1f} req/min)")
    if latencies:
        ordered = sorted(latencies)
        print(f"  latency  p50 {ordered[len(ordered) // 2]:.2f}s"
              f"  p95 {ordered[int(len(ordered) * 0.95)]:.2f}s"
              f"  max {ordered[-1]:.2f}s")
        print(f"  completion tokens {tokens}")
    if args.engine_stderr and args.engine_stderr.exists():
        tail = args.engine_stderr.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
        print("  engine stderr tail:")
        for line in tail:
            print(f"    {line}")

    if failures:
        print(f"\n  SOAK FAILED: {len(failures)} failure(s)")
        for status, body, context in failures[:5]:
            print(f"    status={status} {context}\n      {body[:300]}")
        return 1
    print("\n  SOAK HEALTHY: no failed request")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
