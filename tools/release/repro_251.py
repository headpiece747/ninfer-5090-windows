#!/usr/bin/env python3
"""Reproduce upstream issue #251: prefix reuse stops once the checkpoint budget is exercised.

**Status: fixed 2026-09-19.** Active-capture admission now reclaims the oldest unpinned private
continuation, so a clean run reports "not reproduced" and a reproduction is a regression, not the
expected result (`docs/research/prefix-state-eviction.md`).

#251 reports, from a Windows sm_89 port with a Qwen3.8-27B artifact and DFlash2, that prefix reuse
works while the engine is fresh and then stops completely for the rest of the engine's life, with
only an engine restart restoring it and no LRU eviction observed.

The shape is simple enough to test directly: several conversations, each asked twice, where the
second request is the first request's text plus one appended line. The first request of a fresh
conversation cannot reuse anything, so its cache must be 0; the second must reuse almost all of
it. The failure is that later conversations stop getting that reuse while -- crucially -- nothing
about the requests changed.

Verdict logic: if the first few conversations reuse and later ones do not, the claim holds and the
engine is degrading as its checkpoint budget fills. If every conversation's second request reuses,
it does not.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

FILLER = (
    "Conversation {c} segment {i}. The engine materialises context pages on demand and retains "
    "checkpoints at session endpoints, so the second request of this conversation, which extends "
    "the first verbatim, should reuse rather than re-prefill. This text exists only to occupy "
    "prompt tokens deterministically and to make each conversation distinct. "
)


def build(tokens: int, conversation: int) -> str:
    parts = []
    written = 0
    index = 0
    limit = tokens * 4
    while written < limit:
        text = FILLER.format(c=conversation, i=index)
        parts.append(text)
        written += len(text)
        index += 1
    return "".join(parts)


def chat(base: str, model: str, messages: list[dict], max_tokens: int) -> dict:
    body = {"model": model, "messages": messages, "max_completion_tokens": max_tokens,
            "stream": False}
    request = urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        raise SystemExit(
            f"HTTP {error.code}: {error.read().decode(errors='replace')[:400]}") from error
    except Exception as error:
        print(f"    CONNECTION FAILURE: {error!r}")
        return {"failed": True, "prompt_tokens": 0, "cached_tokens": 0, "elapsed_s": 0.0,
                "text": ""}
    usage = payload.get("usage") or {}
    return {
        "failed": False,
        "prompt_tokens": usage.get("prompt_tokens") or 0,
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
        "elapsed_s": time.monotonic() - started,
        "text": (payload.get("choices") or [{}])[0].get("message", {}).get("content", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8086")
    ap.add_argument("--model", required=True)
    ap.add_argument("--conversations", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=26000,
                    help="#251 used ~26,300-token conversations")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    print(f"repro #251  model={args.model}  conversations={args.conversations}"
          f"  ~{args.tokens:,} tokens each")
    rows = []
    for conversation in range(1, args.conversations + 1):
        first = [{"role": "user", "content": build(args.tokens, conversation)}]
        a = chat(args.base, args.model, first, args.max_tokens)
        second = [*first,
                  {"role": "assistant", "content": a["text"] or "Acknowledged."},
                  {"role": "user", "content": "One more line appended to the same conversation."}]
        b = chat(args.base, args.model, second, args.max_tokens)
        reuse = b["cached_tokens"] / max(1, b["prompt_tokens"])
        rows.append((conversation, a, b, reuse))
        print(f"  conv {conversation}: first prompt {a['prompt_tokens']:>7}"
              f" cache {a['cached_tokens']:>7}"
              f"  |  second prompt {b['prompt_tokens']:>7}"
              f" cache {b['cached_tokens']:>7} ({reuse:.1%} reuse)"
              f"  TTFT-ish {a['elapsed_s']:>5.1f}s/{b['elapsed_s']:>5.1f}s")

    if any(row[1].get("failed") or row[2].get("failed") for row in rows):
        print("\n  a request failed outright; see the message above")
        return 1

    reuse = [row[3] for row in rows]
    healthy = [value for value in reuse if value > 0.5]
    starved = [index + 1 for index, value in enumerate(reuse) if value < 0.1]
    print(f"  reuse per conversation: {', '.join(f'{value:.1%}' for value in reuse)}")
    if healthy and starved:
        print(f"\n  REPRODUCED: reuse held for {len(healthy)} conversation(s) and then stopped"
              f" from conversation {starved[0]} onward -- re-prefilling from the root for the"
              f" rest of the engine's life unless it is restarted")
        return 1
    if not healthy:
        print("\n  INCONCLUSIVE: no conversation reused at all, which is a different failure")
        return 1
    print("\n  not reproduced: every conversation's second request reused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
