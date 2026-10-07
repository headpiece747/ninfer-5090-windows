#!/usr/bin/env python3
"""Reproduce upstream issue #251: prefix reuse stops once the checkpoint budget is exercised.

**Status: fixed 2026-09-19, re-opened 2026-10-06.** Active-capture admission reclaimed the oldest
unpinned private continuation, so a clean run reported "not reproduced"; upstream's `b9114396` then
replaced that cache, the merge took upstream's version, and the cliff is back -- measured here, and
located at the replacement's value gate rather than at a missing eviction path
(`docs/research/prefix-state-eviction.md`, last two sections). A reproduction is therefore the
expected result again, not a regression.

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

Two ways to run it. Attach to a lane you started yourself (`--base`, `--model`), or let it start one
with the shipped pool flags (`--serve`, `--artifact`), which prints the exact command it used and the
request log it read. The second form is what makes a result reproducible from this file alone; the
first is for a loop you are already driving.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
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


def start_serve(args: argparse.Namespace) -> tuple[subprocess.Popen[bytes], Path]:
    """Start a lane with the pool flags under test, and wait for /health."""
    request_log = Path(args.request_log)
    request_log.parent.mkdir(parents=True, exist_ok=True)
    request_log.unlink(missing_ok=True)
    command = [
        str(Path(args.serve).resolve()),
        str(Path(args.artifact).resolve()),
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--max-context", str(args.max_context),
        "--kv-capacity", args.kv_capacity,
        "--device-state-slots", str(args.device_state_slots),
        "--host-context-mib", str(args.host_context_mib),
        "--max-concurrency", "1",
        "--kv-dtype", "fp8",
        "--request-log-jsonl", str(request_log),
        *args.extra.split(),
    ]
    print("lane command:", " ".join(command), flush=True)
    log_path = Path(args.lane_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("wb")
    process = subprocess.Popen(command, cwd=str(REPO_ROOT), stdout=handle, stderr=subprocess.STDOUT)
    deadline = time.time() + args.startup_timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise SystemExit(
                f"the lane exited during startup with status {process.returncode}; "
                f"see {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as reply:
                if reply.status == 200:
                    print(f"lane ready (log {log_path})", flush=True)
                    return process, request_log
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    process.kill()
    raise SystemExit(f"the lane never became ready; see {log_path}")


def model_id(base: str, fallback: str) -> str:
    """The id the lane advertises, so a caller does not have to know it."""
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=10) as reply:
            listing = json.loads(reply.read().decode())
        entries = listing.get("data") or []
        if entries and isinstance(entries[0].get("id"), str):
            return entries[0]["id"]
    except Exception:  # noqa: BLE001 - fall back to what the caller passed
        pass
    return fallback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8086",
                    help="attach to a lane already running here")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--conversations", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=26000,
                    help="#251 used ~26,300-token conversations")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=2,
                    help="turns per conversation; 2 is the classic shape (ask, then extend)")
    ap.add_argument("--interleave", action="store_true",
                    help="round-robin across conversations instead of asking each one twice in a row; "
                         "this is the shape several subagent sessions produce, and the one "
                         "docs/opencode-settings.md recorded as unmeasured")
    ap.add_argument("--serve", help="ninfer-serve path; starts a lane instead of attaching")
    ap.add_argument("--artifact", help="artifact for the lane this harness starts")
    ap.add_argument("--port", type=int, default=8086)
    ap.add_argument("--max-context", type=int, default=262144)
    ap.add_argument("--kv-capacity", default="auto", help="as the shipped lanes pass it")
    ap.add_argument("--device-state-slots", type=int, default=1)
    ap.add_argument("--host-context-mib", default="8192", help="as the shipped lanes pass it")
    ap.add_argument("--extra", default="", help="further lane flags, space separated")
    ap.add_argument("--startup-timeout", type=float, default=600.0)
    ap.add_argument("--lane-log", default="profiles/bench/repro_251-lane.log")
    ap.add_argument("--request-log", default="profiles/bench/repro_251-requests.jsonl")
    args = ap.parse_args()

    process = None
    base = args.base
    if args.serve:
        if not args.artifact:
            raise SystemExit("--serve needs --artifact")
        process, request_log = start_serve(args)
        base = f"http://127.0.0.1:{args.port}"
    model = model_id(base, args.model)

    try:
        print(f"repro #251  model={model}  conversations={args.conversations}"
              f"  ~{args.tokens:,} tokens each  rounds={args.rounds}"
              f"  interleave={'yes' if args.interleave else 'no'}", flush=True)
        conversation_ids = list(range(1, args.conversations + 1))
        history: dict[int, list[dict]] = {
            c: [{"role": "user", "content": build(args.tokens, c)}] for c in conversation_ids
        }
        results: dict[tuple[int, int], dict] = {}

        def one_turn(c: int, turn: int) -> None:
            messages = history[c]
            if turn > 1:
                messages = [*messages,
                            {"role": "user",
                             "content": f"Turn {turn}: one more line appended to this conversation."}]
            answer = chat(base, model, messages, args.max_tokens)
            results[(c, turn)] = answer
            if not answer.get("failed"):
                history[c] = [*messages,
                              {"role": "assistant", "content": answer["text"] or "Acknowledged."}]

        if args.interleave:
            for turn in range(1, args.rounds + 1):
                for c in conversation_ids:
                    one_turn(c, turn)
        else:
            for c in conversation_ids:
                for turn in range(1, args.rounds + 1):
                    one_turn(c, turn)

        rows = []
        for c in conversation_ids:
            for turn in range(1, args.rounds + 1):
                answer = results[(c, turn)]
                reuse = answer["cached_tokens"] / max(1, answer["prompt_tokens"])
                rows.append((c, turn, answer, reuse))
                print(f"  conv {c:>3} turn {turn}: prompt {answer['prompt_tokens']:>7}"
                      f" cache {answer['cached_tokens']:>7} ({reuse:>5.1%} reuse)"
                      f"  {answer['elapsed_s']:>5.1f}s", flush=True)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
            print(f"lane stopped; its own request log is {request_log}", flush=True)

    if any(row[2].get("failed") for row in rows):
        print("\n  a request failed outright; see the message above")
        return 1

    # Turn 1 of a fresh conversation cannot reuse anything; every later turn must. The verdict is
    # about the later turns only, so a healthy first-turn zero is not a failure.
    later = [row for row in rows if row[1] > 1]
    reuse = [row[3] for row in later]
    healthy = [value for value in reuse if value > 0.5]
    starved = [(row[0], row[1]) for row in later if row[3] < 0.1]
    per_conversation = {row[0]: row[3] for row in later}
    print("  reuse per conversation (last turn): "
          + ", ".join(f"{c}:{value:.1%}" for c, value in sorted(per_conversation.items())))
    if healthy and starved:
        print(f"\n  REPRODUCED: reuse held for {len(healthy)} request(s) and then stopped from "
              f"conversation {starved[0][0]} turn {starved[0][1]} onward -- re-prefilling from the "
              f"root for the rest of the engine's life unless it is restarted")
        return 1
    if not healthy:
        print("\n  NO REUSE AT ALL: not one request reused. Two causes look like this and a run "
              "cannot tell them apart on its own: the cache is off, or the pools cannot hold the "
              "live set -- which is what the interleaved shape produces past ~52 conversations at "
              "the shipped 8 GiB quota, where every request reports preferred_reused_tokens 0 "
              "(docs/research/prefix-state-eviction.md). Rerun with fewer conversations or a larger "
              "--host-context-mib to find out which.")
        return 1
    print(f"\n  not reproduced: every later turn reused ({len(later)} request(s), "
          f"minimum {min(reuse):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
