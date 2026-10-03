#!/usr/bin/env python3
"""Is a SHARED PREFIX served, or is only exact-request replay?

`tools/bench/warm_lane_sweep.py` found that a growing conversation with a genuinely shared prefix
returns `prefix_cache_hit_tokens = 0` and path `root`, while an identical repeat takes
`private_response_replay`. That is two observations and no cause. This is the loop that decides
between "the shared-prefix path is not taken" and "the sweep's condition does not reach it".

THE CONTROL IS THE POINT. Every earlier instrument here could report zero hits whether the cache
worked or not, because nothing ever demonstrated that the same measurement can SEE a hit. So this
sends two requests first that are known to hit -- an exact repeat, which the engine answers from
`private_response_replay` -- and FAILS if even those read zero. Without that arm, a green run means
nothing at all: it is equally consistent with a working cache and with a harness pointed at the wrong
field.

Three conditions, each a separate verdict, all in one lane so they share a configuration:

  repeat    the identical body twice            -> must hit (the control)
  grow      conversation(n) then conversation(n+1), sharing n-1 messages -> the question
  shuffle   the same messages in a DIFFERENT order, sharing nothing     -> must not hit
  marked    the same growing pair, with an explicit prompt_cache_breakpoint -> the fix

`shuffle` is the negative control: a measurement that reported hits for a conversation sharing no
prefix would be reporting something other than a prefix match.

WHY `marked` EXISTS. `/v1/chat/completions` sets `allow_engine_automatic_shared_prefixes = false`
for every request (openai_common.cpp:177, with the reason at 175: OpenAI defines the automatic write
policy itself), so the ENGINE never volunteers a shared prefix on this route. Plain string content
also carries no marker of its own -- `parse_openai_prompt_cache_breakpoint` reads it off a content
PART, and a string content is not a part. So a request with neither offers no shared-prefix write
candidate, and the growing conversation correctly gets `root`. The documented way to ask for one is
`prompt_cache_breakpoint:{"mode":"explicit"}` on a content part (docs/serving.md, OpenAI prompt
caching). `marked` is that request. If `marked` hits and `grow` does not, the cache works and the
original sweep was asking a question the protocol does not answer by default.

Usage: check_shared_prefix_reuse.py --artifact PATH [--exe PATH]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from first_request_lane import REPO_ROOT, serve_command, wait_until_ready  # noqa: E402

BODY_WORDS = 400
MESSAGES = 6
SYSTEM = "You are a concise assistant that answers in one sentence."


def conversation(count: int, order: list[int] | None = None) -> list[dict[str, str]]:
    """`count` alternating turns, each ~BODY_WORDS words long.

    `order` permutes the turns so `shuffle` can build a conversation that shares no prefix with the
    one already cached, while keeping the same total content.
    """
    filler = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet " * 40)[: BODY_WORDS * 6]
    turns = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"{filler} turn-{index}"}
        for index in range(count)
    ]
    turns.append({"role": "user", "content": "Reply with one short sentence."})
    if order is not None:
        turns = [turns[index] for index in order] + [turns[-1]]
    return turns


def diverged(count: int, tag: str) -> list[dict[str, str]]:
    """`count - 1` turns identical to `conversation(count)`, then a DIFFERENT final USER ask.

    A private response replay is keyed on the whole prompt, so rewording the tail defeats it and the
    shared prefix is the only remaining route.

    The tail must stay a USER turn. `ResidentPrefixIdentity::matches` (prefix_identity.cpp:218)
    compares `rewrite_execution_frontiers`, and those are derived from ASSISTANT blocks in the
    rendered text (native_render.cpp:550, `if (role == "assistant")`). A different number of
    assistant turns inside the counted prefix makes the two identities disagree and the shared
    prefix is refused -- which is correct behaviour, and the reason this harness cannot simply
    change the tail's content.
    """
    turns = conversation(count)[:-1]
    return turns + [{"role": "user", "content": f"{tag} Reply with one short sentence."}]


def system_then(first_user: str, tail: list[dict[str, str]], ask: str) -> list[dict[str, str]]:
    """System turn, a conversation-unique FIRST USER turn, a shared middle, and a shared final ask.

    ADR-0009's shape, and the middle part is the part that matters. Its two sessions "share a system
    message and one tool definition, differing in their first user turn". The divergence has to come
    EARLY, because the shared prefix has to end at the system message: the protocol's automatic target
    is the end of the last message, so a prefix that runs to the end of the prompt is published at the
    whole-prompt frontier, where `private_response_replay` wins the valuation instead
    (profiles.py:388-392, and ADR-0009's third arm: identical prompts are offered the shared
    candidate and `private_response_replay` takes it).

    My first version differed only in the FINAL ask, so its shared region ran to the end of the
    prompt, and it got 0 hit tokens on both conversations -- the exact third arm, not the second.
    """
    return (
        [{"role": "system", "content": SYSTEM}]
        + [{"role": "user", "content": first_user}]
        + list(tail)
        + [{"role": "user", "content": ask}]
    )


def as_part(message: dict[str, str], marked: bool) -> dict[str, Any]:
    """Turn a string-content message into a single-part message, optionally marked.

    A breakpoint is read off a content PART, so a request built from plain strings can never carry
    one. Converting only the messages that should end at a boundary is what makes the marker land
    where the shared prefix actually ends.
    """
    part: dict[str, Any] = {"type": "text", "text": message["content"]}
    if marked:
        part["prompt_cache_breakpoint"] = {"mode": "explicit"}
    return {"role": message["role"], "content": [part]}


def body_for(messages: list[dict[str, str]], seed: int, marked_upto: int = -1) -> dict[str, Any]:
    """Build a request body. `marked_upto` marks the first N messages as parts with a breakpoint."""
    if marked_upto < 0:
        out = list(messages)
    else:
        out = [as_part(m, index < marked_upto) for index, m in enumerate(messages)]
    return {
        "model": "qwen3.8-27b-quasar-v3-dflash2-vision",
        "messages": out,
        "max_tokens": 16,
        "temperature": 0.0,
        "seed": seed,
        "enable_thinking": False,
        "stream": False,
    }


def post(port: int, body: dict[str, Any], timeout_s: float) -> None:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--exe", type=Path, default=REPO_ROOT / "build" / "apps" / "ninfer-serve.exe")
    parser.add_argument("--port", type=int, default=8280)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "profiles" / "shared_prefix")
    arguments = parser.parse_args()

    arguments.log_dir.mkdir(parents=True, exist_ok=True)
    request_log = arguments.log_dir / "shared_prefix.requests.jsonl"
    server_log = arguments.log_dir / "shared_prefix.log"
    if request_log.exists():
        request_log.unlink()

    command, _ = serve_command(arguments.exe, arguments.artifact, arguments.port, False, request_log)
    handle = server_log.open("wb")
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
    try:
        if not wait_until_ready(arguments.port, process, arguments.timeout):
            print(f"  FAIL: lane never became ready; see {server_log}")
            return 1
        base = conversation(MESSAGES)
        # 1. the control: identical body twice. Known to take private_response_replay.
        for _ in range(2):
            post(arguments.port, body_for(base, 1), arguments.timeout)
        # 2. the question: one more turn, sharing every preceding turn exactly, unmarked.
        post(arguments.port, body_for(conversation(MESSAGES + 1), 1), arguments.timeout)
        # 3. the negative control: same turns, reversed, so no prefix is shared.
        post(arguments.port, body_for(conversation(MESSAGES, order=list(reversed(range(MESSAGES)))), 1),
             arguments.timeout)
        # 4. the documented write: seed a marked prefix at the shared frontier, then extend it.
        # Mark every message INCLUDING the final ask. The frontier has to be where the shared
        # prefix actually ends: `conversation(n)` is n-1 turns plus the ask, so marking fewer than
        # all of them puts the frontier mid-conversation, past the shared region. That is the first
        # version's bug, and it reported "still not measurable" for a request that was simply marked
        # in the wrong place.
        marked = len(base)
        post(arguments.port, body_for(base, 1, marked_upto=marked), arguments.timeout)
        post(arguments.port, body_for(conversation(MESSAGES + 1), 1, marked_upto=marked),
             arguments.timeout)
        # 5. the warm shape: seed the marked prefix, then send a request whose TAIL DIFFERS. A
        # private replay is keyed on the whole prompt, so this cannot replay; the shared prefix is
        # the only remaining route, and the new tail must actually be encoded and decoded.
        post(arguments.port, body_for(diverged(MESSAGES, "ALPHA"), 1, marked_upto=marked),
             arguments.timeout)
        # 6. ADR-0009's shape: a system turn carrying the declaration, then TWO DIFFERENT
        # conversations that share everything up to the system message and diverge after it. This is
        # the only shape in which `private_response_replay` cannot win the valuation, because the
        # two prompts are not the same prompt.
        shared_tail = conversation(MESSAGES)[1:-1]
        for tag in ("ALPHA", "BRAVO"):
            messages = system_then(
                f"{tag}: what is a split-K GEMM?", shared_tail, "Reply with one short sentence."
            )
            post(arguments.port, body_for(messages, 1, marked_upto=1), arguments.timeout)
    except urllib.error.HTTPError as error:
        print(f"  FAIL: the engine refused a request: {error.code} "
              f"{error.read().decode('utf-8', 'replace')[:200]}")
        return 1
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
        handle.close()

    if not request_log.is_file():
        print(f"  FAIL: no request log at {request_log}")
        return 1
    done = []
    for line in request_log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("event") == "request_done":
                done.append(record)
    # Nine requests are sent; done[0] is the first repeat (the seed), so the reported rows are
    # 1-indexed positions 2..9. The last two are the two shared conversations.
    if len(done) != 9:
        print(f"  FAIL: expected 9 completed requests, the log holds {len(done)}")
        return 1

    conditions = ["repeat(control)", "grow(question)", "shuffle(negative)", "marked seed",
                  "marked grow(fix)", "marked diverge", "shared conv A", "shared conv B(warm)"]
    picked = [done[1], done[2], done[3], done[4], done[5], done[6], done[7], done[8]]
    rows = []
    for label, record in zip(conditions, picked, strict=True):
        result = record["result"]
        rows.append((label, result["prompt_tokens"], result["prefix_cache_hit_tokens"],
                     result["prefix_reuse_path"]))

    header = f"  {'condition':20s} {'prompt_tok':>11} {'hit':>7}  path"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, tokens, hit, path in rows:
        print(f"  {label:20s} {tokens:11d} {hit:7d}  {path}")
    print()

    repeat_hit, grow_hit, shuffle_hit = rows[0][2], rows[1][2], rows[2][2]

    if repeat_hit == 0:
        print("  INCONCLUSIVE: the CONTROL did not hit. The loop cannot see a hit at all, so a zero")
        print("  on the question proves nothing. Do not read the rest of this output as a verdict.")
        return 2
    print(f"  control is valid: an exact repeat hits {repeat_hit} tokens, so the measurement works.")

    if shuffle_hit != 0:
        print(f"  FAIL: the negative control reported {shuffle_hit} hit tokens for a conversation")
        print("  sharing no prefix, so 'hit tokens' is not reporting a prefix match.")
        return 1

    if grow_hit == 0:
        print("  VERDICT, unmarked: the shared-prefix path is NOT taken. A conversation sharing every")
        print("  preceding turn exactly got 0 hit tokens and path root, while an exact repeat hit.")
    else:
        print(f"  VERDICT, unmarked: a shared prefix IS served. {grow_hit} hit tokens, path {rows[1][3]}.")

    marked_hit = rows[4][2]
    print()
    if marked_hit == 0:
        print("  VERDICT, marked: STILL 0 hit tokens with an explicit prompt_cache_breakpoint.")
        print("  So the marker is not reaching a write candidate either, and the cause is not the")
        print("  protocol's automatic policy. This loop is red for a real reason.")
        return 1
    print(f"  VERDICT, marked: a shared prefix IS served once the request asks for one. "
          f"{marked_hit} hit tokens, path {rows[4][3]}.")
    if grow_hit == 0:
        print()
        print("  SO: the cache is not broken. /v1/chat/completions suppresses the engine's automatic")
        print("  shared prefixes and plain string content carries no marker, so a growing conversation")
        print("  with no explicit breakpoint correctly gets root.")

    print()
    warm_hit, warm_path = rows[7][2], rows[7][3]
    if warm_hit == 0 or warm_path != "shared_stable_prefix":
        print(f"  WARM TTFT: NOT measurable. The second shared conversation got {warm_hit} hit tokens")
        print(f"  on path {warm_path}.")
        return 1
    print("  WARM TTFT IS MEASURABLE. Two different conversations sharing a declared system prefix:")
    print(f"  the second hit {warm_hit} tokens on path {warm_path} -- a cached prefix plus a real")
    print("  encode and decode of its own tail. That is exactly the shape ADR-0012's 'prefill served")
    print("  from cache' names, and it confirms ADR-0009's reading rather than contradicting it.")
    print()
    print("  tools/bench/warm_lane_sweep.py must therefore build TWO conversations that share a")
    print("  declared system prefix and diverge after it. Seeding with a shorter version of the SAME")
    print("  conversation cannot measure warm TTFT, because private_response_replay wins there.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
