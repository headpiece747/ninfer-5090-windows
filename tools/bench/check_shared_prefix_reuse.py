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
import os
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

BODY_WORDS = 400
MESSAGES = 6
SYSTEM = "You are a concise assistant that answers in one sentence."

# ADR-0009's working recipe names THREE shared things: "two sessions that share a system message and
# one tool definition, differing in their first user turn". The tool definition is the ingredient every
# arm of mine has been missing, and it is not cosmetic:
#
#   - `explicit_write_slots` is `kMaximumExplicitPromptCacheMarkers` (four) minus one when the
#     automatic target does not coincide with an explicit one (openai_common.cpp:147-149). A request
#     declaring a breakpoint on a system message and carrying tools spends its slots differently from
#     one that does not.
#   - The frontend's leading boundary comes from `leading_instruction_boundary(message_roles)`, and
#     the tool contract is built before it (frontend.cpp:733-738), so tools move where the shared
#     region ends.
#
# Measured without a tool: shared_reuse_candidates 0, shared_reuse_key_mismatch 0 -- no shared entry
# reached a reuse plan at all. This arm is the recipe ADR-0009 actually recorded as served.
TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_term",
        "description": "Look up a technical term and return its definition.",
        "parameters": {
            "type": "object",
            "properties": {"term": {"type": "string", "description": "The term to look up."}},
            "required": ["term"],
        },
    },
}


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


def body_for(
    messages: list[dict[str, str]],
    seed: int,
    marked_upto: int = -1,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a request body. `marked_upto` marks the first N messages as parts with a breakpoint."""
    if marked_upto < 0:
        out = list(messages)
    else:
        out = [as_part(m, index < marked_upto) for index, m in enumerate(messages)]
    body: dict[str, Any] = {
        "model": "qwen3.8-27b-quasar-v3-dflash2-vision",
        "messages": out,
        "max_tokens": 16,
        "temperature": 0.0,
        "seed": seed,
        "enable_thinking": False,
        "stream": False,
    }
    if tools:
        body["tools"] = tools
        # An OpenAI request carrying tools needs a choice, and the default forces a function call.
        body["tool_choice"] = "none"
    return body


def post(port: int, body: dict[str, Any], timeout_s: float, protocol: str = "openai") -> None:
    path = "/v1/messages" if protocol == "anthropic" else "/v1/chat/completions"
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s):
        pass


def anthropic_body(
    first_user: str, tail: list[dict[str, str]], ask: str, system_text: str | None = None
) -> dict[str, Any]:
    """The same two-conversation shape on /v1/messages, with cache_control on the SYSTEM block.

    The discriminator for a real defect, not a harness one. On the OpenAI route a declared breakpoint
    lands on a content PART, and the marker that produces -- MessagePartBoundary -- resolves to no
    frontier at all: native_render.cpp:629-631 leaves `LeadingInstructionBoundary` and
    `MessagePartBoundary` unset, and so does the Jinja path by design ("the same answer the Jinja path
    gives when the layout cannot place them"). The OpenAI parser can set cache_boundary_after only on
    a content part (openai_chat_request.cpp:352), so on that protocol a declared boundary never reaches
    a frontier.

    The Anthropic parser sets it on the MESSAGE (anthropic_messages_request.cpp:266, 332), which
    produces MessageBoundary -- and that location does resolve. If this arm hits and the OpenAI one
    does not, the cause is located in the marker location rather than in the cache.
    """
    system = {
        "type": "text",
        "text": system_text or SYSTEM,
        "cache_control": {"type": "ephemeral"},
    }
    return {
        "model": "qwen3.8-27b-quasar-v3-dflash2-vision",
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": False,
        # The shared region is the SYSTEM block plus any tail that both requests carry identically.
        # With an empty tail the region is just the system message, which is only ~50 tokens -- enough
        # to be served, but not enough to measure a TTFT against. Pass the shared turns here.
        "system": [system],
        "messages": [{"role": "user", "content": f"{first_user} {ask}"}]
        + [{"role": "assistant", "content": turn["content"]} for turn in tail],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--exe", type=Path, default=REPO_ROOT / "build" / "apps" / "ninfer-serve.exe")
    parser.add_argument("--port", type=int, default=8280)
    parser.add_argument(
        "--stats-interval-ms",
        type=int,
        default=500,
        help="throughput record interval; the cache-selection counters ride on it",
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "profiles" / "shared_prefix")
    arguments = parser.parse_args()

    arguments.log_dir.mkdir(parents=True, exist_ok=True)
    request_log = arguments.log_dir / "shared_prefix.requests.jsonl"
    server_log = arguments.log_dir / "shared_prefix.log"
    if request_log.exists():
        request_log.unlink()

    command, _ = serve_command(arguments.exe, arguments.artifact, arguments.port, False, request_log)
    # A throughput record is the ONLY place the context-cache selection counters are written, and it
    # is emitted either on the --log-stats-interval timer or in the shutdown tail. The tail alone is
    # not enough in practice: the periodic record keeps the counters available even if the graceful
    # stop is skipped, and it makes them land mid-run rather than only at the end. Measured without
    # this flag: 0 throughput records, the counters absent, and a wrong conclusion drawn from it.
    command += ["--log-stats-interval-ms", str(arguments.stats_interval_ms)]
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
        #
        # The system block is deliberately LARGE. At the original 79 tokens the shared prefix was
        # served but far too small to compare against ADR-0012's 229-message projection. The block is
        # byte-identical in both requests; only the user turn below it differs, so all of it is shared.
        big_system = SYSTEM + " " + (
            "You answer from the supplied context only, you prefer one sentence, and you never "
            "repeat the question. "
        ) * 400
        shared_tail = conversation(MESSAGES)[1:-1]
        for tag in ("ALPHA", "BRAVO"):
            messages = system_then(
                f"{tag}: what is a split-K GEMM?", shared_tail, "Reply with one short sentence."
            )
            post(arguments.port, body_for(messages, 1, marked_upto=1, tools=[TOOL]), arguments.timeout)
        # 7. the DISCRIMINATOR: the same shape on /v1/messages, where the declaration lands on the
        # system MESSAGE and so resolves to a MessageBoundary. If this hits and the OpenAI arms do
        # not, the cause is the marker location the OpenAI parser can produce.
        for tag in ("GAMMA", "DELTA"):
            post(
                arguments.port,
                # NO shared tail: the two requests differ in their first user turn, so the region the
                # system block opens is the only thing they share. That is the whole point of the
                # arm -- a shared prefix that ends where the two conversations part company.
                anthropic_body(
                    f"{tag}: what is a split-K GEMM?", [], "Reply briefly.", system_text=big_system
                ),
                arguments.timeout,
                protocol="anthropic",
            )
    except urllib.error.HTTPError as error:
        print(f"  FAIL: the engine refused a request: {error.code} "
              f"{error.read().decode('utf-8', 'replace')[:200]}")
        return 1
    finally:
        # stop_lane, not terminate(): the context-cache selection counters this loop exists to read
        # are flushed in the shutdown throughput record, and TerminateProcess on Windows skips it.
        stop_lane(process)
        handle.close()

    if not request_log.is_file():
        print(f"  FAIL: no request log at {request_log}")
        return 1
    json_lines = [
        json.loads(line) for line in request_log.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    done = [record for record in json_lines if record.get("event") == "request_done"]
    # Nine requests are sent; done[0] is the first repeat (the seed), so the reported rows are
    # 1-indexed positions 2..9. The last two are the two shared conversations.
    if len(done) != 11:
        print(f"  FAIL: expected 11 completed requests, the log holds {len(done)}")
        return 1

    conditions = ["repeat(control)", "grow(question)", "shuffle(negative)", "marked seed",
                  "marked grow(fix)", "marked diverge", "shared conv A", "shared conv B",
                  "anthropic A", "anthropic B(warm)"]
    picked = [done[1], done[2], done[3], done[4], done[5], done[6], done[7], done[8],
              done[9], done[10]]
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

    # The selection counters say WHICH gate refused, which no per-request field can. They are only
    # on a throughput record, so this is also the check that the lane emitted one at all.
    #
    # Order matters and the first version got it wrong: a periodic record can predate the final
    # request, so a warm HIT above was reported alongside all-zero counters, which read as a
    # contradiction. Only report counters from a record at least as late as the last request, and
    # say how many requests they cover, so a stale record cannot be quoted as the attribution.
    throughput = [r for r in json_lines if r.get("event") == "throughput"]
    if not throughput:
        print("  The selection counters are absent: no throughput record was written, so this run")
        print("  cannot say WHY a read was refused. Re-run with a larger --stats-interval-ms.")
        return 1
    last_request_ms = max(
        (r.get("timestamp_unix_ms") or 0) for r in json_lines if r.get("event") == "request_done"
    )
    fresh = [r for r in throughput if (r.get("timestamp_unix_ms") or 0) >= last_request_ms]
    if not fresh:
        # The counters explain a REFUSAL. They are not needed when the per-request fields already
        # show the hit, and demanding a fresh record there would fail a run whose verdict is solid.
        stale_counters = (throughput[-1].get("context_cache") or {}).get("selections") or {}
        print()
        print("  NOTE: the last throughput record predates the final request, so its counters do not")
        print("  cover this run and are not quoted as the attribution. Raise --stats-interval-ms to")
        print(f"  see them (the stale record holds shared_stable_prefix="
              f"{stale_counters.get('shared_stable_prefix', '(absent)')}).")
        print()
    else:
        selections = (fresh[-1].get("context_cache") or {}).get("selections") or {}
        print()
    if fresh:
        print("  cache selection counters (from a throughput record at least as late as the last"
              " request):")
        for name in (
            "shared_stable_prefix",
            "shared_reuse_candidates",
            "shared_reuse_declined",
            "shared_reuse_key_mismatch",
        ):
            print(f"    {name:28s} {selections.get(name, '(absent)')}")
        print()
    # The discriminator, read before the verdict: a hit on one protocol and not the other locates the
    # cause in the marker location the parser can produce, not in the cache.
    anthropic_hit, anthropic_path = rows[9][2], rows[9][3]
    print()
    print("  DISCRIMINATOR: the same shape on /v1/messages (declaration on the system MESSAGE) got")
    print(f"  {anthropic_hit} hit tokens on path {anthropic_path}.")
    if anthropic_hit != 0:
        print("  So the cache serves a shared prefix when the declaration resolves to a frontier,")
        print("  and the OpenAI arms fail because parse_openai_prompt_cache_breakpoint can only set")
        print("  the boundary on a content PART -- MessagePartBoundary -- which native_render.cpp:629")
        print("  leaves unset by design, so no frontier exists to write at.")
    else:
        print("  So it is not the marker location either: both protocols decline. The cause is below")
        print("  the frontend, and the counters are the next place to look.")

    # Warm TTFT is measurable on whichever arm reached a real cached-prefix decode. Prefer the
    # OpenAI shape, and fall back to the Anthropic one, because the latter is the arm that works.
    warm_source = "openai"
    warm_hit, warm_path = rows[7][2], rows[7][3]
    if warm_hit == 0 and anthropic_hit != 0:
        warm_source = "anthropic"
        warm_hit, warm_path = anthropic_hit, anthropic_path
        print()
        print(f"  Warm TTFT is taken from the {warm_source} arm: it is the one that reaches a cached")
        print("  prefix with a real decode behind it.")

    warm_ok = warm_hit != 0 and warm_path == "shared_stable_prefix"
    if not warm_ok:
        # Only quote a counter as the attribution when a record actually covers the run. Quoting a
        # stale one is how this script once printed all-zero counters directly beneath a warm hit.
        if not fresh:
            print("  NOT ATTRIBUTED: no throughput record covers the final request, so the counters")
            print("  cannot say which gate refused. Raise --stats-interval-ms.")
        elif selections.get("shared_reuse_key_mismatch"):
            print("  ATTRIBUTED: a shared index entry was skipped before planning -- no shortlist key")
            print("  matched at its frontier (resource_manager.h:329). The incoming prompt does not")
            print("  hash to the resident prefix there, so the pair above never became a candidate.")
        elif selections.get("shared_reuse_declined"):
            print("  ATTRIBUTED: a shared entry reached the plan and the Program refused it.")
        elif selections.get("shared_reuse_candidates"):
            print("  ATTRIBUTED: a shared entry reached the plan unrefused, and another path won.")
        else:
            print("  ATTRIBUTED: no shared entry reached a reuse plan at all in this run.")
        print()
        print(f"  WARM TTFT: NOT measurable. The second shared conversation got {warm_hit} hit tokens")
        print(f"  on path {warm_path}.")
        return 1

    print(f"  WARM TTFT IS MEASURABLE, on the {warm_source} arm. Two different conversations sharing a")
    print(f"  declared system prefix: the second hit {warm_hit} tokens on path {warm_path} -- a cached")
    print("  prefix with a real encode and decode of its own tail behind it. That is exactly the shape")
    print("  ADR-0012's 'prefill served from cache' names.")
    print()
    print("  For warm_lane_sweep.py, two things follow and both were the wrong way round before:")
    print("    - the declaration must resolve to a frontier, so on /v1/chat/completions it has to be")
    print("      carried by a MESSAGE, which the OpenAI parser cannot do; a content part resolves to")
    print("      MessagePartBoundary, which the renderer leaves unset. Use /v1/messages, or accept")
    print("      that the OpenAI route has no declared shared prefix at all.")
    print("    - the two requests must be DIFFERENT conversations. Seeding with a shorter version of")
    print("      the same one takes private_response_replay, which never decodes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
