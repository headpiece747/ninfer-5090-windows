#!/usr/bin/env python3
"""The profile table: one source of truth for what ships.

A **profile** is one shippable combination of artifact, spec route, vision and ceiling. This
module holds the table and the serve argument list, so the launcher generator, the measurement
harnesses and the verifier all describe the same thing instead of restating it.

Measured 2026-09-20 on an RTX 5090 (32 GB) through the shipped launcher's own flag set: the
`profile` mode of v3_profile_matrix.py, which composes profiles.ordered_flags. Each value therefore
describes the configuration a launcher actually starts, including its --device-state-slots. An
earlier probe omitted that flag and published a runtime/free pair for a configuration no launcher
starts.

The four values come from ONE interleaved window, three rounds in lane order (quasar dflash2, quasar
mtp4, nvfp4-full dflash2, nvfp4-full mtp5). Interleaving is not optional here: this machine's decode
varies by up to ~9% between windows, and both an earlier "state slots cost 14% of decode" and an
earlier "the fused artifact is 9% faster" were time-ordered comparisons -- they measured the window,
not the variable. Compare alternatives by alternating them; never measure one, then the other.

--device-state-slots is 1 for every profile, from a record: the reclaim that landed 2026-09-19
removed the #251 cliff, so all twelve conversations reuse at 1 as well as at 8 (`slots_sweep_*.txt`
under the bench records). Interleaved against 8, the value 1 costs 1.3 GiB less runtime and raises
acceptance (62.5% against 58.6%) at the same decode. A larger value buys retained-state capacity for
interleaved conversations, which nothing in this repo measures -- treat raising it as an unverified
trade, not as a fix.
Decode varies ~10% run to run and free VRAM ~0.2 GiB with whatever else holds the card.
Every value below is backed by a record in matrix_v3.jsonl under the launcher's own file name, and
every value is measured on the **published** artifact -- the one `download_model.py`'s pin resolves
to, hash-verified. A locally-upgraded copy is not a substitute: it binds the DFlash2 draft attention
differently (ADR-0003), so it produces different tokens. Its throughput is the same -- but do not
compare A against B by measuring one after the other, or this machine's ~8% between-window drift
will read as a finding. That happened here; ADR-0003's amendment records it. Interleave.
The PROFILES table is the only copy: this docstring deliberately restates no further figures,
because a hand-copied number is a drift site. The example table that used to stand here was already stale by
the time the launchers gained --device-state-slots, one day after those measurements.

All four are vision-only and all four reach the native context; the with-vs-without-vision
comparison that justified that is recorded in docs/adr/0004. Text-only variants existed only for
the retired NVFP4 image, which charged 16,384-27,008 tokens for it.

--lm-head-draft is a measured choice per artifact, not a convention: all four shipped profiles set
it today, and ADR-0005 records what it is worth on each -- including the one where it costs
headroom to buy speed. Route, depth and this flag are measurements, never convention.
"""
from __future__ import annotations

from typing import Any, Iterable

# The QUASAR artifact is our own quantisation-aware-trained image; NVFP4-full is the fuller one
# published by cometkim. "ninfer" alone is ambiguous and should not be used. See CONTEXT.md.
QUASAR = "qwen3_8_27b_nvfp4qat.v3.ninfer"
NVFP4FULL = "qwen3_8_27b_nvfp4full.v3.ninfer"

PROFILES: list[dict[str, Any]] = [
    dict(file="start_quasar_v3_dflash2_vision.bat", port=8086, art=QUASAR, device_state_slots=1,
         label="QUASAR QAT + DFlash2 + Vision", model_id="qwen3.8-27b-quasar-v3-dflash2-vision",
         spec="dflash2", draft=7, vision=True, lm_head=True, ctx=262144,
         tok=343.4, acc="62.5%", runtime="10.7 GiB", free="2.7 GiB",
         note="Fastest QUASAR lane at full context, at one state slot."),
    dict(file="start_quasar_v3_mtp4_vision.bat", port=8087, art=QUASAR, device_state_slots=1,
         label="QUASAR QAT + MTP4 + Vision", model_id="qwen3.8-27b-quasar-v3-mtp4-vision",
         spec="mtp", draft=4, vision=True, lm_head=True, ctx=262144,
         tok=219.5, acc="58.3%", runtime="10.4 GiB", free="3.3 GiB",
         note="Lower-VRAM QUASAR profile. MTP depth 4 measured fastest of 2-5 on QUASAR."),
    dict(file="start_ninfer_v3_dflash2_vision.bat", port=8088, art=NVFP4FULL, device_state_slots=1,
         label="NVFP4-full + DFlash2 + Vision", model_id="qwen3.8-27b-nvfp4-v3-dflash2-vision",
         spec="dflash2", draft=7, vision=True, lm_head=True, ctx=262144,
         tok=344.6, acc="63.7%", runtime="10.7 GiB", free="2.0 GiB",
         note="Second artifact, same reach as QUASAR: 262,144 with Vision, at one state slot."),
    dict(file="start_ninfer_v3_mtp5_vision.bat", port=8089, art=NVFP4FULL, device_state_slots=1,
         label="NVFP4-full + MTP5 + Vision", model_id="qwen3.8-27b-nvfp4-v3-mtp5-vision",
         spec="mtp", draft=5, vision=True, lm_head=True, ctx=262144,
         tok=253.5, acc="64.2%", runtime="10.4 GiB", free="2.6 GiB",
         note="MTP lane on the second artifact. Depth 5 measured fastest of 2-5 here."),
]

# Flags every profile ships, in the order the launcher renders them. A value of None marks a
# flag that takes no argument. Keep this list in step with the launcher template: the generator
# renders it verbatim, and regeneration is expected to be byte-identical.
INVARIANT_FLAGS: list[tuple[str, str | None]] = [
    ("--kv-capacity", "auto"),
    ("--kv-dtype", "fp8"),
    ("--prefill-chunk", "8192"),
    ("--max-concurrency", "1"),
    ("--host-state-slots", "16"),
    ("--host-kv-mib", "8192"),
    ("--max-shared-prefixes", "7"),
    ("--max-private-continuations", "8"),
    ("--max-long-anchors-per-continuation", "4"),
    ("--context-cache-policy", "rolling"),
    ("--preserve-thinking", None),
    ("--default-thinking-budget", "4096"),
    ("--pending-timeout-ms", "600000"),
]


def by_file(name: str) -> dict[str, Any]:
    for profile in PROFILES:
        if profile["file"] == name:
            return profile
    raise KeyError(f"no profile shipping {name}")


def by_model_id(model_id: str) -> dict[str, Any]:
    for profile in PROFILES:
        if profile["model_id"] == model_id:
            return profile
    raise KeyError(f"no profile with model id {model_id}")


def varying_flags(profile: dict[str, Any]) -> list[str]:
    """The per-profile flags, each rendered as one line on the launcher's continuation chain."""
    out: list[str] = []
    if profile["vision"]:
        out.append("--vision")
    if profile["spec"] != "none":
        out.append(f"--spec {profile['spec']}")
        out.append(f"--draft-tokens {profile['draft']}")
        if profile["lm_head"]:
            out.append("--lm-head-draft")
    return out


def ordered_flags(profile: dict[str, Any]) -> list[tuple[str, str | None]]:
    """Every flag the launcher passes, in shipped order.

    The generator renders this verbatim and regeneration is checked for byte-identity, so the
    positions matter: the per-profile flags come first, then the bound ones.
    """
    out: list[tuple[str, str | None]] = [(token, None) for token in varying_flags(profile)]
    # --device-state-slots is VRAM-bound, so it stays a per-profile field rather than a shared
    # constant, but every profile ships 1. It is "extra checkpoint capacity beyond active lanes":
    # how many conversations can keep a cached state at once. Until 2026-09-19 the engine had no
    # eviction, so once these were exhausted prefix reuse stopped permanently until restart
    # (upstream issue #251, "no LRU eviction observed"; reproduced here with 1: reuse held for 3
    # conversations, then every later request re-prefilled from the root, silently). Active-capture
    # admission now reclaims the oldest unpinned private continuation to free a slot, so the cliff
    # is gone and the value is a capacity/VRAM trade rather than a survival requirement: 12/12
    # conversations reuse at 1, 2, 4 and 8 alike, while 8 costs ~1.3 GiB of runtime and 13.6% of
    # decode on QUASAR DFlash2. A larger value buys retained-state capacity for interleaved
    # conversations, which nothing here measures -- see the module docstring.
    #
    # --host-state-slots is the bound that decides whether a conversation's prefixes survive at all,
    # and it has to cover the live shared prefixes plus the live private continuations: the five-prompt
    # resend test gives 3/5 hits at 59.1% at 8 with the private bound at 8 (five prefixes plus five
    # continuations against eight slots, so the last two prefixes are evicted), 5/5 at 98.5% at 12, and
    # 5/5 at 98.6% either way when the private bound drops to 2 -- the private bound is the wrong knob,
    # because an agent session forks on every tool call and each fork is a continuation that needs a
    # slot. Measured 2026-09-21 through start_bounds.cmd; raising this costs nothing at startup (the
    # same 2,740 MiB free and 10,996 MiB reserved at 8, 12 and 16) because it only decides how much of
    # the already-pinned 8 GiB host KV may be used.
    #
    # --context-cache-policy rolling ships enabled: once the pools are full, publishing a conversation's
    # newest checkpoint means replacing a resident, and by default that needs two matching reuse domains
    # or explicit evidence -- which one append-only conversation never has, so its reusable frontier
    # pins (measured: 52,723 tokens, TTFT 2.6 s -> 15.1 s at 65k -> 117k context). `rolling` takes the
    # standing from the lineage the request already proved, and the frontier then tracks the
    # conversation (104,283 of 117,180, TTFT 4.1 s). It was proposed upstream as opt-in for shared
    # multi-tenant servers, where it can evict a prefix another conversation still wants; this is a
    # local single-owner product, and two saturating conversations measured byte-identical to the
    # default policy, so there is no second tenant to protect here.
    out.extend([("--host", "127.0.0.1"),
                ("--port", str(profile["port"])),
                ("--model-id", profile["model_id"]),
                ("--max-context", str(profile["ctx"])),
                ("--device-state-slots", str(profile["device_state_slots"]))])
    out.extend(INVARIANT_FLAGS)
    return out


def launcher_args(profile: dict[str, Any], port: int | None = None,
                  model_id: str | None = None, max_context: int | None = None) -> list[str]:
    """The serve argument list for a profile, exactly as the launcher passes it.

    This is the interface the measurement harnesses should use. When they build their own
    argument list instead, their records stop describing what ships -- which happened: the
    cache bounds, the thinking budget and the model ids were all missing from the harness, so
    its numbers came from flags no launcher ships.

    Three overrides exist because a harness genuinely differs from a launcher:

    - ``port``: a harness must bind its own port, because the shipped one may already be
      serving. Every harness overrides it.
    - ``model_id``: the engine enforces the id on every request, so a harness needs one it
      controls. Every harness overrides it.
    - ``max_context``: the effort probe uses a small context. One caller.

    Anything else a harness needs -- a request log, a different thinking budget -- it appends,
    which is why this returns a plain list.
    """
    args: list[str] = []
    for flag, value in ordered_flags(profile):
        if flag == "--port" and port is not None:
            value = str(port)
        elif flag == "--model-id" and model_id is not None:
            value = model_id
        elif flag == "--max-context" and max_context is not None:
            value = str(max_context)
        if value is None and " " not in flag:
            args.append(flag)
        else:
            args.extend(flag.split())
            if value is not None:
                args.append(value)
    return args


def cli_args(profile: dict[str, Any]) -> list[str]:
    """The flag subset the offline CLI accepts, for test_prompt.bat and test_vision.bat.

    The CLI is a different interface from the server: it takes a prompt or messages file and
    rejects --host, --port, --model-id, --max-concurrency, the state slots, the host KV pool,
    the cache bounds and the timeouts. Generating the server's argument list into QUASAR_ARGS
    was wrong for exactly that reason, and running the helper is what showed it.

    The flags kept here are the ones both interfaces share, plus the proposal head, which the
    CLI does accept and the earlier hand-written value omitted.
    """
    out: list[str] = []
    if profile["vision"]:
        out.append("--vision")
    if profile["spec"] != "none":
        out.extend(["--spec", profile["spec"], "--draft-tokens", str(profile["draft"])])
        if profile["lm_head"]:
            out.append("--lm-head-draft")
    out.extend(["--max-context", str(profile["ctx"]),
                "--kv-capacity", "auto",
                "--kv-dtype", "fp8",
                "--prefill-chunk", "8192"])
    return out


def all_profiles() -> Iterable[dict[str, Any]]:
    return iter(PROFILES)
