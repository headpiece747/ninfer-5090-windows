#!/usr/bin/env python3
"""The profile table: one source of truth for what ships.

A **profile** is one shippable combination of artifact, spec route, vision and ceiling. This
module holds the table and the serve argument list, so the launcher generator, the measurement
harnesses and the verifier all describe the same thing instead of restating it.

Measured 2026-09-28 on an RTX 5090 (32 GB) through the shipped launcher's own flag set: the
`profile` mode of v3_profile_matrix.py, which composes profiles.ordered_flags. Each value therefore
describes the configuration a launcher actually starts, including its --device-state-slots. An
earlier probe omitted that flag and published a runtime/free pair for a configuration no launcher
starts.

The eight values come from ONE interleaved window, three rounds, each round visiting all eight lanes
in a rotated order so every lane held a different position. Interleaving is not optional here: this
machine's decode varies by up to ~9% between windows, and both an earlier "state slots cost 14% of
decode" and an earlier "the fused artifact is 9% faster" were time-ordered comparisons -- they
measured the window, not the variable. Compare alternatives by alternating them; never measure one,
then the other. With the harness corrected (below), every lane's three rounds span 1.5% or less, each
returns one digest across all three, and no position effect: position 0 and position 7 read the same.

**The harness used to measure its own warmup.** The first full-length decode after a server start is a
transient: it returns faster than every later identical request, and different and shorter text, while
requests 2..n are byte-identical. On the NVIDIA MTP5 lane it read 261.7 tok/s against 169.8 for
requests 2-7, so averaging it into three runs published ~200 tok/s for a lane that delivers ~166 --
and that inflated figure is the one this table used to carry. A 16-token probe does not reach the
state the transient affects, so `measure_decode` now discards one full-length warmup and
`parse_spec_jsonl` skips it, because the engine appends to its request log for the whole session. That
harness fix, not a change in the lanes, is the whole difference between the 2026-09-20 numbers and
these. Every acceptance figure moved for the same reason: the transient request was in the
denominator and the numerator.

--device-state-slots is 1 for every profile, from a record: the reclaim that landed 2026-09-19
removed the #251 cliff, so all twelve conversations reuse at 1 as well as at 8 (`slots_sweep_*.txt`
under the bench records). Interleaved against 8, the value 1 costs 1.3 GiB less runtime and raises
acceptance (62.5% against 58.6%) at the same decode. A larger value buys retained-state capacity for
interleaved conversations, which nothing in this repo measures -- treat raising it as an unverified
trade, not as a fix.
Decode varies ~10% between windows, and free VRAM ~0.2 GiB with whatever else holds the card. Within
one interleaved window, with the warmup transient excluded, it varies 1.5% or less.
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

import os
from pathlib import Path
from typing import Any, Iterable


def template_path() -> str:
    """The lane's template in the source tree.

    A launcher resolves `%TEMPLATE%` beside itself; a harness runs in the source tree, so it points
    at the same maintained file there. Every consumer of `INVARIANT_FLAGS` must go through this:
    leaving the literal cmd variable on a command line is a startup failure, not a measurement.
    """
    return str(Path(__file__).resolve().parents[2] / "tools" / "chat_templates" / "qwen3_8.jinja")

# The QUASAR artifact is our own quantisation-aware-trained image; NVFP4-full is the fuller one
# published by cometkim. "ninfer" alone is ambiguous and should not be used. See CONTEXT.md.
QUASAR = "qwen3_8_27b_nvfp4qat.v3.ninfer"
NVFP4FULL = "qwen3_8_27b_nvfp4full.v3.ninfer"
# The same unsloth line with its nine BF16 exception parents encoded to NVFP4. It is a separate image
# rather than a replacement because the two encodings measure better on different routes, which is
# the same rule the DFlash2 draft encodes: a pattern is measured per target. On this line the BF16
# exceptions are worth +21.4 acceptance points to the MTP head and cost the DFlash2 route 11.6, and
# only the no-exception build fits the native 262,144 with Vision.
NVFP4FULLNOEX = "qwen3_8_27b_nvfp4full_noex.v3.ninfer"
# UkisAI's Swift 1.5 finetune, re-encoded by this port. It carries both Swift lanes.
#
# Swift 1.0's image is gone from this table rather than renamed: `SWIFT` was withdrawn in the same
# wave that moved the lanes, because a constant naming a retired artifact is reachable surface that
# reads as current. Its file is kept under models/_superseded/ and the matrix harness still addresses
# it there, since every Swift 1.5 figure in docs/research/swift15-lane-measurement.md is quoted
# against it as a same-day control.
#
# Two things distinguish this build from Swift 1.0's and both are measured rather than inherited.
# Its ModelOpt export is structurally identical -- 401 sites, the same 193 NVFP4 MLP and 208 FP8
# attention/GDN names, only the producer string moved from 0.47.0rc0 to 0.47.0rc1.dev90 -- so
# qwen3_8_27b_nvfp4_swift needed no change. And its DFlash2 draft is encoded to NVFP4, which Swift
# 1.0's left at Q8 because on Swift 1.0 that change lost 3.2 acceptance points; on Swift 1.5 it wins
# on three domains of five and, more decisively, is 0.77 GiB smaller, which is exactly the margin
# that puts the native 262,144 back within reach on the DFlash2 + Vision lane.
SWIFT15 = "qwen3_8_27b_nvfp4swift15.v3.ninfer"
NVIDIA = "qwen3_8_27b_nvfp4nvidia.v3.ninfer"

PROFILES: list[dict[str, Any]] = [
    dict(file="start_quasar_v3_dflash2_vision.bat", port=8086, art=QUASAR, device_state_slots=1,
         label="QUASAR QAT + DFlash2 + Vision", model_id="qwen3.8-27b-quasar-v3-dflash2-vision",
         spec="dflash2", draft=7, vision=True, lm_head=True, ctx=262144,
         tok=319.2, acc="55.0%", runtime="10.3 GiB", free="2.82 GiB",
         note="Fastest QUASAR lane at full context, at one state slot. Re-measured 2026-09-30 "
              "across all eight lanes. The 2026-09-24 figures this replaces (310.3/52.5%, runtime "
              "10.3 GiB) were taken before the workspace grew and no longer describe this lane; at "
              "the shipped configuration every DFlash2 lane reads 11.6 GiB and about 1.50 GiB free, "
              "and this row is the measured one. Two earlier figures on this lane, 343.4/62.5% and "
              "the 45.7% published-file reading, had already been withdrawn as not reproducing."),
    dict(file="start_quasar_v3_mtp4_vision.bat", port=8087, art=QUASAR, device_state_slots=1,
         label="QUASAR QAT + MTP4 + Vision", model_id="qwen3.8-27b-quasar-v3-mtp4-vision",
         spec="mtp", draft=4, vision=True, lm_head=True, ctx=262144,
         tok=221.2, acc="59.6%", runtime="9.96 GiB", free="3.40 GiB",
         note="Depth 5 shipped here from 2026-09-28 records that no longer reproduce, and re-swept "
              "2026-09-30 over depths 1-5 on five domains at three interleaved rounds each, this "
              "depth changes to 4. d5 wins two domains and loses three, badly: code 183.0 against "
              "d4's 220.4 and chinese 113.2 against 129.0. On maximin -- the rule this table is "
              "chosen by, since the worst domain decides -- d4's worst case is 0.0% and d5's is "
              "-17.0%, so d5's best case does not buy its worst. d5 is still faster on prose (121.8 "
              "against 116.5) and dialogue (159.6 against 155.1), which is the reversal that makes "
              "the code domain alone the wrong single domain to read."),
    dict(file="start_ninfer_v3_dflash2_vision.bat", port=8088, art=NVFP4FULLNOEX, device_state_slots=1,
         label="NVFP4-full + DFlash2 + Vision", model_id="qwen3.8-27b-nvfp4-v3-dflash2-vision",
         spec="dflash2", draft=7, vision=True, lm_head=True, ctx=262144,
         tok=296.4, acc="48.7%", runtime="10.3 GiB", free="2.85 GiB",
         note="This lane is REFUSED at startup on the BF16-exception image and was, until "
              "2026-09-30. The refusal's own arithmetic: at 262,144 with Vision it needs 11.63 GiB "
              "of reservation plus 1 GiB of automatic headroom against 12.33 GiB available after "
              "weights -- short by 308 MiB. Encoding the nine BF16 exception parents to NVFP4 is "
              "0.7 GiB and clears it, at 17.2 GiB of device weights against 17.9. That costs "
              "0.087 % perplexity overall (5.002751 against 4.998419, same binary same day) and "
              "buys back the native context plus 24.8 % throughput and 11.6 acceptance points on "
              "this route. The exceptions stay on the MTP lane below, where they are worth far more "
              "than they cost here. Note that the artifact fix is not the only thing standing between this lane and "
              "its context: before the tiled saturation guard, every DFlash2 lane read 11.6 GiB and "
              "about 1.50 GiB free, so the split into two images remained necessary and the margin on "
              "this class of lane was thin. The guard now returns about 1.35 GiB, and "
              "`--prefill-chunk 4096` -- measured and rejected, it returns the same memory for -2.3% "
              "prefill -- is no longer needed to improve it."),
    dict(file="start_ninfer_v3_mtp4_vision.bat", port=8089, art=NVFP4FULL, device_state_slots=1,
         label="NVFP4-full + MTP4 + Vision", model_id="qwen3.8-27b-nvfp4-v3-mtp4-vision",
         spec="mtp", draft=4, vision=True, lm_head=True, ctx=262144,
         tok=190.3, acc="51.4%", runtime="9.96 GiB", free="2.96 GiB",
         note="MTP lane on the second artifact, and the one lane where the BF16 exception "
              "projections earn their keep: encoding them to NVFP4 to fit the DFlash2 lane's "
              "context costs that lane acceptance points and throughput, measured interleaved "
              "against the no-exception build. Re-swept 2026-09-30 over depths 1-5 on five domains "
              "at three interleaved rounds each, this depth changes from 5 to 4: d4 wins prose "
              "(120.3 against 110.0), chinese (125.3 against 117.4) and dialogue (166.3 against "
              "140.9), and loses only code (190.6 against 204.8) and repetition by 2.9%, so its "
              "worst case is -2.9% against d5's -7.5%. **The first pass of this sweep read the wrong "
              "artifact**: `widths --art ninfer` resolves to cometkim's base `qwen3_8_27b_nvfp4`, "
              "which is not a shipping lane, while this lane ships `nvfp4full`. Swept on the shipped "
              "image the verdict is the same, but it had to be re-measured to be known -- the two "
              "images differ by 26 tok/s on the same configuration, which is wider than the depth "
              "effect being decided. This lane's 2.96 GiB free is the fleet's tightest, because it "
              "carries the BF16 exceptions and so has the heaviest weights at the same runtime."),
    dict(file="start_swift_v3_dflash2_vision.bat", port=8090, art=SWIFT15, device_state_slots=1,
         label="Swift 1.5 + DFlash2 + Vision", model_id="qwen3.8-27b-swift15-v3-dflash2-vision",
         spec="dflash2", draft=7, vision=True, lm_head=True, ctx=262144,
         tok=362.0, acc="63.0%", runtime="10.3 GiB", free="2.85 GiB",
         note="Swift 1.5 replaces Swift 1.0 on both Swift lanes; these figures are measured "
              "2026-09-30 with `profile` mode through this launcher's own flags. Width 7 was measured "
              "against every window 1-15 on the code domain and re-measured on four others: 13 is "
              "33% faster on code and slower on chinese, prose and dialogue, so the width that "
              "maximises the worst domain is the one already shipped. This image encodes the DFlash2 "
              "draft to NVFP4 where Swift 1.0's left it at Q8, which is 0.77 GiB smaller and is "
              "exactly the margin that puts the native 262,144 back in reach -- the Q8 build of this "
              "same checkpoint is REFUSED at 262,144 with Vision. The encoding is measured per "
              "target and it reverses here: on Swift 1.0 the same change lost 3.2 acceptance points, "
              "so it was measured rather than assumed. The Q8-draft build of this same checkpoint is REFUSED at "
               "262,144 with Vision, so this encoding is load-bearing today; with the tiled "
               "saturation guard in place this lane now reads 2.85 GiB free, so whether the Q8 build "
               "would now also fit is worth re-probing -- if it would, the encoding choice and the 13.7 "
               "acceptance points it buys are both open to re-examination."),
    dict(file="start_swift_v3_mtp4_vision.bat", port=8091, art=SWIFT15, device_state_slots=1,
         label="Swift 1.5 + MTP4 + Vision", model_id="qwen3.8-27b-swift15-v3-mtp4-vision",
         spec="mtp", draft=4, vision=True, lm_head=True, ctx=262144,
         tok=218.3, acc="58.7%", runtime="9.96 GiB", free="3.43 GiB",
         note="Depth 4 re-measured 2026-09-30 against depths 1-5 on four domains. MTP is hard-capped "
              "at 5 by kMaximumMtpDraftTokens, so docs/active-work.md item 8's proposed window of 10 "
              "cannot be run on this tree at all. Depth 5 is faster on code and slower on prose, "
              "dialogue and repetition, so depth 4 stands. This lane is unaffected by the draft "
              "encoding above, which the measurement confirms: the Q8 and NVFP4 builds read 215.7 "
              "and 216.0 tok/s on the same configuration."),
    dict(file="start_nvidia_v3_dflash2_vision.bat", port=8092, art=NVIDIA, device_state_slots=1,
         label="NVIDIA ModelOpt + DFlash2 + Vision", model_id="qwen3.8-27b-nvidia-v3-dflash2-vision",
         spec="dflash2", draft=7, vision=True, lm_head=True, ctx=262144,
         tok=337.5, acc="56.2%", runtime="10.3 GiB", free="2.91 GiB",
         note="NVIDIA's ModelOpt quantization of the base model, built by this port: its NVFP4 MLP "
              "imported on all 64 layers and its FP8 attention re-encoded from the BF16 base. Same "
              "full-corpus perplexity as the official stock at 20% smaller, with no FP8 tensor where "
              "that has 146, and this lane reaches the full context where it caps below. The "
              "349.2/61.5% recorded until 2026-09-30 was taken 2026-09-24, before the workspace "
              "grew; this lane now reads 10.3 GiB and 2.91 GiB free, and its acceptance is 56.2% with "
              "a byte-identical digest across the tiled saturation guard, so the gap is the stale "
              "record rather than the configuration."),
    dict(file="start_nvidia_v3_mtp4_vision.bat", port=8093, art=NVIDIA, device_state_slots=1,
         label="NVIDIA ModelOpt + MTP4 + Vision", model_id="qwen3.8-27b-nvidia-v3-mtp4-vision",
         spec="mtp", draft=4, vision=True, lm_head=True, ctx=262144,
         tok=210.1, acc="53.1%", runtime="9.96 GiB", free="3.45 GiB",
         note="Depth 4 measured fastest of 2-5 here, and the largest correction in the table: d5 read "
              "228.3 on the 2026-09-17 records, which the warmup transient accounts for almost "
              "entirely, and measures 167.8 with it excluded. Depth 4 against depth 5 is 223.4 "
              "against 167.8, +33%, and d4 also accepts 61.4% against 38.2% -- the one lane where "
              "acceptance and throughput agree, which is why the contaminated figure looked "
              "plausible. Re-measured 2026-09-28, interleaved two rounds. **The 223.4/61.4% no "
              "longer reproduces either**: 2026-09-30 reads 208.7 at 53.1%, and 53.1% is what this "
              "lane returns at --prefill-chunk 4096 as well, with a byte-identical digest, so the "
              "depth comparison rests on the same stale population as QUASAR's and has not been "
              "re-measured since."),
]

# Flags every profile ships, in the order the launcher renders them. A value of None marks a
# flag that takes no argument. Keep this list in step with the launcher template: the generator
# renders it verbatim, and regeneration is expected to be byte-identical.
#
# `--prefill-chunk` is 8192, and 4096 was measured and rejected. It is the width the unified
# workspace reservation is maximised over (`startup.cpp`: `max_width = min(prefill_chunk,
# capacity)`), so it is a direct memory lever: 4096 takes the attention workspace from 2.246 GiB to
# 1.130 GiB and lifts free VRAM across the eight lanes from 1.51-3.43 GiB to 2.83-3.33 GiB, with KV
# capacity unchanged at 262,144 and pages 4,096/4,096. It costs prefill, measured three interleaved
# rounds per arm at depth ~19.2k: 13,077 tok/s against 12,770, so -2.3%, the two clusters disjoint
# (0.37% and 0.52% within-arm spread). Decode is unaffected -- three lanes measured at both widths
# returned identical acceptance and byte-identical digests.
#
# It is not used because **262,144 is served at 8192 on all eight lanes**, verified by starting each
# one through its own launcher. The 4096 experiment bought margin, not context, and margin was not
# the binding constraint; it traded measured throughput for headroom that nothing was short of.
# Recorded here because the margin it bought is real and the finding behind it is not the flag: the
# workspace is oversized by the split-planning rule, not by this width. See the next comment.
#
# The rule worth changing was upstream's, and it was not this one. `mxfp8_tiled_partition` selects
# splits by wave balance alone, and more splits always balance better, so it walks to its cap of 8 --
# while each split costs a full FP32 accumulator of head_dim depth per head per width. At 24 heads and
# head_dim 256 that cap held 1.62 GiB of partials for work that was never short of occupancy: 1536
# query tiles against a threshold of 136. Adding the term FlashAttention's rule carries -- 1 split
# once the tiles already fill 80% of the SMs, because "we also don't want too many splits as that
# would incur more HBM reads/writes" -- returns 1.3 GiB of free VRAM per lane at no prefill cost: three
# interleaved rounds per arm gave +3.4% prefill at depth ~45k with disjoint arms, and -0.3% at ~131k
# with overlapping arms. Output is unchanged -- identical acceptance and byte-identical digests on all
# eight lanes -- and all five KV storage types pass the FP64 oracle. The earlier attribution to
# `causal_partition_target`'s flat 2*SM budget was wrong: that governs the small-width families, whose
# buffers are tens of MiB, not the tiled prefill family that dominates. See
# docs/research/swift15-lane-measurement.md.
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
    # The template the lane renders with. The launcher resolves %TEMPLATE% beside itself; a harness
    # gets the source-tree path from launcher_args. Without this the lane inherits whichever template
    # its artifact happens to embed, and the two shipped artifacts embed different ones.
    ("--chat-template", "%TEMPLATE%"),
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
    # Which states occupy the pool, from the request log at 8/8: private_owners_evicted 7,
    # shared_owners_evicted 0, peak host_state_slots 8. The shared prefixes are never evicted, they are
    # never retained -- the slot is gone before the capture can claim it -- and all the pressure is the
    # private side, where the engine now retains continuations and reclaims the oldest only under
    # pressure (upstream #251). That reclaim keeps the current request working, not every prefix, so the
    # pool has to be sized for the states the engine actually keeps rather than for the prefixes alone.
    # The same bounds gave 5/5 on 2026-09-17 (commit 93181817); #251 landed on 2026-09-19 and changed
    # how many states a conversation keeps, and nothing re-ran the claim.
    #
    # 16 is derived rather than chosen: the other two bounds admit at most 7 shared prefixes plus 8
    # private continuations, so 15 covers every configuration they permit and 16 is that plus one.
    # Measured on the agent shape -- one 16,154-token prefix, six forks, then the original prompt again
    # -- the prefix survives the fan-out at 8, 12 and 16 alike, and the forks are what is evicted, 4, 3
    # and 2 as the pool grows. That is the intended priority, and it is why the private bound is the
    # wrong knob to lower: an agent session forks on every tool call.
    #
    # What the pool costs, from our own startup lines: 1.46 GiB at 8 slots, 2.19 at 12 and 2.92 at 16
    # -- about 187 MiB per retained state, linear across the three points. An independent NVFP4 build
    # for the same Gated-DeltaNet architecture quotes roughly 75 MB per sequence for the recurrent
    # state alone, which is a different accounting and not a corroboration; do not cite it as one.
    #
    # --context-cache-policy rolling ships enabled: once the pools are full, publishing a conversation's
    # newest checkpoint means replacing a resident, and by default that needs two matching reuse domains
    # or explicit evidence -- which one append-only conversation never has, so its reusable frontier
    # pins (measured: 52,723 tokens, TTFT 2.6 s -> 15.1 s at 65k -> 117k context). `rolling` values a
    # capture against replacing any resident and lets the fold decide, so the frontier then tracks
    # conversation (104,283 of 117,180, TTFT 4.1 s). It was proposed upstream as opt-in for shared
    # multi-tenant servers, where it can evict a prefix another conversation still wants; this is a
    # local single-owner product, and two saturating conversations measured byte-identical to the
    # default policy, so there is no second tenant to protect here.
    #
    # --spec dflash2's draft is trained on the stock model, so it is tied to that distribution rather
    # than to the engine. Measured here with dflash2_real_test: acceptance is 21/21 on the QUASAR QAT
    # artifact and 20/31 on NVFP4-full, both non-stock weights, so quantisation-aware training and
    # requantisation keep the draft valid. A behaviour finetune does not: a third-party conversion
    # recipe reports 3-5% acceptance on one, decode falling from ~180 to ~70 tok/s, and recommends
    # --spec mtp there because the MTP head ships inside the finetune. The choice between the dflash2
    # and mtp lanes is therefore about what an artifact's weights are, not about which route is
    # faster.
    #
    # What is NOT settled, measured 2026-09-21 and left open deliberately rather than guessed:
    #
    # Prefix sharing here is conditional on the frontend declaring a stable prefix. Two conversations
    # carrying 12,000 byte-identical tokens -- once inside the user turn, once as a shared system
    # message -- both reported `root` and zero cached, so a common prefix nobody declared is never
    # shared. A radix tree shares it automatically, which makes automatic discovery the one advantage
    # of that design this engine lacks; how often a real workload carries such a prefix is unmeasured,
    # and that is what would size the decision.
    #
    # Answered 2026-09-22 from a live 51-request agent log: such a prefix is not rare, it is every
    # conversation switch. An agent carries its system prompt and twelve tool definitions identically
    # on every request, at a tool boundary the frontend does declare, and the log shows every cache
    # hit as `private endpoint`, none shared, with a conversation switch costing a 59.6 s cold
    # prefill at 171,953 tokens.
    # The log is recorded at docs/research/agent-session-field-log-2026-09-22.md, with the startup
    # configuration it was measured under and every request's cache path.
    #
    # Resolved 2026-09-22: the boundary the frontend declares is not the boundary this protocol uses.
    # The OpenAI request path clears allow_engine_automatic_shared_prefixes -- the protocol defines
    # its own write policy -- so the frontend's three automatic shared opportunities are never
    # declared, and the serve layer's automatic target is the end of the last message, which is the
    # whole prompt. A request whose prompt differs before that point is skipped at the shortlist key
    # before a plan is asked; one whose prompt matches is offered the candidate and loses the
    # selection to private_response_replay. A declared boundary
    # (prompt_cache_breakpoint: {mode:"explicit"}) is published and served: two sessions sharing a
    # system message reused 301 of 344 tokens as `shared prefix`. So a shared owner is not "never
    # retained" -- shared_reuse_candidates counts the index entry reaching the plan -- and the
    # degradation below is real but not the cause. Three counters in RuntimeStats carry the decision:
    # shared_reuse_candidates, shared_reuse_declined, shared_reuse_key_mismatch.
    # docs/adr/0009-shared-prefix-served-where-declared.md records the attribution.
    #
    # Measured out along the way, each by rebuilding and re-running the same reproduction: the pools
    # (host KV 653 MiB of 8 GiB, 2 of 16 state slots, so neither is the constraint); the
    # materialization search budget, whose absolute grant ceiling of 250 ms raised to 5 s left
    # shared_stable_prefix at 0 and search_budget_exhaustions at 1, unchanged and reverted; the
    # shared retention weight, RetentionClass::SharedStable raised from 0 to 16, unchanged and
    # reverted; and crediting the engine's own structural boundary in the projection's credit test,
    # unchanged. All four were readings of the code and none survived the measurement.
    #
    # Whether recency beats the fold is unresolved. A diagnostic recency policy (oldest resident as the
    # victim, value gate bypassed) did change the eviction behaviour -- two shared owners evicted where
    # the fold evicts none -- while leaving the retained set identical, which contradicts the model of
    # what it does. The next attempt needs a per-entry use stamp and attribution of which entry was
    # evicted before it can conclude anything, so no conclusion is recorded here.
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
        if value == "%TEMPLATE%":
            value = template_path()
        if value is None and " " not in flag:
            args.append(flag)
        else:
            args.extend(flag.split())
            if value is not None:
                args.append(value)
    return args


def launcher_env(profile: dict[str, Any] | None = None) -> dict[str, str]:
    """The environment a profile's server must run under, exactly as the launcher sets it.

    The same argument as `launcher_args`: a harness that starts Serve has to start it the way the lane
    starts it, or its numbers describe a configuration nobody ships. A launcher renders these as `set`
    lines; a harness must pass them to `Popen`.

    This is where the lanes pin the CUDA wait schedule while the engine's own default stays upstream's
    (`NINFER_CUDA_SYNC`, see docs/cli.md). Measured 2026-09-25 with six interleaved Serve processes per
    condition: `spin` costs 0.18 to 0.31 of a core and buys no measurable latency -- TTFT 1.3 to 1.9 ms
    apart against 3.0 to 15.5 ms spreads, both idle and with half the machine's logical processors held
    busy by host work, which is the condition NVIDIA names under which a spin-wait should pay.
    docs/research/prompt-preparation-cost.md carries the measurement and its limits.

    The profile may be None, for a harness that composes its own command line without a table profile.
    Today the environment is the same for every lane -- it is a property of the wait schedule on this
    port, not of one profile's flags -- so the parameter is the seam if that ever has to change.
    """
    return {"NINFER_CUDA_SYNC": "blocking"}


def launcher_environment(profile: dict[str, Any] | None = None) -> dict[str, str]:
    """`launcher_env` merged over this process's environment, ready for `Popen(env=...)`.

    A harness must call this rather than `launcher_env` directly: `Popen` replaces the environment
    wholesale, so passing only the lane's own variables would drop PATH and the rest.
    """
    return {**os.environ, **launcher_env(profile)}


def cli_args(profile: dict[str, Any]) -> list[str]:
    """The flag subset the offline CLI accepts, for test_prompt.bat and test_vision.bat.

    The CLI is a different interface from the server: it takes a prompt or messages file and
    rejects --host, --port, --model-id, --max-concurrency, the state slots, the host KV pool,
    the cache bounds and the timeouts. Generating the server's argument list into QUASAR_ARGS
    was wrong for exactly that reason, and running the helper is what showed it.

    The flags kept here are the ones both interfaces share, plus the proposal head, which the
    CLI does accept and the earlier hand-written value omitted.

    The template is written as a cmd variable rather than an absolute path, unlike every other
    consumer of `template_path()`. The difference is the consumer: this list is rendered into
    `launcher_env.bat`, a file that ships and is compared as text, so an absolute path would bake one
    machine's checkout into it and make the comparison fail on any other. The launchers resolve the
    same variable beside themselves, which is the shipped convention.
    """
    out: list[str] = []
    # The lane's template, resolved beside the launcher so the CLI harness renders with the same file
    # the launcher ships, wherever the archive was extracted or the repository lives.
    out.extend(["--chat-template", "%TEMPLATE%"])
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
