#!/usr/bin/env python3
"""The profile table: one source of truth for what ships.

A **profile** is one shippable combination of artifact, spec route, vision and ceiling. This
module holds the table and the serve argument list, so the launcher generator, the measurement
harnesses and the verifier all describe the same thing instead of restating it.

Measured 2026-09-18 on an RTX 5090 (32 GB), fp8 KV, --kv-capacity auto, --prefill-chunk 8192.
Every value below is backed by a record in matrix_v3.jsonl.

  artifact      spec      vision  lm-head  context   decode     acceptance
  QUASAR        dflash2   yes     ON       262,144   331.3      61.8%   <- flagship
  QUASAR        mtp d4    yes     ON       262,144   225.4      65.3%
  NVFP4-full    dflash2   yes     ON       262,144   326.6      63.7%
  NVFP4-full    mtp d5    yes     ON       262,144   236.3      64.2%

All four are vision-only and all four reach the native context: Vision measured free on both
artifacts (331.3 against 333.0; 326.6 against 325.0, at the same ceiling either way). Text-only
variants existed only for the retired NVFP4 image, which charged 16,384-27,008 tokens for it.

--lm-head-draft is per profile, not uniform: it is worth +9% (DFlash2) and +18% (MTP d4) on
QUASAR, ~+2% on NVFP4-full DFlash2 at a cost of 0.33 GiB of headroom, and +34% on NVFP4-full
MTP d5. Route, depth and this flag are measurements, never convention -- see ADR-0005.
"""
from __future__ import annotations

from typing import Any, Iterable

# The QUASAR artifact is our own quantisation-aware-trained image; NVFP4-full is the fuller one
# published by cometkim. "ninfer" alone is ambiguous and should not be used. See CONTEXT.md.
QUASAR = "qwen3_8_27b_nvfp4qat.v3.ninfer"
NVFP4FULL = "qwen3_8_27b_nvfp4full.v3.ninfer"

PROFILES: list[dict[str, Any]] = [
    dict(file="start_quasar_v3_dflash2_vision.bat", port=8086, art=QUASAR,
         label="QUASAR QAT + DFlash2 + Vision", model_id="qwen3.8-27b-quasar-v3-dflash2-vision",
         spec="dflash2", draft=7, vision=True, lm_head=True, ctx=262144,
         tok=331.3, acc="61.8%", runtime="10.7 GiB", free="2.52 GiB",
         note="Flagship: fastest measured configuration, at full context."),
    dict(file="start_quasar_v3_mtp4_vision.bat", port=8087, art=QUASAR,
         label="QUASAR QAT + MTP4 + Vision", model_id="qwen3.8-27b-quasar-v3-mtp4-vision",
         spec="mtp", draft=4, vision=True, lm_head=True, ctx=262144,
         tok=225.4, acc="65.3%", runtime="10.4 GiB", free="3.08 GiB",
         note="Lower-VRAM QUASAR profile. MTP depth 4 measured fastest of 2-5 on QUASAR."),
    dict(file="start_ninfer_v3_dflash2_vision.bat", port=8088, art=NVFP4FULL,
         label="NVFP4-full + DFlash2 + Vision", model_id="qwen3.8-27b-nvfp4-v3-dflash2-vision",
         spec="dflash2", draft=7, vision=True, lm_head=True, ctx=262144,
         tok=326.6, acc="63.7%", runtime="10.7 GiB", free="1.74 GiB",
         note="Second artifact, same reach as QUASAR: 262,144 with Vision. This is the\nREM  tightest profile in the set; dropping --lm-head-draft buys 0.33 GiB at ~2% slower."),
    dict(file="start_ninfer_v3_mtp5_vision.bat", port=8089, art=NVFP4FULL,
         label="NVFP4-full + MTP5 + Vision", model_id="qwen3.8-27b-nvfp4-v3-mtp5-vision",
         spec="mtp", draft=5, vision=True, lm_head=True, ctx=262144,
         tok=236.3, acc="64.2%", runtime="10.4 GiB", free="2.28 GiB",
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
    ("--device-state-slots", "1"),
    ("--host-state-slots", "8"),
    ("--host-kv-mib", "8192"),
    ("--max-shared-prefixes", "7"),
    ("--max-private-continuations", "8"),
    ("--max-long-anchors-per-continuation", "4"),
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
    out.extend([("--host", "127.0.0.1"),
                ("--port", str(profile["port"])),
                ("--model-id", profile["model_id"]),
                ("--max-context", str(profile["ctx"]))])
    out.extend(INVARIANT_FLAGS)
    return out


def launcher_args(profile: dict[str, Any], port: int | None = None,
                  model_id: str | None = None, max_context: int | None = None) -> list[str]:
    """The serve argument list for a profile, exactly as the launcher passes it.

    This is the interface the measurement harnesses should use. When they build their own
    argument list instead, their records stop describing what ships -- which happened: the
    cache bounds, the thinking budget and the model ids were all missing from the harness, so
    its numbers came from flags no launcher ships.

    Three overrides exist because a harness genuinely differs from a launcher, and each has
    more than one caller, so the seam is real rather than hypothetical:

    - ``port``: a harness must bind its own port, because the shipped one may already be
      serving. Every harness overrides it.
    - ``model_id``: the engine enforces the id on every request, so a harness needs one it
      controls. Every harness overrides it.
    - ``max_context``: the matrix probes a descending ladder and the effort probe uses a small
      context. Two callers.

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


def all_profiles() -> Iterable[dict[str, Any]]:
    return iter(PROFILES)
