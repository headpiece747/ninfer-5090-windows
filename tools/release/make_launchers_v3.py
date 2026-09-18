#!/usr/bin/env python3
"""Generate the v3 Windows launchers from one measured table.

Single source of truth: every number here was measured on this machine with the exact
arg set the launcher ships (see matrix_v3.jsonl). Re-run this after re-measuring rather
than hand-editing six files, so the launchers cannot silently drift from the data.

Measured 2026-09-17, RTX 5090 32 GB, fp8 KV, --kv-capacity auto, prefill-chunk 8192.

  artifact  spec     vision  lm-head  ctx       decode    acceptance
  quasar    dflash2  yes     ON       262,144   331.3     61.8%     <- flagship
  quasar    mtp      yes     ON       262,144   225.4     65.3%   (draft 4)
  nvfp4     dflash2  no      OFF      180,224   258.2     59.4%   (draft 7)
  nvfp4     dflash2  yes     OFF      163,840   257.2     59.4%   (draft 7)
  nvfp4     mtp      no      ON       240,000   206.0     61.7%   (draft 5)
  nvfp4     mtp      yes     ON       212,992   204.8     61.7%   (draft 5)

--lm-head-draft is deliberately per profile: it is worth +9% (DFlash2) and +18% (MTP)
on QUASAR, and +34% on nvfp4 MTP, but on nvfp4 DFlash2 it costs ~13% throughput, 11pp
acceptance and 16,384 of context, so it is off there.

QUASAR is vision-only by choice: vision measured free on it (331.3 vs 333.0 tok/s, and
262,144 either way), so there is no reason to ship a degraded no-vision variant.
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(r"C:\AI\ninfer-v3-windows")
V3 = r"C:\AI\ninfer-v3-windows"
MODELS = r"C:\AI\models"

QUASAR = "qwen3_8_27b_nvfp4qat.v3.ninfer"
# cometkim's fuller-NVFP4 image: 18.07 GiB on disk, 17.03 GiB of device weights with
# DFlash2, and it reaches the native 262,144 in every combination where our own nvfp4 image
# is capped at 240,000 by its 22.1 GB of weights. Verified against SHA-256 ac98cd39... here.
NVFP4FULL = "qwen3_8_27b_nvfp4full.v3.ninfer"

# Four profiles: two artifacts, both spec routes, Vision everywhere, all at 262,144.
PROFILES = [
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

TEMPLATE = """@echo off
REM ============================================================================
REM  {label}
REM
REM  Measured on this machine (32 GB RTX 5090), fp8 KV at the ceiling below:
REM      context {ctx_h}   decode {tok} tok/s   draft acceptance {acc}
REM      runtime {runtime}   free VRAM {free}
REM
REM  {note}
REM
REM  Requires the FFmpeg runtime DLLs beside the executable (staged by
REM  build_v3.cmd); without them the process exits 0xC0000135.
REM
REM  The three context-cache bounds are deliberate. With max-concurrency 1 the defaults
REM  are max(1,4) shared, 2 private and 2 anchors; measured on five distinct ~530-token
REM  prompts resent, that gave 1/5 round-2 hits at a 19.8% token-level hit rate, with
REM  four of five prompts re-prefilling in full on every call and no error. Raising the
REM  shared bound alone changed nothing; all three together gave 5/5 hits at 99.1%.
REM  They cost no context or VRAM: KV stays 262,144 and runtime stays 10.7 GiB.
REM ============================================================================
setlocal

REM Resolve beside this launcher first, so the released archive is portable wherever it is
REM extracted, then fall back to the source tree so the same file works while developing.
set "SERVE=%~dp0ninfer-serve.exe"
if not exist "%SERVE%" set "SERVE={v3}\\build\\apps\\ninfer-serve.exe"
set "MODEL=%~dp0models\\{art}"
if not exist "%MODEL%" set "MODEL={models}\\{art}"

if not exist "%SERVE%" (
    echo [ERROR] Engine not found.
    echo         Expected ninfer-serve.exe beside this launcher,
    echo         or a source build at {v3}\\build\\apps\\ninfer-serve.exe
    pause
    exit /b 1
)
if not exist "%MODEL%" (
    echo [ERROR] Artifact not found.
    echo         Expected %~dp0models\\{art}
    echo         or {models}\\{art}
    echo         Run download_model.bat to fetch it.
    pause
    exit /b 1
)

"%SERVE%" "%MODEL%" ^{flags}
  --host 127.0.0.1 ^
  --port {port} ^
  --model-id {model_id} ^
  --max-context {ctx} ^
  --kv-capacity auto ^
  --kv-dtype fp8 ^
  --prefill-chunk 8192 ^
  --max-concurrency 1 ^
  --device-state-slots 1 ^
  --host-state-slots 8 ^
  --host-kv-mib 8192 ^
  --max-shared-prefixes 7 ^
  --max-private-continuations 8 ^
  --max-long-anchors-per-continuation 4 ^
  --preserve-thinking ^
  --default-thinking-budget 4096 ^
  --pending-timeout-ms 600000

pause
"""


def flags_of(p: dict) -> list[str]:
    out = []
    if p["vision"]:
        out.append("--vision")
    if p["spec"] != "none":
        out.append(f"--spec {p['spec']}")
        out.append(f"--draft-tokens {p['draft']}")
        if p["lm_head"]:
            out.append("--lm-head-draft")
    return out


def main() -> int:
    written = []
    for p in PROFILES:
        flags = flags_of(p)
        # Every flag continues the ^ chain; the template's --host line follows them.
        flag_block = "".join(f"\n  {f} ^" for f in flags)
        text = TEMPLATE.format(
            label=p["label"], note=p["note"], ctx=p["ctx"], ctx_h=f"{p['ctx']:,}",
            tok=p["tok"], acc=p["acc"], runtime=p["runtime"], free=p["free"],
            v3=V3, models=MODELS, art=p["art"], port=p["port"],
            model_id=p["model_id"], flags=flag_block,
        )
        path = OUT / p["file"]
        path.write_text(text, encoding="utf-8", newline="\r\n")
        written.append(path.name)
    for name in written:
        print(f"  wrote {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
