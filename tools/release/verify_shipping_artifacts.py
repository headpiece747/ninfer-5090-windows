#!/usr/bin/env python3
"""Re-verify every artifact this port has published an image for, and report drift from its baseline.

`docs/perplexity-baseline.md` is the authority for the recorded figures; the table below is a copy of
its most recent column, and this script exists because a copy that is only read does not fail.

**Why it is a script and not a note.** The project's own rule is that a numerics change re-states every
recorded figure and that the tree will not tell you: a `silu` correction once moved a published
perplexity from 1.606336 to 1.609905 and nothing failed, no check moved, and an ADR audit found it
only by reading. Two sessions have now found a recorded figure that no longer reproduced. So the check
is executable, it runs over every image rather than the one being worked on, and it exits non-zero.

**What "working" means here, and what it does not.** Two checks per artifact:

  structure  `tools/convert/verify_artifact.py --only all`, the artifact-conventions section 2
             contract: the complete ordered directory walked, every binding's ranges ordered and in
             range, and every NVFP4 weight and stored input divisor a positive finite FP32 word. This
             is a host-side walk of the container; it does not need the GPU and does not load weights.

  quality    a full-corpus `ninfer-perplexity` run at the shipped protocol (4096 context, 2048 stride,
             fp8 KV), compared against the recorded baseline. This loads the artifact onto the card,
             so it is the slow half and the reason the whole pass is serialised.

Perplexity is a comparison between builds of the *same* source checkpoint, and only that. Two
different finetunes differ in perplexity for reasons that have nothing to do with either build -- the
Swift 1.0 and Swift 1.5 lanes sit 1.30 % apart because they are different models, which is why both
are listed here against their own recorded value rather than against each other.

A run outside the tolerance is reported, not fixed. The recorded figure is the thing under test.

Usage:
  python tools/release/verify_shipping_artifacts.py              # structure for all, quality for all
  python tools/release/verify_shipping_artifacts.py --structure-only
  python tools/release/verify_shipping_artifacts.py --only quasar nvfp4full
  python tools/release/verify_shipping_artifacts.py --tolerance 0.005
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
MODELS = Path(r"C:\AI\models")
PERPLEXITY = ROOT / "build" / "apps" / "ninfer-perplexity.exe"
CORPUS = ROOT / "eval" / "corpora" / "perplexity-1m" / "manifest.json"
OUTPUT_ROOT = ROOT / "profiles" / "perplexity"

# artifact filename -> (label, recorded overall perplexity, directory, shipping lane?)
#
# The recorded values were re-taken 2026-10-07 on the post-merge engine, under the corrected tokenizer:
# every one of them had read 4.3-6.2% below its 2026-09-29 value, and the cause was the instrument, not
# the models -- upstream's tokenizer rewrite (b9114396/a8e212ac, merged 2026-10-04) fixed an
# over-segmentation of 319 tokens on this corpus, and the engine now matches the reference tokenizer
# exactly (see the note at the top of docs/perplexity-baseline.md). Values from before that merge are
# not comparable with these. `directory` is where the file actually lives: superseded images are verified from
# `_superseded`, because the instruction this script answers is "check all previous models", and a
# superseded build that is not checked is a build whose drift nobody would notice. `shipping` says
# whether a launcher starts it -- qwen3_8_27b_nvfp4 is the retired image that still sits in the models
# directory reading like a shipping lane by filename, which has already cost one session of
# benchmarking, so it is verified and labelled rather than ignored.
#
# Swift 1.5's recorded figure is the Q8-draft build's 5.000654, and the shipping image is the
# NVFP4-draft build of the same checkpoint. Those must agree exactly: causal scoring loads only the
# text stack, so a recipe that differs solely in how the dflash2 component is encoded cannot move
# this number. If it does, the variant touched something it should not have.
SUPERSEDED = MODELS / "_superseded"
ARTIFACTS: dict[str, tuple[str, float, Path, bool]] = {
    "qwen3_8_27b_nvfp4qat.v3.ninfer": ("QUASAR QAT", 4.684860, MODELS, True),
    "qwen3_8_27b_nvfp4full.v3.ninfer": ("NVFP4-full", 4.724219, MODELS, True),
    "qwen3_8_27b_nvfp4nvidia.v3.ninfer": ("NVIDIA ModelOpt", 4.686759, MODELS, True),
    "qwen3_8_27b_nvfp4swift15.v3.ninfer": ("Swift 1.5", 4.755739, MODELS, True),
    "qwen3_8_27b_nvfp4swift.v3.ninfer": ("Swift 1.0", 4.725703, SUPERSEDED, False),
    "qwen3_8_27b_nvfp4.v3.ninfer": ("retired nvfp4", 4.615687, SUPERSEDED, False),
}

# The recipe each artifact was built by, for the structural verifier's own cross-check. The Swift
# recipe needs all three of its sources named, because it reads the divisor from the ModelOpt
# checkpoint and the weights from the BF16 finetune; passing the finetune as --recipe-base fails with
# "no activation scale to derive the divisor from", which is a startup-time error message from a
# verifier and reads like a corrupt artifact.
SOURCES = {
    "qwen3_8_27b_nvfp4qat.v3.ninfer": ("qwen3_8_27b_nvfp4_qat", r"C:\AI\models\hf-src\Qwen3.8-27B-NVFP4-QUASAR", {}),
    "qwen3_8_27b_nvfp4full.v3.ninfer": ("qwen3_8_27b_nvfp4_unsloth", r"C:\AI\models\hf-src\Qwen3.8-27B-NVFP4-unsloth", {}),
    "qwen3_8_27b_nvfp4swift.v3.ninfer": ("qwen3_8_27b_nvfp4_swift", r"C:\AI\models\hf-src\Swift-Qwen3.8-27B-NVFP4",
                                         {"swift_bf16": r"C:\AI\models\hf-src\Swift-Qwen3.8-27b",
                                          "dflash2": r"C:\AI\models\hf-src\Qwen3.8-27B-DFlash2"}),
    "qwen3_8_27b_nvfp4swift15.v3.ninfer": ("qwen3_8_27b_nvfp4_swift", r"C:\AI\models\hf-src\Swift-1.5-Qwen3.8-27b-NVFP4",
                                           {"swift_bf16": r"C:\AI\models\hf-src\Swift-1.5-Qwen3.8-27b",
                                            "dflash2": r"C:\AI\models\hf-src\Qwen3.8-27B-DFlash2"}),
    "qwen3_8_27b_nvfp4nvidia.v3.ninfer": ("qwen3_8_27b_nvfp4_nvidia", r"C:\AI\models\hf-src\Qwen3.8-27B-NVFP4-nvidia", {}),
    "qwen3_8_27b_nvfp4.v3.ninfer": ("qwen3_8_27b_nvfp4", r"C:\AI\models\hf-src\Qwen3.8-27B-NVFP4", {}),
}


def structure(name: str, directory: Path) -> tuple[bool, str]:
    """The container walk. Returns (ok, one line of evidence)."""
    path = directory / name
    if not path.exists():
        return False, f"missing artifact {path}"
    recipe = SOURCES.get(name)
    argv = [PYTHON, str(ROOT / "tools" / "convert" / "verify_artifact.py"), str(path),
            "--only", "all"]
    if recipe is not None:
        argv += ["--recipe", recipe[0], "--recipe-base", recipe[1]]
        for key, value in recipe[2].items():
            argv += ["--source", f"{key}={value}"]
    done = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT), check=False)
    tail = [line.strip() for line in (done.stdout + done.stderr).splitlines() if line.strip()]
    evidence = next((line for line in reversed(tail) if "PASS" in line or "FAIL" in line),
                    tail[-1] if tail else "no output")
    return done.returncode == 0, evidence


def quality(name: str, tag: str, directory: Path) -> tuple[bool, str, float | None]:
    """The full corpus at the shipped protocol. Returns (ran, summary line, perplexity or None)."""
    path = directory / name
    if not path.exists():
        return False, f"missing artifact {path}", None
    out_dir = OUTPUT_ROOT / "verify-shipping" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("report.json"):
        stale.unlink()
    done = subprocess.run(
        [str(PERPLEXITY), str(path), "--corpus", str(CORPUS), "--kv-dtype", "fp8",
         "--output", str(out_dir)],
        capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT), check=False)
    report = out_dir / "report.json"
    if not report.exists():
        tail = [line.strip() for line in done.stderr.splitlines() if line.strip()]
        return False, tail[-1] if tail else f"exit {done.returncode}, no report", None
    payload = json.loads(report.read_text(encoding="utf-8"))
    return True, (f"{payload['overall']['scored_tokens']:,} tokens, "
                  f"{payload['timing']['scored_tokens_per_second']:.0f} tok/s"), \
        float(payload["overall"]["perplexity"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=sorted(ARTIFACTS), help="restrict to these")
    parser.add_argument("--structure-only", action="store_true",
                        help="skip the GPU half; the container walk needs no device")
    parser.add_argument("--quality-only", action="store_true", help="skip the container walk")
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="relative drift from the recorded baseline before it is reported "
                             "(default 0.01, the +/-1%% band the perplexity authority uses)")
    args = parser.parse_args(argv)

    names = args.only or list(ARTIFACTS)
    started = time.time()
    rows: list[dict] = []
    print(f"{'artifact':<34} {'kind':<9} {'struct':<7} {'ppl':>10} {'recorded':>10} {'drift':>9}")
    for name in names:
        label, recorded, directory, shipping = ARTIFACTS[name]
        tag = name.removesuffix(".v3.ninfer").replace(".ninfer", "")
        row: dict = {"artifact": name, "label": label, "shipping": shipping,
                     "directory": str(directory), "recorded": recorded}
        if not args.quality_only:
            ok, evidence = structure(name, directory)
            row["structure_ok"] = ok
            row["structure_evidence"] = evidence
        else:
            row["structure_ok"] = None
        if not args.structure_only:
            ran, summary, value = quality(name, tag, directory)
            row["quality_ran"] = ran
            row["quality_summary"] = summary
            row["perplexity"] = value
            if value is not None:
                row["drift"] = (value - recorded) / recorded
        rows.append(row)
        struct = {True: "PASS", False: "FAIL", None: "skip"}[row["structure_ok"]]
        value = row.get("perplexity")
        drift = row.get("drift")
        print(f"{name:<34} {label:<9} {struct:<7} "
              f"{(f'{value:.6f}' if value is not None else 'skip'):>10} "
              f"{recorded:>10.6f} "
              f"{(f'{drift * 100:+.3f}%' if drift is not None else '-'):>9}")
        if not args.structure_only and row.get("quality_ran") is not None:
            print(f"{'':<34} {'':<9} {'':<7}   {row.get('quality_summary', '')}")
        if not args.structure_only and row.get("quality_ran") is False:
            print(f"{'':<34} {'':<9} {'':<7} ! {row.get('quality_summary', '')}")
        if not args.quality_only and row["structure_ok"] is False:
            print(f"{'':<34} {'':<9} {'':<7} ! {row.get('structure_evidence', '')}")

    out = OUTPUT_ROOT / "verify-shipping" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tolerance": args.tolerance, "artifacts": rows}, indent=2) + "\n",
                   encoding="utf-8")

    bad_structure = [r["artifact"] for r in rows if r["structure_ok"] is False]
    not_run = [r["artifact"] for r in rows if r.get("quality_ran") is False]
    drifted = [f"{r['artifact']} {r['drift'] * 100:+.3f}%" for r in rows
               if r.get("drift") is not None and abs(r["drift"]) > args.tolerance]
    print(f"\nsummary: {len(rows)} artifact(s) | structure failures {len(bad_structure)} | "
          f"perplexity not run {len(not_run)} | outside +/-{args.tolerance * 100:.1f}% "
          f"{len(drifted)}")
    for item in bad_structure:
        print(f"  STRUCTURE FAIL  {item}")
    for item in not_run:
        print(f"  NOT RUN         {item}")
    for item in drifted:
        print(f"  DRIFT           {item}")
    print(f"json: {out}")
    print(f"elapsed {time.time() - started:.0f}s")
    return 1 if (bad_structure or not_run or drifted) else 0


if __name__ == "__main__":
    raise SystemExit(main())
