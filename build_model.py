#!/usr/bin/env python3
"""Build one shipped image from its public source checkpoints, with this port's own converter.

The archive carries the converter (`tools/convert` and `tools/artifact`) and the chat template, so a
fresh install can produce any of the five images without fetching a prebuilt `.ninfer` from anyone. The
sources are public and ungated; `download_model.py` fetches them at the revisions pinned there and records
what it fetched in `sources.json` beside them, which is what makes a build reproducible.

Usage:
    build_model.py --list                     # the five lines, their recipes and their sources
    build_model.py --line quasar              # fetch what is missing, then convert
    build_model.py --line quasar --dry-run    # print the converter command and stop
    build_model.py --line swift15 --dest D    # write the image and its report under D

A conversion takes minutes and needs a CUDA device. The image lands in `models/` beside this file unless
--dest says otherwise, which is where the launchers look for it. Nothing here fetches a `.ninfer`: the
only downloads are source checkpoints, and the only artifact is the one this machine builds.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from download_model import HF_SRC, SOURCES, fetch, report  # noqa: E402

# line -> the recipe, the source that is `--model`, the named sources, and the image's identity.
#
# The recipes and their source sets are the ones the artifact reference records per line, and `--name` is
# the value the shipped image carries, so a rebuild differs from what this port ships only in the 16-byte
# artifact id -- a random `uuid4()` (tools/artifact/writer.py:141) -- if it differs at all.
LINES: dict[str, dict[str, object]] = {
    "quasar": {
        "recipe": "qwen3_8_27b_nvfp4_qat",
        "primary": "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4",
        "sources": {"bf16": "Qwen/Qwen3.8-27B", "dflash2": "z-lab/Qwen3.8-27B-DFlash2"},
        "artifact": "qwen3_8_27b_nvfp4qat.v3.ninfer",
        "name": "qwen3.8-27b",
        "note": "QUASAR's quantization-aware-trained codes imported whole; both lanes run this image",
    },
    "full": {
        "recipe": "qwen3_8_27b_nvfp4_unsloth",
        "primary": "Qwen/Qwen3.8-27B",
        "sources": {"quantized": "unsloth/Qwen3.8-27B-NVFP4",
                    "dflash2": "z-lab/Qwen3.8-27B-DFlash2"},
        "artifact": "qwen3_8_27b_nvfp4full.v3.ninfer",
        "name": "qwen3.8-27b",
        "note": "the unsloth line with its nine BF16 exception parents; the MTP lane runs this image",
    },
    "noex": {
        "recipe": "qwen3_8_27b_nvfp4_unsloth_noex",
        "primary": "Qwen/Qwen3.8-27B",
        "sources": {"quantized": "unsloth/Qwen3.8-27B-NVFP4",
                    "dflash2": "z-lab/Qwen3.8-27B-DFlash2"},
        "artifact": "qwen3_8_27b_nvfp4full_noex.v3.ninfer",
        "name": "nvfp4full-noex",
        "note": "the same line with every projection at NVFP4; the DFlash2 lane runs this image",
    },
    "swift15": {
        "recipe": "qwen3_8_27b_nvfp4_swift15_nvdraft",
        "primary": "ukisai/Swift-1.5-Qwen3.8-27b-NVFP4",
        "sources": {"swift_bf16": "ukisai/Swift-1.5-Qwen3.8-27b",
                    "dflash2": "z-lab/Qwen3.8-27B-DFlash2"},
        "artifact": "qwen3_8_27b_nvfp4swift15.v3.ninfer",
        "name": "swift15-nvdraft",
        "note": "the Swift 1.5 finetune, MLP imported and the rest re-encoded, with the NVFP4 draft",
    },
    "nvidia": {
        "recipe": "qwen3_8_27b_nvfp4_nvidia",
        "primary": "Qwen/Qwen3.8-27B",
        "sources": {"quantized": "nvidia/Qwen3.8-27B-NVFP4",
                    "dflash2": "z-lab/Qwen3.8-27B-DFlash2"},
        "artifact": "qwen3_8_27b_nvfp4nvidia.v3.ninfer",
        "name": "qwen3.8-27b",
        "note": "NVIDIA's ModelOpt codes and per-site scales; both lanes run this image",
    },
}


def local_dir(root: str, repo: str) -> str:
    return os.path.join(root, SOURCES[repo][0])


def template_path() -> Path:
    """The maintained template: `chat_templates/` in an extracted archive, `tools/chat_templates/` here."""
    for candidate in (HERE / "chat_templates" / "qwen3_8.jinja",
                      HERE / "tools" / "chat_templates" / "qwen3_8.jinja"):
        if candidate.exists():
            return candidate
    raise SystemExit("chat_template.jinja is not beside this script or under tools/")


def converter_command(line: str, root: str, out: Path) -> list[str]:
    spec = LINES[line]
    command = [sys.executable, "-m", "tools.convert",
               "--model", local_dir(root, str(spec["primary"])),
               "--recipe", str(spec["recipe"])]
    for name, repo in sorted(dict(spec["sources"]).items()):
        command += ["--source", f"{name}={local_dir(root, str(repo))}"]
    command += ["--components", "text,vision,mtp,dflash2",
                "--resource", f"chat_template.jinja={template_path()}",
                "--name", str(spec["name"]), "--device", "cuda", "--out", str(out)]
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--line", choices=sorted(LINES), help="which image to build")
    parser.add_argument("--list", action="store_true", help="show the five lines and exit")
    parser.add_argument("--dest", default=None, help="where the image lands (default: models/ here)")
    parser.add_argument("--hf-src", default=None, help="source checkpoint root (default: hf-src here)")
    parser.add_argument("--dry-run", action="store_true", help="print the converter command and stop")
    args = parser.parse_args()

    if args.list or not args.line:
        for name, spec in LINES.items():
            print(f"{name:<8} {spec['artifact']}")
            print(f"         recipe {spec['recipe']}  --model {spec['primary']}")
            print(f"         sources {', '.join(f'{k}={v}' for k, v in sorted(dict(spec['sources']).items()))}")
            print(f"         {spec['note']}")
        return 0

    spec = LINES[args.line]
    root = args.hf_src or os.environ.get("NINFER_HF_SRC") or str(HERE / "hf-src")
    dest = Path(args.dest) if args.dest else HERE / "models"
    out = dest / str(spec["artifact"])

    print(f"[INFO] line {args.line}: {spec['note']}")
    print(f"[INFO] sources under {root}; the image will be written to {out}")

    if not args.dry_run:
        needed = [str(spec["primary"]), *dict(spec["sources"]).values()]
        missing = [repo for repo in needed if not os.path.isdir(local_dir(root, repo))]
        for repo in missing:
            print(f"[INFO] fetching {repo}")
            if not fetch(repo, root):
                print(f"[FAILED] could not fetch {repo}", file=sys.stderr)
                return 1
        if missing:
            print()
            report(root, sorted(set(needed)))
        if out.exists():
            print(f"[FAILED] {out} already exists; move it aside or choose another --dest",
                  file=sys.stderr)
            return 1

    command = converter_command(args.line, root, out)
    print("[INFO] " + " ".join(command))
    if args.dry_run:
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(command, cwd=str(HERE), check=False)
    if done.returncode != 0:
        print(f"[FAILED] converter exited {done.returncode}", file=sys.stderr)
        return done.returncode
    print(f"[OK] {out} and its .conversion.json are ready; start a launcher for this line.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
