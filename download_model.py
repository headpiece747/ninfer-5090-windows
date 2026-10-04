#!/usr/bin/env python3
"""Fetch the SOURCE checkpoints that every shipped artifact is built from.

This used to download two prebuilt `.ninfer` artifacts from a third party's Hugging Face account.
That was wrong twice over, and both faults are why this script was rewritten rather than edited:

  1. It contradicted how the product is actually made. Every `.ninfer` artifact this port ships is
     built LOCALLY by this port's own converter -- `converter: ninfer-v3` in each artifact's
     conversion report -- from Hugging Face source checkpoints under `hf-src/`. The prebuilt files
     were predecessors, not inputs.
  2. Its pins were stale even as downloads. It pinned sha256 `ac98cd39...` for the fuller-NVFP4
     artifact while the local build of that same recipe is `f8dc6470...` (RELEASE_NOTES.md). So the
     script would have fetched superseded bytes and called them verified.

So this fetches SOURCES now. Each artifact's conversion report names exactly which of these it
consumed, and the `sources` block there is the authority -- this list mirrors those directories:

    qwen3_8_27b_nvfp4full.v3.ninfer.conversion.json
      "converter": "ninfer-v3",
      "recipe": "qwen3_8_27b_nvfp4_unsloth_noex",
      "sources": { base: ...\\hf-src\\Qwen3.8-27B,
                   dflash2: ...\\hf-src\\Qwen3.8-27B-DFlash2,
                   quantized: ...\\hf-src\\Qwen3.8-27B-NVFP4-unsloth }

Usage:
    download_model.py                 # fetch every source below
    download_model.py --source unsloth
    download_model.py --verify-only   # report what is present, fetch nothing
    download_model.py --dest PATH     # override the hf-src root

Verification changed with the target. A prebuilt artifact was one file and could carry a sha256; a
source repository is a tree whose contents move, so pinning a single digest would be meaningless and
a floating `main` would not be reproducible. `--verify-only` therefore reports presence and size per
source, which is the property that actually matters here -- a missing source fails the converter with
a clear error, whereas a subtly different source produces a subtly different artifact. Record the
exact commit you convert from alongside the conversion report; that is what makes a build
reproducible, and this script cannot do it for you.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HF_SRC = r"C:\AI\models\hf-src"

# repo_id -> (local directory under HF_SRC, what consumes it)
SOURCES: dict[str, tuple[str, str]] = {
    "Qwen/Qwen3.8-27B": ("Qwen3.8-27B", "the BF16 base every line is built from"),
    "z-lab/Qwen3.8-27B-DFlash2": ("Qwen3.8-27B-DFlash2", "the DFlash2 draft companion, all eight lanes"),
    "unsloth/Qwen3.8-27B-NVFP4": ("Qwen3.8-27B-NVFP4-unsloth", "nvfp4full and nvfp4full_noex"),
    "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4": ("Qwen3.8-27B-NVFP4-QUASAR", "nvfp4qat (QUASAR)"),
    "nvidia/Qwen3.8-27B-NVFP4": ("Qwen3.8-27B-NVFP4-nvidia", "nvfp4nvidia (NVIDIA ModelOpt)"),
    "ukisai/Swift-Qwen3.8-27b": ("Swift-Qwen3.8-27b", "the Swift 1.5 finetune, before quantisation"),
    "ukisai/Swift-Qwen3.8-27B-NVFP4": ("Swift-Qwen3.8-27B-NVFP4", "nvfp4swift15 (Swift 1.5)"),
}


def dir_stats(path: str) -> tuple[int, int]:
    """(file count, total bytes) for a directory, or (0, 0) if absent."""
    if not os.path.isdir(path):
        return 0, 0
    count = 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            count += 1
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return count, total


def report(root: str, repos: list[str]) -> bool:
    ok = True
    print(f"{'source':<40} {'files':>7} {'GiB':>8}  state")
    for repo in repos:
        local, why = SOURCES[repo]
        path = os.path.join(root, local)
        count, total = dir_stats(path)
        if count == 0:
            ok = False
            state = "MISSING"
        else:
            state = "present"
        print(f"{repo:<40} {count:>7} {total / (1 << 30):>8.2f}  {state}  ({why})")
    return ok


def fetch(repo: str, root: str) -> bool:
    local, _why = SOURCES[repo]
    dest = os.path.join(root, local)
    # huggingface_hub 1.x ignores HF_HUB_ENABLE_HF_TRANSFER, and on 0.x it raises when hf_transfer is
    # absent -- which the broad except below would report as a failed download rather than a
    # fallback, so it is set only when the module is genuinely importable.
    import importlib.util

    if importlib.util.find_spec("hf_transfer"):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "  huggingface_hub is not installed. Install it, or fetch these repositories by hand\n"
            "  into the directories named above.",
            file=sys.stderr,
        )
        return False

    print(f"[INFO] {repo} -> {dest}")
    started = time.time()
    try:
        snapshot_download(repo_id=repo, local_dir=dest)
    except Exception as exc:  # noqa: BLE001 - report any transport/auth/disk failure the same way
        print(f"  FAILED: {exc}", file=sys.stderr)
        return False
    print(f"  done in {time.time() - started:.0f}s")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the SOURCE checkpoints every shipped .ninfer artifact is built from."
    )
    parser.add_argument("--source", choices=sorted(SOURCES), action="append",
                        help="fetch one source (repeatable); default is all of them")
    parser.add_argument("--dest", default=None, help="override the hf-src root")
    parser.add_argument("--verify-only", action="store_true",
                        help="report which sources are present; fetch nothing")
    args = parser.parse_args()

    root = args.dest or os.environ.get("NINFER_HF_SRC") or HF_SRC
    repos = args.source or sorted(SOURCES)

    print(f"[INFO] Source root: {root}")
    print("[INFO] These are INPUTS to the converter, not the shipped artifacts. Each shipped")
    print("       .ninfer is built locally by this port ('converter: ninfer-v3' in its")
    print("       .conversion.json), and that report names which sources it consumed.")
    print()

    if args.verify_only:
        ok = report(root, repos)
        print()
        print("All required sources present." if ok else "Some sources are missing.")
        return 0 if ok else 1

    os.makedirs(root, exist_ok=True)
    failed = [repo for repo in repos if not fetch(repo, root)]

    print()
    print("[INFO] Summary:")
    report(root, repos)
    print()

    if failed:
        print(f"[FAILED] {len(failed)} of {len(repos)} source(s): {', '.join(failed)}", file=sys.stderr)
        return 1
    print("All sources fetched. Convert them with tools/convert; do not look for a prebuilt artifact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())