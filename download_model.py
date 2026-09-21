#!/usr/bin/env python3
"""Fetch a v3 artifact for the launchers, verifying it before and after.

Two problems with the original: it was hardcoded to one artifact, and it saved the file
under the repository's name (qwen3_8_27b_nvfp4qat.ninfer) while every launcher looks for the
.v3. form. A user following the README would download 17 GiB and then be told the artifact
was missing.

Artifact naming is also not uniform across publishers, and it changes: cometkim's fuller-NVFP4
repository ships v3, and the QUASAR QAT repository has since been republished as v3 as well, so
both download and run directly. The offline upgrader still ships in this archive for anyone
holding a pre-existing v2 copy, which a v3 engine rejects outright.

Usage: download_model.py [--artifact quasar|nvfp4full] [--dest PATH] [--verify-only]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

LOCAL_DIR = r"C:\AI\models"

ARTIFACTS = {
    "quasar": {
        "repo": "cometkim/Qwen3.8-27B-nvfp4qat-NInfer",
        "source": "qwen3_8_27b_nvfp4qat.ninfer",
        "target": "qwen3_8_27b_nvfp4qat.v3.ninfer",
        "size": 18_638_510_576,
        "sha256": "8b86901a8cd2a297a3d737e470c793b67e5ce65b49131c48c2f2f0b346fd943c",
        "container": "v3",
    },
    "nvfp4full": {
        "repo": "cometkim/Qwen3.8-27B-nvfp4full-NInfer",
        "source": "qwen3_8_27b_nvfp4full.ninfer",
        "target": "qwen3_8_27b_nvfp4full.v3.ninfer",
        "size": 19_407_229_188,
        "sha256": "ac98cd392c84a04b2a21c2f5c3988dece88d20a697ba1de663fb32d5998b8ee9",
        "container": "v3",
    },
}


def compute_sha256(path: str) -> str:
    digest = hashlib.sha256()
    total = os.path.getsize(path)
    done = 0
    started = time.time()
    with open(path, "rb") as handle:
        while chunk := handle.read(64 * 1024 * 1024):
            digest.update(chunk)
            done += len(chunk)
            print(f"\rVerifying SHA-256: {done / total * 100:5.1f}% "
                  f"({done / 1024 ** 3:.2f}/{total / 1024 ** 3:.2f} GiB) at "
                  f"{(done / 1024 ** 2) / max(time.time() - started, 0.001):.0f} MB/s...",
                  end="", flush=True)
    print()
    return digest.hexdigest()


def verify_file(path: str, spec: dict) -> bool:
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return False
    size = os.path.getsize(path)
    print(f"Checking file: {path}")
    print(f"File size: {size} bytes ({size / 1024 ** 3:.2f} GiB)")
    if size != spec["size"]:
        print(f"[FAIL] Size mismatch: expected {spec['size']}, got {size}")
        return False
    print("[OK] Size matches expected size.")
    print(f"Computing SHA-256 (expected: {spec['sha256']})...")
    actual = compute_sha256(path)
    print(f"Computed SHA-256: {actual}")
    if actual.lower() == spec["sha256"].lower():
        print("[SUCCESS] SHA-256 verified.")
        return True
    print(f"[FAIL] SHA-256 mismatch.\n  Expected: {spec['sha256']}\n  Actual:   {actual}")
    print()
    print("  Two causes, and they need different fixes:")
    print("    1. The download is corrupt or partial. Delete the file and download again.")
    print("    2. The publisher replaced the file under the same name. Then this pin is stale:")
    print(f"       update ARTIFACTS[{spec['source']!r}] in this script with the hash above,")
    print("       and check whether the container changed too -- a v3 file must not be fed to")
    print("       the v2 upgrader, and a v2 file must not be handed to a v3 engine.")
    return False


def print_upgrade_instructions() -> None:
    """Tell the user where the upgrader is, for a pre-existing v2 file.

    No pinned artifact ships v2 any more, so this no longer fires on a fresh download; it is
    kept for the case it was written for -- a copy downloaded before the republish.
    """
    tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upgrade_ninfer_v2_to_v3.py")
    print()
    print("[ACTION REQUIRED] A v2 container is rejected outright by a v3 engine.")
    print("  Upgrade it with the tool shipped in this archive:")
    print()
    print(f'    "{sys.executable}" "{tool}" INPUT.ninfer OUTPUT.v3.ninfer')


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify a NInfer v3 model artifact.")
    parser.add_argument("--artifact", choices=sorted(ARTIFACTS), default="quasar",
                        help="quasar (recommended) or nvfp4full; both ship v3")
    parser.add_argument("--dest", default=None, help="Override the destination model file path.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify the existing file, do not download.")
    args = parser.parse_args()

    spec = ARTIFACTS[args.artifact]
    target = os.path.abspath(args.dest or os.environ.get("MODEL")
                             or os.path.join(LOCAL_DIR, spec["target"]))
    target_dir = os.path.dirname(target)

    print(f"[INFO] Artifact: {args.artifact} ({spec['container']} container, from {spec['repo']})")
    print(f"[INFO] Target:   {target}")

    if args.verify_only:
        return 0 if verify_file(target, spec) else 1

    if os.path.exists(target) and verify_file(target, spec):
        print("Already downloaded and verified.")
        if spec["container"] != "v3":
            print_upgrade_instructions()
        return 0

    # huggingface_hub 1.x ignores HF_HUB_ENABLE_HF_TRANSFER, and on 0.x it raises when
    # hf_transfer is not installed -- which the broad except below would turn into a
    # failed download rather than a fallback.
    import importlib.util
    if importlib.util.find_spec("hf_transfer"):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import hf_hub_download

    print(f"Downloading {spec['source']} from {spec['repo']}...")
    started = time.time()
    os.makedirs(target_dir, exist_ok=True)
    try:
        path = hf_hub_download(repo_id=spec["repo"], filename=spec["source"],
                               local_dir=target_dir)
    except Exception as error:  # noqa: BLE001
        print(f"Error downloading: {error}")
        return 1
    print(f"Downloaded in {time.time() - started:.1f}s: {path}")

    # The repositories name their files differently from the launchers' convention.
    if os.path.abspath(path) != target:
        os.replace(path, target)  # overwrites atomically; no remove window
        print(f"Renamed to the launcher convention: {target}")

    if not verify_file(target, spec):
        return 1

    if spec["container"] != "v3":
        print_upgrade_instructions()
    return 0


if __name__ == "__main__":
    sys.exit(main())
