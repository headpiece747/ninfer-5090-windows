import argparse
import hashlib
import os
import sys
import time

EXPECTED_SIZE = 18_638_209_796
EXPECTED_SHA256 = "df3c9c3a3660d688f0c2158d54fef8c66f21d723bd7a1b0b12acc908e86d12b7"
REPO_ID = "cometkim/Qwen3.8-27B-nvfp4qat-NInfer"
FILENAME = "qwen3_8_27b_nvfp4qat.ninfer"
LOCAL_DIR = r"C:\ai\models"
FILE_PATH = os.path.join(LOCAL_DIR, FILENAME)


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    total_size = os.path.getsize(path)
    processed = 0
    start = time.time()
    with open(path, "rb") as f:
        while chunk := f.read(64 * 1024 * 1024):
            h.update(chunk)
            processed += len(chunk)
            pct = (processed / total_size) * 100
            elapsed = time.time() - start
            mb_s = (processed / (1024 * 1024)) / max(elapsed, 0.001)
            print(
                f"\rVerifying SHA-256: {pct:5.1f}% ({processed / (1024**3):.2f}/{total_size / (1024**3):.2f} GiB) at {mb_s:.0f} MB/s...",
                end="",
                flush=True,
            )
    print()
    return h.hexdigest()


def verify_file(path: str) -> bool:
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return False

    size = os.path.getsize(path)
    print(f"Checking file: {path}")
    print(f"File size: {size} bytes ({size / (1024**3):.2f} GiB)")
    if size != EXPECTED_SIZE:
        print(f"[FAIL] Size mismatch: expected {EXPECTED_SIZE}, got {size}")
        return False
    print("[OK] Size matches expected size.")

    print(f"Computing SHA-256 hash (expected: {EXPECTED_SHA256})...")
    actual_sha = compute_sha256(path)
    print(f"Computed SHA-256: {actual_sha}")
    if actual_sha.lower() == EXPECTED_SHA256.lower():
        print("[SUCCESS] SHA-256 checksum verified perfectly!")
        return True
    else:
        print(f"[FAIL] SHA-256 mismatch!\n  Expected: {EXPECTED_SHA256}\n  Actual:   {actual_sha}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download and verify Qwen 3.8 27B QUASAR QAT model.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify the existing file without downloading.")
    args = parser.parse_args()

    print(f"[INFO] Python interpreter: {sys.executable} (Python {sys.version.split()[0]})")

    if os.path.exists(FILE_PATH) and os.path.getsize(FILE_PATH) == EXPECTED_SIZE:
        if args.verify_only:
            ok = verify_file(FILE_PATH)
            sys.exit(0 if ok else 1)
        else:
            print(f"Found existing file at {FILE_PATH}.")
            ok = verify_file(FILE_PATH)
            if ok:
                print("File is already fully downloaded and verified!")
                sys.exit(0)
            print("Existing file hash did not match. Re-downloading...")

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import hf_hub_download

    print(f"Starting ultra-fast download of {FILENAME} from {REPO_ID} to {LOCAL_DIR}...")
    start_time = time.time()
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=LOCAL_DIR,
        )
    except Exception as e:
        print(f"Error downloading: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time
    size = os.path.getsize(path)
    print(f"Download complete in {elapsed:.1f}s!")
    print(f"File path: {path}")
    print(f"File size: {size} bytes ({size / (1024**3):.2f} GiB)")

    if not verify_file(path):
        sys.exit(1)


if __name__ == "__main__":
    main()
