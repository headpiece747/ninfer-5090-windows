"""Assert the shipped context-cache bounds retain the working set they are documented to retain.

The README's bounds claim is measured on "five distinct ~530-token prompts sent and resent". That
claim went stale once already and nothing noticed: the same bounds gave 5/5 on 2026-09-17, the
retention rework landed on 2026-09-19, and from then on the same bounds gave 3/5 -- silent, with no
error and no signal beyond the cache column. So the shape is repeated here, through the profile
table's own `launcher_args`, on every release check. Ten requests, about twenty seconds.

Exit code 1 if fewer than four of the five resends reuse anything.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import profiles  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SERVE = REPO / "build" / "apps" / "ninfer-serve.exe"
SERVE_FALLBACK = Path(r"C:\AI\ninfer-v3-windows\build\apps\ninfer-serve.exe")
MODEL = Path(r"C:\AI\models\qwen3_8_27b_nvfp4qat.v3.ninfer")
LOG = Path(r"C:\Users\tobia\AppData\Local\Temp\opencode\check_cache_capacity.jsonl")
MODEL_ID = "qwen3.8-27b-quasar-v3-dflash2-vision"
REQUIRED_HITS = 4

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]


def filler(tag: int) -> str:
    return " ".join("%s%d" % (WORDS[(i * 7 + tag) % len(WORDS)], i) for i in range(115))


def send(port: int, tag: int) -> None:
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "Reply with a single word."},
            {"role": "user",
             "content": "Topic %d.\n%s\n\nReply with the word ok." % (tag, filler(tag))},
        ],
        "max_tokens": 4,
        "temperature": 0,
    }
    request = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % port,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        json.loads(response.read().decode("utf-8"))


def main() -> int:
    serve = SERVE if SERVE.exists() else SERVE_FALLBACK
    profile = profiles.by_model_id(MODEL_ID)
    port = int(profile["port"])
    LOG.unlink(missing_ok=True)
    argv = [str(serve), str(MODEL)] + profiles.launcher_args(profile) + [
        "--request-log-jsonl", str(LOG)]
    server = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            time.sleep(2)
            try:
                with urllib.request.urlopen(
                        "http://127.0.0.1:%d/health" % port, timeout=3) as probe:
                    if probe.status == 200:
                        break
            except (urllib.error.URLError, OSError):
                continue
        else:
            print("  FAIL: the server did not become ready")
            return 1

        for tag in range(5):
            send(port, tag)
        for tag in range(5):
            send(port, tag)
        time.sleep(7)  # the request log flushes per interval

        done = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines()]
        resends = [record for record in done if record.get("event") == "request_done"][5:]
        hits = sum(1 for record in resends if record["result"]["prefix_cache_hit_tokens"] > 0)
        cached = sum(record["result"]["prefix_cache_hit_tokens"] for record in resends)
        prompts = sum(record["result"]["prompt_tokens"] for record in resends)
        rate = 100.0 * cached / prompts if prompts else 0.0
        bounds = dict(profiles.ordered_flags(profile))
        print("  bounds  host-state-slots=%s private=%s shared=%s"
              % (bounds.get("--host-state-slots"), bounds.get("--max-private-continuations"),
                 bounds.get("--max-shared-prefixes")))
        print("  resend  %d/5 hits, token-level %.1f%%" % (hits, rate))
        if hits < REQUIRED_HITS:
            print("  FAIL: the shipped bounds retain fewer prefixes than they are documented to")
            return 1
        print("  PASS")
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    sys.exit(main())
