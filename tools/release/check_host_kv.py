#!/usr/bin/env python3
"""Did v2 know something about host KV? Compare v2's 16 GiB / 16 slots with v3's 8 GiB / 8.

Host KV is pinned system RAM, not VRAM, so it costs the GPU nothing. It holds offloaded and
cached conversation states, which is what a prefix hit reuses. v2 allocated twice ours over
16 slots; v3 runs the documented defaults.

Method mirrors the prefix-cache test that produced the 99.1% figure: five distinct ~530
token prompts, sent then resent, counting how many get a cache hit and what fraction of
prompt tokens were cached.

Usage: check_host_kv.py <host-kv-mib> <host-state-slots>
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiles import PROFILES, launcher_args  # noqa: E402

CWD = r"C:\AI\ninfer-v3-windows"
EXE = CWD + r"\build\apps\ninfer-serve.exe"
PROFILE = PROFILES[0]  # QUASAR DFlash2, the profile the cache bounds were measured on
MODEL = rf"C:\AI\models\{PROFILE['art']}"
MODEL_ID = "hostkv-probe"
PORT = 8106
BASE = f"http://127.0.0.1:{PORT}"

BODY = (
    "Review this module and answer in one short sentence.\n\n"
    "def schedule(items, budget):\n"
    "    total = 0\n"
    "    chosen = []\n"
    "    for item in sorted(items, key=lambda i: i.cost):\n"
    "        if total + item.cost > budget:\n"
    "            continue\n"
    "        total += item.cost\n"
    "        chosen.append(item)\n"
    "    return chosen, total\n\n"
)


from engine import kill_servers, stop_engine  # noqa: E402


def main() -> int:
    host_kv = sys.argv[1] if len(sys.argv) > 1 else "8192"
    slots = sys.argv[2] if len(sys.argv) > 2 else "8"

    kill_servers()
    time.sleep(3)
    log = Path(r"C:\AI\bench") / f"hostkv_{host_kv}_{slots}.txt"
    # The profile's shipped flags, with this harness's own port and model id. The two host
    # values under test are appended, since they are what this script varies.
    args = [EXE, MODEL] + launcher_args(PROFILE, port=PORT, model_id=MODEL_ID) + [
        "--host-state-slots", slots, "--host-kv-mib", host_kv]
    handle = log.open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(args, cwd=CWD, stdin=subprocess.DEVNULL, stdout=handle,
                            stderr=subprocess.STDOUT)
    ready = False
    for _ in range(120):
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(f"{BASE}/v1/models", timeout=5) as r:
                r.read()
            ready = True
            break
        except Exception:  # noqa: BLE001
            time.sleep(2)
    if not ready:
        print(f"   host-kv-mib={host_kv} slots={slots}: FAILED TO START")
        stop_engine(proc)
        return 1

    cache_line = ""
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "context cache |" in line:
            cache_line = line.strip()

    prompts = [BODY + f"# variant {i}\n" + (f"value = process(record, index={i})\n" * 40)
               for i in range(5)]

    def send(text: str):
        body = {"model": MODEL_ID, "messages": [{"role": "user", "content": text}],
                "max_tokens": 8}
        req = urllib.request.Request(BASE + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())

    for prompt in prompts:
        send(prompt)

    hits, cached_total, prompt_total = 0, 0, 0
    for prompt in prompts:
        response = send(prompt)
        details = response["usage"].get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens", 0)
        hits += 1 if cached > 0 else 0
        cached_total += cached
        prompt_total += response["usage"]["prompt_tokens"]

    tail = cache_line.split("states")[-1].strip() if "states" in cache_line else cache_line[-60:]
    rate = 100.0 * cached_total / prompt_total if prompt_total else 0.0
    print(f"   host-kv-mib={host_kv:<6} slots={slots:<3} hits {hits}/5  "
          f"token hit rate {rate:5.1f}%   | {tail[:60]}")

    stop_engine(proc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
