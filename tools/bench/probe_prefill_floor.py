"""Probe the fixed ~46 ms base in the `prefill` phase, on a served lane.

The open item (docs/research/lane-coverage-2026-10-08.md) recorded a ~45 ms floor on near-zero-work
requests and ruled out KV page allocation and per-conversation restores. This is the instrument that
reproduces it and separates it from its neighbours: it starts a lane with its request log on, fires
requests whose prompts differ (an identical repeat takes `private_response_replay` and never decodes), and
prints the phase fields by name from the log's own schema.

Measured 2026-10-10 on the NVIDIA lane as it ships (fp8, DFlash2 d9), one variable per run:

    prompt 10 chars   (77 tokens)    prefill 45.6 ms    device_wait 16-31 ms
    prompt 2,000      (325 tokens)   prefill 70.4 ms
    prompt 20,000     (2,575 tokens) prefill 214.7 ms   device_wait 164.6 ms

so `prefill` reads a ~45 ms floor at 77 tokens and grows by 64-100 us/token above it (the two-point
slope between consecutive sizes), and `computed_prefill_tokens` equals `prompt_tokens` exactly at every
size -- there is no fixed token count. A second, separate component was bimodal at 15.6 ms and showed up in
`queue_wait_seconds`, in `initial_binding_seconds`, or in neither (`total` minus the phases), on different
requests: the engine's scheduling poll (`queue_cv_.wait_for(1 ms)`, src/runtime/engine/engine_core.h:2355)
has no predicate and resolves on the system timer, which is 15.625 ms in a process that never calls
`timeBeginPeriod`.
`apps/serve/main.cpp` raises it, and the patched binary reads a flat 2.0-2.6 ms `initial_binding` against a
bimodal 0.9/16.0-17.0 before (interleaved control: build-test's unpatched binary, same lane, artifact and
requests, `queue_wait` 18.8/13.0/13.5/15.2/15.0 and `total` mean 77.3 ms against 56.3).

Ruled out, each by a single-variable run: a fixed-token prefill (above); CUDA graph capture
(`--no-cuda-graph`: 49.3 vs 45.6 ms); the vision path (`--vision` dropped: 46.4 vs 45.6); KV page
allocation, per-conversation restores and the context length (the document's own runs: 61.2 ms at
`--kv-capacity auto` against 63.4 ms at `--max-context 32768`). The prefix path is *inverted*:
`--no-prefix-reuse` makes the floor worse (117-163 ms), so reuse is what keeps it at 46.

The base is host-visible wall time inside `staged.elapsed_seconds`
(src/models/qwen3_5/program/prefill.cpp:749) and it scales with neither the prompt nor the context; the
draft backend is not it either (the mtp4 lane reads the same floor as dflash2, 43.7-45.2 against
44.2-46.4 ms). What remains is the ordinary cost of one chunk: the weights are read whatever the token
count, and the chunk's kernels are submitted from the host.

Counts are printed raw, never through the seconds formatter: the first version of this probe printed a
token count through `* 1000`, and a 76-token prompt read as "76000.0 prefill tokens" -- a phantom that
cost real time before it was explained.

Usage:
    python tools/bench/probe_prefill_floor.py [--requests N] [--prompt CHARS]
    python tools/bench/probe_prefill_floor.py --serve <other build> ...   (A/B two binaries)
    PROBE_NO_GRAPH=1 | PROBE_NO_VISION=1 | PROBE_NO_PREFIX=1 | PROBE_SPEC=mtp | PROBE_DRAFT0=1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[2]
SERVE = REPO / "build" / "apps" / "ninfer-serve.exe"
ARTIFACT = r"C:\AI\models\qwen3_8_27b_nvfp4nvidia.v3.ninfer"
LOG_DIR = Path(os.environ.get("PROBE_LOG_DIR", REPO / "profiles" / "bench" / "prefill-floor"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
PORT = 18090
MODEL_ID = "qwen3.6-27b"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--requests", type=int, default=3)
parser.add_argument("--prompt", type=int, default=10, help="characters in the prompt")
parser.add_argument("--lane", default="nvidia_v3_dflash2_vision",
                    help="substring of the launcher profile file to run")
parser.add_argument("--artifact", default=ARTIFACT, help="artifact path to serve")
parser.add_argument("--serve", default=str(SERVE), help="server binary (A/B two builds)")
args = parser.parse_args()

sys.path.insert(0, str(REPO / "tools" / "release"))
from profiles import PROFILES, launcher_args  # noqa: E402

profile = next(p for p in PROFILES if args.lane in p["file"])
print(f"lane: {profile['file']}", flush=True)
run_stamp = time.strftime("%Y%m%dT%H%M%S")
request_log = LOG_DIR / f"probe-prefill-floor-{run_stamp}.jsonl"
flags = launcher_args(profile, port=PORT, model_id=MODEL_ID) + [
    "--request-log-jsonl", str(request_log)]
for env_name, flag in (("PROBE_NO_GRAPH", "--no-cuda-graph"), ("PROBE_NO_PREFIX", "--no-prefix-reuse")):
    if os.environ.get(env_name):
        flags.append(flag)
        print(f"probe: {flag} (from {env_name})", flush=True)
if os.environ.get("PROBE_NO_VISION"):
    flags = [f for f in flags if f != "--vision"]
    print("probe: --vision dropped (from PROBE_NO_VISION)", flush=True)
if os.environ.get("PROBE_SPEC"):
    flags += ["--spec", os.environ["PROBE_SPEC"]]
    print(f"probe: --spec {os.environ['PROBE_SPEC']} appended (later flag wins)", flush=True)
if os.environ.get("PROBE_DRAFT0"):
    flags += ["--draft-tokens", "0"]
    print("probe: --draft-tokens 0 appended", flush=True)

lane_log_path = LOG_DIR / f"probe-prefill-floor-lane-{run_stamp}.log"
lane_log = lane_log_path.open("w", encoding="utf-8", errors="replace")
print(f"lane log: {lane_log_path}", flush=True)
proc = subprocess.Popen([args.serve, args.artifact, *flags], cwd=str(REPO), stdout=lane_log,
                        stderr=subprocess.STDOUT)
try:
    deadline = time.time() + 240
    while time.time() < deadline:
        time.sleep(2.0)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=3) as response:
                model = json.loads(response.read())["data"][0]["id"]
                break
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            if proc.poll() is not None:
                raise SystemExit(f"lane exited early; see {lane_log_path}")
    else:
        raise SystemExit(f"lane never listened; see {lane_log_path}")
    print(f"lane up on {PORT} as {model}", flush=True)

    for index in range(args.requests):
        body = json.dumps({"model": model,
                           "messages": [{"role": "user",
                                         "content": "x" * args.prompt + f" probe {index} {time.time_ns()}"}],
                           "max_tokens": 1, "temperature": 0}).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        started = time.time()
        with urllib.request.urlopen(request, timeout=600) as response:
            json.loads(response.read())
        wall = (time.time() - started) * 1000.0

        records = [json.loads(line) for line in request_log.read_text(encoding="utf-8",
                                                                      errors="replace").splitlines()
                   if line.strip()]
        done = [r for r in records if r.get("event") == "request_done"]
        if not done:
            print(f"  request {index + 1}: wall {wall:6.1f} ms; no request_done record yet", flush=True)
            continue
        record = done[-1]
        timings = record.get("timings_seconds") or {}
        first = record.get("first_output_timing") or {}
        engine = record.get("engine_timing") or {}
        result = record.get("result") or {}

        def as_ms(value: object) -> str:
            return f"{value * 1000:.1f}" if isinstance(value, (int, float)) else str(value)[:60]

        shown = "  ".join(f"{k}={as_ms(v)}" for k, v in timings.items())
        print(f"  request {index + 1}: wall {wall:6.1f} ms   timings[{shown}] ms", flush=True)
        print(f"      engine[device_wait={as_ms(engine.get('device_wait_exposed_seconds'))} "
              f"queue_wait={as_ms(engine.get('queue_wait_seconds'))}] ms   "
              f"first[elapsed={as_ms(first.get('elapsed_seconds'))} "
              f"binding={as_ms(first.get('initial_binding_seconds'))}]", flush=True)
        first_simple = {k: v for k, v in first.items() if k != "context_transfers"}
        print(f"      result[prompt_tokens={result.get('prompt_tokens')} "
              f"computed_prefill_tokens={result.get('computed_prefill_tokens')} "
              f"prefix_cache_hit_tokens={result.get('prefix_cache_hit_tokens')} "
              f"prefix_reuse_path={result.get('prefix_reuse_path')}]\n"
              f"      speculative={str(record.get('speculative'))[:90]}\n"
              f"      host_exposed={str(engine.get('host_exposed_seconds'))[:420]}\n"
              f"      first_full={str(first_simple)[:420]}", flush=True)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
    lane_log.close()
    print("lane stopped", flush=True)
