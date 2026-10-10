"""Drive `chunked_context_soak.py` against a lane this script starts and stops.

The soak is a client by design, so owning the lane is this script's job. Two launch details are not
cosmetic and are why this exists as a tool rather than as a note (`AGENTS.md` carries the rule):

  * **Detached.** A run launched by a shell here dies with the shell's process tree when opencode evicts
    an idle location -- about hourly, and reported upstream (`anomalyco/opencode` #51828 and siblings).
    This script is meant to be started with `CREATE_BREAKAWAY_FROM_JOB | DETACHED_PROCESS` from a launcher
    that exits, so its parent chain reaches no shell; `tools/scripts/check_console_free.py` is unrelated
    to that (the console is a separate hazard).
  * **Its own logs.** The lane log and the soak log are separate files, and the driver's own output is
    only a summary, so a driver that dies still leaves the measurement behind. The soak's completion
    notification is lost anyway when a run outlives ~60 minutes of location inactivity, so the log file
    is the report.

Usage:
    python tools/bench/soak_lane.py --cycles 10 --tag short
    python tools/bench/soak_lane.py --lane nvidia_v3_mtp4_vision --artifact C:\\AI\\models\\<file>.ninfer
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
SERVE = REPO / "build" / "apps" / "ninfer-serve.exe"
LOG_DIR = REPO / "profiles" / "bench" / "soak"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--lane", default="quasar_v3_dflash2_vision",
                    help="substring of the launcher profile file to run")
parser.add_argument("--artifact", default=r"C:\AI\models\qwen3_8_27b_nvfp4qat.v3.ninfer",
                    help="artifact path to serve")
parser.add_argument("--port", type=int, default=18190)
parser.add_argument("--cycles", type=int, default=10)
parser.add_argument("--chunks", type=int, default=6)
parser.add_argument("--chunk-chars", type=int, default=100_000)
parser.add_argument("--decode", type=int, default=2048)
parser.add_argument("--tag", default="run")
args = parser.parse_args()

LOG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "tools" / "release"))
from profiles import PROFILES, launcher_args  # noqa: E402

profile = next(p for p in PROFILES if args.lane in p["file"])
flags = launcher_args(profile, port=args.port, model_id="qwen3.8-27b-soak")
lane_log_path = LOG_DIR / f"lane-{args.tag}.log"
soak_log_path = LOG_DIR / f"soak-{args.tag}.log"
print(f"lane: {profile['file']}  artifact: {args.artifact}", flush=True)
print(f"lane log: {lane_log_path}", flush=True)
print(f"soak log: {soak_log_path}", flush=True)

with lane_log_path.open("w", encoding="utf-8", errors="replace") as lane_log:
    proc = subprocess.Popen([str(SERVE), args.artifact, *flags],
                            cwd=str(REPO), stdout=lane_log, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 300
        while time.time() < deadline:
            time.sleep(2.0)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=3) as r:
                    json.loads(r.read())
                    break
            except (urllib.error.URLError, OSError, json.JSONDecodeError):
                if proc.poll() is not None:
                    raise SystemExit(f"lane exited early; see {lane_log_path}")
        else:
            raise SystemExit(f"lane never listened; see {lane_log_path}")
        print(f"lane up on {args.port} at {time.strftime('%H:%M:%S')}; starting the soak", flush=True)

        started = time.time()
        with soak_log_path.open("w", encoding="utf-8", errors="replace") as soak_log:
            code = subprocess.call(
                [sys.executable, "-u", str(REPO / "tools" / "bench" / "chunked_context_soak.py"),
                 "--port", str(args.port), "--cycles", str(args.cycles), "--chunks", str(args.chunks),
                 "--chunk-chars", str(args.chunk_chars), "--decode", str(args.decode)],
                cwd=str(REPO), stdout=soak_log, stderr=subprocess.STDOUT)
        print(f"soak rc={code} after {(time.time() - started) / 60:.1f} min", flush=True)
        for line in soak_log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-10:]:
            print(f"    {line}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("lane stopped", flush=True)
