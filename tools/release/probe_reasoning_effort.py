#!/usr/bin/env python3
"""Map which reasoning_effort values the engine actually accepts, and why others fail.

The settings grid showed `high` failing on every model at 0.0s with an HTTP error, which is
a request rejection rather than a model failure. The harness recorded only the exception
type, so this probe sends one request per effort value and prints the status and body.

The launchers pass --default-thinking-budget 4096, so the leading hypothesis is that a
higher effort demands a larger thinking budget than the server was started with.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiles import PROFILES, launcher_args  # noqa: E402

CWD = r"C:\AI\ninfer-v3-windows"
EXE = CWD + r"\build\apps\ninfer-serve.exe"
PROFILE = PROFILES[0]  # QUASAR DFlash2; the effort question is not profile-specific
MODEL = rf"C:\AI\models\{PROFILE['art']}"
PORT = 8105
EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh"]
PROMPT = "Reply with the single word OK."


from engine import kill_servers, start_engine, stop_engine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=None,
                        help="probe this .ninfer instead of a shipped profile")
    parser.add_argument("--spec", default="mtp", choices=["none", "mtp", "dflash", "dflash2"],
                        help="draft route when --artifact is given")
    parser.add_argument("--model-id", default="effort-probe")
    parser.add_argument("--port", type=int, default=PORT)
    return parser.parse_args()


def main() -> int:
    options = parse_args()
    base = f"http://127.0.0.1:{options.port}"
    kill_servers()
    time.sleep(3)
    if options.artifact is not None:
        # An artifact that is not a shipped profile: the invariant flags the launchers pass, and a
        # small context because this probe only needs one short reply.
        args = [EXE, str(options.artifact), "--host", "127.0.0.1", "--port", str(options.port),
                "--model-id", options.model_id, "--max-context", "32768", "--kv-capacity", "auto",
                "--kv-dtype", "fp8", "--max-concurrency", "1", "--preserve-thinking",
                "--default-thinking-budget", "4096"]
        if options.spec != "none":
            args += ["--spec", options.spec, "--draft-tokens", "3", "--lm-head-draft"]
        log_path = Path(r"C:\AI\bench\effort_probe_artifact.txt")
    else:
        # The profile's shipped flags. Context is overridden small because this probe only needs
        # one short reply, and the port and model id are this harness's own.
        args = [EXE, MODEL] + launcher_args(PROFILE, port=options.port,
                                            model_id=options.model_id, max_context=32768)
        log_path = Path(r"C:\AI\bench\effort_probe.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(args, cwd=CWD, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT)
    for _ in range(120):
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as r:
                r.read()
            break
        except Exception:  # noqa: BLE001
            time.sleep(2)

    for effort in EFFORTS:
        body = {"model": options.model_id,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": 24, "reasoning_effort": effort}
        request = urllib.request.Request(base + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=600) as r:
                response = json.loads(r.read())
            message = response["choices"][0]["message"]
            reasoning = len(message.get("reasoning_content") or "")
            print(f"   {effort:<8} HTTP 200  {time.time()-started:5.1f}s  "
                  f"reasoning_chars={reasoning:<6} completion={response['usage']['completion_tokens']}")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            # The engine's error text can contain characters cp1252 cannot encode, which
            # would otherwise kill the probe while reporting.
            safe = detail.encode("ascii", "replace").decode("ascii")
            print(f"   {effort:<8} HTTP {error.code}  {time.time()-started:5.1f}s  {safe[:230]}")
        except Exception as error:  # noqa: BLE001
            print(f"   {effort:<8} {type(error).__name__}: {str(error)[:150]}")

    stop_engine(proc, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
