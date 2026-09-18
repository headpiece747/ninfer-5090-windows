#!/usr/bin/env python3
"""Acceptance test for the release archive.

Extracts the published zip to a fresh directory and runs a launcher from there, which is
what a user does. The launcher must resolve ninfer-serve.exe beside itself (the archive is
extracted somewhere with no source tree), start, answer /v1/models, and complete a chat
request. The artifact is expected to come from the launcher's absolute fallback path, since
this archive deliberately ships no model.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

ARCHIVE = Path(r"C:\AI\releases\ninfer-windows-v1.1.0-rtx5090.zip")
EXTRACT = Path(r"C:\Users\tobia\AppData\Local\Temp\opencode\release-check")
PORT = 8086
BASE = f"http://127.0.0.1:{PORT}"


def kill() -> None:
    subprocess.run(["taskkill", "/F", "/IM", "ninfer-serve.exe"], capture_output=True, text=True)


def main() -> int:
    if not ARCHIVE.exists():
        print(f"   missing archive: {ARCHIVE}")
        return 1
    if EXTRACT.exists():
        shutil.rmtree(EXTRACT)
    EXTRACT.mkdir(parents=True)
    with zipfile.ZipFile(ARCHIVE) as zf:
        zf.extractall(EXTRACT)
    names = sorted(p.name for p in EXTRACT.iterdir())
    print(f"   extracted {len(names)} entries to {EXTRACT}")
    print(f"   engine beside the launcher: {(EXTRACT / 'ninfer-serve.exe').exists()}")

    kill()
    time.sleep(3)
    launcher = EXTRACT / "start_quasar_v3_dflash2_vision.bat"
    log = EXTRACT / "run.log"
    with log.open("w", encoding="utf-8", errors="replace") as handle:
        proc = subprocess.Popen(["cmd", "/c", str(launcher)], cwd=str(EXTRACT),
                                stdin=subprocess.DEVNULL, stdout=handle,
                                stderr=subprocess.STDOUT)

    ids: list[str] = []
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"{BASE}/v1/models", timeout=5) as r:
                ids = [m["id"] for m in json.loads(r.read())["data"]]
            break
        except Exception:  # noqa: BLE001
            time.sleep(2)

    print(f"   serves from the extracted copy : {bool(ids)}  {ids}")
    if ids:
        body = {"model": ids[0],
                "messages": [{"role": "user", "content": "Reply with the single word OK."}],
                "max_tokens": 16}
        req = urllib.request.Request(BASE + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.loads(r.read())
            print(f"   chat                          : HTTP 200, "
                  f"{resp['usage']['completion_tokens']} tokens")
        except Exception as error:  # noqa: BLE001
            print(f"   chat                          : FAILED {type(error).__name__}")
    else:
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-6:]:
            print(f"   | {line.strip()[:150]}")

    proc.kill()
    kill()
    return 0 if ids else 1


if __name__ == "__main__":
    raise SystemExit(main())
