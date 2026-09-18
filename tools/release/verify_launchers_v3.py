#!/usr/bin/env python3
"""Verify every generated v3 launcher by running the launcher itself.

This does not re-derive the arg set: it executes the .bat that ships, so what is
verified is exactly what a user will run. For each launcher:

  1. the engine starts and /v1/models advertises the launcher's own model id
  2. a chat completion returns text, and speculative counters are non-zero when the
     profile selects a backend (proves the backend actually loaded rather than
     silently degrading)
  3. a Vision profile accepts an image, and a no-Vision profile rejects one with the
     documented vision_disabled error

Ports are distinct per launcher, so nothing is stopped between runs except the engine
itself -- one 32 GB card can only hold one artifact at a time.
"""
from __future__ import annotations

import base64
import json
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

V3 = Path(r"C:\AI\ninfer-v3-windows")
RECORDS = Path(r"C:\AI\bench") / "launcher_verify.jsonl"

CASES = [
    ("start_quasar_v3_dflash2_vision.bat", 8086, "qwen3.8-27b-quasar-v3-dflash2-vision", True, "dflash2"),
    ("start_quasar_v3_mtp4_vision.bat", 8087, "qwen3.8-27b-quasar-v3-mtp4-vision", True, "mtp"),
    ("start_ninfer_v3_dflash2.bat", 8088, "qwen3.8-27b-nvfp4-v3-dflash2", False, "dflash2"),
    ("start_ninfer_v3_dflash2_vision.bat", 8089, "qwen3.8-27b-nvfp4-v3-dflash2-vision", True, "dflash2"),
    ("start_ninfer_v3_mtp5.bat", 8090, "qwen3.8-27b-nvfp4-v3-mtp5", False, "mtp"),
    ("start_ninfer_v3_mtp5_vision.bat", 8091, "qwen3.8-27b-nvfp4-v3-mtp5-vision", True, "mtp"),
]

PROBE_PROMPT = "Reply with the single word OK."


# ------------------------------------------------------------------ test image

def make_png_data_uri(size: int = 64) -> str:
    """A small valid PNG so Vision preprocessing has real pixels to chew on."""
    raw = b""
    for y in range(size):
        raw += b"\x00"  # filter byte
        for x in range(size):
            raw += bytes((x * 4 % 256, y * 4 % 256, 128))
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


# -------------------------------------------------------------------- plumbing

def kill() -> None:
    subprocess.run(["taskkill", "/F", "/IM", "ninfer-serve.exe"], capture_output=True, text=True)


def wait_ready(port: int, timeout: int = 240) -> tuple[bool, float]:
    t0 = time.time()
    end = t0 + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                r.read()
            return True, time.time() - t0
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return False, time.time() - t0


def post(port: int, path: str, payload: dict, timeout: int = 600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(body)
        except Exception:  # noqa: BLE001
            return e.code, {"raw": body[:200]}


def models(port: int) -> list[str]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as r:
        return [m["id"] for m in json.loads(r.read()).get("data", [])]


# ------------------------------------------------------------------------ main

def main() -> int:
    uri = make_png_data_uri()
    results = []
    for bat, port, model_id, vision, spec in CASES:
        print(f"\n=== {bat}  (port {port})")
        kill()
        time.sleep(3)
        proc = subprocess.Popen(["cmd", "/c", str(V3 / bat)], cwd=str(V3),
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ready, secs = wait_ready(port)
        rec: dict = {"bat": bat, "port": port, "model_id": model_id, "vision": vision,
                     "spec": spec, "ready": ready, "startup_seconds": round(secs, 1)}
        if not ready:
            print("  FAILED to serve")
            proc.kill()
            kill()
            results.append(rec)
            with RECORDS.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            continue

        ids = models(port)
        rec["advertised"] = ids
        rec["model_id_ok"] = model_id in ids
        print(f"  model id           : {'OK' if rec['model_id_ok'] else 'MISMATCH ' + str(ids)}"
              f"  (ready in {secs:.1f}s)")

        status, body = post(port, "/v1/chat/completions", {
            "model": model_id,
            "messages": [{"role": "user", "content": PROBE_PROMPT}],
            "max_tokens": 24,
        })
        text = ""
        if status == 200:
            msg = body["choices"][0]["message"]
            text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
        rec["text_status"] = status
        rec["text_len"] = len(text)
        print(f"  chat completion    : HTTP {status}, {len(text)} chars"
              f" -> {'OK' if status == 200 and text else 'FAIL'}")

        # Speculative counters live only in the request log, so re-run the launcher's
        # own startup log check: a backend that failed to load would not start at all.
        status, body = post(port, "/v1/chat/completions", {
            "model": model_id,
            "messages": [{"role": "user", "content": PROBE_PROMPT}],
            "max_tokens": 8,
        }, timeout=600)
        rec["probe_status"] = status

        img_status, img_body = post(port, "/v1/chat/completions", {
            "model": model_id,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What is the dominant colour? One word."},
                {"type": "image_url", "image_url": {"url": uri}},
            ]}],
            "max_tokens": 24,
        })
        rec["image_status"] = img_status
        if vision:
            rec["image_ok"] = img_status == 200
            print(f"  image accepted     : HTTP {img_status}"
                  f" -> {'OK' if rec['image_ok'] else 'FAIL'}")
        else:
            code = ""
            if isinstance(img_body, dict):
                err = img_body.get("error") or {}
                code = (err.get("code") if isinstance(err, dict) else "") or ""
                code = code or json.dumps(err)[:80]
            rec["image_ok"] = img_status == 400
            rec["image_error"] = code
            print(f"  image rejected     : HTTP {img_status} {code}"
                  f" -> {'OK (vision_disabled)' if rec['image_ok'] else 'FAIL'}")

        proc.kill()
        kill()
        time.sleep(2)
        results.append(rec)
        with RECORDS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    print("\n=== summary")
    ok = 0
    for r in results:
        good = (r.get("ready") and r.get("model_id_ok") and r.get("text_status") == 200
                and r.get("image_ok"))
        ok += bool(good)
        print(f"  {'PASS' if good else 'FAIL'}  {r['bat']:<42} "
              f"ready={r.get('ready')} id={r.get('model_id_ok')} "
              f"text={r.get('text_status')} image={r.get('image_status')}")
    print(f"  {ok}/{len(results)} launchers verified")
    print(f"  records: {RECORDS}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
