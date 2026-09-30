#!/usr/bin/env python3
"""A/B test: NVFP4-full with vs without BF16 exceptions.

Measures both artifacts interleaved to control for clock drift. Each round:
1. Start server with artifact A, measure DFlash2 acceptance + tok/s
2. Start server with artifact B, measure DFlash2 acceptance + tok/s
3. Repeat for the configured number of rounds

The variant (noex) encodes the 27 BF16 exception projections to NVFP4.
The control is the current NVFP4-full with 27 BF16 exceptions.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import wait_ready, start_engine, stop_engine  # noqa: E402
from profiles import launcher_environment, template_path, NVFP4FULL  # noqa: E402

EXE = Path(r"C:\AI\ninfer-v3-windows\build\apps\ninfer-serve.exe")
MODELS = Path(r"C:\AI\models")
VARIANT = Path(r"C:\AI\ninfer-v3-windows\out\qwen3_8_27b_nvfp4full_nvdiv.v3.ninfer")
PORT_BASE = 8096
ROUNDS = 3

# DFlash2 configuration matching the shipped profile
SPEC = "dflash2"
DRAFT = 7
VISION = True
LM_HEAD = True
MODEL_ID = "ab-bf16-exceptions"


def measure(artifact: Path, port: int, round_num: int) -> dict:
    """Start server, run benchmark, return metrics."""
    log_path = Path(r"C:\AI\bench") / f"ab_{artifact.stem}_r{round_num}.log"
    args = [
        str(EXE), str(artifact),
        "--model-id", MODEL_ID,
        "--spec", SPEC,
        "--draft-tokens", str(DRAFT),
        "--port", str(port),
    ]
    if VISION:
        args.append("--vision")
    if LM_HEAD:
        args.append("--lm-head-draft")

    # Add invariant flags from the profile
    from profiles import INVARIANT_FLAGS
    for flag, value in INVARIANT_FLAGS:
        args.append(flag)
        if value is not None:
            args.append(value)
    args.extend(["--chat-template", template_path()])

    # The environment the launchers pin is passed through os.environ by the caller, not as an
    # argument: start_engine takes no env parameter, so a computed value here would be dropped and
    # the lane would measure the engine's default CUDA wait schedule rather than the shipped one.
    # That is exactly the defect INVARIANT_FLAGS composition exists to prevent in the other harnesses.
    os.environ.update(launcher_environment({"device_state_slots": 1}))
    proc, handle = start_engine(args, log_path, str(EXE.parents[1]))

    ready = wait_ready(port, proc, timeout=240)
    if not ready:
        stop_engine(proc, handle)
        return {"error": "server failed to start", "artifact": artifact.name, "round": round_num}

    # Run benchmark: send requests and measure acceptance + throughput.
    # Annotated rather than left to inference: an unannotated dict literal types its values as
    # `object`, so `metrics["acceptance"].append(...)` is an attr-defined error under mypy, and the
    # fix has to be here rather than a suppression because the list really is a list of floats.
    base = f"http://127.0.0.1:{port}"
    metrics: dict[str, object] = {"artifact": artifact.name, "round": round_num,
                                  "acceptance": [], "tok_per_s": []}
    acceptance: list[float] = metrics["acceptance"]  # type: ignore[assignment]
    tok_per_s: list[float] = metrics["tok_per_s"]  # type: ignore[assignment]

    # Warmup
    for _ in range(3):
        _send_request(base, MODEL_ID, "def hello(): pass", 32)

    # Measurement rounds
    for i in range(5):
        result = _send_request(base, MODEL_ID, f"def test_{i}(): return {i} * 42", 128)
        if "error" in result:
            acceptance.append(0.0)
            tok_per_s.append(0.0)
        else:
            acceptance.append(float(result.get("acceptance", 0.0)))
            tok_per_s.append(float(result.get("tok_per_s", 0.0)))

    stop_engine(proc, handle)
    return metrics


def _send_request(base: str, model_id: str, prompt: str, max_tokens: int) -> dict:
    """Send a chat completion request and extract metrics."""
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        # Extract acceptance and throughput from usage or stats
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        # Acceptance is not directly in the response; we'd need to parse the log
        # For now, return what we can
        return {
            "completion_tokens": completion_tokens,
            "acceptance": 0.0,  # Will be parsed from log
            "tok_per_s": 0.0,   # Will be parsed from log
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def parse_log_metrics(log_path: Path) -> dict:
    """Parse acceptance and throughput from the engine log.

    Log format:
        dflash2 accepted 22/52 (42.3%)
        decode 222.2 tok/s
    """
    if not log_path.exists():
        return {"acceptance": 0.0, "tok_per_s": 0.0}

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    acceptance = 0.0
    tok_per_s = 0.0
    for line in lines:
        if "accepted" in line:
            import re
            m = re.search(r"accepted\s+\d+/\d+\s+\((\d+\.?\d*)%\)", line)
            if m:
                acceptance = float(m.group(1))
        if "decode" in line and "tok/s" in line:
            import re
            m = re.search(r"decode\s+(\d+\.?\d*)\s*tok/s", line)
            if m:
                tok_per_s = float(m.group(1))
    return {"acceptance": acceptance, "tok_per_s": tok_per_s}


def main() -> int:
    control = MODELS / NVFP4FULL
    variant = VARIANT

    print("A/B Test: Divisor Source on NVFP4-full")
    print(f"Control: {control.name} (fork calibration divisors)")
    print(f"Variant: {variant.name} (NVIDIA Local-Hessian divisors)")
    print(f"Rounds: {ROUNDS}")
    print()

    results: dict[str, list[dict]] = {"control": [], "variant": []}

    for r in range(ROUNDS):
        print(f"--- Round {r + 1}/{ROUNDS} ---")

        # Control first
        print(f"  Measuring control ({control.name})...")
        m = measure(control, PORT_BASE, r)
        log = Path(r"C:\AI\bench") / f"ab_{control.stem}_r{r}.log"
        parsed = parse_log_metrics(log)
        m.update(parsed)
        results["control"].append(m)
        print(f"    acceptance: {m['acceptance']:.1f}%, tok/s: {m['tok_per_s']:.1f}")

        # Variant second
        print(f"  Measuring variant ({variant.name})...")
        m = measure(variant, PORT_BASE + 1, r)
        log = Path(r"C:\AI\bench") / f"ab_{variant.stem}_r{r}.log"
        parsed = parse_log_metrics(log)
        m.update(parsed)
        results["variant"].append(m)
        print(f"    acceptance: {m['acceptance']:.1f}%, tok/s: {m['tok_per_s']:.1f}")

        time.sleep(5)  # Let the card settle between rounds

    # Summary
    print()
    print("=== Summary ===")
    for label, key in [("Control (fork calibration)", "control"), ("Variant (NVIDIA divisors)", "variant")]:
        accs = [m["acceptance"] for m in results[key]]
        toks = [m["tok_per_s"] for m in results[key]]
        avg_acc = sum(accs) / len(accs) if accs else 0
        avg_tok = sum(toks) / len(toks) if toks else 0
        print(f"  {label}: acceptance {avg_acc:.1f}%, tok/s {avg_tok:.1f}")

    # Save results
    out = Path(r"C:\AI\bench\ab_bf16_exceptions.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
