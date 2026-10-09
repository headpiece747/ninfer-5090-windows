#!/usr/bin/env python3
"""Measure speculative acceptance and decode on real text, at a lane's own configuration.

Why this instrument exists, in one paragraph, because a bench corpus cannot answer this question from
either end of its range: `ninfer_bench -n` generates after a one-token seed, so speculation has no
context to be predictable from (measured 0.085-0.129 acceptance), and the same corpus behind a real
prompt saturates it (0.991 in four consecutive runs), because the corpus is synthetic and tiled. Real
domains sit at 0.44-0.55, and nothing measured there until this tool.

Design decisions, each one a lesson already paid for:

  * **Greedy by default** (`--temperature 0`). At temperature 1.0 the generated text differs per run, and
    acceptance depends on the text: two rounds of the SAME artifact read 47.4% and 31.6%, a 16-point
    swing that cannot resolve any effect. Greedy makes each arm's generation identical, so the shipped
    arm's two rounds agree exactly and the acceptance statistic carries no noise. Use a non-zero
    temperature only to measure the level rather than to compare arms.
  * **The model id is read from `/v1/models` per arm**, never hard-coded. Artifacts carry their own id
    (this one's NVIDIA build serves `qwen3.8-27b-fp8attn`), and a hard-coded id is a 404 that reads like
    a broken artifact.
  * **One port per arm, and the serve is stopped, never a profiler killed.** A profiler terminated
    instead of its subject writes no report and leaves the subject alive holding the port, which is how
    a whole profile came back void.
  * **Acceptance is parsed tolerantly** of thousands separators: `accepted 351/1,112 (31.6%)` silently
    failing a `\\d+` pattern reads exactly like a missing line.

The figure to sanity-check before trusting a run is the absolute acceptance level: near 0.99 means the
prompt is too predictable, near 0.10 means too little context, and both mean the protocol is measuring
something other than the lanes' regime.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

ACCEPT = re.compile(r"accepted ([\d,]+)/([\d,]+) \(([\d.]+)%\)")
DECODE = re.compile(r"decode ([\d.]+) tok/s")
REPO = Path(__file__).resolve().parents[2]
SERVE = REPO / "build" / "apps" / ("ninfer-serve.exe" if sys.platform == "win32" else "ninfer-serve")
DEFAULT_TEMPLATE = REPO / "tools" / "chat_templates" / "qwen3_8.jinja"

LANE_FLAGS = ["--vision", "--spec", "dflash2", "--draft-tokens", "7", "--lm-head-draft",
              "--max-context", "262144", "--device-state-slots", "1", "--kv-capacity", "auto",
              "--kv-dtype", "fp8", "--prefill-chunk", "8192", "--max-concurrency", "1",
              "--host-context-mib", "8192", "--preserve-thinking", "--default-thinking-budget", "4096"]


def default_text() -> str:
    """Real prose from the corpus the perplexity authority names, assembled from its own data files."""
    root = REPO / "eval" / "corpora" / "perplexity-1m"
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file() or candidate.suffix not in (".txt", ".jsonl"):
            continue
        body = candidate.read_text(encoding="utf-8", errors="replace")
        if candidate.suffix == ".jsonl":
            pieces = []
            for line in body.splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = record.get("text") or record.get("content")
                if isinstance(text, str):
                    pieces.append(text)
                if sum(len(piece) for piece in pieces) > 60000:
                    break
            body = "\n".join(pieces)
        if len(body) >= 40000:
            return body
    raise SystemExit("no real-text source found under eval/corpora/perplexity-1m; pass --text")


def wait_for_port(port: int, deadline: float) -> bool:
    while time.time() < deadline:
        with socket.socket() as probe:
            probe.settimeout(1.0)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1.0)
    return False


def get_json(port: int, path: str) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    connection.request("GET", path)
    payload = json.loads(connection.getresponse().read().decode("utf-8", errors="replace"))
    connection.close()
    return payload


def post_json(port: int, body: dict, timeout: float) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    connection.request("POST", "/v1/chat/completions", body=json.dumps(body).encode("utf-8"),
                       headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    payload = json.loads(response.read().decode("utf-8", errors="replace"))
    connection.close()
    return {"status": response.status, "payload": payload}


def run_arm(artifact: Path, port: int, blocks: list[str], args: argparse.Namespace,
            run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "serve.log"
    result_path = run_dir / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    flags = [*LANE_FLAGS, "--chat-template", str(args.template)]
    if args.draft_tokens != 7:
        # The lane's own depth is 7. A depth A/B is the same artifact under two flag sets, so the flag
        # is substituted rather than appended, and the shipped value stays the default.
        lane_flags = list(LANE_FLAGS)
        lane_flags[lane_flags.index("--draft-tokens") + 1] = str(args.draft_tokens)
        flags = [*lane_flags, "--chat-template", str(args.template)]
    model_id = None
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        process = subprocess.Popen([str(SERVE), str(artifact), *flags, "--host", "127.0.0.1",
                                    "--port", str(port)], cwd=str(REPO), stdout=handle,
                                   stderr=subprocess.STDOUT)
        try:
            if not wait_for_port(port, time.time() + args.startup_timeout):
                raise SystemExit(f"{artifact.name}: serve never listened on {port}")
            time.sleep(2.0)
            model_id = (get_json(port, "/v1/models").get("data") or [{}])[0].get("id")
            for index, block in enumerate(blocks, 1):
                answer = post_json(port, {"model": model_id,
                                          "messages": [{"role": "user", "content": block}],
                                          "max_tokens": args.max_tokens,
                                          "temperature": args.temperature}, args.request_timeout)
                print(f"        prompt {index}: http {answer['status']}", flush=True)
        finally:
            time.sleep(2.0)
            process.terminate()      # the SERVE is stopped; nothing else is killed
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                pass

    body = log_path.read_text(encoding="utf-8", errors="replace")
    records = [{"accepted": int(a.replace(",", "")), "drafted": int(d.replace(",", "")),
                "percent": float(p)} for a, d, p in ACCEPT.findall(body)]
    result = {"artifact": artifact.name,
              "model_id": model_id,
              "requests": len(records),
              "accepted": sum(record["accepted"] for record in records),
              "drafted": sum(record["drafted"] for record in records),
              "acceptance": (sum(record["accepted"] for record in records)
                             / max(1, sum(record["drafted"] for record in records))),
              "decode": [float(value) for value in DECODE.findall(body)]}
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, action="append", required=True,
                        help="repeatable: one arm per artifact, labelled by its file stem")
    parser.add_argument("--label", action="append", default=[],
                        help="repeatable, positionally matched to --artifact, to name the arms")
    parser.add_argument("--text", type=Path, help="real-text source; defaults to the perplexity corpus")
    parser.add_argument("--prompts", type=int, default=3, help="fixed prompts to pool over")
    parser.add_argument("--prompt-chars", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--draft-tokens", type=int, default=7,
                        help="the draft depth to run the lane at; 7 is the shipped DFlash2 lane's")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 makes each arm's generation identical, which is what makes arms comparable")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--port-base", type=int, default=18120)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    text = args.text.read_text(encoding="utf-8", errors="replace") if args.text else default_text()
    blocks = [text[i * args.prompt_chars:(i + 1) * args.prompt_chars] for i in range(args.prompts)]
    if any(len(block) < 100 for block in blocks):
        raise SystemExit("not enough real text for that many prompts")
    labels = [args.label[index] if index < len(args.label) else artifact.stem
              for index, artifact in enumerate(args.artifact)]
    print(f"    {len(args.artifact)} arm(s) x {args.rounds} round(s) x {args.prompts} prompt(s), "
          f"temperature {args.temperature}, max_tokens {args.max_tokens}")

    results: dict[str, list[dict]] = {label: [] for label in labels}
    for round_index in range(1, args.rounds + 1):
        for index, (label, artifact) in enumerate(zip(labels, args.artifact)):
            print(f"\n=== {label} round {round_index}: {artifact.name}", flush=True)
            results[label].append(run_arm(artifact, args.port_base + index,
                                          blocks, args, args.output_dir / f"{label}-r{round_index}"))

    print()
    print(f"    {'arm':<22} {'round':>5} {'accepted':>9} {'drafted':>8} {'accept %':>9} {'decode':>8}")
    for label in labels:
        for round_index, record in enumerate(results[label], 1):
            decode = sum(record["decode"]) / len(record["decode"]) if record["decode"] else 0.0
            print(f"    {label[:22]:<22} {round_index:>5} {record['accepted']:>9} {record['drafted']:>8} "
                  f"{record['acceptance'] * 100:>8.1f}% {decode:>8.1f}")
    if len(labels) == 2:
        first = sum(r["acceptance"] for r in results[labels[0]]) / len(results[labels[0]])
        second = sum(r["acceptance"] for r in results[labels[1]]) / len(results[labels[1]])
        print()
        print(f"    pooled mean: {labels[0]} {first * 100:.2f}%, {labels[1]} {second * 100:.2f}% "
              f"-> {second / first - 1:+.1%} for {labels[1]}")
    print()
    print("    sanity: near 0.99 is a too-predictable prompt and near 0.10 too little context; real "
          "domains read 0.44-0.55")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
