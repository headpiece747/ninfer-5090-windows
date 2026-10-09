"""Settle a draft-depth choice: paired, interleaved, at the served sampling.

Why not one invocation per depth. At temperature 1.0 the generated text differs per run by construction,
so an A/B whose arms are separate invocations measures the depth plus the text plus whatever the card was
doing between them. `tools/release/v3_profile_matrix.py` measured this lane's depth that way -- one
invocation per domain, three rounds inside each -- and its own documentation says only within-invocation
comparisons are usable at that temperature. This harness instead pairs the two depths inside one session,
alternating which comes first so drift cancels, and repeats the pair enough times for a margin of 1-2%.

The protocol is the matrix's, with only the prompt changed: temperature 1.0 with the model's own sampling
defaults (the server fills top_p/top_k from the loaded model), max_tokens 400, the same artifact. The
prompts are traffic-like: three 12,000-character slices of real prose, real code, or long Chinese.

Idempotent: a run whose result.json exists is skipped, so an interrupted sweep resumes.

Usage:
    python tools/bench/paired_depth_sweep.py --artifact <path> --depth-a 7 --depth-b 9 --pairs 6 \
        --output-dir profiles/bench/<name> \
        --treatment prose-12000=-:12000:3 \
        --treatment code-12000=profiles/bench/.../inputs/repo-code-3x12000.txt:12000:3
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
INSTRUMENT = REPO / "tools" / "bench" / "realtext_acceptance.py"


def parse_treatments(values: list[str]) -> list[tuple[str, str | None, int, int]]:
    """LABEL=TEXT:CHARS:PROMPTS, where TEXT '-' means the instrument's default corpus."""
    out = []
    for value in values:
        label, _, rest = value.partition("=")
        text, _, tail = rest.partition(":")
        chars, _, prompts = tail.partition(":")
        if not label or not text or not chars or not prompts:
            raise SystemExit(f"bad --treatment {value!r}; expected LABEL=TEXT:CHARS:PROMPTS")
        out.append((label, None if text == "-" else text, int(chars), int(prompts)))
    return out


def run_one(artifact: Path, depth: int, treatment, args, run_dir: Path) -> dict:
    label, text, chars, prompts = treatment
    result = run_dir / f"d{depth}" / f"d{depth}-r1" / "result.json"
    if result.is_file():
        print(f"    {label} pair {run_dir.name}: d{depth} already measured", flush=True)
        return json.loads(result.read_text(encoding="utf-8"))
    command = [PYTHON, str(INSTRUMENT), "--artifact", str(artifact), "--label", f"d{depth}",
               "--prompt-chars", str(chars), "--prompts", str(prompts), "--rounds", "1",
               "--temperature", str(args.temperature), "--max-tokens", str(args.max_tokens),
               "--draft-tokens", str(depth), "--output-dir", str(run_dir / f"d{depth}")]
    if text is not None:
        command += ["--text", str(REPO / text)]
    completed = subprocess.run(command, cwd=str(REPO), capture_output=True, encoding="utf-8",
                               errors="replace", check=False)
    if completed.returncode != 0:
        raise SystemExit(f"{label} d{depth}: instrument failed\n{(completed.stderr or '')[-600:]}")
    return json.loads(result.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--depth-a", type=int, default=7)
    parser.add_argument("--depth-b", type=int, default=9)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--treatment", action="append", required=True, metavar="LABEL=TEXT:CHARS:PROMPTS")
    args = parser.parse_args()
    treatments = parse_treatments(args.treatment)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"    {args.artifact.name}: depths {args.depth_a} vs {args.depth_b}, {args.pairs} pairs, "
          f"temperature {args.temperature}, max_tokens {args.max_tokens}")

    summary: dict[str, list[tuple[float, float]]] = {}
    for treatment in treatments:
        label = treatment[0]
        summary[label] = []
        for pair in range(1, args.pairs + 1):
            # Alternate which depth runs first, so a drifting card affects both arms equally.
            order = (args.depth_a, args.depth_b) if pair % 2 else (args.depth_b, args.depth_a)
            run_dir = args.output_dir / f"{label}-pair{pair}"
            print(f"\n=== {label} pair {pair} order {order}", flush=True)
            rates = {}
            for depth in order:
                record = run_one(args.artifact, depth, treatment, args, run_dir)
                decode = record["decode"]
                rates[depth] = statistics.fmean(decode) if decode else float("nan")
                print(f"    d{depth}: accept {record['acceptance'] * 100:.1f}%  "
                      f"decode {rates[depth]:.1f} tok/s  ({len(decode)} record(s))", flush=True)
            summary[label].append((rates[args.depth_a], rates[args.depth_b]))

    print("\n=== paired result, decode tok/s")
    print(f"    {'treatment':<22} {'pairs':>5} {f'd{args.depth_a}':>9} {f'd{args.depth_b}':>9} "
          f"{'mean ratio':>11} {'worst':>8} {'best':>8}")
    for label, pairs in summary.items():
        if not pairs:
            continue
        first = statistics.fmean(pair[0] for pair in pairs)
        second = statistics.fmean(pair[1] for pair in pairs)
        ratios = [pair[1] / pair[0] for pair in pairs if pair[0]]
        print(f"    {label:<22} {len(pairs):>5} {first:>9.1f} {second:>9.1f} "
              f"{statistics.fmean(ratios) - 1:>10.1%} {min(ratios) - 1:>7.1%} {max(ratios) - 1:>7.1%}")
    print(f"\n    a positive mean ratio means depth {args.depth_b} is faster on that treatment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
