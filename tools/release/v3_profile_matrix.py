#!/usr/bin/env python3
"""v3 profile matrix: context ceilings, MTP depth sweep, and verification.

Every measurement is taken the way a launcher will actually start the engine, so the
numbers written into the launchers come from the same arg set the launcher ships.

Modes
  ceiling  probe a descending max-context ladder per profile; record what serves and
           what is refused, with the engine's own runtime/free-VRAM accounting
  sweep    for one profile, measure decode tok/s and spec statistics at each draft
           depth, plus a deterministic digest for output-preservation checks
  verify   start one profile exactly as its launcher will and check it end to end
  profile  measure one *shipped* profile by name (--file, default all four) through
           profiles.launcher_args; --device-state-slots overrides its slot count. This is
           the mode every value in profiles.PROFILES is measured with.
  correct  is speculative decoding output-preserving on this artifact? Greedy control
           against each spec configuration, compared by digest.

Records append to matrix_v3.jsonl so a long sweep can be resumed or inspected.

Notes
  - Kills every ninfer-serve.exe before each start; only one 32 GB card is present.
  - Waits for VRAM to fall before starting, so a leaked process cannot silently
    turn a real refusal into a false one.
  - --request-log-jsonl captures full-precision per-request records, which is the
    measurement substrate for acceptance and throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import kill_servers, wait_ready  # noqa: E402
from profiles import PROFILES, QUASAR, NVFP4FULL, SWIFT, NVIDIA, INVARIANT_FLAGS, by_file, launcher_args, template_path  # noqa: E402

EXE = Path(__file__).resolve().parents[2] / "build" / "apps" / "ninfer-serve.exe"
MODELS = Path(r"C:\AI\models")
OUT = Path(r"C:\AI\bench")
PORT = 8095
BASE = f"http://127.0.0.1:{PORT}"
RECORDS = OUT / "matrix_v3.jsonl"
CURRENT_MODEL_ID = ""  # the engine enforces --model-id, so requests must match it

# The two shipped artifacts come from the module that names them. "ninfer" is the earlier nvfp4
# image, which no launcher starts and which this matrix probes deliberately (see CEILINGS below),
# so it is the one filename that has to stay literal here.
ARTS = {
    "quasar": QUASAR,
    "ninfer": "qwen3_8_27b_nvfp4.v3.ninfer",
    # cometkim's fuller-NVFP4 profile: 18.07 GiB, NVFP4 DFlash2 module, upstream-shaped
    # draft bindings (no fused query_key_value), 17.03 GiB device weights with DFlash2.
    "nvfp4full": NVFP4FULL,
    # UkisAI's Swift finetune, re-encoded by this port: the attention and GDN projections its source
    # keeps in FP8 are encoded to NVFP4 from the finetune's BF16 export, and both W8 endpoints are Q8,
    # so the artifact is all-NVFP4 and reaches the full context. The build it replaces -- the one
    # published from the source's own FP8 import -- sits above the envelope and reached 240,000 (MTP)
    # and 180,224 (DFlash2) at fp8 KV. See docs/maintainer/artifact-conventions.md section 1.
    "swift": SWIFT,
    # Built from NVIDIA's ModelOpt NVFP4 checkpoint: its NVFP4 MLP is imported and its FP8 attention
    # and linear-attention are re-encoded from the BF16 base, with the divisors derived from that
    # checkpoint's own per-site `input_scale`. The line it replaces is `ninfer` above, which carries
    # 146 FP8 tensors and caps below the full context.
    "nvidia": NVIDIA,
}
LADDER = [262144, 240000, 212992, 180224, 163840, 131072]
MTP_DEPTHS = [2, 3, 4, 5]

# Measured 2026-09-17. Key is (artifact, spec, vision, lm_head_draft).
#
# --lm-head-draft is NOT uniformly good: measured, it is worth +9% (DFlash2) and +18%
# (MTP d4) on QUASAR, but on nvfp4 it costs ~14% throughput, 11pp acceptance and a full
# ladder step of context (163,840 -> 180,224 on DFlash2). So the ceilings below differ
# per artifact AND per flag, and both must be probed with the flag the launcher ships.
CEILINGS = {
    ("quasar", "mtp", False, True): 262144,
    ("quasar", "mtp", True, True): 262144,
    ("quasar", "dflash2", False, True): 262144,
    ("quasar", "dflash2", True, True): 262144,
    ("ninfer", "mtp", False, True): 240000,
    ("ninfer", "mtp", True, True): 212992,
    ("ninfer", "dflash2", False, True): 163840,
    ("ninfer", "dflash2", True, True): 131072,
    # nvfp4 with the flag off: higher, because the optimized head is not resident.
    ("ninfer", "mtp", False, False): 240000,
    ("ninfer", "mtp", True, False): 212992,
    # nvfp4full (cometkim's fuller-NVFP4 profile), measured 2026-09-17: reachable at the
    # native 262,144 in every combination, where our nvfp4 image tops out at 240,000 for
    # MTP and 163,840/131,072 for DFlash2 with and without Vision.
    ("nvfp4full", "mtp", False, True): 262144,
    ("nvfp4full", "mtp", True, True): 262144,
    ("nvfp4full", "dflash2", False, True): 262144,
    ("nvfp4full", "dflash2", True, True): 262144,
    # swift (UkisAI's finetune, re-encoded by this port), measured 2026-09-24 on the re-encoded
    # artifact. Re-encoding its FP8 attention and GDN to NVFP4 from the finetune's BF16 source took
    # device weights from 18.90 GiB to 15.3, which is what puts every combination back at the full
    # native context; the same lanes measured 240,000 and 180,224 while the FP8 codes were imported.
    ("swift", "mtp", False, True): 262144,
    ("swift", "mtp", True, True): 262144,
    ("swift", "dflash2", False, True): 262144,
    ("swift", "dflash2", True, True): 262144,
    # nvidia (NVIDIA's ModelOpt checkpoint, built by this port), measured 2026-09-24 with `ceiling`
    # mode, which renders the launcher's own flags. Every combination reaches the full native context:
    # importing the MLP and re-encoding the rest from the BF16 base keeps device weights well inside
    # the envelope, and 262,144 is what the artifact it replaces cannot reach at all.
    ("nvidia", "mtp", False, True): 262144,
    ("nvidia", "mtp", True, True): 262144,
    ("nvidia", "dflash2", False, True): 262144,
    ("nvidia", "dflash2", True, True): 262144,
}


def ceiling_of(art: str, spec: str, vision: bool, lm_head: bool) -> int:
    """Measured ceiling for the exact flag combination, falling back to the
    with-flag measurement when that combination has not been probed yet."""
    return (CEILINGS.get((art, spec, vision, lm_head))
            or CEILINGS.get((art, spec, vision, True))
            or 131072)

CODE_PROMPT = (
    "Write a Python module with: a dataclass Point(x, y), a function distance(a, b) "
    "returning Euclidean distance, and a function closest_pair(points) returning the two "
    "closest points. Include type hints. Code only, no explanation."
)
PROBE_PROMPT = "Reply with the single word OK."


# ---------------------------------------------------------------- process control

def gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False).stdout
    return int("".join(c for c in out if c.isdigit()) or 0)


def wait_free(limit: int = 2000, timeout: int = 120) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if gpu_used_mib() < limit:
            return True
        time.sleep(3)
    return False


def start(args: list[str], log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=str(EXE.parent))
    return proc, fh


# ---------------------------------------------------------------------- requests

def post(payload: dict, timeout: int = 1800) -> tuple[dict, float]:
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    return resp, time.time() - t0


def run_once_gen(prompt: str, max_tokens: int, sampling: str = "default") -> tuple[int, float, str]:
    """Send one completion.

    sampling "default" is the realistic profile run; "zero" pins temperature 0; "none"
    sends no sampling fields at all, which is the only way the server's --greedy flag
    governs (the docs are explicit that request fields override server flags).
    """
    body = {
        "model": CURRENT_MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if sampling == "default":
        body["temperature"] = 0.6
        body["top_p"] = 0.95
    elif sampling == "zero":
        body["temperature"] = 0
        body["top_p"] = 1
    resp, dt = post(body)
    tokens = resp["usage"]["completion_tokens"]
    msg = resp["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    return tokens, dt, text


def measure_decode(runs: int = 3, jsonl: Path | None = None, greedy: bool = False) -> dict:
    """One warmup, `runs` realistic decode runs, then one deterministic pass.

    Acceptance is taken only from the realistic runs: the deterministic pass uses
    temperature 0, which inflates draft acceptance and would flatter every depth
    equally. The digest comes from that deterministic pass and is what proves spec
    decoding is output-preserving across depths.
    """
    ct, dt, _ = run_once_gen(PROBE_PROMPT, 16, "default")
    warmup = ct / dt if dt else 0.0

    rates = []
    for _ in range(runs):
        ct, dt, _ = run_once_gen(CODE_PROMPT, 400, "default")
        rates.append(ct / dt if dt else 0.0)

    out = {
        "warmup_tps": round(warmup, 1),
        "decode_tps": [round(r, 1) for r in rates],
        "decode_avg": round(sum(rates) / len(rates), 1) if rates else 0.0,
    }

    if jsonl is not None:
        out.update(parse_spec_jsonl(jsonl))

    _, _, text = run_once_gen(CODE_PROMPT, 400, "none" if greedy else "zero")
    out["digest"] = hashlib.sha256(text.encode()).hexdigest()[:16]
    out["digest_tokens"] = len(text)
    return out


# ------------------------------------------------------------------- log parsing

def _to_gib(part: str) -> float:
    """'runtime 9.49 GiB' or 'free 584.7 MiB' -> GiB. The engine mixes units."""
    fields = part.split()
    try:
        value = float(fields[1])
        unit = fields[2] if len(fields) > 2 else "GiB"
        return round(value if unit.startswith("Gi") else value / 1024.0, 3)
    except Exception:  # noqa: BLE001
        return 0.0


def parse_capacity(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        if "capacity |" not in line:
            continue
        try:
            seg = line.split("capacity |", 1)[1]
            kv = seg.split("KV", 1)[1].split("tokens", 1)[0].strip()
            out["kv_tokens"] = int(kv.replace(",", ""))
            for part in seg.split("|"):
                part = part.strip()
                if part.startswith("runtime"):
                    out["runtime_gib"] = _to_gib(part)
                elif part.startswith("free"):
                    out["free_gib"] = _to_gib(part)
                elif part.startswith("pages"):
                    out["pages"] = part.split()[1]
        except Exception:  # noqa: BLE001
            pass
        break
    return out


def parse_spec(text: str) -> list[str]:
    hits = []
    for line in text.splitlines():
        low = line.lower()
        if any(k in low for k in ("accept", "speculative", "draft token", "proposal")):
            hits.append(line.strip()[:200])
    return hits[-8:]


def parse_spec_jsonl(path: Path) -> dict:
    """Aggregate speculative and decode counters from the request log.

    Accepted/drafted is the acceptance rate behind the throughput number: a depth can
    raise decode tok/s while lowering acceptance, and only the ratio shows which.
    """
    if not path.exists():
        return {}
    acc = dra = rnd = fallback = gens = 0
    backend = window = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        sp = rec.get("speculative")
        if not sp:
            continue
        acc += sp.get("accepted_tokens", 0)
        dra += sp.get("drafted_tokens", 0)
        rnd += sp.get("rounds", 0)
        fallback += sp.get("fallback_steps", 0)
        backend = sp.get("backend", backend)
        window = sp.get("draft_window", window)
        gens += 1
    if not gens:
        return {}
    return {
        "spec_backend": backend,
        "spec_window": window,
        "spec_rounds": rnd,
        "spec_accepted": acc,
        "spec_drafted": dra,
        "spec_fallback_steps": fallback,
        "accept_rate": round(acc / dra, 4) if dra else 0.0,
        "gen_records": gens,
    }


def refusal_reason(text: str) -> str:
    for line in reversed(text.splitlines()):
        if "FATAL" in line or "ERROR" in line or "refus" in line.lower():
            return line.strip()[:220]
    return ""


# ------------------------------------------------------------------ one profile run

ART_BY_FILE = {artifact: key for key, artifact in ARTS.items()}


def profile_args(profile: dict, log_jsonl: Path,
                 slots: str | None = None) -> tuple[list[str], str]:
    """Arguments for one *shipped* profile, composed through profiles.launcher_args.

    build_args explores combinations no launcher ships, so it omits per-profile flags such as
    --device-state-slots. Measuring a shipped profile through it published a runtime/free pair for
    a configuration nobody starts. This path cannot diverge from the launcher: it renders the same
    flag list through the same function the launcher generator uses.

    `slots` overrides the profile's --device-state-slots so the value can be chosen from a record.
    """
    args = [str(EXE), str(MODELS / profile["art"])]
    args += launcher_args(profile, port=PORT)
    if slots is not None:
        at = args.index("--device-state-slots")
        args[at + 1] = slots
    args += ["--log-stats-interval-ms", "2000", "--request-log-jsonl", str(log_jsonl),
             "--seed", "1234"]
    return args, profile["model_id"]


def build_args(art: str, spec: str, draft: int, vision: bool, max_context: int,
               log_jsonl: Path, greedy: bool = False, kv_capacity: str = "auto",
               lm_head: bool = True, kv_dtype: str = "fp8") -> list[str]:
    """Arguments for one probe run.

    This harness explores combinations the four shipped profiles do not cover -- spec "none",
    arbitrary draft depths, and a descending context ladder -- so it cannot take a profile
    directly. It composes the invariant flags from profiles.INVARIANT_FLAGS instead, which is
    what stops the cache bounds, the thinking budget and the pending timeout going missing
    again: they were absent here, so every record this harness produced came from flags no
    launcher ships while its own header claimed the opposite.
    """
    a = [str(EXE), str(MODELS / ARTS[art]),
         "--host", "127.0.0.1", "--port", str(PORT),
         "--model-id", f"{art}-v3-{spec}",
         "--max-context", str(max_context)]
    # The per-profile flags this probe varies, then the shipped invariants.
    if spec != "none":
        a += ["--spec", spec, "--draft-tokens", str(draft)]
        if lm_head:
            a += ["--lm-head-draft"]
    if vision:
        a += ["--vision"]
    a += ["--kv-capacity", kv_capacity]
    for flag, value in INVARIANT_FLAGS:
        if flag == "--kv-capacity":
            continue  # already emitted above, from the caller's value
        if flag == "--kv-dtype":
            value = kv_dtype  # a probe may vary it; every launcher ships fp8
        elif value == "%TEMPLATE%":
            # This path composes the shipped flags itself instead of going through launcher_args, so
            # it has to resolve the launcher's cmd variable too. Left literal it is a startup failure
            # that the ladder reports as a context refusal, which is a measurement of nothing.
            value = template_path()
        a.append(flag) if value is None else a.extend([flag, value])
    a += ["--log-stats-interval-ms", "2000",
          "--request-log-jsonl", str(log_jsonl),
          "--seed", "1234"]
    if greedy:
        a += ["--greedy"]
    return a


def run_profile(art: str = "", spec: str = "", draft: int = 0, vision: bool = False,
                max_context: int = 0, measure: bool = True, greedy: bool = False,
                lm_head: bool = True, profile: dict | None = None,
                slots: str | None = None, kv_dtype: str = "fp8") -> dict:
    if profile is None:
        tag = (f"{art}-v3-{spec}-d{draft}{'-vision' if vision else ''}-ctx{max_context}"
               f"{'' if lm_head else '-nolmh'}")
    else:
        tag = profile["file"].removesuffix(".bat")
    log = OUT / f"sweep_{tag}.txt"
    jsonl = OUT / f"req_{tag}.jsonl"
    jsonl.unlink(missing_ok=True)  # the server appends; a stale file would average runs
    kill_servers()
    freed = wait_free()
    kv_before = gpu_used_mib()

    global CURRENT_MODEL_ID
    if profile is None:
        CURRENT_MODEL_ID = f"{art}-v3-{spec}"
        args = build_args(art, spec, draft, vision, max_context, jsonl, greedy=greedy,
                          lm_head=lm_head, kv_dtype=kv_dtype)
    else:
        args, CURRENT_MODEL_ID = profile_args(profile, jsonl, slots=slots)
        art, spec, draft = ART_BY_FILE[profile["art"]], profile["spec"], profile["draft"]
        vision, max_context = bool(profile["vision"]), profile["ctx"]
    proc, fh = start(args, log)
    ready = wait_ready(PORT, proc)
    time.sleep(3)  # let the capacity/stats lines flush
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""

    record: dict = {
        "tag": tag, "art": art, "spec": spec, "draft": draft, "vision": vision,
        "max_context": max_context, "ready": ready,
        "vram_before_mib": kv_before, "vram_freed": freed,
        "log": log.name,
    }
    if profile is not None:
        record["device_state_slots"] = slots or str(profile["device_state_slots"])
    record.update(parse_capacity(text))
    record["spec_lines"] = parse_spec(text)

    if ready and measure:
        try:
            record.update(measure_decode(jsonl=jsonl, greedy=greedy))
        except urllib.error.HTTPError as e:
            record["measure_error"] = f"HTTP {e.code} {e.read().decode('utf-8', 'replace')[:150]}"
        except Exception as e:  # noqa: BLE001
            record["measure_error"] = f"{type(e).__name__}: {e}"
        try:
            record["vram_peak_mib"] = gpu_used_mib()
        except Exception:  # noqa: BLE001
            pass
    elif not ready:
        record["refusal"] = refusal_reason(text)

    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
    fh.close()
    kill_servers()

    with RECORDS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def show(rec: dict) -> None:
    head = rec["tag"]
    if rec.get("ready"):
        cap = f"KV {rec.get('kv_tokens', 0):,}"
        print(f"  SERVES  {head:<42} {cap:>14} | runtime {rec.get('runtime_gib', '?')} GiB"
              f" | free {rec.get('free_gib', '?')} GiB")
        if "decode_avg" in rec:
            acc = (f"accept {rec['accept_rate'] * 100:5.1f}% "
                   f"({rec['spec_accepted']}/{rec['spec_drafted']})"
                   if "accept_rate" in rec else "accept n/a")
            print(f"          decode {rec['decode_avg']:>7.1f} tok/s  (runs "
                  f"{rec['decode_tps']}) | {acc} | digest {rec.get('digest', '')}")
        else:
            print("          no generation")
    else:
        print(f"  REFUSED {head:<42} {rec.get('refusal', '')[:95]}")


# ------------------------------------------------------------------------- modes

def mode_ceiling(arts: list[str], specs: list[str], visions: list[bool],
                 lm_head: bool = True) -> None:
    for art in arts:
        for spec in specs:
            for vision in visions:
                depth = 5 if spec == "mtp" else 7
                print(f"\n=== ceiling: {art} / {spec} / vision={vision} / lm_head={lm_head}")
                for ctx in LADDER:
                    rec = run_profile(art, spec, depth, vision, ctx, measure=False,
                                      lm_head=lm_head)
                    show(rec)
                    if rec["ready"]:
                        print(f"     -> CEILING {ctx:,}")
                        break
                else:
                    print("     -> no candidate served")


def mode_sweep(art: str, vision: bool, lm_head: bool = True) -> None:
    """Every MTP depth at this profile's measured ceiling, plus a DFlash2 baseline,
    so the depth choice and the backend choice are answered from one comparable run set."""
    ctx = ceiling_of(art, "mtp", vision, lm_head)
    print(f"\n=== MTP depth sweep: {art} / vision={vision} / lm_head={lm_head} @ {ctx:,}")
    for depth in MTP_DEPTHS:
        rec = run_profile(art, "mtp", depth, vision, ctx, measure=True, lm_head=lm_head)
        show(rec)
    dctx = ceiling_of(art, "dflash2", vision, lm_head)
    print(f"  --- DFlash2 baseline @ {dctx:,}")
    show(run_profile(art, "dflash2", 7, vision, dctx, measure=True, lm_head=lm_head))


def mode_correct(art: str, vision: bool) -> None:
    """Record the greedy digest and decode rate of every speculative configuration.

    The depth sweep showed depths producing different digests, but request-level
    temperature 0 is not exact argmax (the server's default top_k still applies), so
    this pass runs the server with --greedy and sends no sampling fields: the only
    configuration where "identical tokens" is a meaningful claim. No-spec is the
    control; every speculative configuration is compared against it.

    ADR-0002 records the answer: speculation is not bit-identical, on any artifact, and
    depth is chosen on measured decode rate rather than on agreement with the control.
    So a run that reports "match control: none" is the expected result, not a defect --
    what it produces is the per-configuration digest and the speculative speed-up.
    """
    print(f"\n=== correctness, greedy, no request sampling: {art} / vision={vision}")
    rows: list[tuple[str, str, int]] = []
    plan = [("none", 0), ("mtp", 2), ("mtp", 3), ("mtp", 4), ("mtp", 5), ("dflash2", 7)]
    for spec, draft in plan:
        ctx = ceiling_of(art, spec, vision, True)
        rec = run_profile(art, spec, draft, vision, ctx, measure=True, greedy=True)
        label = f"{spec} d{draft}" if spec != "none" else "none (control)"
        rows.append((label, rec.get("digest", "?"), rec.get("digest_tokens", 0)))
        print(f"  {label:<16} digest {rec.get('digest', '?')} "
              f"| {rec.get('digest_tokens', 0):>5} chars | decode {rec.get('decode_avg')} tok/s")
    control = rows[0][1]
    match = [r[0] for r in rows[1:] if r[1] == control]
    differ = [r[0] for r in rows[1:] if r[1] != control]
    print(f"  -> match control: {match if match else 'none'}")
    print(f"  -> differ from control: {differ if differ else 'none'}")


def mode_verify(art: str, spec: str, draft: int, vision: bool, max_context: int,
                lm_head: bool = True, kv_dtype: str = "fp8") -> None:
    print(f"\n=== verify: {art} / {spec} d{draft} / vision={vision} / ctx {max_context:,}"
          f" / lm-head-draft={lm_head} / kv {kv_dtype}")
    rec = run_profile(art, spec, draft, vision, max_context, measure=True, lm_head=lm_head,
                      kv_dtype=kv_dtype)
    show(rec)
    for line in rec.get("spec_lines", [])[-4:]:
        print(f"           | {line[:150]}")


def mode_profile(name: str, slots: str | None = None) -> None:
    """Measure one shipped profile exactly as its launcher starts it.

    `slots` overrides the profile's --device-state-slots so the value can be chosen from a record;
    the default is still whatever the profile table ships.
    """
    rec = run_profile(profile=by_file(name), measure=True, slots=slots)
    show(rec)
    for line in rec.get("spec_lines", [])[-4:]:
        print(f"           | {line[:150]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ceiling", "sweep", "verify", "correct", "profile"])
    ap.add_argument("--art", dest="arts", action="append",
                    choices=sorted(ARTS), help="repeatable; default both")
    ap.add_argument("--spec", dest="specs", action="append",
                    choices=["mtp", "dflash2"], help="ceiling mode; default both")
    ap.add_argument("--vision", action="store_true", help="sweep/verify: vision profile")
    ap.add_argument("--no-vision", action="store_true", help="sweep/verify: text profile")
    ap.add_argument("--draft", type=int, default=None)
    ap.add_argument("--max-context", type=int, default=262144)
    ap.add_argument("--no-lm-head", action="store_true",
                    help="verify mode: omit --lm-head-draft")
    ap.add_argument("--file", dest="files", action="append",
                    help="profile mode: shipped launcher file; repeatable, default all four")
    ap.add_argument("--device-state-slots", dest="slots", default=None,
                    help="profile mode: override the profile's slot count to choose the value")
    ap.add_argument("--kv-dtype", default="fp8",
                    choices=["bf16", "int8", "fp8", "nvfp4", "k8v4"],
                    help="probe modes: vary the KV dtype the launchers ship as fp8")
    args = ap.parse_args()

    if not EXE.exists():
        print(f"missing engine: {EXE}")
        return 1

    if args.mode == "ceiling":
        mode_ceiling(args.arts or sorted(ARTS), args.specs or ["mtp", "dflash2"],
                     [False, True], lm_head=not args.no_lm_head)
    elif args.mode == "sweep":
        visions = [True] if args.vision else [False] if args.no_vision else [False, True]
        for art in (args.arts or sorted(ARTS)):
            for vision in visions:
                mode_sweep(art, vision, lm_head=not args.no_lm_head)
    elif args.mode == "correct":
        visions = [True] if args.vision else [False] if args.no_vision else [False, True]
        for art in (args.arts or sorted(ARTS)):
            for vision in visions:
                mode_correct(art, vision)
    elif args.mode == "profile":
        for name in (args.files or [p["file"] for p in PROFILES]):
            mode_profile(name, slots=args.slots)
    else:
        spec = (args.specs or ["mtp"])[0]
        mode_verify((args.arts or ["quasar"])[0], spec,
                    args.draft if args.draft is not None else (5 if spec == "mtp" else 7),
                    args.vision, args.max_context, lm_head=not args.no_lm_head,
                    kv_dtype=args.kv_dtype)

    print(f"\nrecords: {RECORDS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
