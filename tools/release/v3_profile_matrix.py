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
from profiles import PROFILES, QUASAR, NVFP4FULL, SWIFT15, NVIDIA, INVARIANT_FLAGS, by_file, launcher_args, launcher_environment, template_path  # noqa: E402

EXE = Path(__file__).resolve().parents[2] / "build" / "apps" / "ninfer-serve.exe"
MODELS = Path(r"C:\AI\models")
ROOT_OUT = Path(__file__).resolve().parents[2] / "out"
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
    # Swift 1.0's image is retired from the shipping table and is addressed below from
    # _superseded, where it is kept because every Swift 1.5 figure is quoted against it.
    # Built from NVIDIA's ModelOpt NVFP4 checkpoint: its NVFP4 MLP is imported and its FP8 attention
    # and linear-attention are re-encoded from the BF16 base, with the divisors derived from that
    # checkpoint's own per-site `input_scale`. The line it replaces is `ninfer` above, which carries
    # 146 FP8 tensors and caps below the full context.
    "nvidia": NVIDIA,
    # Swift 1.5, the same recipe and the same three sources as "swift" above with the finetune's
    # revision advanced. Its ModelOpt export is structurally identical to Swift 1.0's -- 401 sites,
    # the same 193 NVFP4 MLP and 208 FP8 attention/GDN names, only the producer version string moved
    # -- so the recipe is unchanged and the build differs only in which weights it read.
    "swift15": SWIFT15,
}
# The Q8-draft build of the same Swift 1.5 checkpoint, kept addressable because it is the control
# every NVFP4-draft figure is quoted against: it is the only variable between them, since the recipe
# differs solely in how the dflash2 component is encoded. It lives in _superseded rather than beside
# the shipping image, because a superseded build under a live filename is a measurement waiting to go
# wrong -- a harness resolving "the Swift artifact" by name would silently pick the wrong one.
SUPERSEDED = Path(r"C:\AI\models\_superseded")
ARTS["swift15q8"] = str(SUPERSEDED / "qwen3_8_27b_nvfp4swift15_q8draft.v3.ninfer")
ARTS["swift"] = str(SUPERSEDED / "qwen3_8_27b_nvfp4swift.v3.ninfer")
# NVFP4-full with its nine BF16 exception parents encoded to NVFP4. Kept addressable because it is
# the candidate that restores the native context on the DFlash2 + Vision lane, and every figure
# quoted for it is measured against the BF16-exception build under the same name below.
ARTS["nvfp4fullnoex"] = "qwen3_8_27b_nvfp4full_noex.v3.ninfer"
# ART_BY_FILE maps a profile's `art` field back to an ARTS key, so it only covers the shipping
# artifacts in the model directory, whose ARTS values are bare filenames. The two superseded entries
# above are absolute paths under _superseded and are deliberately unreachable through profile mode:
# a retired artifact must not be startable by naming a lane. Sweeps reach them by --art directly,
# which is what the Swift 1.0 and Q8-draft controls in the measurement record use.
ARTS_BY_FILE = {artifact: key for key, artifact in ARTS.items()
                if not Path(artifact).is_absolute()}
LADDER = [262144, 240000, 212992, 180224, 163840, 131072]
MTP_DEPTHS = [2, 3, 4, 5]

# Every draft window each backend will accept, from the startup validation rather than from this
# file's own convenience: startup.cpp raises "MTP draft window must be in [1,5]" against
# kMaximumMtpDraftTokens = 5, and "masked draft window must be in [1,15]" for DFlash/DFlash2. An
# earlier revision of docs/active-work.md item 8 proposed raising the MTP window to 10 on the
# strength of a fork that did it; this engine refuses to start at 6, so that item's sweep cannot be
# run on this tree at all and the MTP axis is 1..5.
WIDTH_LIMITS = {"mtp": 5, "dflash2": 15, "none": 0}

# Re-measured 2026-09-30, same binary, same card, one session, and the DFlash2 rows moved.
#
# The 2026-09-17/24 numbers below said every DFlash2 combination reached the native 262,144. On
# 2026-09-30 two of them do not: `swift` and `nvfp4full` are REFUSED at 262,144 with Vision and
# serve 240,000 instead, on this tree's own launcher flag set. The cause is not the artifacts --
# device weights are 18.0 GiB on the Swift build and the refusal is "minimum Engine runtime
# reservation" -- it is that runtime grew from 10.6/10.7 GiB on 2026-09-24 to 11.5/11.6 GiB on
# 2026-09-30, and the DFlash2+Vision lanes had under a GiB of margin left. `quasar` and `nvidia`
# still serve 262,144 on 1.51 and 1.37 GiB free, which is the whole margin between serving and being
# refused.
#
# The consequence is not confined to this file: tools/release/profiles.py carried ctx=262144 for
# `start_swift_v3_dflash2_vision` and `start_ninfer_v3_dflash2_vision`, and `profile` mode measured
# the first of them REFUSED through the real launcher. A ceiling table is only as good as the day it
# was taken, which is the rule docs/active-work.md item 1 already records for the merge.
#
# Only the rows measured on 2026-09-30 are updated. The MTP rows for quasar, nvfp4full and nvidia
# were not re-probed and still carry the 2026-09-17 values; they are marked rather than silently
# presented as current.
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
    # nvfp4full (cometkim's fuller-NVFP4 profile), measured 2026-09-30: the MTP rows still carry
    # 2026-09-17, the DFlash2 rows were re-probed. Vision + DFlash2 is refused at 262,144.
    ("nvfp4full", "mtp", False, True): 262144,
    ("nvfp4full", "mtp", True, True): 262144,
    ("nvfp4full", "dflash2", False, True): 262144,
    ("nvfp4full", "dflash2", True, True): 240000,
    # nvfp4fullnoex: the same line with its nine BF16 exception parents encoded to NVFP4, measured
    # 2026-09-30. 17.2 GiB of device weights against the BF16 build's 17.9, and that 0.7 GiB is what
    # the refusal was about: at 262,144 with Vision the BF16 build needs 11.63 GiB of reservation plus
    # 1 GiB of automatic headroom against 12.33 GiB available after weights, short by 308 MiB, while
    # this build reads 1.51 GiB free on the same configuration -- the same margin QUASAR has.
    ("nvfp4fullnoex", "mtp", False, True): 262144,
    ("nvfp4fullnoex", "mtp", True, True): 262144,
    ("nvfp4fullnoex", "dflash2", False, True): 262144,
    ("nvfp4fullnoex", "dflash2", True, True): 262144,
    # swift (UkisAI's 1.0 finetune, re-encoded by this port), re-measured 2026-09-30. The three
    # 262,144 rows hold; Vision + DFlash2 is refused at 262,144 and serves 240,000, where the
    # 2026-09-24 probe recorded 262,144 at 10.7 GiB. That is a runtime change, not a source change:
    # the same probe on the same artifact today reads 11.5 GiB.
    ("swift", "mtp", False, True): 262144,
    ("swift", "mtp", True, True): 262144,
    ("swift", "dflash2", False, True): 262144,
    ("swift", "dflash2", True, True): 240000,
    # swift15: Swift 1.5 with the DFlash2 draft encoded to NVFP4, measured 2026-09-30. All four
    # combinations reach the native 262,144, on 1.51 GiB free with Vision and DFlash2 -- which is
    # where the Q8-draft build of the same checkpoint is REFUSED at 262,144 and serves 240,000
    # instead. 0.77 GiB of device weight is exactly that margin: 17.65 GiB against 18.42.
    ("swift15", "mtp", False, True): 262144,
    ("swift15", "mtp", True, True): 262144,
    ("swift15", "dflash2", False, True): 262144,
    ("swift15", "dflash2", True, True): 262144,
    # swift15q8: the same Swift 1.5 checkpoint with its DFlash2 draft left at Q8, measured
    # 2026-09-30. It is the control the NVFP4-draft figures are quoted against, and the only
    # difference between the two is that component's encoding -- the MTP head is untouched, which is
    # why the MTP rows read 262,144 here and 262,144 on the shipping image.
    ("swift15q8", "mtp", False, True): 262144,
    ("swift15q8", "mtp", True, True): 262144,
    ("swift15q8", "dflash2", False, True): 262144,
    ("swift15q8", "dflash2", True, True): 240000,
    # nvidia (NVIDIA's ModelOpt checkpoint, built by this port), re-measured 2026-09-30: all four
    # combinations still reach the native 262,144, on 1.71 GiB free text-only and 1.37 with Vision.
    ("nvidia", "mtp", False, True): 262144,
    ("nvidia", "mtp", True, True): 262144,
    ("nvidia", "dflash2", False, True): 262144,
    ("nvidia", "dflash2", True, True): 262144,
}


# Key is (artifact, spec, vision, lm_head_draft).
#
# --lm-head-draft is NOT uniformly good: measured, it is worth +9% (DFlash2) and +18%
# (MTP d4) on QUASAR, but on nvfp4 it costs ~14% throughput, 11pp acceptance and a full
# ladder step of context (163,840 -> 180,224 on DFlash2). So the ceilings below differ
# per artifact AND per flag, and both must be probed with the flag the launcher ships.


def ceiling_of(art: str, spec: str, vision: bool, lm_head: bool) -> int:
    """Measured ceiling for the exact flag combination, falling back to the
    with-flag measurement when that combination has not been probed yet."""
    return (CEILINGS.get((art, spec, vision, lm_head))
            or CEILINGS.get((art, spec, vision, True))
            or 131072)

DECODE_TOKENS = 400

# The sampling configuration Qwen3.8-27B's own model card specifies, for the mode it defaults to
# and the mode its coding benchmarks are measured in. Sourced from the checkpoint's README.md, not
# chosen here:
#
#   "Qwen3.8 models operate in thinking mode by default"
#   "We recommend using the following sets of sampling parameters for generation:
#    - Thinking Mode: temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0
#    - Instruct (or non-thinking) mode: temperature=0.7, top_p=0.80, top_k=20, min_p=0.0,
#      presence_penalty=1.5"
#
# The coding benchmarks settle which of the two applies: SWE-bench Pro and DeepSWE 1.1 are both
# "evaluated with the Claude Code harness at temp=1.0, top_p=0.95, and a 256K context window", which is
# the thinking set, and the checkpoint's shipped generation_config.json carries temperature 1.0,
# top_k 20, top_p 0.95. The instruct set is for enable_thinking: false, which is not how code
# generation is evaluated.
#
# frontend.cpp's default_sampling() already encodes both sets, so these are the engine's own defaults
# rather than this port's. The table is sent explicitly because a measurement that silently depends on
# a server default is one nobody can see, and an earlier revision of this bench sent 0.6 / 0.95 with no
# presence_penalty -- a point the card specifies for neither mode.
#
# presence_penalty is 0.0 here because it is the card's coding value, and it also happens to be the
# cheaper one to speculate under. The drafter proposes from the selector's own scores and reads only
# temperature and seed out of the sampling config (candidate_selector_path.cu, draw_rank), while the
# verify path applies the request's penalty overlay to the target. The two distributions therefore
# differ, and a q that predicts p poorly costs acceptance: measured on QUASAR dflash2, 4.18 tokens per
# round at penalty 0 against 2.45 at 1.5, and sglang measured the same effect on DFlash
# (accept_len 3.227 at repetition_penalty 1.0, 1.311 at 1.5).
#
# That is an efficiency difference and not a correctness one, and the distinction matters enough to
# state precisely. The accept test is min(1, p/q) and the correction is normalize(max(0, p - q))
# (speculative_round.cuh: speculative_sparse_warp_accept), which is the standard speculative-sampling
# identity and holds for ANY normalised q, not only one derived from p the same way. p is normalised
# over its truncated support by sampling_normalize_support, which renormalises after the top_p and
# min_p cuts, and q is normalised over the 16 candidates by draw_rank. Tokens outside the candidate set
# have q = 0, so they are never proposed, and the residual term assigns them their full p. So the
# emitted token is distributed exactly as p -- the penalised, truncated target -- however mismatched q
# is. The penalty asymmetry therefore cannot change what the engine samples, only how many draft
# tokens it keeps. An earlier revision of this comment called it a defect; it is not one.
#
# repetition_penalty appears in the card at 1.0 and is deliberately absent: this engine's request
# contract has no such field (types.h carries temperature, top_p, top_k, min_p, presence_penalty and
# frequency_penalty), and 1.0 is the identity, so sending it would add an unsupported key for no
# behavioural change.
DOCUMENTED_SAMPLING: dict[str, object] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
}

CODE_PROMPT = (
    "Write a Python module with: a dataclass Point(x, y), a function distance(a, b) "
    "returning Euclidean distance, and a function closest_pair(points) returning the two "
    "closest points. Include type hints. Code only, no explanation."
)
PROBE_PROMPT = "Reply with the single word OK."

# The domain a throughput figure was measured on, and the prompt that defines it. A measurement
# carries the workload it was taken on, the way a startup figure carries its slot count, and this
# bench carried neither: measure_decode hardcoded CODE_PROMPT and the record had no domain field, so
# every tok/s in the profile table was a single synthetic code prompt and no record said so. That is
# not cosmetic. Code is the most favourable domain for speculation -- measured 3.18 to 5.71 tokens per
# round against 1.18 to 1.85 on Chinese -- so a code-prompt figure sits at the high end of the range,
# and every width and depth decision taken from it reversed once other domains were included.
#
# "repetition" is there for a specific reason rather than as filler: it is copy-heavy by construction,
# which is the workload the ngram copy selector is for, so it is the domain its break-even has to be
# measured on rather than assumed from a code prompt.
DOMAINS: dict[str, str] = {
    "code": CODE_PROMPT,
    "prose": (
        "Explain how a modern CPU out-of-order execution unit speculates past a branch and then "
        "recovers when the prediction was wrong. Write four paragraphs of continuous prose."
    ),
    "chinese": (
        "请详细解释现代处理器中的乱序执行和分支预测是如何协同工作的，"
        "并说明它们对程序性能的实际影响。请写四段连续的文字。"
    ),
    "dialogue": (
        "You are helping a colleague debug a failing test. They say: the suite passes locally and "
        "fails in CI about one run in five, always on the same test, and never on a clean machine. "
        "Ask them the questions you would need answered, one at a time, and say why each one matters."
    ),
    "repetition": (
        "Complete the following pattern exactly, copying it without variation:\n"
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
        "alpha bravo charlie delta echo"
    ),
}
DEFAULT_DOMAIN = "code"


def domain_prompt(name: str) -> str:
    """The prompt for a domain label, refusing an unknown one rather than measuring something else."""
    if name not in DOMAINS:
        raise SystemExit(f"unknown domain {name!r}; choose from {', '.join(sorted(DOMAINS))}")
    return DOMAINS[name]


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


def start(profile: dict | None, args: list[str], log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("w", encoding="utf-8", errors="replace")
    # The lane's environment, so what this harness measures is what the launcher ships. The profiles
    # pin the CUDA wait schedule; a server started without it would measure the engine's default.
    proc = subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=str(EXE.parent), env=launcher_environment(profile))
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

    sampling "default" is the realistic profile run and carries the sampling configuration
    Qwen3.8-27B's own model card specifies for its default mode -- temperature 1.0, top_p 0.95,
    top_k 20, min_p 0.0, presence_penalty 0.0. That is the set its coding benchmarks are measured
    under, and the set the checkpoint's own generation_config.json ships.

    An earlier revision sent the instruct set (0.7 / 0.80 / presence_penalty 1.5) on the reasoning that
    coding is non-thinking. The card says the opposite: the model defaults to thinking mode and the
    coding benchmarks run at temp 1.0 / top_p 0.95. The instruct set also turned on a penalty the
    drafter does not see, which costs acceptance but does not change what is sampled -- see
    DOCUMENTED_SAMPLING, where the mechanism is worked through.

    "zero" pins temperature 0; "none" sends no sampling fields at all, which is the only way the
    server's --greedy flag governs (the docs are explicit that request fields override server flags).
    """
    body = {
        "model": CURRENT_MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if sampling == "default":
        body.update(DOCUMENTED_SAMPLING)
    elif sampling == "zero":
        body["temperature"] = 0
        body["top_p"] = 1
    resp, dt = post(body)
    tokens = resp["usage"]["completion_tokens"]
    msg = resp["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    return tokens, dt, text


def measure_decode(runs: int = 3, jsonl: Path | None = None, greedy: bool = False,
                   domain: str = DEFAULT_DOMAIN, sampling_label: str = "documented") -> dict:
    """A probe, one discarded full-length warmup, `runs` realistic decode runs, one deterministic pass.

    Acceptance is taken only from the realistic runs: the deterministic pass uses
    temperature 0, which inflates draft acceptance and would flatter every depth
    equally. The digest comes from that deterministic pass and is what proves spec
    decoding is output-preserving across depths.

    The discarded warmup is not optional. The first full-length decode after a server start is a
    transient: it returns faster than every later identical request and different, shorter text,
    while requests 2..n are byte-identical. Measured on the NVIDIA MTP5 lane it reads 261.7 tok/s
    against 169.8 for requests 2-7, so averaging it into three runs reported ~200 tok/s for a lane
    that delivers ~170, and the profile matrix recorded that inflated figure. The transient needs a
    full-length decode to clear it -- a 16-token probe does not reach the state it affects, which is
    why the probe below is not sufficient on its own.
    """
    ct, dt, _ = run_once_gen(PROBE_PROMPT, 16, "default")
    warmup = ct / dt if dt else 0.0
    prompt = domain_prompt(domain)
    run_once_gen(prompt, DECODE_TOKENS, "default")

    rates = []
    for _ in range(runs):
        ct, dt, _ = run_once_gen(prompt, DECODE_TOKENS, "default")
        rates.append(ct / dt if dt else 0.0)

    out = {
        "warmup_tps": round(warmup, 1),
        "decode_tps": [round(r, 1) for r in rates],
        "decode_avg": round(sum(rates) / len(rates), 1) if rates else 0.0,
        # The workload and the sampling, in the record that carries the number. A decode figure
        # without these is not a measurement of anything in particular: the domain moves tokens per
        # round by more than a factor of two and the sampling moves acceptance by tens of points.
        "domain": domain,
        "sampling": sampling_label,
    }

    if jsonl is not None:
        # The 16-token probe emits no speculative record, so the discarded full-length warmup is
        # the only leading request record in the log.
        out.update(parse_spec_jsonl(jsonl, skip=1))

    # The digest has to come from the same workload as the throughput it sits beside, or the record
    # describes two different runs in one entry: a prose record carrying a digest of code-prompt text
    # reads as though both figures came from the same request. The prompt is a parameter for the same
    # reason the sampling is. A digest is only ever compared within one sweep, so changing the prompt
    # does not invalidate anything held elsewhere.
    _, _, text = run_once_gen(prompt, 400, "none" if greedy else "zero")
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


def parse_spec_jsonl(path: Path, skip: int = 0) -> dict:
    """Aggregate speculative and decode counters from the request log.

    Accepted/drafted is the acceptance rate behind the throughput number: a depth can
    raise decode tok/s while lowering acceptance, and only the ratio shows which.

    `skip` drops that many leading request records. The engine appends to its log for the whole
    session, so the requests a caller warms up with are in the file alongside the measured ones; a
    warmup that exists to absorb a startup transient would otherwise contribute its acceptance to
    the figure the transient is meant to be excluded from.
    """
    if not path.exists():
        return {}
    acc = dra = rnd = fallback = gens = seen = 0
    backend = window = ""
    per_position: list[int] = []
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
        seen += 1
        if seen <= skip:
            continue
        acc += sp.get("accepted_tokens", 0)
        dra += sp.get("drafted_tokens", 0)
        rnd += sp.get("rounds", 0)
        fallback += sp.get("fallback_steps", 0)
        backend = sp.get("backend", backend)
        window = sp.get("draft_window", window)
        gens += 1
        # Per-position acceptance, summed. Issue 119 measured this on the NVFP4 MTP lane and found
        # the loss concentrated in the deeper positions (84/72/58 against groupwise's 93/85/75), so
        # the aggregate alone hides where a wider window stops paying. The engine already records
        # it per request; summing keeps the aggregate rate and the profile consistent with each
        # other rather than reporting two different runs.
        for i, count in enumerate(sp.get("accepted_per_position") or []):
            while len(per_position) <= i:
                per_position.append(0)
            per_position[i] += int(count)
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
        "accept_per_position": per_position,
        "gen_records": gens,
    }


def refusal_reason(text: str) -> str:
    for line in reversed(text.splitlines()):
        if "FATAL" in line or "ERROR" in line or "refus" in line.lower():
            return line.strip()[:220]
    return ""


# ------------------------------------------------------------------ one profile run

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
                slots: str | None = None, kv_dtype: str = "fp8",
                domain: str = DEFAULT_DOMAIN) -> dict:
    if profile is None:
        tag = (f"{art}-v3-{spec}-d{draft}{'-vision' if vision else ''}-ctx{max_context}"
               f"{'' if lm_head else '-nolmh'}")
    else:
        tag = profile["file"].removesuffix(".bat")
    # The domain goes in the tag as well as the record, so two domains of the same lane do not
    # overwrite each other's log or request file. A sweep that varies the domain and writes one file
    # per lane silently keeps only the last domain's request log, and the acceptance numbers read
    # back from it then belong to a workload the record does not name.
    if domain != DEFAULT_DOMAIN:
        tag = f"{tag}-{domain}"
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
        art, spec, draft = ARTS_BY_FILE[profile["art"]], profile["spec"], profile["draft"]
        vision, max_context = bool(profile["vision"]), profile["ctx"]
    proc, fh = start(profile, args, log)
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
            record.update(measure_decode(jsonl=jsonl, greedy=greedy, domain=domain))
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


def per_position_profile(recs: list[dict]) -> str:
    """Accepted-drafted rate at each draft position, as a compact string.

    Each position's denominator is the number of drafts actually offered at that position, which
    is why this cannot be recovered by dividing the position's accepts by the window: a round that
    rejects early never offers position 7. Summing accepts and drafts per position across the
    rounds gives the rate for the positions that were reached, and the two disagree in exactly the
    way that matters -- a wide window's late positions are reached rarely and accepted rarely.
    """
    accepted: list[int] = []
    drafted: list[int] = []
    for rec in recs:
        counts = rec.get("accept_per_position")
        if not counts:
            continue
        # This record's own accepts, before any cross-record accumulation: the denominator for
        # position i is position i-1 *of the same round*, so reading it back out of the running
        # total would charge later records for earlier ones' rounds.
        own = [int(c) for c in counts]
        while len(accepted) < len(own):
            accepted.append(0)
            drafted.append(0)
        for i, count in enumerate(own):
            accepted[i] += count
            # Position 0 is offered on every round; position i is offered on every round that got
            # past position i-1, so a round that rejects early never offers the later positions.
            drafted[i] += own[i - 1] if i else max(1, rec.get("spec_rounds", 1))
    out = []
    for i, acc in enumerate(accepted):
        if drafted[i] <= 0:
            out.append("-")
        else:
            out.append(f"{acc / drafted[i] * 100:.0f}")
    return "/".join(out)


def mode_widths(art: str, drafts_by_spec: dict[str, list[int]], vision: bool,
                rounds: int, domain: str, lm_head: bool = True) -> None:
    """Every draft window each backend accepts, interleaved and rotated.

    Grouping the widths and reading them in order measures this card's clock drift, not the width:
    ADR-0003 records two findings that died that way -- a 14% slot-count claim and a 9% artifact
    claim, both committed before an interleaved run disproved them. So round r visits the
    configurations starting at offset r * n / rounds, which gives every configuration a different
    position in every round, and a configuration's rounds are compared against each other rather
    than against its neighbours.

    The non-speculative lane is in the same rotation on purpose. A width is only interesting
    against what it is speeding up, and a no-spec run started inside the same window is the only
    control that cannot have been taken while the card was in a different state.
    """
    configs: list[tuple[str, int]] = [("none", 0)]
    for spec, drafts in drafts_by_spec.items():
        limit = WIDTH_LIMITS[spec]
        for draft in drafts:
            if not 1 <= draft <= limit:
                raise SystemExit(f"{spec} draft window {draft} is outside the [1,{limit}] the "
                                 f"startup validation accepts")
            configs.append((spec, draft))

    collected: dict[tuple[str, int], list[dict]] = {}
    n = len(configs)
    for r in range(rounds):
        offset = (r * n) // rounds
        order = configs[offset:] + configs[:offset]
        print(f"\n=== widths round {r + 1}/{rounds} for {art} / vision={vision} / domain={domain}"
              f" / lm_head={lm_head} (rotation offset {offset})")
        for spec, draft in order:
            label = "none (control)" if spec == "none" else f"{spec} d{draft}"
            ctx = ceiling_of(art, spec, vision, lm_head) if spec != "none" else 262144
            rec = run_profile(art, spec, draft, vision, ctx, measure=True, lm_head=lm_head,
                              domain=domain)
            collected.setdefault((spec, draft), []).append(rec)
            show(rec)
            print(f"          ^ {label}")

    print(f"\n=== width summary: {art} / vision={vision} / domain={domain} / {rounds} rounds")
    print(f"  {'config':<16} {'tok/s':>18} {'spread':>7} {'accept':>8} {'tok/round':>10} {'rounds':>8}")
    ranked: list[tuple[float, str]] = []
    for (spec, draft), recs in collected.items():
        rates = [r["decode_avg"] for r in recs if r.get("ready") and "decode_avg" in r]
        if not rates:
            print(f"  {spec + ' d' + str(draft):<16} {'no measurement':>18}")
            continue
        acc = [r["accept_rate"] for r in recs if "accept_rate" in r]
        tpr = [r["spec_accepted"] / r["spec_rounds"] for r in recs
               if r.get("spec_rounds") and r.get("spec_accepted")]
        rnd = [r["spec_rounds"] for r in recs if r.get("spec_rounds")]
        avg = sum(rates) / len(rates)
        spread = (max(rates) - min(rates)) / avg * 100.0 if len(rates) > 1 and avg else 0.0
        label = "none (control)" if spec == "none" else f"{spec} d{draft}"
        print(f"  {label:<16} {avg:>18.1f} {spread:>6.1f}% "
              f"{(sum(acc) / len(acc) * 100 if acc else float('nan')):>7.1f}% "
              f"{(sum(tpr) / len(tpr) if tpr else float('nan')):>10.2f} "
              f"{(sum(rnd) // len(rnd) if rnd else 0):>8}")
        profile = per_position_profile(recs)
        if profile:
            print(f"  {'':<16} accept per draft position: {profile}")
        ranked.append((avg, label))
    ranked.sort(reverse=True)
    if ranked:
        best_rate, best = ranked[0]
        control = next((r for r, l in ranked if l == "none (control)"), None)
        print(f"\n  TOKEN SPEED CEILING: {best} at {best_rate:.1f} tok/s"
              + (f"  ({best_rate / control:.2f}x the non-speculative control's {control:.1f})"
                 if control and control > 0 else ""))


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


def mode_profile(name: str, slots: str | None = None,
                 domain: str = DEFAULT_DOMAIN) -> None:
    """Measure one shipped profile exactly as its launcher starts it.

    `slots` overrides the profile's --device-state-slots so the value can be chosen from a record;
    the default is still whatever the profile table ships. `domain` is the workload the decode figure
    is taken on, and it is recorded alongside the figure.
    """
    rec = run_profile(profile=by_file(name), measure=True, slots=slots, domain=domain)
    show(rec)
    for line in rec.get("spec_lines", [])[-4:]:
        print(f"           | {line[:150]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ceiling", "widths", "sweep", "verify", "correct", "profile"])
    ap.add_argument("--art", dest="arts", action="append",
                    choices=sorted(ARTS), help="repeatable; default both")
    ap.add_argument("--spec", dest="specs", action="append",
                    choices=["mtp", "dflash2"], help="ceiling mode; default both")
    ap.add_argument("--vision", action="store_true", help="sweep/verify: vision profile")
    ap.add_argument("--no-vision", action="store_true", help="sweep/verify: text profile")
    ap.add_argument("--draft", action="append", default=None,
                    help="widths mode: draft windows for the selected backends. Either a bare list "
                         "('7,9,13') applied to every selected backend, or one scoped to a backend "
                         "('dflash2=7,9,13'), repeatable. Repeatable flags append, so two bare "
                         "--draft values are a union rather than one per backend -- which is how an "
                         "earlier revision of this tool handed MTP the DFlash2 widths and was "
                         "refused by the width check below, which is the check earning its place.")
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
    ap.add_argument("--domain", dest="domains", action="append", choices=sorted(DOMAINS),
                    help="profile mode: the workload to measure on, repeatable, default the one "
                         "domain the table was measured on. Recorded in every result, because a "
                         "decode figure without its workload is not interpretable -- see DOMAINS.")
    ap.add_argument("--rounds", type=int, default=2,
                    help="widths mode: interleaved passes over every configuration, each rotated "
                         "so no configuration keeps a position. One round cannot show a width's "
                         "own spread, and this card's first-half drift reaches 5.8-7.6%%.")
    ap.add_argument("--all-widths", action="store_true",
                    help="widths mode: sweep every window each selected backend accepts")
    args = ap.parse_args()

    # Parsed once here rather than inside the widths branch, because verify mode reads a width from
    # the same option. `--draft` appends, so a bare value and a scoped one both accumulate into these.
    bare: list[int] = []
    scoped: dict[str, list[int]] = {}
    for entry in args.draft or []:
        if "=" in entry:
            spec_name, _, raw = entry.partition("=")
            scoped.setdefault(spec_name.strip(), []).extend(
                int(part) for part in raw.split(",") if part.strip())
        else:
            bare.extend(int(part) for part in entry.split(",") if part.strip())

    if not EXE.exists():
        print(f"missing engine: {EXE}")
        return 1

    if args.mode == "ceiling":
        mode_ceiling(args.arts or sorted(ARTS), args.specs or ["mtp", "dflash2"],
                     [False, True], lm_head=not args.no_lm_head)
    elif args.mode == "widths":
        specs = args.specs or ["dflash2", "mtp"]
        unknown = sorted(set(scoped) - set(WIDTH_LIMITS))
        if unknown:
            raise SystemExit(f"unknown backend(s) in --draft: {', '.join(unknown)}; "
                             f"choose from {', '.join(sorted(WIDTH_LIMITS))}")
        drafts: dict[str, list[int]] = {}
        for spec in specs:
            if spec in scoped:
                drafts[spec] = scoped[spec]
            elif bare:
                drafts[spec] = list(bare)
            elif args.all_widths:
                drafts[spec] = list(range(1, WIDTH_LIMITS[spec] + 1))
            else:
                drafts[spec] = MTP_DEPTHS if spec == "mtp" else [7]
        visions = [True] if args.vision else [False] if args.no_vision else [True]
        for art in (args.arts or sorted(ARTS)):
            for vision in visions:
                mode_widths(art, drafts, vision, args.rounds,
                            (args.domains or [DEFAULT_DOMAIN])[0],
                            lm_head=not args.no_lm_head)
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
            for domain in (args.domains or [DEFAULT_DOMAIN]):
                mode_profile(name, slots=args.slots, domain=domain)
    else:
        spec = (args.specs or ["mtp"])[0]
        mode_verify((args.arts or ["quasar"])[0], spec,
                    (scoped.get(spec, bare) or [5 if spec == "mtp" else 7])[0],
                    args.vision, args.max_context, lm_head=not args.no_lm_head,
                    kv_dtype=args.kv_dtype)

    print(f"\nrecords: {RECORDS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
