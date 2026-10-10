"""Drive a serving lane with a growing conversation, fed in chunks, and find out if it dies.

#208 reports intermittent `cudaErrorIllegalAddress` while running Qwen3.8-27B NVFP4 with MTP
speculative decoding under sustained agentic workloads. Reporter 2 supplies the shape: "reading a
large file in chunks to fill up my context. It failed at different points for both of my runs."

So this tool fills a conversation in chunks and then decodes at the filled window, which is what the
reports describe and what a large-prompt benchmark does not do. It ends by asking the server for
`/v1/models`: if the lane stops answering, the run reproduced a crash, and the engine's own log carries
the reason (open a lane with its output captured before diagnosing one, or the only evidence is a line
that says nothing).

Two traps, both met while writing this, are handled structurally:

- The decode turn appends a distinct user message, because an identical request body is served from the
  stored response and never decodes -- the run then reports a plausible 0.1 s with the previous
  completion. A turn whose completion count is far below `--decode` and whose prompt did not grow is
  the signature of that, and it is why every turn prints both.
- The model id is read from `/v1/models` rather than assumed. A wrong alias answers 404 and no request
  reaches the engine.

It is also a leak check, and the verdict is a **slope with a confidence interval**, not a duration. The
criterion is the one the field uses: fit the per-cycle floor of the engine's own counters and call it a
leak when the 95% interval excludes zero. The counters come from `/metrics` -- `device_kv_used_pages`
("occupied physical Main KV pages, including retained history") and `host_context_used_bytes` -- because
they are **exact**, where RSS carries tens of MiB of jitter that only a four-hour run can resolve a slope
above. That is why a run can be short: the length is set by the smallest growth worth detecting, not by
convention, and an exact counter answers it in a handful of cycles. RSS and device memory are still
sampled, as the coarse backstop for anything the counters do not model. `--selftest` falsifies the
estimator in both directions before a run is trusted.

The lane's pid is found from the port it listens on, or passed with `--pid`.

Usage:
    python tools/bench/chunked_context_soak.py --port 8188 --cycles 3 --chunks 6 \
        --chunk-chars 100000 --decode 2048
    python tools/bench/chunked_context_soak.py --port 8188 --cycles 120 --pid 1234
    python tools/bench/chunked_context_soak.py --selftest
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

WORDS = ("the quick brown fox jumps over the lazy dog while the engine attends to every token "
         "in a long context window and the page directory tracks each key value pair carefully ")


def chunk(cycle: int, index: int, target_chars: int) -> str:
    """Deterministic filler, distinct per cycle so a later run is not a cache replay."""
    unit = f"[c{cycle} k{index}] " + WORDS
    return (unit * (target_chars // len(unit) + 1))[:target_chars]


def fetch_model(base: str) -> str:
    with urllib.request.urlopen(f"{base}/v1/models", timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["data"][0]["id"]


def listening_pid(port: int) -> int | None:
    """The pid listening on `port`, so the soak samples the lane it is actually driving.

    `--pid` wins; this is the convenience path. The two `netstat` dialects disagree on everything but
    the port number, and Linux needs `-p` to print the pid at all.
    """
    command = ["netstat", "-ano"] if sys.platform == "win32" else ["netstat", "-ltnp"]
    try:
        output = subprocess.run(command, capture_output=True, text=True, timeout=30,
                                encoding="utf-8", check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 4 or not fields[1].endswith(f":{port}"):
            continue
        candidate = fields[-1]
        head = candidate.split("/", 1)[0]
        if head.isdigit():
            return int(head)
    return None


def host_rss_bytes(pid: int) -> int | None:
    """The process's resident set: ctypes on Windows, /proc on Linux, None when neither answers.

    The ctypes call needs explicit `argtypes`; without them the handle is truncated and the call
    fails, which is a failure this repository has already paid for once.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters),
                                               wintypes.DWORD]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return None
            return int(counters.WorkingSetSize)
        finally:
            kernel32.CloseHandle(handle)
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def device_memory_bytes(pid: int) -> int | None:
    """The process's device memory, from nvidia-smi's own compute-apps query (MiB -> bytes)."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, encoding="utf-8", check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2 or not fields[0].isdigit() or int(fields[0]) != pid:
            continue
        try:
            return int(float(fields[1])) * 1024 * 1024
        except ValueError:
            return None
    return None


def system_device_memory_bytes() -> int | None:
    """System-wide device memory in use, for when the per-process query answers nothing.

    WDDM can report no compute processes at all, so this keeps a leak check alive; it includes every
    other process on the card, which is why every sample prints which scope it read.
    """
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, encoding="utf-8", check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in output.splitlines():
        try:
            return int(float(line.strip())) * 1024 * 1024
        except ValueError:
            continue
    return None


def fetch_gauges(base: str, timeout: float = 10.0) -> dict[str, float]:
    """The engine's own live counters, read from `/metrics`.

    These are the reason a leak check does not need hours: RSS carries tens of MiB of jitter, so only a
    long run can resolve a slope above it, while `device_kv_used_pages` (occupied physical Main KV pages,
    including retained history) and `host_context_used_bytes` are exact counts. A per-request leak raises
    their per-cycle floor immediately, and a per-cycle floor series is what the regression below reads.
    """
    wanted = ("device_kv_used_pages", "host_context_used_bytes", "device_state_used_slots",
              "host_context_peak_bytes")
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError):
        return {}
    gauges: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("ninfer_") or " " not in line:
            continue
        name, _, value = line.partition(" ")
        short = name[len("ninfer_"):]
        if short in wanted:
            try:
                gauges[short] = float(value)
            except ValueError:
                continue
    return gauges


def slope_with_ci(points: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    """Ordinary least squares slope with its 95% interval, on (x, y) pairs.

    The field's criterion for a soak verdict is exactly this: a slope whose interval excludes zero is a
    leak, one that straddles zero is stable within noise. The normal approximation (1.96) is used and
    stated; it is honest for the sample counts a soak produces and it is falsified by `--selftest`.
    """
    n = len(points)
    if n < 4:
        return None
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    sxx = sum((x - mean_x) ** 2 for x, _ in points)
    if sxx == 0.0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in points)
    slope = sxy / sxx
    residuals = [(y - mean_y - slope * (x - mean_x)) ** 2 for x, y in points]
    residual_variance = sum(residuals) / (n - 2)
    standard_error = (residual_variance / sxx) ** 0.5
    half_width = 1.96 * standard_error
    return slope, slope - half_width, slope + half_width


def cycle_floors(samples: list[dict], key: str) -> list[tuple[float, float]]:
    """Per-cycle minimum of one gauge, against its cycle number: the floor a leak lifts.

    A floor, not a mean: each cycle ends holding its own conversation, and a healthy engine releases that
    history when the next cycle starts, so the minimum per cycle is the value a leak would raise.
    """
    floors: dict[int, float] = {}
    for entry in samples:
        value = entry.get(key)
        cycle = entry.get("cycle")
        if value is None or cycle is None:
            continue
        if cycle not in floors or value < floors[cycle]:
            floors[cycle] = value
    return [(float(cycle), value) for cycle, value in sorted(floors.items()) if cycle >= 0]


def growth_summary(samples: list[dict]) -> str:
    """First versus last and the second half's slope: the shape a leak check reports.

    The whole-run slope includes warmup -- pools filling, caches admitting -- so the number that
    answers "is anything leaking" is the slope after the midpoint, where a run that is merely warming
    reads flat.
    """
    usable = [entry for entry in samples if entry["rss"] is not None or entry["vram"] is not None]
    if len(usable) < 2:
        return "no growth check: fewer than two samples answered"
    first, last = usable[0], usable[-1]
    hours = max((last["at"] - first["at"]) / 3600.0, 1e-9)
    midpoint = first["at"] + (last["at"] - first["at"]) / 2.0
    tail = [entry for entry in usable if entry["at"] >= midpoint]
    tail_hours = max((tail[-1]["at"] - tail[0]["at"]) / 3600.0, 1e-9) if len(tail) >= 2 else 0.0

    def slope(begin: int, end: int, span: float) -> str:
        return f"{(end - begin) / 2**20 / span:+.1f} MiB/h"

    parts: list[str] = []
    for key, name in (("rss", "host RSS"), ("vram", f"device ({last.get('scope', '?')})")):
        begin, end = first[key], last[key]
        if begin is None or end is None:
            parts.append(f"{name}: unavailable")
            continue
        line = (f"{name} {begin / 2**20:.0f} -> {end / 2**20:.0f} MiB "
                f"({slope(begin, end, hours)} over {hours:.2f} h)")
        if tail_hours > 0.0 and tail[0][key] is not None and tail[-1][key] is not None:
            line += f", second half {slope(tail[0][key], tail[-1][key], tail_hours)}"
        parts.append(line)
    return "; ".join(parts)


def selftest() -> int:
    """Falsify the estimator in both directions before trusting it on a run.

    A flat series must read as flat within noise and a rising one as a leak. An instrument that cannot
    fail is not evidence that it works, so this runs before any measurement.
    """
    flat = [(float(cycle), 1000.0 + (cycle % 3) * 4.0) for cycle in range(30)]
    rising = [(float(cycle), 1000.0 + 12.0 * cycle + (cycle % 3) * 4.0) for cycle in range(30)]
    failures = 0
    for name, series, expect_leak in (("flat", flat, False), ("rising", rising, True)):
        verdict = slope_with_ci(series)
        if verdict is None:
            print(f"selftest {name}: estimator returned nothing")
            failures += 1
            continue
        slope, low, high = verdict
        leak = low > 0 or high < 0
        print(f"selftest {name}: slope {slope:+.1f} [{low:+.1f}, {high:+.1f}] -> "
              f"{'leak' if leak else 'flat within noise'}")
        if leak != expect_leak:
            failures += 1
    print(f"selftest: {'ok' if failures == 0 else f'{failures} failure(s)'}")
    return 1 if failures else 0


def turn(base: str, model: str, messages: list[dict[str, str]], max_tokens: int,
         label: str) -> dict:
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0}).encode()
    request = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=3600) as response:
        payload = json.loads(response.read().decode("utf-8"))
    usage = payload.get("usage", {})
    print(f"    {label}: prompt={usage.get('prompt_tokens')} "
          f"completion={usage.get('completion_tokens')} {time.time() - started:.1f}s", flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--cycles", type=int, default=3,
                        help="conversations; each one refills from empty")
    parser.add_argument("--chunks", type=int, default=6, help="chunks per conversation")
    parser.add_argument("--chunk-chars", type=int, default=100_000,
                        help="characters per chunk, about four per token")
    parser.add_argument("--decode", type=int, default=2048,
                        help="max tokens for the sustained-decode turn")
    parser.add_argument("--json", action="store_true", help="print each response object")
    parser.add_argument("--pid", type=int, default=None,
                        help="lane process id to sample (default: the pid listening on --port)")
    parser.add_argument("--selftest", action="store_true",
                        help="exercise the slope estimator on synthetic floors and exit")
    args = parser.parse_args()
    if args.selftest:
        return selftest()

    base = f"http://127.0.0.1:{args.port}"
    try:
        model = fetch_model(base)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
        print(f"no lane answering {base} ({type(error).__name__}): start ninfer-serve first, "
              "with its output captured", file=sys.stderr)
        return 1
    print(f"    model id: {model}", flush=True)

    pid = args.pid if args.pid is not None else listening_pid(args.port)
    samples: list[dict] = []

    def sample(label: str, cycle: int) -> None:
        gauges = fetch_gauges(base)
        entry: dict = {"at": time.time(), "cycle": cycle, **gauges}
        if pid is not None:
            vram = device_memory_bytes(pid)
            scope = "process"
            if vram is None:
                vram = system_device_memory_bytes()
                scope = "system"
            entry["rss"] = host_rss_bytes(pid)
            entry["vram"] = vram
            entry["scope"] = scope
        samples.append(entry)
        rss = f"{entry['rss'] / 2**20:.0f}" if entry.get("rss") is not None else "?"
        shown = f"{entry['vram'] / 2**20:.0f}" if entry.get("vram") is not None else "?"
        kv = f"{gauges['device_kv_used_pages']:.0f}" if "device_kv_used_pages" in gauges else "?"
        host = (f"{gauges['host_context_used_bytes'] / 2**20:.0f}"
                if "host_context_used_bytes" in gauges else "?")
        print(f"      {label}: host RSS {rss} MiB, device {shown} MiB ({entry.get('scope', 'n/a')}), "
              f"kv_pages {kv}, host_ctx {host} MiB", flush=True)

    if pid is None:
        print("    memory sampling off: no pid found for the port (pass --pid)", flush=True)
    else:
        print(f"    memory sampling: pid {pid}", flush=True)
        sample("baseline", -1)

    for cycle in range(args.cycles):
        print(f"  cycle {cycle}: {args.chunks} chunks of {args.chunk_chars:,} chars", flush=True)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "You are a careful assistant."}]
        for index in range(args.chunks):
            messages.append({"role": "user", "content": chunk(cycle, index, args.chunk_chars)})
            payload = turn(base, model, messages, 32, f"c{cycle} k{index}")
            sample(f"c{cycle} k{index}", cycle)
            if args.json:
                print(f"      {json.dumps(payload)[:400]}")
        messages.append({"role": "user", "content":
                         f"Cycle {cycle}: summarise every chunk above in one long paragraph and "
                         "keep writing until you reach the length limit."})
        payload = turn(base, model, messages, args.decode, f"c{cycle} decode")
        sample(f"c{cycle} decode", cycle)
        if args.json:
            print(f"      {json.dumps(payload)[:400]}")

    print(f"    growth: {growth_summary(samples)}", flush=True)
    for key in ("device_kv_used_pages", "host_context_used_bytes"):
        points = cycle_floors(samples, key)
        verdict = slope_with_ci(points)
        if verdict is None:
            print(f"    {key}: no verdict ({len(points)} cycle floors)", flush=True)
            continue
        slope, low, high = verdict
        state = "LEAK (interval excludes zero)" if low > 0 or high < 0 else "flat within noise"
        print(f"    {key}: {len(points)} cycle floors, slope {slope:+.1f} per cycle "
              f"[{low:+.1f}, {high:+.1f}] 95% -> {state}", flush=True)
    try:
        fetch_model(base)
        print("lane still answering /v1/models: no crash in this run")
        return 0
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
        print(f"LANE STOPPED ANSWERING ({type(error).__name__}): read the engine log for the "
              "cudaError, this run reproduced a crash", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
