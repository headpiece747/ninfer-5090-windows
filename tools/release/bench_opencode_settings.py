#!/usr/bin/env python3
"""Find the best opencode settings per model.

Every earlier measurement was engine throughput. This measures what opencode actually
cares about: given a coding task, does the model produce code that passes tests, how long
does it take, and does reasoning_effort change either.

Scoring is objective and automatic: each task ships hidden assertions, the model's reply is
executed against them, and the run is scored pass/fail. No LLM judge.

The dial is `reasoning_effort`, the one per-request setting our engine exposes
(none|minimal|low|medium|high|xhigh, per src/serve/openai_chat_request.cpp:840) and the one
opencode can set per model via options.reasoningEffort.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CWD = r"C:\AI\ninfer-v3-windows"
EXE = CWD + r"\build\apps\ninfer-serve.exe"
MODELS = r"C:\AI\models"
RECORDS = Path(r"C:\AI\bench\opencode_settings.jsonl")

# The shipped profiles, keyed for this harness. The values come from profiles.py, so this is a
# view rather than a second copy: the harness needs its own ports (a launcher may already hold
# the shipped one) and short keys for --models, and nothing else differs.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiles import PROFILES as SHIPPED, launcher_args  # noqa: E402

_HARNESS_PORTS = [8101, 8102, 8103, 8104]
PROFILES = {
    profile["file"].replace("start_", "").replace("_vision.bat", "").replace("_", "-"):
        dict(profile, port=port)
    for profile, port in zip(SHIPPED, _HARNESS_PORTS)
}

# The artifact's chat template (artifact:chat_template.jinja line 55) implements exactly
# these four. `minimal` and `high` raise "Unexpected reasoning effort" and return HTTP 400,
# even though the engine's own validation message lists all six.
EFFORTS = ["none", "low", "medium", "xhigh"]

TASKS = [
    dict(
        name="lru_cache",
        prompt=(
            "Write a Python function `lru( capacity )` returning a `get(key, loader)`/"
            " `put(key, value)` pair implementing a fixed-capacity LRU cache. `get` calls "
            "`loader(key)` on a miss and returns None if there is no loader. Evict the "
            "least-recently-used entry when over capacity; a hit or a put refreshes "
            "recency. Reply with one ```python code block only."
        ),
        tests="""
get, put = lru(2)
put("a", 1); put("b", 2)
assert get("a", None) == 1, "hit"
put("c", 3)
assert get("b", None) is None, "b should have been evicted"
assert get("a", None) == 1 and get("c", None) == 3, "a and c must survive"
get2, _ = lru(1)
assert get2("z", lambda k: k.upper()) == "Z", "loader on miss"
assert get2("z", None) == "Z", "then cached from the loader"
""",
    ),
    dict(
        name="merge_intervals",
        prompt=(
            "Write a Python function `merge(intervals)` taking a list of [start, end] pairs "
            "(unordered, possibly touching or nested, possibly empty) and returning the "
            "minimal list of merged intervals sorted by start. Treat touching intervals "
            "([1,2] and [2,3]) as mergeable. Reply with one ```python code block only."
        ),
        tests="""
assert merge([]) == []
assert merge([[1, 2]]) == [[1, 2]]
assert merge([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
assert merge([[5, 6], [1, 2]]) == [[1, 2], [5, 6]], "unordered input"
assert merge([[1, 2], [2, 3]]) == [[1, 3]], "touching must merge"
assert merge([[1, 10], [2, 3]]) == [[1, 10]], "nested"
""",
    ),
    dict(
        name="parse_duration",
        prompt=(
            "Write a Python function `seconds(text)` converting strings like '1h30m', '45s', "
            "'2d5h', '10m' into a total number of seconds (int). Parts may repeat in any "
            "order and must be summed; unknown units, a missing number, or an empty string "
            "must raise ValueError. Reply with one ```python code block only."
        ),
        tests="""
assert seconds("45s") == 45
assert seconds("10m") == 600
assert seconds("1h30m") == 5400
assert seconds("2d5h") == 190800
assert seconds("1m1h") == 3660, "order independent"
assert seconds("1h1h") == 7200, "repeat must sum"
for bad in ["", "5", "5x", "h", "1h30"]:
    try:
        seconds(bad); raise AssertionError(f"{bad!r} should raise")
    except ValueError:
        pass
""",
    ),
    dict(
        name="fix_off_by_one",
        prompt=(
            "This function is meant to return the indices of the k largest values in `xs`, "
            "descending by value, but it is wrong. Reply with one ```python code block only "
            "containing the corrected function, keeping the name and signature.\n\n"
            "def top_k(xs, k):\n"
            "    ordered = sorted(range(len(xs)), key=lambda i: xs[i])\n"
            "    return ordered[:k]\n"
        ),
        tests="""
assert top_k([1, 5, 3], 2) == [1, 2], "descending by value"
assert top_k([1, 5, 3], 0) == []
assert top_k([1, 5, 3], 5) == [1, 2, 0], "k larger than input"
assert top_k([], 2) == []
""",
    ),
    dict(
        name="alloc_budget",
        prompt=(
            "Write a Python function `schedule(items, budget)` where each item is a "
            "(cost, value) tuple. Return the maximum total value achievable with total cost "
            "<= budget, choosing each item at most once (0/1 knapsack). Costs and the budget "
            "are positive ints; value may be 0. Items with a cost above the budget are "
            "unusable. An empty list yields 0. Reply with one ```python code block only."
        ),
        tests="""
assert schedule([], 10) == 0
assert schedule([(5, 10)], 4) == 0, "unaffordable"
assert schedule([(5, 10)], 5) == 10, "exact fit"
assert schedule([(3, 5), (4, 7), (5, 9)], 8) == 12, "3+5 vs 5+9: pick the better pair"
assert schedule([(1, 1)] * 5, 3) == 3
assert schedule([(2, 3), (2, 3), (2, 3)], 4) == 6, "two of three fit"
assert schedule([(10, 100), (1, 60), (1, 60)], 2) == 120, "cheap pair beats one costly"
""",
    ),
    dict(
        name="tokenize_expr",
        prompt=(
            "Write a Python function `tokens(text)` that splits an arithmetic expression into "
            "a list of tokens: multi-digit integers become int, operators + - * / ( ) each "
            "become their own string token, and whitespace is ignored. Any other character "
            "must raise ValueError. Negative numbers are NOT part of a numeric token (treat "
            "'-' as an operator). Reply with one ```python code block only."
        ),
        tests="""
assert tokens("1+2") == [1, "+", 2]
assert tokens("12 * 34") == [12, "*", 34]
assert tokens("(1 + 2) * 3") == ["(", 1, "+", 2, ")", "*", 3]
assert tokens("  7  ") == [7]
assert tokens("") == []
assert tokens("1-2") == [1, "-", 2], "minus is an operator"
assert tokens("10/2") == [10, "/", 2]
assert tokens("1+2*3") == [1, "+", 2, "*", 3], "precedence is not this function's job"
for bad in ["1a", "1.5", "1+@", "$"]:
    try:
        tokens(bad); raise AssertionError(f"{bad!r} should raise")
    except ValueError:
        pass
""",
    ),
    # --- harder, multi-step tasks: these need real work, not recall -------------
    dict(
        name="evaluate",
        prompt=(
            "Write a Python function `evaluate(text)` evaluating an arithmetic expression "
            "given as a string, supporting + - * / , parentheses, and standard precedence with "
            "left associativity. Return an int when the result is integral, else a float. "
            "Reject malformed input (empty string, unbalanced parentheses, dangling or doubled "
            "operators, unknown characters) with ValueError. Unary minus is allowed at the "
            "start of the expression or right after '('. Reply with one ```python code block "
            "only."
        ),
        tests="""
assert evaluate("1+2*3") == 7
assert evaluate("(1+2)*3") == 9
assert evaluate("2*(3+4)-5") == 9
assert evaluate("10/4") == 2.5
assert evaluate("8/2/2") == 2
assert evaluate("7") == 7
assert evaluate("  3 + 4  ") == 7
assert evaluate("-5+2") == -3, "unary minus"
assert evaluate("2*(-3)") == -6, "unary minus after paren"
assert evaluate("((2))") == 2
assert evaluate("1+2*3-4") == 3
for bad in ["", "1+", "(1", "1+*2", "abc", "1..2", "()", "1/"]:
    try:
        evaluate(bad); raise AssertionError(f"{bad!r} should raise")
    except ValueError:
        pass
""",
    ),
    dict(
        name="edit_distance",
        prompt=(
            "Write a Python function `distance(a, b)` returning the Levenshtein edit distance "
            "between two strings: the minimum number of single-character insertions, deletions "
            "or substitutions to turn a into b. Reply with one ```python code block only."
        ),
        tests="""
assert distance("", "") == 0
assert distance("abc", "") == 3
assert distance("", "abc") == 3
assert distance("abc", "abc") == 0
assert distance("kitten", "sitting") == 3
assert distance("flaw", "lawn") == 2
assert distance("ab", "ba") == 2
assert distance("a", "b") == 1
assert distance("intention", "execution") == 5
""",
    ),
    dict(
        name="rooms_needed",
        prompt=(
            "Write a Python function `rooms(meetings)` taking a list of [start, end] intervals "
            "and returning the minimum number of rooms needed so no two meetings in the same "
            "room overlap. A meeting ending at t and another starting at t do NOT overlap and "
            "may share a room. An empty list needs 0 rooms; a zero-length meeting still needs "
            "a room. Reply with one ```python code block only."
        ),
        tests="""
assert rooms([]) == 0
assert rooms([[1, 2]]) == 1
assert rooms([[1, 5], [2, 6], [3, 7]]) == 3
assert rooms([[1, 2], [2, 3]]) == 1, "touching may share"
assert rooms([[1, 10], [2, 3], [4, 5]]) == 2
assert rooms([[5, 6], [1, 2]]) == 1, "unordered input"
assert rooms([[1, 4], [2, 5], [7, 9]]) == 2
assert rooms([[1, 1]]) == 1, "a zero-length meeting needs one room"
assert rooms([[1, 1], [5, 5]]) == 1, "zero-length meetings at different times share"
""",
    ),
    dict(
        name="debug_first_index",
        prompt=(
            "This function should return the index of the FIRST occurrence of target in a "
            "sorted list, or -1 when it is absent, but it has a bug. Reply with one ```python "
            "code block only containing the corrected function, keeping the name and "
            "signature.\n\n"
            "def first_index(sorted_xs, target):\n"
            "    lo, hi = 0, len(sorted_xs) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if sorted_xs[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return lo\n"
        ),
        tests="""
assert first_index([1, 2, 2, 3], 2) == 1, "first of duplicates"
assert first_index([1, 2, 2, 3], 9) == -1, "absent must be -1"
assert first_index([1, 2, 2, 3], 0) == -1
assert first_index([], 1) == -1, "empty"
assert first_index([5], 5) == 0
assert first_index([5], 1) == -1
assert first_index([1, 1, 1], 1) == 0
assert first_index([1, 3], 2) == -1
""",
    ),
]


def kill() -> None:
    subprocess.run(["taskkill", "/F", "/IM", "ninfer-serve.exe"], capture_output=True, text=True)


def start(profile: dict, thinking_budget: int = 4096) -> tuple[subprocess.Popen, Path]:
    kill()
    time.sleep(3)
    port = profile["port"]
    log = Path(r"C:\AI\bench") / f"settings_{port}.txt"
    jsonl = Path(r"C:\AI\bench") / f"settings_{port}.jsonl"
    jsonl.unlink(missing_ok=True)
    args = [EXE, str(Path(MODELS) / profile["art"])] + launcher_args(
        profile, port=port, model_id="settings-probe") + [
        "--request-log-jsonl", str(jsonl),
        "--default-thinking-budget", str(thinking_budget)]
    handle = log.open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(args, cwd=CWD, stdin=subprocess.DEVNULL, stdout=handle,
                            stderr=subprocess.STDOUT)
    for _ in range(120):
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                r.read()
            return proc, jsonl
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return proc, jsonl


def extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return match.group(1) if match else text


def score(task: dict, code: str) -> tuple[bool, str]:
    namespace: dict = {}
    try:
        exec(compile(code, "<model>", "exec"), namespace)  # noqa: S102
    except Exception as error:  # noqa: BLE001
        return False, f"exec: {type(error).__name__}: {str(error)[:70]}"
    try:
        exec(compile(task["tests"], "<tests>", "exec"), namespace)  # noqa: S102
    except AssertionError as error:
        return False, f"assert: {str(error)[:80]}"
    except Exception as error:  # noqa: BLE001
        return False, f"test error: {type(error).__name__}: {str(error)[:70]}"
    return True, "pass"


def ask(port: int, prompt: str, effort: str, output_tokens: int = 2048,
        timeout: int = 900) -> tuple[str, float, dict]:
    body = {"model": "settings-probe",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": output_tokens, "temperature": 0.2}
    if effort != "default":
        body["reasoning_effort"] = effort
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as r:
        response = json.loads(r.read())
    elapsed = time.time() - started
    message = response["choices"][0]["message"]
    text = (message.get("content") or "")
    reasoning = message.get("reasoning_content") or ""
    usage = response.get("usage", {})
    return text, elapsed, {"reasoning_chars": len(reasoning),
                           "completion": usage.get("completion_tokens", 0),
                           "prompt": usage.get("prompt_tokens", 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(PROFILES))
    ap.add_argument("--efforts", nargs="*", default=EFFORTS)
    ap.add_argument("--tasks", nargs="*", default=[t["name"] for t in TASKS])
    ap.add_argument("--thinking-budget", type=int, default=4096,
                    help="server cap on reasoning tokens; the chat template sets none")
    ap.add_argument("--output-tokens", type=int, default=2048,
                    help="completion cap; reasoning tokens are billed against it")
    args = ap.parse_args()

    wanted = [t for t in TASKS if t["name"] in args.tasks]
    results: list[dict] = []
    for name in args.models:
        profile = PROFILES[name]
        print(f"\n=== {name} ({profile['spec']} d{profile['draft']}, {profile['ctx']:,} ctx)")
        proc, jsonl = start(profile, args.thinking_budget)
        if proc.poll() is not None:
            print("   FAILED to start")
            continue
        for effort in args.efforts:
            for task in wanted:
                try:
                    text, elapsed, extra = ask(profile["port"], task["prompt"], effort,
                                               args.output_tokens)
                    ok, detail = score(task, extract_code(text))
                except Exception as error:  # noqa: BLE001
                    ok, detail, elapsed, extra = False, f"{type(error).__name__}", 0.0, {}
                record = dict(model=name, effort=effort, task=task["name"], pass_=ok,
                              detail=detail, seconds=round(elapsed, 1),
                              output_tokens=args.output_tokens, thinking_budget=args.thinking_budget, **extra)
                results.append(record)
                mark = "PASS" if ok else "FAIL"
                print(f"   {effort:<6} {task['name']:<18} {mark}  {elapsed:5.1f}s  "
                      f"reason={extra.get('reasoning_chars', 0):>5}  {detail[:44]}")
                with RECORDS.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
        proc.kill()
        kill()

    print("\n=== summary (pass count / mean seconds by effort) ===")
    for model in args.models:
        for effort in args.efforts:
            rows = [r for r in results if r["model"] == model and r["effort"] == effort]
            if not rows:
                continue
            passed = sum(1 for r in rows if r["pass_"])
            mean = sum(r["seconds"] for r in rows) / len(rows)
            reason = sum(r.get("reasoning_chars", 0) for r in rows) / len(rows)
            print(f"   {model:<16} {effort:<6} {passed}/{len(rows)} pass  "
                  f"{mean:5.1f}s mean  {reason:6.0f} reasoning chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
