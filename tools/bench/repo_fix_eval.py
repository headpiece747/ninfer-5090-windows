"""A private SWE-style evaluation for this repository.

The shape is SWE-bench's: an issue, a patch, the repository's own tests decide. What changes is the
environment -- there is no container, because the environment this repository needs (MSVC 14.51,
CUDA 13.3, Ninja, ctest) is already on this machine, and a container would be a worse copy of it.

Three decisions, each forced by something measured:

* **The instance is a commit that touched `src/` and `tests/`.** The harness reverts the commit's
  *source* changes and keeps its *test* changes, so the test's new case is present and failing -- the
  same construction SWE-bench uses for FAIL_TO_PASS. Reverting the test as well would delete the target
  and leave nothing to score.
* **No worktree, no checkout of a parent commit.** `build-test/` is already configured and built, so
  the incremental rebuild of one dispatch plus one shape file is minutes, where a fresh worktree costs
  30-60 minutes of configure and full build. The working tree *is* the environment; the patch is applied
  with `git apply` and every touched path is restored in a `finally`.
* **The issue text comes from the test's own new case, not the commit message.** This repository's
  commit bodies explain the fix in detail; using them would leak the answer into the prompt.

Two traps this had to clear, both already earned elsewhere in this repository: a Python process does not
inherit the Visual Studio environment (`cl.exe` then fails on the first standard header), so builds run
through a `vcvars64.bat` call in a real `.cmd` file -- inline `cmd /c "call ... && cmake ..."` gets its
embedded quotes re-quoted by `subprocess` and the wrapper silently does nothing; and the port-delta
ratchet blocks commits, so run this on a clean tree.

Usage:
    python tools/bench/repo_fix_eval.py --commit <sha> --test <ctest -R name> [--gold]
    python tools/bench/repo_fix_eval.py --commit <sha> --test <name> --lane http://127.0.0.1:18080
    python tools/bench/repo_fix_eval.py --commit <sha> --test <name> --validate-only

`--gold` applies the commit's own source diff instead of asking a model: it validates the loop (apply,
build, test) before any model is involved. Measured 2026-10-09 on 15cba227: the reverted tree fails
`ninfer_linear_(add_)?bf16_a16_test`, the 94-line gold patch applies, builds and resolves it, and the
tree is restored with zero residual changes.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "build-test"
TMP = Path(tempfile.gettempdir())
VCVARS = Path(r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build"
              r"\vcvars64.bat")


def run(args: list[str], cwd: Path = REPO, timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=str(cwd), capture_output=True, encoding="utf-8",
                          errors="replace", timeout=timeout, check=False)


def git(*args: str) -> subprocess.CompletedProcess:
    return run(["git", *args])


def msvc(args: list[str], timeout: int = 7200) -> subprocess.CompletedProcess:
    """Run a build or test command inside the MSVC environment, the way the recipes do."""
    script = TMP / "repo_fix_eval_step.cmd"
    body = '@echo off\r\n' f'call "{VCVARS}" >nul 2>&1\r\n' + " ".join(args) + "\r\n"
    script.write_text(body, encoding="utf-8", newline="")
    return subprocess.run(["cmd", "/c", str(script)], cwd=str(REPO), capture_output=True,
                          encoding="utf-8", errors="replace", timeout=timeout, check=False)


def changed_paths(commit: str) -> tuple[list[str], list[str]]:
    out = git("show", "--name-only", "--format=", commit).stdout
    paths = [line.strip() for line in out.splitlines() if line.strip()]
    src = [p for p in paths if not p.startswith("tests/") and not p.startswith("docs/")]
    tests = [p for p in paths if p.startswith("tests/")]
    return src, tests


def build() -> tuple[int, str]:
    result = msvc(["cmake", "--build", f'"{BUILD}"', "-j"])
    return result.returncode, ((result.stdout or "") + (result.stderr or ""))[-600:]


def test(name: str) -> tuple[int, str]:
    result = msvc(["ctest", "--test-dir", f'"{BUILD}"', "-R", f'"{name}"', "--output-on-failure"])
    return result.returncode, ((result.stdout or "") + (result.stderr or ""))[-800:]


def ask(lane: str, prompt: str) -> str:
    with urllib.request.urlopen(f"{lane}/v1/models", timeout=30) as response:
        model = json.loads(response.read())["data"][0]["id"]
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.2, "max_tokens": 8192}).encode()
    request = urllib.request.Request(f"{lane}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        payload = json.loads(response.read())
    return payload["choices"][0]["message"]["content"] or ""


def extract_patch(text: str) -> str:
    fenced = re.search(r"```(?:diff|patch)?\n(.*?)```", text, re.DOTALL)
    body = fenced.group(1) if fenced else text
    start = body.find("diff --git")
    return body[start:] if start >= 0 else body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--commit", required=True)
    parser.add_argument("--test", required=True, help="the ctest -R pattern that must fail, then pass")
    parser.add_argument("--prompt", type=Path, help="issue text; default is derived from the commit")
    parser.add_argument("--lane", default="http://127.0.0.1:18080")
    parser.add_argument("--validate-only", action="store_true",
                        help="revert, build, confirm the test fails, restore -- no patch at all")
    parser.add_argument("--gold", action="store_true",
                        help="apply the commit's own source diff instead of asking a model")
    args = parser.parse_args()

    src, tests = changed_paths(args.commit)
    if not src or not tests:
        raise SystemExit(f"instance is invalid: src={len(src)} tests={len(tests)} (both are needed)")
    subject = git("log", "-1", "--format=%s", args.commit).stdout.strip()
    print(f"    instance {args.commit[:8]}: {subject}")
    print(f"    src paths: {len(src)}   test paths: {len(tests)}   test pattern: {args.test}")

    touched = src + tests
    status = git("status", "--porcelain", "--", *touched).stdout.strip()
    residue = [line for line in status.splitlines() if line.strip()]
    if residue:
        raise SystemExit(f"refusing to run: the instance's paths are not clean ({len(residue)})")

    exit_code = 1
    try:
        revert = git("revert", "--no-commit", args.commit)
        if revert.returncode != 0:
            raise SystemExit(f"revert failed: {revert.stderr.strip()[-300:]}")
        keep = git("checkout", args.commit, "--", *tests)
        if keep.returncode != 0:
            raise SystemExit(f"could not keep the test changes: {keep.stderr.strip()[-300:]}")

        code, tail = build()
        if code != 0:
            raise SystemExit(f"the reverted tree does not build:\n{tail}")
        code, tail = test(args.test)
        if code == 0:
            raise SystemExit("instance is invalid: the test passes without the fix")
        print("    validated: the test fails without the fix")
        if args.validate_only:
            exit_code = 0
            return exit_code

        if args.gold:
            gold = git("show", args.commit, "--", *src).stdout
            patch = gold[gold.find("diff --git"):] if "diff --git" in gold else ""
            print(f"    gold patch: {len(patch.splitlines())} lines")
        else:
            prompt = args.prompt.read_text(encoding="utf-8") if args.prompt else (
                f"The repository at this commit cannot satisfy the case that tests/{args.test} covers.\n"
                f"The failing test is `{args.test}`; run it for the exact failure.\n"
                f"Implement the change in the source tree so that it passes. Reply with a unified diff "
                f"(git format) only, no prose and no code fences around it.")
            print(f"    asking the lane {args.lane} for a patch")
            patch = extract_patch(ask(args.lane, prompt))
        patch_file = TMP / "repo_fix_eval.patch"
        patch_file.write_text(patch, encoding="utf-8", newline="\n")
        apply_result = git("apply", "--verbose", str(patch_file))
        print(f"    patch: {len(patch.splitlines())} lines, apply rc={apply_result.returncode}")
        if apply_result.returncode != 0:
            print(f"    apply stderr: {apply_result.stderr.strip()[-300:]}")
            return 1
        code, tail = build()
        print(f"    build after the patch: rc={code}")
        if code != 0:
            print(f"    {tail[-400:]}")
            return 1
        code, tail = test(args.test)
        print(f"    test after the patch: rc={code}  {'RESOLVED' if code == 0 else 'NOT RESOLVED'}")
        if code != 0:
            print(f"    {tail[-400:]}")
        exit_code = 0 if code == 0 else 1
    finally:
        git("reset", "--quiet")
        git("checkout", "HEAD", "--", *touched)
        left = [line for line in git("status", "--porcelain").stdout.splitlines()
                if line.strip() and ".audit/" not in line]
        print(f"    restored; residual changes: {len(left)}"
              + (f" -> {left[:3]}" if left else ""))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
