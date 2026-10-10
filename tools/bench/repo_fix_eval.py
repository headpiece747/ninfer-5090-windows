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
    python tools/bench/repo_fix_eval.py --commit <sha> --test <name> \
        --agent local-ninfer-v3/<model> --agent-bin <desktop>/cli/2.0.26/opencode-cli.exe

**The agent path, measured 2026-10-09 on the same instance.** `--agent` drives the opencode CLI, which
gives the model tools and the real tree. It investigated -- `git status`, `git diff`, a build -- completed
`rc=0`, and left no change on the instance's paths, so the harness reported "nothing to score". That is a
negative result with a visible shape, and it is the right shape for this eval: the field's 32B-class
resolve rates are quoted *with* a scaffold, for a reason. The one-shot path, asked the same question,
returned a diff for a file that does not exist. Both results are recorded because the difference between
them is the point.

`--gold` applies the commit's own source diff instead of asking a model: it validates the loop (apply,
build, test) before any model is involved. Measured 2026-10-09 on 15cba227: the reverted tree fails
`ninfer_linear_(add_)?bf16_a16_test`, the 94-line gold patch applies, builds and resolves it, and the
tree is restored with zero residual changes.

**Measured on the model path the same day, and it is the interesting half.** Asked once over HTTP -- no
tools, no repository access -- the model returned a 54-line diff for a file that does not exist
(`src/ops/linear_bf16_a16.cpp`, with a fabricated `index` line), and `git apply` correctly refused it.
That is not a harness defect: a SWE-style task needs repository access, which is why the field's
harnesses at this scale are agents (mini-SWE-agent, OpenHands) and why a 32B-class resolve rate like
OpenHands-LM's 37.2% is quoted *with* a scaffold. The useful shape for `--lane` is an agent that can read
the tree; the one-shot path stays as the cheap control it is.

A model's diff can also be malformed in ways git repairs rather than rejects, so apply falls back to
`--recount` and then `--3way`. `--recount` is proven against a count-corrupted hunk -- plain apply says
`corrupt patch`, `--recount` applies it -- and its scope is measured: it recomputes counts, not start
lines, so a hunk whose content moved is a different failure it cannot fix.
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
    """Run a build or test command inside the MSVC environment, the way the recipes do.

    The child asks to break away from any job object: this shell's background commands run inside one
    (measured 2026-10-09, `IsProcessInJob` true), and a job close kills its members silently -- no flush,
    no traceback -- so a long build started that way would die with its session and take its own log with
    it. Breakaway is permitted here; where it is not, the fallback keeps the call working.
    """
    script = TMP / "repo_fix_eval_step.cmd"
    body = '@echo off\r\n' f'call "{VCVARS}" >nul 2>&1\r\n' + " ".join(args) + "\r\n"
    script.write_text(body, encoding="utf-8", newline="")
    flags = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    try:
        return subprocess.run(["cmd", "/c", str(script)], cwd=str(REPO), capture_output=True,
                              encoding="utf-8", errors="replace", timeout=timeout, check=False,
                              creationflags=flags)
    except OSError:
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
    parser.add_argument("--agent", metavar="MODEL",
                        help="ask an opencode agent with repository access instead of one HTTP call; it "
                             "edits the tree and the patch is the diff it leaves, so there is no apply "
                             "step. The provider's baseURL decides which lane must be serving -- "
                             "local-ninfer-v3 points at 127.0.0.1:8086, the QUASAR lane")
    parser.add_argument("--agent-bin", default="opencode",
                        help="the opencode CLI to drive; the one on PATH may be a V1 build that rejects "
                             "the V2 config ('Unrecognized keys: providers, media, snapshots'), so point "
                             "this at the desktop's bundled 2.x opencode-cli.exe when that happens")
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
        failure_output = tail
        print("    validated: the test fails without the fix")
        if args.validate_only:
            exit_code = 0
            return exit_code

        already_applied = False
        if args.agent:
            agent_prompt = args.prompt.read_text(encoding="utf-8") if args.prompt else (
                f"`{args.test}` fails in this repository. Its output was:\n\n{failure_output}\n\n"
                f"Fix the source so the test passes, without changing the test. Work in the tree.")
            print(f"    asking the opencode agent {args.agent} (it edits the tree directly)")
            agent = subprocess.run([args.agent_bin, "run", "--model", args.agent, agent_prompt],
                                   cwd=str(REPO), capture_output=True, encoding="utf-8",
                                   errors="replace", timeout=2400, check=False)
            agent_tail = ((agent.stdout or "") + (agent.stderr or "")).strip()
            print(f"    agent rc={agent.returncode}; tail: {agent_tail[-300:]}")
            git("add", "-N", "--", *src)
            diff = git("diff", "--", *src).stdout
            patch = diff[diff.find("diff --git"):] if "diff --git" in diff else ""
            print(f"    the agent's diff: {len(patch.splitlines())} lines")
            already_applied = True
        elif args.gold:
            gold = git("show", args.commit, "--", *src).stdout
            patch = gold[gold.find("diff --git"):] if "diff --git" in gold else ""
            print(f"    gold patch: {len(patch.splitlines())} lines")
        else:
            prompt = args.prompt.read_text(encoding="utf-8") if args.prompt else (
                f"The repository at this commit cannot satisfy the case that tests/{args.test} covers.\n"
                f"The failing test is `{args.test}`, and its output was:\n\n{failure_output}\n\n"
                f"Implement the change in the source tree so that it passes. Reply with a unified diff "
                f"(git format) only, no prose and no code fences around it.")
            print(f"    asking the lane {args.lane} for a patch")
            patch = extract_patch(ask(args.lane, prompt))
        if already_applied and not patch:
            print("    the agent left no tracked change on the instance's paths; nothing to score")
            return 1
        if already_applied:
            apply_result = subprocess.CompletedProcess([], 0)
        else:
            patch_file = TMP / "repo_fix_eval.patch"
            patch_file.write_text(patch, encoding="utf-8", newline="\n")
            apply_result = git("apply", "--verbose", str(patch_file))
        if apply_result.returncode != 0:
            # A model's diff is often malformed in ways git can repair rather than reject: --recount
            # rebuilds the hunk headers, --3way uses the blob context. SWE-bench's own harness carries
            # the same kind of fallback (git apply, then patch with fuzz) for exactly this reason.
            print(f"    apply rc={apply_result.returncode}; retrying with --recount, then --3way")
            apply_result = git("apply", "--recount", str(patch_file))
            if apply_result.returncode != 0:
                apply_result = git("apply", "--3way", str(patch_file))
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
        left = [line for line in git("status", "--porcelain", "--", *touched).stdout.splitlines()
                if line.strip()]
        others = [line for line in git("status", "--porcelain").stdout.splitlines()
                  if line.strip() and ".audit/" not in line]
        print(f"    restored; residual changes on the instance's paths: {len(left)}"
              + (f" -> {left[:3]}" if left else "")
              + (f"; other tree changes: {len(others)}" if others else ""))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
