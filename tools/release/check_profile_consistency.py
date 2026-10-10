#!/usr/bin/env python3
"""Check every remaining profile consumer against the profile table.

profiles.py is the source of truth, but some consumers still hold their own copy: opencode's
provider entries, the doc tables, the verifier's case list, the harnesses' profile dicts and the
packager's file list. Moving all of them onto the module is the next slice; until then this gate
makes the duplication checked rather than silent.

It is written to have caught the drifts that actually happened: two doc tables still naming
retired launchers at sub-262k ceilings, a stale profile dict in a bench harness, and the
packager shipping a file list the module no longer produces.

Exit code 1 if any consumer disagrees.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_launchers_v3 import render  # noqa: E402
from profiles import PROFILES, QUASAR, NVFP4FULL, cli_args, launcher_args, template_path  # noqa: E402
from v3_profile_matrix import build_args  # noqa: E402
from bench_opencode_settings import PROFILES as BENCH_PROFILES  # noqa: E402

WT = Path(__file__).resolve().parents[2]
OPENCODE = Path(r"C:\Users\tobia\.config\opencode\opencode.json")

RETIRED = ["start_ninfer_v3_dflash2.bat", "start_ninfer_v3_mtp5.bat",
           "start_quasar_fp8_vision.bat"]

failures: list[str] = []
notes: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"   {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(f"{label}: {detail}")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def main() -> int:
    print(f"\n=== launcher generator agreement ({len(PROFILES)} profiles) ===")
    for profile in PROFILES:
        path = WT / profile["file"]
        check(f"{profile['file']} exists", path.exists())
        if not path.exists():
            continue
        # One comparison replaces the per-fact substring checks. The launcher is a generated
        # file, so byte-identity against the table's own render is strictly stronger than
        # looking for each fact in its text, and it is the only check that fails when the
        # template or the flag order changes rather than a value.
        #
        # render() now emits CRLF itself and the generator writes it unchanged, so this is a straight
        # byte comparison. Converting here instead made the verdict depend on the platform -- the two
        # sides converted LF differently and the gate passed on one machine and failed on the other by
        # one byte per line.
        expected = render(profile).encode("utf-8")
        shipped = path.read_bytes()
        check(f"{profile['file']} is byte-identical to the table's render",
              shipped == expected,
              f"{len(shipped)} bytes on disk against {len(expected)} rendered")

    print("\n=== the measurement harness measures what ships ===")
    harness = read(WT / "tools" / "release" / "v3_profile_matrix.py")
    verifier = read(WT / "tools" / "release" / "verify_launchers_v3.py")
    check("matrix composes shipped flags through profiles.launcher_args",
          "launcher_args" in harness,
          "the harness must render the profile's own flag list, not rebuild one")
    for name in ("v3_profile_matrix.py", "check_host_kv.py", "probe_reasoning_effort.py",
                 "bench_opencode_settings.py"):
        body = read(WT / "tools" / "release" / name)
        check(f"{name} starts Serve with the lane's environment",
              "launcher_environment(" in body,
              "a server started without the table's environment measures a configuration nobody "
              "ships: the profiles pin the CUDA wait schedule, and Popen would otherwise inherit "
              "whatever the harness happened to have")
    check("matrix can measure a shipped profile by name", "mode_profile" in harness)
    check("the table's stated provenance names that mode",
          "`profile` mode of v3_profile_matrix.py" in read(WT / "tools" / "release" / "profiles.py"),
          "the provenance sentence and the harness mode must agree")
    check("launcher verifier drains VRAM before starting", "wait_free()" in verifier,
          "an accounting line taken with a leftover process resident is not comparable")

    print("\n=== the recorded digests agree across the documents ===")
    # The downloader pins no digest any more, so the digests live in three documents and nothing
    # compared them; a digest quoted in one of them that the reference does not record is either
    # stale or invented. The reference's identity blocks are the source.
    reference = read(WT / "docs" / "maintainer" / "qwen3.8-27b-artifact.md")
    recorded = {m[:8] for m in re.findall(r"sha256\s*=\s*([0-9a-f]{64})", reference)}
    check("the reference records the shipping digests", len(recorded) >= 5,
          "one identity block per artifact")
    quoted = set(re.findall(r"`([0-9a-f]{8})…`", read(WT / "README.md")))
    quoted |= set(re.findall(r"`([0-9a-f]{8})…`", read(WT / "RELEASE_NOTES.md")))
    for short in sorted(quoted):
        check(f"{short} quoted in a doc is a digest the reference records",
              short in recorded, "a digest no identity block records is stale or invented")
    check("the docs quote at least one digest", bool(quoted), "or the check above is vacuous")

    print("\n=== doc tables quote the table ===")
    for doc_path in (WT / "README.md", WT / "RELEASE_NOTES.md"):
        text = read(doc_path)
        for profile in PROFILES:
            check(f"{doc_path.name} quotes {profile['file']} at {round(profile['tok'])} tok/s",
                  f"{round(profile['tok'])} tok/s" in text,
                  "a doc table is a transcription site: it must match profiles.PROFILES")
            check(f"{doc_path.name} quotes {profile['file']}'s acceptance {profile['acc']}",
                  f"| {profile['acc']} |" in text)

    print("\n=== opencode providers ===")
    if not OPENCODE.exists():
        notes.append("opencode.json not found; skipped")
    else:
        config = json.loads(OPENCODE.read_text(encoding="utf-8"))
        entries = {}
        for provider in (config.get("providers") or config.get("provider") or {}).values():
            for model_id, model in provider.get("models", {}).items():
                base = (provider.get("settings") or provider.get("options") or {}).get("baseURL", "")
                if base.startswith("http://127.0.0.1:80"):
                    # reasoningEffort belongs on the model. ProviderConfig.options declares a fixed
                    # property list (apiKey, baseURL, enterpriseUrl, setCacheKey, timeout,
                    # headerTimeout, chunkTimeout) and reasoningEffort is not in it, so a
                    # provider-level value is silently ignored rather than applied to its models.
                    # models.<id>.options is a free-form object, which is where AI SDK call options
                    # go. Reading both would hide exactly that mistake.
                    effort = (model.get("settings") or model.get("options") or {}).get("reasoningEffort")
                    provider_effort = ((provider.get("settings") or provider.get("options") or {})
                           .get("reasoningEffort"))
                    entries[model_id] = (base, model, effort, provider_effort)
        for profile in PROFILES:
            found = entries.get(profile["model_id"])
            check(f"opencode has {profile['model_id']}", found is not None)
            if not found:
                continue
            base, model, effort, provider_effort = found
            check(f"{profile['model_id']} port {profile['port']}",
                  str(profile["port"]) in base, base)
            check(f"{profile['model_id']} sets reasoningEffort on the model, not the provider",
                  provider_effort is None,
                  f"provider.options.reasoningEffort={provider_effort!r} is ignored by the schema")
            limit = model.get("limit", {})
            check(f"{profile['model_id']} context {profile['ctx']}",
                  limit.get("context") == profile["ctx"], str(limit))
            check(f"{profile['model_id']} accepts images",
                  "image" in ((model.get("capabilities") or {}).get("input")
                              or (model.get("modalities") or {}).get("input") or []))
            check(f"{profile['model_id']} output > thinking budget",
                  limit.get("output", 0) > 4096, str(limit.get("output")))
            # The lanes default to the model's own default reasoning effort. It used to be "none",
            # which turned thinking off and therefore selected the non-thinking sampling preset --
            # temperature 0.7, top_p 0.80, presence_penalty 1.5.
            #
            # The reason to move to xhigh is the card, which names xhigh the default, for complex
            # tasks demanding thorough analysis, and which measures its coding benchmarks at
            # temperature 1.0 and top_p 0.95 -- the thinking set. That alone is sufficient.
            #
            # The non-thinking set is also the slower one to speculate under, which is a second and
            # independent reason to prefer xhigh but is not a correctness matter. The drafter reads
            # only temperature and seed out of the sampling config (candidate_selector_path.cu) while
            # the verify path applies the penalty overlay, so the drafter's q is not derived from the
            # penalised p. Measured on QUASAR dflash2: 4.18 tokens per round at penalty 0 against 2.45
            # at 1.5. It does not change what is sampled -- the accept test is min(1, p/q) with a
            # normalize(max(0, p - q)) correction, which is the speculative-sampling identity and holds
            # for any normalised q, and p is renormalised after the top_p and min_p cuts by
            # sampling_normalize_support. An earlier revision of this comment called the asymmetry a
            # defect; it is an acceptance-efficiency difference and this gate does not rest on it.
            check(f"{profile['model_id']} defaults to the model's reasoning effort",
                  effort == "xhigh", f"reasoningEffort={effort!r}")

    print("\n=== verifier case list ===")
    verifier = read(WT / "tools" / "release" / "verify_launchers_v3.py")
    # The verifier derives its cases from the table now, so the per-profile checks that used to
    # assert the literals are gone with them. Asserting that a copy exists is what kept it there:
    # removing the literals turned those thirteen checks red, which is the proof of that claim.
    check("verifier derives its cases from the module",
          "from profiles import PROFILES" in verifier)
    check("verifier does not restate the launcher file names",
          "start_quasar_v3_" not in verifier and "start_ninfer_v3_" not in verifier)
    check("verifier does not restate a ceiling", "262144" not in verifier)

    print("\n=== docs ===")
    for doc in ("README.md", "RELEASE_NOTES.md"):
        body = read(WT / doc)
        for profile in PROFILES:
            check(f"{doc} names {profile['file']}", profile["file"] in body)
        # The Spec column pairs a launcher with its draft count, and it has been left behind twice
        # when a row's depth changed: nvfp4fullnoex shipped depth 9 beside depth-7 figures, and the
        # ninfer row read "DFlash2 (7)" while its figures were the depth-9 ones. Checked against the
        # module now, because a figure and a setting that disagree is a defect whichever one is right.
        spec_labels = {"dflash2": "DFlash2", "mtp": "MTP"}
        for profile in PROFILES:
            if profile["spec"] not in spec_labels:
                continue
            want = f"{spec_labels[profile['spec']]} ({profile['draft']})"
            row = next((line for line in body.splitlines()
                        if line.startswith("| `") and f"`{profile['file']}`" in line), "")
            check(f"{doc} pairs {profile['file']} with {want}", want in row,
                  f"row reads: {row.strip()[:100]}")
        # A prose count of a configuration drifts silently: "one lane ships k8v4" became four and then
        # five in a single day, each time caught by reading rather than by a gate. Assert the sentence's
        # number against the rows.
        words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
                 8: "eight"}
        stated = re.search(r"\*\*(\w+) lanes ship `--kv-dtype k8v4`\*\*", body)
        k8v4_rows = len([p for p in PROFILES if p.get("kv_dtype") == "k8v4"])
        check(f"{doc} states the k8v4 lane count correctly",
              stated is not None
              and stated.group(1).lower() == words.get(k8v4_rows, str(k8v4_rows)),
              f"doc says {stated.group(1) if stated else 'nothing'}, rows have {k8v4_rows}")
        for retired in RETIRED:
            check(f"{doc} free of retired {retired}", retired not in body)
        check(f"{doc} free of sub-262k ceilings", "163,840" not in body and "180,224" not in body)

    print("\n=== packager file list ===")
    packager = read(WT / "tools" / "release" / "package_release.py")
    # The four start_*.bat come from the table, so checking that their names appear in the
    # packager would assert the copy this change removed. The hand-written three remain.
    for name in ("launcher_env.bat", "download_model.bat", "download_model.py", "build_model.py"):
        check(f"packager stages {name}", name in packager)
    # The converter travels with the archive, or a fresh install can fetch sources and not build them.
    check("packager stages the converter packages",
          "TOOL_PACKAGES" in packager and '"convert"' in packager and '"artifact"' in packager)
    check("the converter packages land under tools/ in the archive",
          'stage / "tools"' in packager,
          "staged at the root they are not importable as tools.convert")
    check("packager derives the launcher list from the module",
          "from profiles import PROFILES" in packager)
    check("packager does not restate the launcher file names",
          "start_quasar_v3_" not in packager and "start_ninfer_v3_" not in packager)
    for retired in RETIRED:
        check(f"packager drops retired {retired}", retired not in packager)

    print("\n=== measured artifacts ===")
    # v3_profile_matrix.py measures the shipped lanes, so it must know both artifacts.
    body = read(WT / "tools" / "release" / "v3_profile_matrix.py")
    for artifact, label, constant in ((QUASAR, "QUASAR", "QUASAR"),
                                      (NVFP4FULL, "NVFP4-full", "NVFP4FULL")):
        # Either the filename is written out, or the file imports the constant that owns it.
        # The matrix does the latter now; asserting only the literal would keep the copy.
        check(f"v3_profile_matrix.py references the {label} artifact",
              artifact in body or ("from profiles import" in body and constant in body))

    # The probe path composes the shipped flags itself, so a flag that moves out of the invariant
    # list can silently stop being passed at all. That happened on 2026-10-09: kv-dtype moved to the
    # per-profile column, the probe's special case for it went dead, and every probe start ran on the
    # engine's default format while its record claimed the requested one. Assert the probe emits every
    # flag it varies, not merely that the module mentions the invariant list.
    import importlib.util

    matrix_spec = importlib.util.spec_from_file_location(
        "matrix_under_test", WT / "tools" / "release" / "v3_profile_matrix.py")
    if matrix_spec is None or matrix_spec.loader is None:
        check("v3_profile_matrix.py is importable by the gate", False)
    else:
        matrix_module = importlib.util.module_from_spec(matrix_spec)
        matrix_spec.loader.exec_module(matrix_module)
        probe = matrix_module.build_args("quasar", "dflash2", 9, True, 262144, "probe.jsonl",
                                         greedy=True, lm_head=True, kv_dtype="fp8")
        probe_flags = {token for token in probe if token.startswith("--")}
        for flag in ("--kv-capacity", "--kv-dtype", "--spec", "--draft-tokens",
                     "--max-context", "--vision", "--lm-head-draft"):
            check(f"the probe emits {flag} even though it varies it", flag in probe_flags)
        # And the other half of the same class: a flag that leaves the invariant list entirely is
        # dropped by the probe while the launchers keep it -- which is how kv-dtype silently stopped
        # being passed at all. Assert the probe's flag set covers a launcher's, allowing the flags it
        # deliberately varies and --device-state-slots, which the probe omits because the server's
        # default is the value every launcher ships (1); that omission is noted rather than fixed here.
        launcher_profile = next(p for p in PROFILES if p["file"] == "start_quasar_v3_dflash2_vision.bat")
        launcher_flags = {token for token in launcher_args(launcher_profile) if token.startswith("--")}
        varied = {"--kv-capacity", "--kv-dtype", "--spec", "--draft-tokens", "--max-context",
                  "--lm-head-draft", "--vision", "--host", "--port", "--model-id",
                  "--device-state-slots"}
        missing = launcher_flags - probe_flags - varied
        check("the probe covers every launcher flag it does not vary", not missing,
              f"missing {sorted(missing)}")

    # download_model.py no longer fetches artifacts at all -- it fetches the SOURCE checkpoints this
    # port builds every artifact from, because each shipped image is produced locally ("converter:
    # ninfer-v3" in its .conversion.json). So the invariant here is the reverse of the one above: it
    # must NOT name a shipped artifact, because doing so is what let it serve stale prebuilt bytes,
    # and it must name every source repository the conversion reports consume.
    download = read(WT / "download_model.py")
    # "Mentions" is not "serves": the docstring quotes the old artifact while explaining what it
    # replaced, so a substring test on the artifact name would fail on prose. What made it a SERVER
    # was the two mechanisms below -- a single-file sha256 pin, and a per-file fetch. Both are gone,
    # and both are what let it install superseded bytes under a verified label.
    check("download_model.py pins no prebuilt artifact digest", '"sha256"' not in download)
    check("download_model.py fetches whole repositories, not single files",
          "hf_hub_download" not in download)
    check("download_model.py downloads sources via snapshot_download",
          "snapshot_download" in download)
    # The source list is read from the module rather than matched as text: a substring test over prose
    # fails on a comment that names the same repository, which is exactly what a comment explaining
    # why the Swift 1.0 pair is no longer fetched does.
    import importlib.util

    spec = importlib.util.spec_from_file_location("download_model_under_test", WT / "download_model.py")
    if spec is None or spec.loader is None:
        check("download_model.py is importable", False)
    else:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        check("download_model.py fetches exactly the shipping sources",
              set(module.SOURCES) == {
                  "Qwen/Qwen3.8-27B", "z-lab/Qwen3.8-27B-DFlash2", "unsloth/Qwen3.8-27B-NVFP4",
                  "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4", "nvidia/Qwen3.8-27B-NVFP4",
                  "ukisai/Swift-1.5-Qwen3.8-27b", "ukisai/Swift-1.5-Qwen3.8-27b-NVFP4"})
        # A floating `main` is what made a build unreproducible before: every source is pinned at a
        # 40-hex commit, fetched at that revision, and the resolved commit is recorded.
        check("every source carries a 40-hex revision pin",
              all(len(entry[2]) == 40 and all(c in "0123456789abcdef" for c in entry[2])
                  for entry in module.SOURCES.values()))
    check("download_model.py fetches at the pinned revision",
          "snapshot_download(repo_id=repo, local_dir=dest, revision=revision)" in download)
    check("download_model.py records the resolved revision", "sources.json" in download)

    env = read(WT / "launcher_env.bat")
    check("launcher_env.bat names the QUASAR artifact", QUASAR in env)

    # QUASAR_ARGS feeds test_prompt.bat and test_vision.bat, which invoke ninfer.exe -- the
    # offline CLI, a different interface from the server. It rejects --host, --port, --model-id,
    # --max-concurrency, the state slots, the host KV pool, the cache bounds and the timeouts.
    # It is generated from profiles.cli_args now; assert it matches that, not the server's list.
    mtp4 = next(p for p in PROFILES if "mtp4" in p["file"])
    line = next((l for l in env.splitlines() if l.startswith('set "QUASAR_ARGS=')), "")
    tokens = line.split("QUASAR_ARGS=", 1)[1].rstrip('"').split() if line else []
    check("QUASAR_ARGS matches the CLI flag set", tokens == cli_args(mtp4),
          f"{len(tokens)} tokens against {len(cli_args(mtp4))} generated")
    for flag in ("--vision", "--lm-head-draft", "--prefill-chunk", "--max-context",
                 "--kv-capacity", "--kv-dtype"):
        check(f"QUASAR_ARGS passes {flag}", flag in tokens)
    for flag in ("--host", "--port", "--model-id", "--max-concurrency",
                 "--host-context-mib", "--pending-timeout-ms"):
        check(f"QUASAR_ARGS omits the server-only {flag}", flag not in tokens)

    print("\n=== retired launchers absent from the tree ===")
    for retired in RETIRED:
        check(f"{retired} not on disk", not (WT / retired).exists())

    print("\n=== harness profile tables ===")
    # The bench harness no longer names the artifacts: it derives its profile dict from the
    # module, which is the point. Assert the derivation, not the literal strings.
    bench = read(WT / "tools" / "release" / "bench_opencode_settings.py")
    check("bench harness derives its profiles from the module",
          "from profiles import PROFILES as SHIPPED" in bench)
    # Derived, and complete: a literal port list zipped against the table truncates silently, so
    # assert the coverage rather than the derivation.
    check("bench harness covers every shipped profile",
          len(BENCH_PROFILES) == len(PROFILES),
          f"{len(BENCH_PROFILES)} keys against {len(PROFILES)} profiles")
    check("bench harness does not restate the artifact filenames",
          "qwen3_8_27b_" not in bench)

    print("\n=== harnesses cross the module seam ===")
    # Each of these built its own argument list, and theirs omitted the cache bounds, the
    # thinking budget and the pending timeout -- so their records came from flags no launcher
    # ships. They must compose from the module instead.
    for name in ("v3_profile_matrix.py", "bench_opencode_settings.py", "check_host_kv.py",
                 "probe_reasoning_effort.py"):
        body = read(WT / "tools" / "release" / name)
        check(f"{name} imports from profiles", "from profiles import" in body)
        check(f"{name} does not hand-write the cache bounds",
              "--host-context-mib" not in body or "INVARIANT_FLAGS" in body
              or "launcher_args" in body)

    # The matrix explores combinations no profile covers, so it composes the invariants rather
    # than calling launcher_args. Assert that specifically -- and assert the composed result, because
    # a text check cannot see the launcher's `%TEMPLATE%` reaching the command line unresolved, which
    # is a startup failure that the context ladder then reports as a refusal.
    matrix = read(WT / "tools" / "release" / "v3_profile_matrix.py")
    check("matrix composes INVARIANT_FLAGS", "INVARIANT_FLAGS" in matrix)
    composed = build_args("quasar", "mtp", 4, True, 262144, Path("unused.jsonl"))
    check("matrix resolves the launcher's cmd variables", "%TEMPLATE%" not in composed,
          " ".join(composed))
    check("matrix points at the maintained template file", template_path() in composed)

    for note in notes:
        print(f"\n   note: {note}")
    print(f"\n=== {len(failures)} disagreement(s) ===")
    for failure in failures:
        print(f"   {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
