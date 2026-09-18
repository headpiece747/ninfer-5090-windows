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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiles import PROFILES, QUASAR, NVFP4FULL  # noqa: E402

WT = Path(r"C:\AI\ninfer-v3-windows")
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
    files = [p["file"] for p in PROFILES]
    print(f"\n=== launcher generator agreement ({len(PROFILES)} profiles) ===")
    for profile in PROFILES:
        shipped = read(WT / profile["file"])
        check(f"{profile['file']} exists", bool(shipped))
        if not shipped:
            continue
        check(f"{profile['file']} passes --max-context {profile['ctx']}",
              f"--max-context {profile['ctx']}" in shipped)
        check(f"{profile['file']} passes --port {profile['port']}",
              f"--port {profile['port']}" in shipped)
        check(f"{profile['file']} declares {profile['model_id']}",
              f"--model-id {profile['model_id']}" in shipped)
        check(f"{profile['file']} passes --vision", "--vision" in shipped)
        check(f"{profile['file']} passes --spec {profile['spec']}",
              f"--spec {profile['spec']}" in shipped)

    print("\n=== opencode providers ===")
    if not OPENCODE.exists():
        notes.append("opencode.json not found; skipped")
    else:
        config = json.loads(OPENCODE.read_text(encoding="utf-8"))
        entries = {}
        for provider in config.get("provider", {}).values():
            for model_id, model in provider.get("models", {}).items():
                base = provider.get("options", {}).get("baseURL", "")
                if base.startswith("http://127.0.0.1:80"):
                    entries[model_id] = (base, model)
        for profile in PROFILES:
            found = entries.get(profile["model_id"])
            check(f"opencode has {profile['model_id']}", found is not None)
            if not found:
                continue
            base, model = found
            check(f"{profile['model_id']} port {profile['port']}",
                  str(profile["port"]) in base, base)
            limit = model.get("limit", {})
            check(f"{profile['model_id']} context {profile['ctx']}",
                  limit.get("context") == profile["ctx"], str(limit))
            check(f"{profile['model_id']} accepts images",
                  "image" in (model.get("modalities", {}).get("input") or []))
            check(f"{profile['model_id']} output > thinking budget",
                  limit.get("output", 0) > 4096, str(limit.get("output")))

    print("\n=== verifier case list ===")
    verifier = read(WT / "tools" / "release" / "verify_launchers_v3.py")
    for profile in PROFILES:
        check(f"verifier covers {profile['file']}", profile["file"] in verifier)
        check(f"verifier expects port {profile['port']}",
              f", {profile['port']}, " in verifier or f"({profile['file']}\", {profile['port']}" in verifier)
    check("verifier requires the native ceiling", "262144" in verifier)

    print("\n=== docs ===")
    for doc in ("README.md", "RELEASE_NOTES.md"):
        body = read(WT / doc)
        for profile in PROFILES:
            check(f"{doc} names {profile['file']}", profile["file"] in body)
        for retired in RETIRED:
            check(f"{doc} free of retired {retired}", retired not in body)
        check(f"{doc} free of sub-262k ceilings", "163,840" not in body and "180,224" not in body)

    print("\n=== packager file list ===")
    packager = read(WT / "tools" / "release" / "package_release.py")
    for name in files:
        check(f"packager stages {name}", name in packager)
    for retired in RETIRED:
        check(f"packager drops retired {retired}", retired not in packager)

    print("\n=== measured artifacts ===")
    # Only files that genuinely have to know both artifacts are checked for both. The launchers
    # and download_model.py cover the pair; the measurement harnesses do too. launcher_env.bat is
    # the CLI test environment and deliberately names one artifact, so it is checked for the
    # QUASAR default only -- adding an unused second variable would be a seam with no caller.
    for doc, path in (("download_model.py", WT / "download_model.py"),
                      ("v3_profile_matrix.py", WT / "tools" / "release" / "v3_profile_matrix.py")):
        body = read(path)
        for artifact, label in ((QUASAR, "QUASAR"), (NVFP4FULL, "NVFP4-full")):
            check(f"{doc} references the {label} artifact", artifact in body)

    env = read(WT / "launcher_env.bat")
    check("launcher_env.bat names the QUASAR artifact", QUASAR in env)
    if "QUASAR_ARGS" in env:
        notes.append("launcher_env.bat QUASAR_ARGS is still a third, divergent MTP4 profile "
                     "missing nine flags the shipped launcher passes; moving it onto the module "
                     "is the next slice")

    print("\n=== retired launchers absent from the tree ===")
    for retired in RETIRED:
        check(f"{retired} not on disk", not (WT / retired).exists())

    print("\n=== harness profile tables ===")
    bench = read(WT / "tools" / "release" / "bench_opencode_settings.py")
    check("bench harness uses the NVFP4-full artifact", NVFP4FULL in bench)
    check("bench harness uses the QUASAR artifact", QUASAR in bench)

    for note in notes:
        print(f"\n   note: {note}")
    print(f"\n=== {len(failures)} disagreement(s) ===")
    for failure in failures:
        print(f"   {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
