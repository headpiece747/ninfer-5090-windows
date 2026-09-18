#!/usr/bin/env python3
"""Update the packager for the four-profile set, and ship the upgrader.

The launcher list still named the retired files. And the QUASAR artifact is published as a v2
container, which a v3 engine rejects, so download_model.py tells the user to run the offline
upgrader -- which then has to actually be in the archive. It lives under tools/, so the
packager gains a second source directory.
"""
from __future__ import annotations

from pathlib import Path

PACKAGER = Path(r"C:\AI\ninfer-v3-windows\tools\release\package_release.py")

OLD_FILES = '''LAUNCHER_FILES = [
    "launcher_env.bat", "download_model.bat", "download_model.py",
    "start_quasar_v3_dflash2_vision.bat", "start_quasar_v3_mtp4_vision.bat",
    "start_ninfer_v3_dflash2.bat", "start_ninfer_v3_dflash2_vision.bat",
    "start_ninfer_v3_mtp5.bat", "start_ninfer_v3_mtp5_vision.bat",
]'''

NEW_FILES = '''LAUNCHER_FILES = [
    "launcher_env.bat", "download_model.bat", "download_model.py",
    "start_quasar_v3_dflash2_vision.bat", "start_quasar_v3_mtp4_vision.bat",
    "start_ninfer_v3_dflash2_vision.bat", "start_ninfer_v3_mtp5_vision.bat",
]

# Shipped from tools/ rather than the root. The QUASAR artifact is published as a v2
# container, so download_model.py sends the user to this offline upgrader; it has to be here.
TOOL_FILES = ["upgrade_ninfer_v2_to_v3.py"]'''

OLD_LOOP = '''    for group, location in ((EXES + DLLS, BUILD), (ROOT_FILES, REPO), (LAUNCHER_FILES, REPO)):'''
NEW_LOOP = '''    for group, location in ((EXES + DLLS, BUILD), (ROOT_FILES, REPO),
                            (LAUNCHER_FILES, REPO), (TOOL_FILES, REPO / "tools")):'''


def main() -> int:
    text = PACKAGER.read_text(encoding="utf-8")
    if "TOOL_FILES" in text:
        print("  packager already updated")
        return 0
    for old, new, label in ((OLD_FILES, NEW_FILES, "file lists"),
                            (OLD_LOOP, NEW_LOOP, "copy loop")):
        if old not in text:
            print(f"  ANCHOR NOT FOUND: {label}")
            return 1
        text = text.replace(old, new, 1)
    PACKAGER.write_text(text, encoding="utf-8", newline="")
    print("  packager: four launchers plus the upgrader from tools/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
