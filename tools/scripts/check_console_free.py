"""Report whether a process tree has a console: any `conhost.exe` whose parent is in the tree.

Why this exists: on 2026-10-10 a controlled experiment showed that closing a console kills every attached
process at once, silently, with no WER report and no event-log entry, and that `DETACHED_PROCESS` does not
remove the console a venv trampoline allocates (`docs/research/silent-shell-death-2026-10-10.md`).
`GetConsoleWindow()` is the in-process test; from outside, a `conhost.exe` whose parent is inside the tree
is the observable proxy -- it matched the in-process test in both arms of that experiment.

Use it to verify a long detached run before trusting it: a tree with no console cannot be killed by a
console teardown, and a tree with one can.

Usage:
    python tools/scripts/check_console_free.py <pid> [<pid> ...]

Exit status: 0 no console in any tree, 1 a console was found, 2 nothing was checked.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

TH32CS_SNAPPROCESS = 0x00000002
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


kernel32.Process32FirstW.restype = wintypes.BOOL
kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
kernel32.Process32NextW.restype = wintypes.BOOL
kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]


def snapshot() -> list[tuple[int, int, str]]:
    handle = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if handle == INVALID_HANDLE_VALUE:
        raise SystemExit("cannot take a process snapshot")
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        rows: list[tuple[int, int, str]] = []
        if not kernel32.Process32FirstW(handle, ctypes.byref(entry)):
            return rows
        while True:
            rows.append((entry.th32ProcessID, entry.th32ParentProcessID, entry.szExeFile))
            if not kernel32.Process32NextW(handle, ctypes.byref(entry)):
                break
        return rows
    finally:
        kernel32.CloseHandle(handle)


def main() -> int:
    roots = [int(argument) for argument in sys.argv[1:]]
    if not roots:
        print("usage: check_console_free.py <pid> [<pid> ...]")
        return 2
    if sys.platform != "win32":
        print("no console model to check on this platform")
        return 2
    rows = snapshot()
    tree = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, parent, _ in rows:
            if parent in tree and pid not in tree:
                tree.add(pid)
                changed = True
    consoles = [(pid, parent) for pid, parent, name in rows
                if name.lower() == "conhost.exe" and parent in tree]
    print(f"tree from {roots}: {len(tree)} processes")
    for pid, parent in consoles:
        print(f"  console: conhost pid {pid}, parented by {parent}, inside the tree")
    if consoles:
        print("RESULT: this tree has a console; a console teardown can kill it")
        return 1
    print("RESULT: no conhost inside the tree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
