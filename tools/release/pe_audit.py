"""Portable Executable (PE) import table inspection for release gates.

Extracts imported DLL names directly from PE32+ (64-bit) headers without external dependencies
like dumpbin or pefile, and audits that non-system DLL dependencies are satisfied by the release archive.
"""
from __future__ import annotations

import struct
from pathlib import Path

SYSTEM_DLL_PREFIXES = ("api-ms-", "ext-ms-")

SYSTEM_DLLS = {
    "advapi32.dll",
    "avicap32.dll",
    "avrt.dll",
    "bcrypt.dll",
    "bcryptprimitives.dll",
    "cfgmgr32.dll",
    "comdlg32.dll",
    "crypt32.dll",
    "d2d1.dll",
    "dnsapi.dll",
    "dwrite.dll",
    "gdi32.dll",
    "imm32.dll",
    "iphlpapi.dll",
    "kernel32.dll",
    "msimg32.dll",
    "msvcp140.dll",
    "msvcp140_atomic_wait.dll",
    "ncrypt.dll",
    "normaliz.dll",
    "ntdll.dll",
    "odbc32.dll",
    "odbccp32.dll",
    "ole32.dll",
    "oleaut32.dll",
    "rpcrt4.dll",
    "sechost.dll",
    "secur32.dll",
    "setupapi.dll",
    "shell32.dll",
    "shlwapi.dll",
    "user32.dll",
    "userenv.dll",
    "usp10.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "version.dll",
    "winhttp.dll",
    "winmm.dll",
    "winspool.drv",
    "ws2_32.dll",
}


def is_system_dll(name: str) -> bool:
    """Return True if the DLL is an expected Windows OS or Universal CRT runtime library."""
    lower = name.lower()
    return lower.startswith(SYSTEM_DLL_PREFIXES) or lower in SYSTEM_DLLS


def pe_imported_dlls(data: bytes) -> list[str]:
    """Parse PE32+ import directory table for all directly imported DLL names."""
    if len(data) < 0x40:
        return []
    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    if pe_off + 24 > len(data) or data[pe_off : pe_off + 4] != b"PE\0\0":
        return []
    num_secs = struct.unpack_from("<H", data, pe_off + 6)[0]
    opt_size = struct.unpack_from("<H", data, pe_off + 20)[0]
    magic = struct.unpack_from("<H", data, pe_off + 24)[0]
    # 0x20b is PE32+ (64-bit)
    if magic != 0x20B or pe_off + 148 > len(data):
        return []
    import_rva = struct.unpack_from("<I", data, pe_off + 144)[0]
    if not import_rva:
        return []
    secs_off = pe_off + 24 + opt_size
    if secs_off + num_secs * 40 > len(data):
        return []
    sections = [
        (
            struct.unpack_from("<II", data, secs_off + i * 40 + 12)[0],
            struct.unpack_from("<I", data, secs_off + i * 40 + 8)[0],
            struct.unpack_from("<I", data, secs_off + i * 40 + 20)[0],
        )
        for i in range(num_secs)
    ]

    def rva2off(rva: int) -> int | None:
        for s_rva, s_vsz, s_raw in sections:
            if s_rva <= rva < s_rva + max(s_vsz, 1):
                return s_raw + (rva - s_rva)
        return None

    desc_off = rva2off(import_rva)
    if desc_off is None:
        return []
    dlls: list[str] = []
    while desc_off + 20 <= len(data):
        desc = struct.unpack_from("<IIIII", data, desc_off)
        if desc == (0, 0, 0, 0, 0):
            break
        name_off = rva2off(desc[3])
        if name_off is not None and name_off < len(data):
            nul = data.find(b"\0", name_off)
            if nul != -1:
                dlls.append(data[name_off:nul].decode("ascii", errors="replace"))
        desc_off += 20
    return dlls


def audit_binary_dependencies(binaries: dict[str, bytes]) -> list[str]:
    """Verify that all non-system DLL dependencies of each binary are satisfied in binaries.

    binaries maps relative filename (e.g. 'ninfer-serve.exe') to file contents bytes.
    Returns a list of error strings for any unresolved dependencies.
    """
    staged_dll_names = {
        Path(name).name.lower() for name in binaries if name.lower().endswith(".dll")
    }
    errors: list[str] = []
    for binary_name, data in binaries.items():
        if not binary_name.lower().endswith((".exe", ".dll")):
            continue
        for imported in pe_imported_dlls(data):
            if is_system_dll(imported):
                continue
            if imported.lower() not in staged_dll_names:
                errors.append(
                    f"{binary_name} imports {imported}, which is not present in the archive"
                )
    return errors
