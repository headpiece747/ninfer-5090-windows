"""Keep large offline transfers from retaining whole artifacts in Linux's page cache."""

from __future__ import annotations

import os
import sys

IO_CHUNK_BYTES = 8 * 1024 * 1024
WRITEBACK_BYTES = 64 * 1024 * 1024

if sys.platform == "win32":
    # sysconf/posix_fadvise/fdatasync are POSIX page-cache controls with no Windows
    # equivalent. Dropping them changes only how long clean pages linger in the cache;
    # the bytes written are identical, so the advice is a no-op and the sync is fsync.
    _PAGE_BYTES = 4096

    def discard_cached_pages(fd: int, offset: int = 0, count: int | None = None) -> None:
        return

    def _sync_fd(fd: int) -> None:
        os.fsync(fd)
else:
    _PAGE_BYTES = os.sysconf("SC_PAGE_SIZE")

    def discard_cached_pages(fd: int, offset: int = 0, count: int | None = None) -> None:
        if count is None:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        elif count > 0:
            begin = offset // _PAGE_BYTES * _PAGE_BYTES
            end = (offset + count + _PAGE_BYTES - 1) // _PAGE_BYTES * _PAGE_BYTES
            os.posix_fadvise(fd, begin, end - begin, os.POSIX_FADV_DONTNEED)

    def _sync_fd(fd: int) -> None:
        os.fdatasync(fd)


# Windows os.open defaults to text mode, which would translate CRLF and stop at 0x1A inside
# artifact payloads. O_BINARY does not exist on POSIX.
BINARY = getattr(os, "O_BINARY", 0)

if sys.platform == "win32":

    def pread(fd: int, count: int, offset: int) -> bytes:
        saved = os.lseek(fd, 0, os.SEEK_CUR)
        try:
            os.lseek(fd, offset, os.SEEK_SET)
            return os.read(fd, count)
        finally:
            os.lseek(fd, saved, os.SEEK_SET)

    def pwrite(fd: int, data, offset: int) -> int:
        saved = os.lseek(fd, 0, os.SEEK_CUR)
        try:
            os.lseek(fd, offset, os.SEEK_SET)
            return os.write(fd, data)
        finally:
            os.lseek(fd, saved, os.SEEK_SET)

else:
    pread = os.pread
    pwrite = os.pwrite


class Writeback:
    """Bound dirty output across all open shards; release clean pages after writeback."""

    def __init__(self) -> None:
        self._bytes = 0
        self._fds: set[int] = set()

    def written(self, fd: int, count: int) -> None:
        self._fds.add(fd)
        self._bytes += count
        if self._bytes >= WRITEBACK_BYTES:
            self.flush()

    def flush(self) -> None:
        for fd in self._fds:
            _sync_fd(fd)
            discard_cached_pages(fd)
        self._fds.clear()
        self._bytes = 0
