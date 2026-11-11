#include "core/io/file.h"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <share.h>
#include <sys/stat.h>
#include <corecrt_io.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::io {
namespace {

#ifdef _WIN32

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation,
                       int error = 0) {
    throw std::system_error(error ? error : (int)::GetLastError(), std::generic_category(),
                            path.string() + ": " + operation);
}

long long position(std::uint64_t value) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<long long>::max()))
        throw std::invalid_argument("file offset exceeds file range");
    return static_cast<long long>(value);
}

#else

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation,
                       int error = errno) {
    throw std::system_error(error, std::generic_category(), path.string() + ": " + operation);
}

off_t position(std::uint64_t value) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()))
        throw std::invalid_argument("file offset exceeds off_t");
    return static_cast<off_t>(value);
}

#endif

} // namespace

#ifdef _WIN32

ReadOnlyFile::ReadOnlyFile(const std::filesystem::path& path, ReadMode mode) : path_(path) {
    if (mode != ReadMode::Buffered && mode != ReadMode::Direct)
        throw std::invalid_argument("invalid file read mode");
    int fd = -1;
    if (mode == ReadMode::Direct) {
        HANDLE handle = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                      nullptr, OPEN_EXISTING,
                                      FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (handle == INVALID_HANDLE_VALUE) fail(path, "open");
        // Wrap the Win32 handle in the CRT descriptor used by the rest of this class.
        // The underlying handle stays unbuffered: _read/_lseeki64 keep the no-cache
        // behaviour when called through the CRT descriptor.
        fd = _open_osfhandle(reinterpret_cast<intptr_t>(handle), _O_RDONLY | _O_BINARY);
        if (fd < 0) {
            ::CloseHandle(handle);
            fail(path, "open");
        }
    } else {
        if (_wsopen_s(&fd, path.c_str(), _O_RDONLY | _O_BINARY | _O_SEQUENTIAL, _SH_DENYNO, 0) != 0)
            fail(path, "open");
        if (fd < 0) fail(path, "open");
    }
    view_.descriptor = fd;
    view_.mode = mode;
    try {
        struct _stat64 status {};
        if (_fstat64(fd, &status) != 0) fail(path, "fstat");
        // st_mode on Windows via MSVC includes _S_IFREG.
        if (status.st_size < 0 || !(status.st_mode & _S_IFREG))
            fail(path, "expected regular file", EINVAL);
        view_.bytes = static_cast<std::uint64_t>(status.st_size);
        if (mode == ReadMode::Direct) {
            view_.memory_alignment = 4096;
            view_.offset_alignment = 4096;
        } else {
            view_.memory_alignment = 1;
            view_.offset_alignment = 1;
        }
    } catch (...) {
        _close(view_.descriptor);
        view_.descriptor = -1;
        throw;
    }
}

ReadOnlyFile::~ReadOnlyFile() {
    if (view_.descriptor >= 0) _close(view_.descriptor);
}

ReadOnlyFile::ReadOnlyFile(ReadOnlyFile&& other) noexcept
    : path_(std::move(other.path_)), view_(std::exchange(other.view_, {})) {}

ReadOnlyFile& ReadOnlyFile::operator=(ReadOnlyFile&& other) noexcept {
    if (this != &other) {
        if (view_.descriptor >= 0) _close(view_.descriptor);
        path_ = std::move(other.path_);
        view_ = std::exchange(other.view_, {});
    }
    return *this;
}

void ReadOnlyFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (view_.mode != ReadMode::Buffered || view_.descriptor < 0)
        throw std::invalid_argument("read_exact requires a buffered file");
    if (offset > view_.bytes || destination.size() > view_.bytes - offset)
        fail(path_, "read exceeds file length", ENODATA);
    const HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(view_.descriptor));
    if (handle == INVALID_HANDLE_VALUE) fail(path_, "read", EBADF);
    while (!destination.empty()) {
        (void)position(offset);
        const DWORD count = static_cast<DWORD>(
            std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024));
        OVERLAPPED ov{};
        ov.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
        ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
        DWORD n       = 0;
        if (!::ReadFile(handle, destination.data(), count, &n, &ov)) {
            const DWORD err = ::GetLastError();
            if (err == ERROR_HANDLE_EOF) fail(path_, "unexpected EOF", ENODATA);
            fail(path_, "read", static_cast<int>(err));
        }
        if (n == 0) fail(path_, "unexpected EOF", ENODATA);
        offset += static_cast<std::uint64_t>(n);
        destination = destination.subspan(static_cast<std::size_t>(n));
    }
}

void validate_direct_read(FileView file, std::uint64_t offset, std::span<std::byte> destination) {
    if (file.descriptor < 0 || file.mode != ReadMode::Direct ||
        !std::has_single_bit(file.memory_alignment) ||
        !std::has_single_bit(file.offset_alignment) || offset % file.offset_alignment ||
        destination.size() % file.offset_alignment ||
        (!destination.empty() &&
         (!destination.data() ||
          reinterpret_cast<std::uintptr_t>(destination.data()) % file.memory_alignment)) ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("invalid direct read range or alignment");
    (void)position(offset);
}

std::size_t ReadOnlyFile::read_direct(std::uint64_t offset,
                                      std::span<std::byte> destination) const {
    validate_direct_read(view_, offset, destination);
    if (destination.empty()) return 0;
    const HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(view_.descriptor));
    if (handle == INVALID_HANDLE_VALUE) fail(path_, "direct read", EBADF);
    OVERLAPPED ov{};
    ov.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
    ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
    DWORD n       = 0;
    if (!::ReadFile(handle, destination.data(), static_cast<DWORD>(destination.size()), &n, &ov)) {
        const DWORD err = ::GetLastError();
        if (err == ERROR_HANDLE_EOF) return 0;
        fail(path_, "direct read", static_cast<int>(err));
    }
    return static_cast<std::size_t>(n);
}

DirectReadWindow direct_read_window(FileView file, std::uint64_t offset, std::size_t bytes) {
    if (file.descriptor < 0 || file.mode != ReadMode::Direct || !bytes ||
        !std::has_single_bit(file.offset_alignment) || offset > file.bytes ||
        bytes > file.bytes - offset)
        throw std::invalid_argument("invalid direct read window");
    const auto alignment = file.offset_alignment;
    const auto prefix = offset % alignment;
    if (bytes > std::numeric_limits<std::size_t>::max() - prefix - (alignment - 1))
        throw std::overflow_error("direct read window size overflow");
    const auto required = bytes + prefix;
    const auto count = (required + alignment - 1) / alignment * alignment;
    (void)position(offset - prefix);
    return {offset - prefix, count, prefix, required};
}

#else

// Original upstream implementation follows

ReadOnlyFile::ReadOnlyFile(const std::filesystem::path& path, ReadMode mode) : path_(path) {
    if (mode != ReadMode::Buffered && mode != ReadMode::Direct)
        throw std::invalid_argument("invalid file read mode");
    view_.descriptor =
        ::open(path.c_str(), O_RDONLY | O_CLOEXEC | (mode == ReadMode::Direct ? O_DIRECT : 0));
    if (view_.descriptor < 0) fail(path, "open");
    view_.mode = mode;
    try {
        struct stat status {};

        if (::fstat(view_.descriptor, &status)) fail(path, "fstat");
        if (status.st_size < 0 || !S_ISREG(status.st_mode))
            fail(path, "expected regular file", EINVAL);
        view_.bytes = static_cast<std::uint64_t>(status.st_size);
        if (mode == ReadMode::Direct) {
            struct statx alignment {};

            const bool reported = ::statx(view_.descriptor, "", AT_EMPTY_PATH | AT_STATX_DONT_SYNC,
                                          STATX_DIOALIGN, &alignment) == 0 &&
                                  (alignment.stx_mask & STATX_DIOALIGN);
            view_.memory_alignment = reported ? alignment.stx_dio_mem_align : 4096;
            view_.offset_alignment = reported ? alignment.stx_dio_offset_align : 4096;
            if (!std::has_single_bit(view_.memory_alignment) ||
                !std::has_single_bit(view_.offset_alignment))
                fail(path, "unsupported direct I/O alignment", EOPNOTSUPP);
            if (!reported && view_.bytes) {
                std::unique_ptr<void, decltype(&std::free)> probe(std::aligned_alloc(4096, 4096),
                                                                  &std::free);
                if (!probe) throw std::bad_alloc();
                (void)read_direct(0, {static_cast<std::byte*>(probe.get()), 4096});
            }
        }
    } catch (...) {
        ::close(view_.descriptor);
        view_.descriptor = -1;
        throw;
    }
}

ReadOnlyFile::~ReadOnlyFile() {
    if (view_.descriptor >= 0) ::close(view_.descriptor);
}

ReadOnlyFile::ReadOnlyFile(ReadOnlyFile&& other) noexcept
    : path_(std::move(other.path_)), view_(std::exchange(other.view_, {})) {}

ReadOnlyFile& ReadOnlyFile::operator=(ReadOnlyFile&& other) noexcept {
    if (this != &other) {
        if (view_.descriptor >= 0) ::close(view_.descriptor);
        path_ = std::move(other.path_);
        view_ = std::exchange(other.view_, {});
    }
    return *this;
}

void ReadOnlyFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (view_.mode != ReadMode::Buffered || view_.descriptor < 0)
        throw std::invalid_argument("read_exact requires a buffered file");
    if (offset > view_.bytes || destination.size() > view_.bytes - offset)
        fail(path_, "read exceeds file length", ENODATA);
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto n     = ::pread(view_.descriptor, destination.data(), count, position(offset));
        if (n < 0) {
            if (errno == EINTR) continue;
            fail(path_, "pread");
        }
        if (!n) fail(path_, "unexpected EOF", ENODATA);
        offset += static_cast<std::uint64_t>(n);
        destination = destination.subspan(static_cast<std::size_t>(n));
    }
}

void validate_direct_read(FileView file, std::uint64_t offset, std::span<std::byte> destination) {
    if (file.descriptor < 0 || file.mode != ReadMode::Direct ||
        !std::has_single_bit(file.memory_alignment) ||
        !std::has_single_bit(file.offset_alignment) || offset % file.offset_alignment ||
        destination.size() % file.offset_alignment ||
        (!destination.empty() &&
         (!destination.data() ||
          reinterpret_cast<std::uintptr_t>(destination.data()) % file.memory_alignment)) ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()))
        throw std::invalid_argument("invalid direct read range or alignment");
    (void)position(offset);
}

std::size_t ReadOnlyFile::read_direct(std::uint64_t offset,
                                      std::span<std::byte> destination) const {
    validate_direct_read(view_, offset, destination);
    if (destination.empty()) return 0;
    ssize_t n;
    do {
        n = ::pread(view_.descriptor, destination.data(), destination.size(), position(offset));
    } while (n < 0 && errno == EINTR);
    if (n < 0) fail(path_, "direct pread");
    return static_cast<std::size_t>(n);
}

DirectReadWindow direct_read_window(FileView file, std::uint64_t offset, std::size_t bytes) {
    if (file.descriptor < 0 || file.mode != ReadMode::Direct || !bytes ||
        !std::has_single_bit(file.offset_alignment) || offset > file.bytes ||
        bytes > file.bytes - offset)
        throw std::invalid_argument("invalid direct read window");
    const auto alignment = file.offset_alignment;
    const auto prefix    = offset % alignment;
    if (bytes > std::numeric_limits<std::size_t>::max() - prefix - (alignment - 1))
        throw std::overflow_error("direct read window size overflow");
    const auto required = bytes + prefix;
    const auto count    = (required + alignment - 1) / alignment * alignment;
    (void)position(offset - prefix);
    return {offset - prefix, count, prefix, required};
}

#endif

} // namespace ninfer::io