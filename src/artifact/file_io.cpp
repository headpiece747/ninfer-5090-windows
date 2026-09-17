#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

#ifdef _WIN32

// Windows port (headpiece747 Windows build). std::filesystem + Win32 synchronous
// positional I/O; FILE_FLAG_NO_BUFFERING provides the equivalent of O_DIRECT.
[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    const auto error = ::GetLastError();
    throw ArtifactError(path.string() + ": " + operation + ": Win32 error " +
                        std::to_string(static_cast<unsigned long>(error)));
}

HANDLE open_read(const std::filesystem::path& path, DWORD extra_flags) {
    // POSIX enforces no share mode, so a reader never blocks a writer. Sharing read, write
    // and delete keeps that behaviour: the artifact tests rewrite a file while the reader
    // holds it, and the engine re-validates on the next open.
    constexpr DWORD kShareMode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
    HANDLE handle = ::CreateFileW(path.c_str(), GENERIC_READ, kShareMode, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | extra_flags, nullptr);
    if (handle == INVALID_HANDLE_VALUE) { fail(path, "CreateFileW"); }
    return handle;
}

std::uint64_t regular_file_bytes(HANDLE handle, const std::filesystem::path& path) {
    LARGE_INTEGER size{};
    if (!::GetFileSizeEx(handle, &size)) { fail(path, "GetFileSizeEx"); }
    if (size.QuadPart < 0) { throw ArtifactError(path.string() + ": expected a regular file"); }
    BY_HANDLE_FILE_INFORMATION info{};
    if (::GetFileInformationByHandle(handle, &info) &&
        (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
        throw ArtifactError(path.string() + ": expected a regular file");
    }
    return static_cast<std::uint64_t>(size.QuadPart);
}

// Synchronous handles: supplying OVERLAPPED still selects the offset, and the call
// does not return until the read completes.
bool read_at(HANDLE handle, std::uint64_t offset, std::byte* data, std::size_t count,
             DWORD& transferred) {
    OVERLAPPED overlapped{};
    overlapped.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32U);
    return ::ReadFile(handle, data, static_cast<DWORD>(count), &transferred, &overlapped) != 0;
}

#else

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}

#endif

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
#ifdef _WIN32
    fd_    = open_read(path_, 0);
    bytes_ = regular_file_bytes(fd_, path_);
#else
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { fail(path_, "open"); }

    struct stat status {};

    if (::fstat(fd_, &status) != 0) {
        const auto error = errno;
        ::close(fd_);
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        ::close(fd_);
        fd_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
#endif
}

InputFile::~InputFile() {
#ifdef _WIN32
    if (direct_fd_ != nullptr) { ::CloseHandle(static_cast<HANDLE>(direct_fd_)); }
    if (fd_ != nullptr) { ::CloseHandle(static_cast<HANDLE>(fd_)); }
#else
    if (direct_fd_ >= 0) { ::close(direct_fd_); }
    if (fd_ >= 0) { ::close(fd_); }
#endif
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
#ifdef _WIN32
        DWORD transferred = 0;
        if (!read_at(static_cast<HANDLE>(fd_), offset, destination.data(), count, transferred)) {
            fail(path_, "ReadFile");
        }
        if (!transferred) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        const auto read = static_cast<std::size_t>(transferred);
#else
        const auto read = ::pread(fd_, destination.data(), count, file_offset(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
#endif
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(read);
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment
#ifndef _WIN32
        || destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())
#endif
    ) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
#ifdef _WIN32
    if (direct_fd_ == nullptr) {
        // FILE_FLAG_NO_BUFFERING is the O_DIRECT equivalent; SEQUENTIAL_SCAN matches the
        // artifact reader's forward-only access pattern.
        direct_fd_ = open_read(path_, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN);
    }
    DWORD transferred = 0;
    if (!read_at(static_cast<HANDLE>(direct_fd_), offset, destination.data(), destination.size(),
                 transferred)) {
        fail(path_, "direct ReadFile");
    }
    return static_cast<std::size_t>(transferred);
#else
    if (direct_fd_ < 0) {
        direct_fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (direct_fd_ < 0) { fail(path_, "open direct"); }
    }
    ssize_t read;
    do {
        read = ::pread(direct_fd_, destination.data(), destination.size(), file_offset(offset));
    } while (read < 0 && errno == EINTR);
    if (read < 0) { fail(path_, "direct pread"); }
    return static_cast<std::size_t>(read);
#endif
}

} // namespace ninfer::artifact
