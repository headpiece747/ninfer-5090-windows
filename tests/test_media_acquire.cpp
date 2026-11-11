// Tests for the media acquisition module.
//
// This module is the only deep one in the Windows port's hot spots: one function, 392 lines
// behind it, and until now no test at all. It was linked into the test binary and never
// exercised. The Bytes, Data and Path kinds are deterministic with their dependencies passed in,
// so they need no network and no adapter to test; only the Url kind does.
//
// The policy branches covered here are the ones that are unreachable in production: neither
// caller sets Policy::media_root, so the path-confinement branch never runs, and the two callers
// disagree about allow_private_network with nothing asserting which value each passes.
#include "product/media_acquire/acquire.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ninfer::product::media_acquire;

int failures = 0;

void require(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

template <typename Fn>
void require_throws(Fn&& fn, const std::string& what) {
    try {
        fn();
        std::cerr << "FAIL: expected a throw: " << what << "\n";
        ++failures;
    } catch (const std::exception&) {
        // expected
    }
}

std::filesystem::path temp_dir() {
    const auto base  = std::filesystem::temp_directory_path();
    const auto stamp = static_cast<unsigned long long>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    for (unsigned long long attempt = 0; attempt < 128; ++attempt) {
        auto candidate = base / ("ninfer-acquire-" + std::to_string(stamp + attempt));
        std::error_code error;
        if (std::filesystem::create_directory(candidate, error)) { return candidate; }
    }
    throw std::runtime_error("cannot create a test directory");
}

void write_file(const std::filesystem::path& path, std::string_view content) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(content.data(), static_cast<std::streamsize>(content.size()));
}

// --- Bytes: the simplest kind, no decoding and no filesystem -------------------------

void test_bytes_kind() {
    Source source;
    source.kind  = SourceKind::Bytes;
    source.bytes = {1, 2, 3, 4};

    const auto bytes = acquire_bytes(source);
    require(bytes == std::vector<std::uint8_t>({1, 2, 3, 4}), "Bytes kind returns its bytes");

    // An empty byte vector is rejected, and the message names the source rather than the budget.
    Source empty;
    empty.kind = SourceKind::Bytes;
    require_throws([&] { acquire_bytes(empty); }, "empty Bytes source");

    // The budget is checked before the copy, so an oversized source never allocates.
    Source big;
    big.kind  = SourceKind::Bytes;
    big.bytes = std::vector<std::uint8_t>(64, 0);
    Policy tight;
    tight.max_bytes = 8;
    try {
        acquire_bytes(big, tight);
        std::cerr << "FAIL: oversized Bytes source was accepted\n";
        ++failures;
    } catch (const Error& error) {
        require(error.kind() == ErrorKind::BudgetExceeded, "oversized Bytes reports BudgetExceeded");
    }
}

// --- Data: base64 decoding, which the media_decode test re-implements separately -------

void test_data_kind() {
    Source source;
    source.kind  = SourceKind::Data;
    source.value = "data:image/png;base64,aGVsbG8=";  // "hello"

    const auto bytes = acquire_bytes(source);
    require(std::string(bytes.begin(), bytes.end()) == "hello", "Data kind decodes base64");

    // Whitespace inside the payload is ignored, which is how wrapped data URIs arrive.
    Source wrapped;
    wrapped.kind  = SourceKind::Data;
    wrapped.value = "data:image/png;base64,aGVs\n bG8=\r\n";
    const auto wrapped_bytes = acquire_bytes(wrapped);
    require(std::string(wrapped_bytes.begin(), wrapped_bytes.end()) == "hello",
            "Data kind ignores whitespace in the payload");

    // A malformed payload must raise invalid_argument, not Error: it is a bad request rather
    // than a resource limit, and the serving path maps the two to different HTTP statuses.
    Source malformed;
    malformed.kind  = SourceKind::Data;
    malformed.value = "data:image/png;base64,!!!!";
    require_throws([&] { acquire_bytes(malformed); }, "malformed base64");

    // A payload that is not a base64 data URI at all.
    Source not_data;
    not_data.kind  = SourceKind::Data;
    not_data.value = "https://example.invalid/x.png";
    require_throws([&] { acquire_bytes(not_data); }, "Data kind without a data: prefix");

    // A data URI that is not base64-encoded is refused rather than guessed at.
    Source not_base64;
    not_base64.kind  = SourceKind::Data;
    not_base64.value = "data:image/png,hello";
    require_throws([&] { acquire_bytes(not_base64); }, "Data kind without ;base64");
}

// --- Path: the branch whose media_root confinement never runs in production -----------

void test_path_kind() {
    const auto directory = temp_dir();
    const auto inside    = directory / "inside.bin";
    const auto outside   = directory.parent_path() / (directory.filename().string() + "-outside.bin");
    write_file(inside, "payload");
    write_file(outside, "secret");

    Source source;
    source.kind  = SourceKind::Path;
    source.value = inside.string();
    const auto bytes = acquire_bytes(source);
    require(std::string(bytes.begin(), bytes.end()) == "payload", "Path kind reads a file");

    // Neither caller sets media_root, so this confinement never runs in production. It is the
    // reason the field exists, so it is worth asserting directly.
    Policy confined;
    confined.media_root = directory;
    const auto confined_bytes = acquire_bytes(source, confined);
    require(std::string(confined_bytes.begin(), confined_bytes.end()) == "payload",
            "Path kind accepts a file inside media_root");

    Source escaping;
    escaping.kind  = SourceKind::Path;
    escaping.value = outside.string();
    require_throws([&] { acquire_bytes(escaping, confined); },
                   "Path kind rejects a file outside media_root");

    // A directory is not a regular file, and a missing path is not either.
    Source directory_source;
    directory_source.kind  = SourceKind::Path;
    directory_source.value = directory.string();
    require_throws([&] { acquire_bytes(directory_source); }, "Path kind rejects a directory");

    Source missing;
    missing.kind  = SourceKind::Path;
    missing.value = (directory / "absent.bin").string();
    require_throws([&] { acquire_bytes(missing); }, "Path kind rejects a missing file");

    // The budget is checked against the file size before the read.
    Policy tight;
    tight.max_bytes = 3;
    try {
        acquire_bytes(source, tight);
        std::cerr << "FAIL: oversized Path source was accepted\n";
        ++failures;
    } catch (const Error& error) {
        require(error.kind() == ErrorKind::BudgetExceeded, "oversized Path reports BudgetExceeded");
    }

    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    std::filesystem::remove(outside, ignored);
}

// --- Policy: the control checks that every kind shares -------------------------------

void test_policy_controls() {
    Source source;
    source.kind  = SourceKind::Bytes;
    source.bytes = {1};

    // A zero byte limit is a programming error, not a resource limit.
    Policy zero;
    zero.max_bytes = 0;
    require_throws([&] { acquire_bytes(source, zero); }, "zero byte limit");

    // A deadline already in the past fails before any work.
    Policy expired;
    expired.deadline = std::chrono::steady_clock::now() - std::chrono::seconds(1);
    try {
        acquire_bytes(source, expired);
        std::cerr << "FAIL: an expired deadline was accepted\n";
        ++failures;
    } catch (const Error& error) {
        require(error.kind() == ErrorKind::DeadlineExceeded, "expired deadline reports DeadlineExceeded");
    }

    // Cancellation is reported as Cancelled, distinct from a deadline.
    Policy cancelled;
    cancelled.is_cancelled = [] { return true; };
    try {
        acquire_bytes(source, cancelled);
        std::cerr << "FAIL: a cancelled request was accepted\n";
        ++failures;
    } catch (const Error& error) {
        require(error.kind() == ErrorKind::Cancelled, "cancelled request reports Cancelled");
    }

    // An empty value with a non-Bytes kind is rejected before any dispatch.
    Source no_value;
    no_value.kind = SourceKind::Path;
    require_throws([&] { acquire_bytes(no_value); }, "empty Path value");
}

// --- Url: only the compile-time refusal is reachable without a network ----------------

void test_url_kind() {
    Source source;
    source.kind  = SourceKind::Url;
    source.value = "https://example.invalid/x.png";

    try {
        acquire_bytes(source);
        std::cerr << "FAIL: a Url source was accepted without a reachable host\n";
        ++failures;
    } catch (const std::exception&) {
        // Unreachable / non-remote environments reject unreachable remote URLs.
    }

#ifdef _WIN32
    Policy no_remote;
    no_remote.allow_remote = false;
    require_throws([&] { acquire_bytes(source, no_remote); }, "allow_remote=false rejects URL");

    Source credentialed;
    credentialed.kind  = SourceKind::Url;
    credentialed.value = "http://user:pass@example.com/x.png";
    require_throws([&] { acquire_bytes(credentialed); }, "credentialed URL rejected");

    WSADATA wsa{};
    require(::WSAStartup(MAKEWORD(2, 2), &wsa) == 0, "WSAStartup for loopback server");
    SOCKET listen_sock = ::socket(AF_INET, SOCK_STREAM, 0);
    require(listen_sock != INVALID_SOCKET, "loopback listen socket created");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    require(::bind(listen_sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0,
            "loopback bind succeeded");
    int addr_len = sizeof(addr);
    require(::getsockname(listen_sock, reinterpret_cast<sockaddr*>(&addr), &addr_len) == 0,
            "loopback getsockname succeeded");
    require(::listen(listen_sock, 8) == 0, "loopback listen succeeded");
    const int port = ::ntohs(addr.sin_port);

    std::atomic<bool> stop{false};
    std::atomic<int> accepted{0};
    std::thread server([&] {
        while (!stop.load()) {
            fd_set rfds;
            FD_ZERO(&rfds);
            FD_SET(listen_sock, &rfds);
            timeval tv{0, 50000};
            if (::select(0, &rfds, nullptr, nullptr, &tv) <= 0) { continue; }
            SOCKET client = ::accept(listen_sock, nullptr, nullptr);
            if (client == INVALID_SOCKET) { continue; }
            accepted.fetch_add(1);
            char req[1024]{};
            const int n = ::recv(client, req, sizeof(req) - 1, 0);
            if (n > 0) {
                std::string_view req_view(req, static_cast<std::size_t>(n));
                std::string response;
                if (req_view.starts_with("GET /ok ")) {
                    response = "HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nPNG99";
                } else if (req_view.starts_with("GET /redir ")) {
                    response = "HTTP/1.1 302 Found\r\nLocation: /ok\r\nContent-Length: 0\r\n"
                               "Connection: close\r\n\r\n";
                } else if (req_view.starts_with("GET /big ")) {
                    response = "HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\n" +
                               std::string(100, 'x');
                } else {
                    response = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                }
                ::send(client, response.data(), static_cast<int>(response.size()), 0);
            }
            ::closesocket(client);
        }
    });

    // Public-only policy blocks loopback before accepting or connecting.
    const std::string base = "http://127.0.0.1:" + std::to_string(port);
    require_throws([&] { acquire_bytes(Source{.kind = SourceKind::Url, .value = base + "/ok"}); },
                   "SSRF blocks loopback by default");
    require(accepted.load() == 0, "SSRF loopback block prevented connection attempt");

    // Loopback enabled explicitly fetches bytes, follows redirects, and enforces HTTP & byte limits.
    Policy loopback;
    loopback.allow_private_network = true;
    const auto fetched =
        acquire_bytes(Source{.kind = SourceKind::Url, .value = base + "/ok"}, loopback);
    require(fetched == std::vector<std::uint8_t>{'P', 'N', 'G', '9', '9'},
            "WinHTTP fetches loopback URL bytes");

    const auto redirected =
        acquire_bytes(Source{.kind = SourceKind::Url, .value = base + "/redir"}, loopback);
    require(redirected == fetched, "WinHTTP follows relative redirect");

    Policy no_redirect = loopback;
    no_redirect.max_redirects = 0;
    try {
        acquire_bytes(Source{.kind = SourceKind::Url, .value = base + "/redir"}, no_redirect);
        std::cerr << "FAIL: max_redirects=0 allowed redirect\n";
        ++failures;
    } catch (const Error& e) {
        require(e.kind() == ErrorKind::RemoteUnavailable, "redirect limit reports RemoteUnavailable");
    }

    try {
        acquire_bytes(Source{.kind = SourceKind::Url, .value = base + "/missing"}, loopback);
        std::cerr << "FAIL: 404 accepted\n";
        ++failures;
    } catch (const Error& e) {
        require(e.kind() == ErrorKind::RemoteUnavailable, "HTTP 404 reports RemoteUnavailable");
    }

    Policy tiny = loopback;
    tiny.max_bytes = 10;
    try {
        acquire_bytes(Source{.kind = SourceKind::Url, .value = base + "/big"}, tiny);
        std::cerr << "FAIL: oversized HTTP body accepted\n";
        ++failures;
    } catch (const Error& e) {
        require(e.kind() == ErrorKind::BudgetExceeded, "oversized URL reports BudgetExceeded");
    }

    stop.store(true);
    server.join();
    ::closesocket(listen_sock);
    ::WSACleanup();
#endif
}

} // namespace

int main() {
    test_bytes_kind();
    test_data_kind();
    test_path_kind();
    test_policy_controls();
    test_url_kind();

    if (failures == 0) {
        std::cout << "OK media_acquire\n";
        return 0;
    }
    std::cerr << failures << " failure(s)\n";
    return 1;
}
