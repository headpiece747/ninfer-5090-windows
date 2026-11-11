#include "product/media_acquire/acquire.h"

#if defined(NINFER_HAVE_LIBCURL)
#include <curl/curl.h>
#endif

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#if !defined(NINFER_HAVE_LIBCURL)
#include <windows.h>
#include <winhttp.h>
#endif
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstddef>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::product::media_acquire {
namespace {

using Clock = std::chrono::steady_clock;

void check_control(const Policy& policy) {
    if (policy.is_cancelled && policy.is_cancelled()) {
        throw Error(ErrorKind::Cancelled, "media acquisition was cancelled");
    }
    if (policy.deadline != Clock::time_point{} && Clock::now() >= policy.deadline) {
        throw Error(ErrorKind::DeadlineExceeded, "media acquisition exceeded request deadline");
    }
}

#if defined(NINFER_HAVE_LIBCURL) || defined(_WIN32)
long bounded_timeout_ms(const Policy& policy, int configured) {
    constexpr std::int64_t kLongMax = std::numeric_limits<long>::max();
    if (policy.deadline == Clock::time_point{}) {
        return configured > 0 ? static_cast<long>(std::min<std::int64_t>(configured, kLongMax))
                              : 0L;
    }
    check_control(policy);
    const auto remaining = std::max<std::int64_t>(
        1,
        std::chrono::duration_cast<std::chrono::milliseconds>(policy.deadline - Clock::now())
            .count());
    const std::int64_t capped =
        configured > 0 ? std::min<std::int64_t>(configured, remaining) : remaining;
    return static_cast<long>(std::min<std::int64_t>(capped, kLongMax));
}

std::chrono::milliseconds next_control_wait_slice(const Policy& policy,
                                                  Clock::time_point op_deadline,
                                                  const char* timeout_message) {
    check_control(policy);
    const auto now = Clock::now();
    if (now >= op_deadline) { throw Error(ErrorKind::RemoteTimeout, timeout_message); }
    auto slice = std::chrono::milliseconds(25);
    if (policy.deadline != Clock::time_point{}) {
        slice = std::min(slice, std::max(std::chrono::milliseconds(1),
                                         std::chrono::duration_cast<std::chrono::milliseconds>(
                                             policy.deadline - now)));
    }
    if (op_deadline != Clock::time_point::max()) {
        slice = std::min(
            slice, std::max(std::chrono::milliseconds(1),
                            std::chrono::duration_cast<std::chrono::milliseconds>(op_deadline - now)));
    }
    return slice;
}
#endif

#ifdef _WIN32
std::wstring utf8_to_wide(std::string_view text) {
    if (text.empty()) { return {}; }
    if (text.find('\0') != std::string_view::npos ||
        text.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("invalid media URL");
    }
    const int needed = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
                                             static_cast<int>(text.size()), nullptr, 0);
    if (needed <= 0) { throw std::invalid_argument("invalid media URL"); }
    std::wstring out(static_cast<std::size_t>(needed), L'\0');
    const int converted = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
                                                static_cast<int>(text.size()), out.data(), needed);
    if (converted != needed) { throw std::invalid_argument("invalid media URL"); }
    return out;
}

std::string wide_to_utf8(std::wstring_view text) {
    if (text.empty()) { return {}; }
    const int needed =
        ::WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0,
                              nullptr, nullptr);
    if (needed <= 0) { throw std::runtime_error("failed to convert wide URL text"); }
    std::string out(static_cast<std::size_t>(needed), '\0');
    const int converted =
        ::WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), out.data(),
                              needed, nullptr, nullptr);
    if (converted != needed) { throw std::runtime_error("failed to convert wide URL text"); }
    return out;
}
#endif

std::vector<std::uint8_t> decode_base64(std::string_view text) {
    static constexpr std::array<std::int8_t, 256> table = [] {
        std::array<std::int8_t, 256> out{};
        out.fill(-1);
        for (int i = 0; i < 26; ++i) {
            out[static_cast<std::size_t>('A' + i)] = static_cast<std::int8_t>(i);
            out[static_cast<std::size_t>('a' + i)] = static_cast<std::int8_t>(26 + i);
        }
        for (int i = 0; i < 10; ++i) {
            out[static_cast<std::size_t>('0' + i)] = static_cast<std::int8_t>(52 + i);
        }
        out[static_cast<std::size_t>('+')] = 62;
        out[static_cast<std::size_t>('/')] = 63;
        return out;
    }();

    std::vector<std::uint8_t> out;
    out.reserve(text.size() * 3 / 4);
    std::uint32_t bits = 0;
    int count          = 0;
    bool padded        = false;
    for (const unsigned char c : text) {
        if (c == '=') {
            padded = true;
            continue;
        }
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') { continue; }
        if (padded || table[c] < 0) { throw std::invalid_argument("malformed base64 media data"); }
        bits = (bits << 6U) | static_cast<std::uint32_t>(table[c]);
        count += 6;
        if (count >= 8) {
            count -= 8;
            out.push_back(static_cast<std::uint8_t>((bits >> count) & 0xffU));
        }
    }
    if (count >= 6) { throw std::invalid_argument("malformed base64 media padding"); }
    return out;
}

#if defined(NINFER_HAVE_LIBCURL) || defined(_WIN32)
bool private_ipv4(std::uint32_t address) {
    const std::uint32_t a = ntohl(address);
    return (a >> 24U) == 0 || (a >> 24U) == 10 || (a >> 24U) == 127 || (a >> 16U) == 0xa9fe ||
           (a >> 20U) == 0xac1 || (a >> 16U) == 0xc0a8 || (a >> 22U) == 0x0191 ||
           (a >> 17U) == 0x6309 || (a >> 24U) >= 224;
}

bool private_address(const sockaddr* address) {
    if (address->sa_family == AF_INET) {
        return private_ipv4(reinterpret_cast<const sockaddr_in*>(address)->sin_addr.s_addr);
    }
    if (address->sa_family != AF_INET6) { return true; }
    const in6_addr& a = reinterpret_cast<const sockaddr_in6*>(address)->sin6_addr;
    if (IN6_IS_ADDR_UNSPECIFIED(&a) || IN6_IS_ADDR_LOOPBACK(&a) || IN6_IS_ADDR_LINKLOCAL(&a) ||
        IN6_IS_ADDR_MULTICAST(&a) || (a.s6_addr[0] & 0xfeU) == 0xfcU) {
        return true;
    }
    if (IN6_IS_ADDR_V4MAPPED(&a)) {
        std::uint32_t v4 = 0;
        std::memcpy(&v4, &a.s6_addr[12], sizeof(v4));
        return private_ipv4(v4);
    }
    return false;
}

struct UrlParts {
    std::string scheme;
    std::string host;
    std::string port;
#if !defined(NINFER_HAVE_LIBCURL)
    std::string path_query;
    std::uint16_t port_num = 80;
#endif
};

#if defined(NINFER_HAVE_LIBCURL)
std::string curlu_part(CURLU* url, CURLUPart part, unsigned flags = 0) {
    char* raw            = nullptr;
    const CURLUcode code = curl_url_get(url, part, &raw, flags);
    if (code != CURLUE_OK || raw == nullptr) { return {}; }
    std::string out(raw);
    curl_free(raw);
    return out;
}

UrlParts parse_url(std::string_view value) {
    std::unique_ptr<CURLU, decltype(&curl_url_cleanup)> url(curl_url(), curl_url_cleanup);
    if (!url ||
        curl_url_set(url.get(), CURLUPART_URL, std::string(value).c_str(), 0) != CURLUE_OK) {
        throw std::invalid_argument("invalid media URL");
    }
    UrlParts out;
    out.scheme             = curlu_part(url.get(), CURLUPART_SCHEME);
    out.host               = curlu_part(url.get(), CURLUPART_HOST);
    out.port               = curlu_part(url.get(), CURLUPART_PORT, CURLU_DEFAULT_PORT);
    const std::string user = curlu_part(url.get(), CURLUPART_USER);
    if ((out.scheme != "http" && out.scheme != "https") || out.host.empty() || out.port.empty() ||
        !user.empty()) {
        throw std::invalid_argument("media URL must be credential-free HTTP(S)");
    }
    return out;
}
#else
std::string lower_ascii(std::string_view text) {
    std::string out(text);
    for (char& ch : out) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return out;
}

UrlParts parse_url(std::string_view value) {
    const auto scheme_sep = value.find("://");
    if (scheme_sep == std::string_view::npos || scheme_sep == 0) {
        throw std::invalid_argument("invalid media URL");
    }
    const std::string scheme = lower_ascii(value.substr(0, scheme_sep));
    if (scheme != "http" && scheme != "https") {
        throw std::invalid_argument("media URL must be credential-free HTTP(S)");
    }

    const std::wstring wide = utf8_to_wide(value);
    const std::size_t cap   = wide.size() + 1;
    std::wstring wscheme(cap, L'\0');
    std::wstring whost(cap, L'\0');
    std::wstring wuser(cap, L'\0');
    std::wstring wpass(cap, L'\0');
    std::wstring wpath(cap, L'\0');
    std::wstring wextra(cap, L'\0');
    URL_COMPONENTSW uc{};
    uc.dwStructSize     = sizeof(uc);
    uc.lpszScheme       = wscheme.data();
    uc.dwSchemeLength   = static_cast<DWORD>(cap);
    uc.lpszHostName     = whost.data();
    uc.dwHostNameLength = static_cast<DWORD>(cap);
    uc.lpszUserName     = wuser.data();
    uc.dwUserNameLength = static_cast<DWORD>(cap);
    uc.lpszPassword     = wpass.data();
    uc.dwPasswordLength = static_cast<DWORD>(cap);
    uc.lpszUrlPath      = wpath.data();
    uc.dwUrlPathLength  = static_cast<DWORD>(cap);
    uc.lpszExtraInfo    = wextra.data();
    uc.dwExtraInfoLength = static_cast<DWORD>(cap);
    if (!::WinHttpCrackUrl(wide.c_str(), static_cast<DWORD>(wide.size()), 0, &uc)) {
        throw std::invalid_argument("invalid media URL");
    }
    if ((uc.nScheme != INTERNET_SCHEME_HTTP && uc.nScheme != INTERNET_SCHEME_HTTPS) ||
        uc.dwHostNameLength == 0 || uc.nPort == 0) {
        throw std::invalid_argument("invalid media URL");
    }
    if (uc.dwUserNameLength != 0 || uc.dwPasswordLength != 0) {
        throw std::invalid_argument("media URL must be credential-free HTTP(S)");
    }
    const auto authority_end = value.find_first_of("/?#", scheme_sep + 3);
    const auto authority =
        value.substr(scheme_sep + 3, authority_end == std::string_view::npos
                                         ? std::string_view::npos
                                         : authority_end - (scheme_sep + 3));
    if (authority.find('@') != std::string_view::npos) {
        throw std::invalid_argument("media URL must be credential-free HTTP(S)");
    }

    std::string host =
        wide_to_utf8(std::wstring_view(whost.data(), static_cast<std::size_t>(uc.dwHostNameLength)));
    if (host.size() >= 2 && host.front() == '[' && host.back() == ']') {
        host = host.substr(1, host.size() - 2);
    }
    std::string path_query =
        wide_to_utf8(std::wstring_view(wpath.data(), static_cast<std::size_t>(uc.dwUrlPathLength)));
    std::string extra =
        wide_to_utf8(std::wstring_view(wextra.data(), static_cast<std::size_t>(uc.dwExtraInfoLength)));
    const auto hash_pos = extra.find('#');
    if (hash_pos != std::string::npos) { extra.resize(hash_pos); }
    path_query += extra;
    if (path_query.empty()) { path_query = "/"; }

    UrlParts out;
    out.scheme     = scheme;
    out.host       = std::move(host);
    out.port       = std::to_string(uc.nPort);
    out.path_query = std::move(path_query);
    out.port_num   = uc.nPort;
    return out;
}
#endif

template <class AddrInfoNode>
std::string select_resolved_address(const AddrInfoNode* raw, bool allow_private) {
    std::string selected;
    for (const AddrInfoNode* it = raw; it != nullptr; it = it->ai_next) {
        if (it->ai_family != AF_INET && it->ai_family != AF_INET6) { continue; }
        if (!allow_private && private_address(it->ai_addr)) { continue; }
        std::array<char, INET6_ADDRSTRLEN> text{};
        const void* bytes =
            it->ai_family == AF_INET
                ? static_cast<const void*>(
                      &reinterpret_cast<const sockaddr_in*>(it->ai_addr)->sin_addr)
                : static_cast<const void*>(
                      &reinterpret_cast<const sockaddr_in6*>(it->ai_addr)->sin6_addr);
        if (inet_ntop(it->ai_family, bytes, text.data(), text.size()) != nullptr) {
            selected = text.data();
            if (it->ai_family == AF_INET) { break; }
        }
    }
    if (selected.empty()) {
        throw std::invalid_argument("media URL resolves only to disallowed network addresses");
    }
    return selected;
}

std::string resolve_public(const UrlParts& url, const Policy& policy) {
    const bool allow_private = policy.allow_private_network;
#ifdef _WIN32
    static std::once_flag winsock_init;
    std::call_once(winsock_init, [] {
        WSADATA data{};
        const int result = ::WSAStartup(MAKEWORD(2, 2), &data);
        if (result != 0) {
            throw Error(ErrorKind::RemoteUnavailable,
                        "failed to initialize Winsock: " + std::to_string(result));
        }
    });

    check_control(policy);
    constexpr std::int64_t kLongMax = std::numeric_limits<long>::max();
    const long connect_ms =
        policy.connect_timeout_ms > 0
            ? static_cast<long>(std::min<std::int64_t>(policy.connect_timeout_ms, kLongMax))
            : 0L;
    const Clock::time_point dns_deadline =
        connect_ms > 0 ? Clock::now() + std::chrono::milliseconds(connect_ms)
                       : Clock::time_point::max();

    struct AsyncDnsGuard {
        HANDLE event         = nullptr;
        HANDLE cancel_handle = nullptr;
        OVERLAPPED ov{};
        PADDRINFOEXW raw     = nullptr;
        bool pending         = false;

        void cancel_and_wait() noexcept {
            if (pending) {
                if (cancel_handle != nullptr) { (void)::GetAddrInfoExCancel(&cancel_handle); }
                if (event != nullptr) { (void)::WaitForSingleObject(event, INFINITE); }
                pending = false;
            }
        }

        ~AsyncDnsGuard() {
            cancel_and_wait();
            if (raw != nullptr) {
                ::FreeAddrInfoExW(raw);
                raw = nullptr;
            }
            if (event != nullptr) {
                ::CloseHandle(event);
                event = nullptr;
            }
        }
    };

    AsyncDnsGuard dns;
    dns.event = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (dns.event == nullptr) {
        throw Error(ErrorKind::RemoteUnavailable, "failed to allocate DNS resolution event");
    }
    dns.ov.hEvent = dns.event;

    const std::wstring wide_host = utf8_to_wide(url.host);
    const std::wstring wide_port = utf8_to_wide(url.port);
    ADDRINFOEXW hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    INT rc = ::GetAddrInfoExW(wide_host.c_str(), wide_port.c_str(), NS_DNS, nullptr, &hints,
                              &dns.raw, nullptr, &dns.ov, nullptr, &dns.cancel_handle);
    if (rc == WSA_IO_PENDING) {
        dns.pending = true;
        for (;;) {
            const auto slice = next_control_wait_slice(
                policy, dns_deadline, "media URL host resolution timed out");
            const DWORD waited =
                ::WaitForSingleObject(dns.event, static_cast<DWORD>(slice.count()));
            if (waited == WAIT_OBJECT_0) {
                dns.pending = false;
                rc          = ::GetAddrInfoExOverlappedResult(&dns.ov);
                break;
            }
            if (waited != WAIT_TIMEOUT) {
                throw Error(ErrorKind::RemoteUnavailable,
                            "failed waiting for media URL host resolution");
            }
        }
    }
    check_control(policy);
    if (rc == WSAETIMEDOUT) {
        throw Error(ErrorKind::RemoteTimeout,
                    "media URL host resolution timed out (" + std::to_string(rc) + ")");
    }
    if (rc != 0) {
        throw Error(ErrorKind::RemoteUnavailable,
                    "failed to resolve media URL host: " + std::to_string(rc));
    }

    return select_resolved_address(dns.raw, allow_private);
#else
    (void)policy;
    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* raw     = nullptr;
    const int rc      = getaddrinfo(url.host.c_str(), url.port.c_str(), &hints, &raw);
    if (rc != 0) {
        throw Error(ErrorKind::RemoteUnavailable,
                    "failed to resolve media URL host: " + std::string(gai_strerror(rc)));
    }
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> addresses(raw, freeaddrinfo);
    return select_resolved_address(raw, allow_private);
#endif
}

#if defined(NINFER_HAVE_LIBCURL)
struct CurlBuffer {
    std::vector<std::uint8_t> bytes;
    std::size_t limit = 0;
};

int curl_progress(void* opaque, curl_off_t, curl_off_t, curl_off_t, curl_off_t) noexcept {
    const auto& policy = *static_cast<const Policy*>(opaque);
    try {
        return (policy.is_cancelled && policy.is_cancelled()) ||
                       (policy.deadline != Clock::time_point{} && Clock::now() >= policy.deadline)
                   ? 1
                   : 0;
    } catch (...) { return 1; }
}

std::size_t curl_write(char* data, std::size_t size, std::size_t count, void* opaque) {
    auto& out = *static_cast<CurlBuffer*>(opaque);
    if (size != 0 && count > std::numeric_limits<std::size_t>::max() / size) { return 0; }
    const std::size_t amount = size * count;
    if (amount > out.limit - std::min(out.limit, out.bytes.size())) { return 0; }
    out.bytes.insert(out.bytes.end(), reinterpret_cast<std::uint8_t*>(data),
                     reinterpret_cast<std::uint8_t*>(data) + amount);
    return amount;
}

std::vector<std::uint8_t> fetch_url(std::string url, const Policy& policy) {
    if (!policy.allow_remote) { throw std::invalid_argument("remote media URLs are disabled"); }
    static std::once_flag init;
    std::call_once(init, [] {
        if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
            throw std::runtime_error("failed to initialize libcurl");
        }
    });

    for (int redirect = 0; redirect <= policy.max_redirects; ++redirect) {
        check_control(policy);
        const UrlParts parts = parse_url(url);
        const std::string ip = resolve_public(parts, policy);
        check_control(policy);
        std::string resolve = parts.host + ":" + parts.port + ":";
        resolve += ip.find(':') == std::string::npos ? ip : "[" + ip + "]";
        curl_slist* resolve_list = curl_slist_append(nullptr, resolve.c_str());
        if (resolve_list == nullptr) { throw std::bad_alloc(); }
        std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> resolve_guard(
            resolve_list, curl_slist_free_all);
        std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(),
                                                                 curl_easy_cleanup);
        if (!curl) { throw std::runtime_error("failed to create libcurl handle"); }
        CurlBuffer buffer{.bytes = {}, .limit = policy.max_bytes};
        curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl.get(), CURLOPT_PROTOCOLS_STR, "http,https");
        curl_easy_setopt(curl.get(), CURLOPT_REDIR_PROTOCOLS_STR, "http,https");
        curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 0L);
        curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT_MS,
                         bounded_timeout_ms(policy, policy.connect_timeout_ms));
        curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT_MS,
                         bounded_timeout_ms(policy, policy.timeout_ms));
        curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl.get(), CURLOPT_SSL_VERIFYPEER, 1L);
        curl_easy_setopt(curl.get(), CURLOPT_SSL_VERIFYHOST, 2L);
        curl_easy_setopt(curl.get(), CURLOPT_PROXY, "");
        curl_easy_setopt(curl.get(), CURLOPT_RESOLVE, resolve_list);
        curl_easy_setopt(curl.get(), CURLOPT_MAXFILESIZE_LARGE,
                         static_cast<curl_off_t>(policy.max_bytes));
        curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, curl_write);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &buffer);
        curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl.get(), CURLOPT_XFERINFOFUNCTION, curl_progress);
        curl_easy_setopt(curl.get(), CURLOPT_XFERINFODATA, &policy);
        curl_easy_setopt(curl.get(), CURLOPT_USERAGENT, "ninfer/vision");
        const CURLcode code = curl_easy_perform(curl.get());
        if (code != CURLE_OK) {
            check_control(policy);
            if (code == CURLE_OPERATION_TIMEDOUT) {
                throw Error(ErrorKind::RemoteTimeout,
                            "media URL fetch timed out: " + std::string(curl_easy_strerror(code)));
            }
            if (code == CURLE_FILESIZE_EXCEEDED || code == CURLE_WRITE_ERROR) {
                throw Error(ErrorKind::BudgetExceeded, "media URL exceeds byte limit");
            }
            throw Error(ErrorKind::RemoteUnavailable,
                        "failed to fetch media URL: " + std::string(curl_easy_strerror(code)));
        }
        long status = 0;
        curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &status);
        if (status >= 200 && status < 300) {
            check_control(policy);
            return buffer.bytes;
        }
        if (status < 300 || status >= 400 || redirect == policy.max_redirects) {
            throw Error(ErrorKind::RemoteUnavailable,
                        "media URL returned HTTP " + std::to_string(status));
        }
        char* next = nullptr;
        curl_easy_getinfo(curl.get(), CURLINFO_REDIRECT_URL, &next);
        if (next == nullptr || *next == '\0') {
            throw Error(ErrorKind::RemoteUnavailable, "media URL redirect has no location");
        }
        url = next;
    }
    throw Error(ErrorKind::RemoteUnavailable, "too many media URL redirects");
}
#elif defined(_WIN32)
bool literal_address_is_private(std::wstring_view text) {
    if (text.size() >= 2 && text.front() == L'[' && text.back() == L']') {
        text = text.substr(1, text.size() - 2);
    }
    std::wstring clean(text);
    const auto zone = clean.find(L'%');
    if (zone != std::wstring::npos) { clean.resize(zone); }

    sockaddr_in v4{};
    if (::InetPtonW(AF_INET, clean.c_str(), &v4.sin_addr) == 1) {
        v4.sin_family = AF_INET;
        return private_address(reinterpret_cast<const sockaddr*>(&v4));
    }
    sockaddr_in6 v6{};
    if (::InetPtonW(AF_INET6, clean.c_str(), &v6.sin6_addr) == 1) {
        v6.sin6_family = AF_INET6;
        return private_address(reinterpret_cast<const sockaddr*>(&v6));
    }
    return true;
}

struct WinHttpHandle {
    HINTERNET handle = nullptr;
    WinHttpHandle()  = default;
    explicit WinHttpHandle(HINTERNET h) : handle(h) {}
    WinHttpHandle(const WinHttpHandle&)            = delete;
    WinHttpHandle& operator=(const WinHttpHandle&) = delete;
    ~WinHttpHandle() {
        if (handle != nullptr) { ::WinHttpCloseHandle(handle); }
    }
};

struct WinHttpRequestContext {
    std::mutex mu;
    std::condition_variable cv;
    bool allow_private_network = false;
    bool blocked_private       = false;
    bool close_started         = false;
    bool send_done             = false;
    bool headers_done          = false;
    bool read_done             = false;
    bool handle_closed         = false;
    DWORD async_error          = 0;
    DWORD bytes_read           = 0;
};

void CALLBACK winhttp_callback(HINTERNET hInternet, DWORD_PTR dwContext, DWORD dwInternetStatus,
                               LPVOID lpvStatusInformation, DWORD dwStatusInformationLength) {
    if (dwContext == 0) { return; }
    auto* ctx = reinterpret_cast<WinHttpRequestContext*>(dwContext);
    switch (dwInternetStatus) {
    case WINHTTP_CALLBACK_STATUS_CONNECTING_TO_SERVER: {
        bool should_abort = false;
        const wchar_t* raw_ip = static_cast<const wchar_t*>(lpvStatusInformation);
        const std::wstring_view ip_view =
            raw_ip != nullptr ? std::wstring_view(raw_ip) : std::wstring_view{};
        {
            std::lock_guard lock(ctx->mu);
            if (!ctx->allow_private_network &&
                (ip_view.empty() || literal_address_is_private(ip_view))) {
                ctx->blocked_private = true;
                if (!ctx->close_started) {
                    ctx->close_started = true;
                    should_abort       = true;
                }
            }
        }
        if (should_abort) { ::WinHttpCloseHandle(hInternet); }
        break;
    }
    case WINHTTP_CALLBACK_STATUS_SENDREQUEST_COMPLETE: {
        std::lock_guard lock(ctx->mu);
        ctx->send_done = true;
        ctx->cv.notify_all();
        break;
    }
    case WINHTTP_CALLBACK_STATUS_HEADERS_AVAILABLE: {
        std::lock_guard lock(ctx->mu);
        ctx->headers_done = true;
        ctx->cv.notify_all();
        break;
    }
    case WINHTTP_CALLBACK_STATUS_READ_COMPLETE: {
        std::lock_guard lock(ctx->mu);
        ctx->bytes_read = dwStatusInformationLength;
        ctx->read_done  = true;
        ctx->cv.notify_all();
        break;
    }
    case WINHTTP_CALLBACK_STATUS_REQUEST_ERROR: {
        const auto* info = static_cast<const WINHTTP_ASYNC_RESULT*>(lpvStatusInformation);
        std::lock_guard lock(ctx->mu);
        ctx->async_error  = info != nullptr ? info->dwError : ERROR_GEN_FAILURE;
        ctx->send_done    = true;
        ctx->headers_done = true;
        ctx->read_done    = true;
        ctx->cv.notify_all();
        break;
    }
    case WINHTTP_CALLBACK_STATUS_HANDLE_CLOSING: {
        std::lock_guard lock(ctx->mu);
        ctx->handle_closed = true;
        ctx->send_done     = true;
        ctx->headers_done  = true;
        ctx->read_done     = true;
        ctx->cv.notify_all();
        break;
    }
    default:
        break;
    }
}

struct AsyncRequestGuard {
    HINTERNET handle             = nullptr;
    WinHttpRequestContext* ctx   = nullptr;
    bool callback_installed      = false;

    void close_and_wait() noexcept {
        if (handle == nullptr) { return; }
        const HINTERNET h = handle;
        handle            = nullptr;
        bool need_close   = true;
        if (ctx != nullptr) {
            std::lock_guard lock(ctx->mu);
            if (ctx->close_started) {
                need_close = false;
            } else {
                ctx->close_started = true;
            }
        }
        if (need_close) { ::WinHttpCloseHandle(h); }
        if (callback_installed && ctx != nullptr) {
            std::unique_lock lock(ctx->mu);
            ctx->cv.wait(lock, [&] { return ctx->handle_closed; });
        }
    }

    ~AsyncRequestGuard() { close_and_wait(); }
};

std::string resolve_redirect_url(const UrlParts& base, std::string_view location) {
    while (!location.empty() && (location.front() == ' ' || location.front() == '\t')) {
        location.remove_prefix(1);
    }
    while (!location.empty() && (location.back() == ' ' || location.back() == '\t' ||
                                 location.back() == '\r' || location.back() == '\n')) {
        location.remove_suffix(1);
    }
    if (location.empty()) {
        throw Error(ErrorKind::RemoteUnavailable, "media URL redirect has no location");
    }
    const auto colon = location.find(':');
    if (colon != std::string_view::npos && colon > 0) {
        const auto slash = location.find_first_of("/?#");
        if (slash == std::string_view::npos || colon < slash) { return std::string(location); }
    }
    if (location.starts_with("//")) { return base.scheme + ":" + std::string(location); }

    const bool default_port =
        (base.scheme == "http" && base.port_num == 80) ||
        (base.scheme == "https" && base.port_num == 443);
    const std::string host_part =
        base.host.find(':') != std::string::npos ? "[" + base.host + "]" : base.host;
    const std::string authority =
        base.scheme + "://" + host_part + (default_port ? std::string{} : ":" + base.port);
    if (location.starts_with('/')) { return authority + std::string(location); }

    const auto q_pos   = base.path_query.find('?');
    const auto base_p  = std::string_view(base.path_query).substr(0, q_pos);
    if (location.starts_with('?')) { return authority + std::string(base_p) + std::string(location); }
    const auto dir_end = base_p.rfind('/');
    const std::string dir =
        dir_end == std::string_view::npos ? "/" : std::string(base_p.substr(0, dir_end + 1));
    return authority + dir + std::string(location);
}

std::string query_header_string(HINTERNET request, DWORD info_level) {
    DWORD bytes = 0;
    if (::WinHttpQueryHeaders(request, info_level, WINHTTP_HEADER_NAME_BY_INDEX, nullptr, &bytes,
                              WINHTTP_NO_HEADER_INDEX) ||
        ::GetLastError() != ERROR_INSUFFICIENT_BUFFER || bytes == 0) {
        return {};
    }
    std::wstring wide(bytes / sizeof(wchar_t), L'\0');
    if (!::WinHttpQueryHeaders(request, info_level, WINHTTP_HEADER_NAME_BY_INDEX, wide.data(),
                               &bytes, WINHTTP_NO_HEADER_INDEX)) {
        return {};
    }
    std::size_t chars = bytes / sizeof(wchar_t);
    while (chars > 0 && wide[chars - 1] == L'\0') { --chars; }
    return wide_to_utf8(std::wstring_view(wide.data(), chars));
}

bool is_redirect_status(DWORD status) noexcept {
    return status == 301 || status == 302 || status == 303 || status == 307 || status == 308;
}

[[noreturn]] void throw_winhttp_failure(DWORD err, const Policy& policy, const char* prefix) {
    check_control(policy);
    if (err == ERROR_WINHTTP_TIMEOUT) {
        throw Error(ErrorKind::RemoteTimeout,
                    "media URL fetch timed out (WinHTTP " + std::to_string(err) + ")");
    }
    throw Error(ErrorKind::RemoteUnavailable,
                std::string(prefix) + " (WinHTTP " + std::to_string(err) + ")");
}

template <class Pred>
DWORD wait_async_step(AsyncRequestGuard& guard, WinHttpRequestContext& ctx, const Policy& policy,
                      Clock::time_point op_deadline, const char* stage, Pred&& done) {
    std::unique_lock lock(ctx.mu);
    for (;;) {
        if (done()) { break; }
        lock.unlock();
        std::chrono::milliseconds wait_slice(25);
        try {
            const std::string timeout_msg = std::string("media URL fetch timed out during ") + stage;
            wait_slice                    = next_control_wait_slice(policy, op_deadline, timeout_msg.c_str());
        } catch (...) {
            guard.close_and_wait();
            throw;
        }
        lock.lock();
        if (done()) { break; }
        ctx.cv.wait_for(lock, wait_slice, [&] { return done(); });
    }
    const bool blocked     = ctx.blocked_private;
    const DWORD err_code   = ctx.async_error;
    const DWORD bytes_read = ctx.bytes_read;
    lock.unlock();
    if (blocked) {
        guard.close_and_wait();
        throw std::invalid_argument("media URL resolves only to disallowed network addresses");
    }
    if (err_code != 0) {
        guard.close_and_wait();
        throw_winhttp_failure(err_code, policy, "failed to fetch media URL");
    }
    return bytes_read;
}

std::vector<std::uint8_t> fetch_url(std::string url, const Policy& policy) {
    if (!policy.allow_remote) { throw std::invalid_argument("remote media URLs are disabled"); }

    for (int redirect = 0; redirect <= policy.max_redirects; ++redirect) {
        check_control(policy);
        const UrlParts parts         = parse_url(url);
        const std::string resolved   = resolve_public(parts, policy);
        check_control(policy);

        // Keep op_deadline strictly governed by policy.timeout_ms (so timeout -> RemoteTimeout)
        // while wait_async_step enforces policy.deadline first (so expired deadline -> DeadlineExceeded).
        // Give WinHTTP a small margin over policy.deadline when deadline < timeout_ms so WinHTTP
        // does not report ERROR_WINHTTP_TIMEOUT a millisecond before policy.deadline elapses.
        constexpr std::int64_t kLongMax = std::numeric_limits<long>::max();
        const long connect_ms =
            policy.connect_timeout_ms > 0
                ? static_cast<long>(std::min<std::int64_t>(policy.connect_timeout_ms, kLongMax))
                : 0L;
        const long req_timeout_ms =
            policy.timeout_ms > 0
                ? static_cast<long>(std::min<std::int64_t>(policy.timeout_ms, kLongMax))
                : 0L;
        const Clock::time_point op_deadline =
            req_timeout_ms > 0 ? Clock::now() + std::chrono::milliseconds(req_timeout_ms)
                               : Clock::time_point::max();
        int winhttp_op_ms = static_cast<int>(req_timeout_ms);
        if (policy.deadline != Clock::time_point{}) {
            const auto remaining = std::max<std::int64_t>(
                1,
                std::chrono::duration_cast<std::chrono::milliseconds>(policy.deadline -
                                                                      Clock::now())
                    .count());
            const auto buffered =
                std::min<std::int64_t>(kLongMax, remaining + 50);
            winhttp_op_ms = req_timeout_ms > 0
                                ? static_cast<int>(std::min<std::int64_t>(req_timeout_ms, buffered))
                                : static_cast<int>(buffered);
        }

        WinHttpHandle session(::WinHttpOpen(L"ninfer/vision", WINHTTP_ACCESS_TYPE_NO_PROXY,
                                            WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS,
                                            WINHTTP_FLAG_ASYNC));
        if (!session.handle) {
            throw Error(ErrorKind::RemoteUnavailable, "failed to open WinHTTP session");
        }
        (void)::WinHttpSetTimeouts(session.handle, connect_ms, connect_ms, winhttp_op_ms,
                                   winhttp_op_ms);

        // For plaintext HTTP, connect directly to the validated IP (like CURLOPT_RESOLVE) with an
        // explicit Host header so multi-record DNS cannot fail on an unchosen private address.
        // For HTTPS, keep the hostname so Schannel performs standard SAN/SNI verification while
        // CONNECTING_TO_SERVER rejects private peers before TCP connect().
        const std::string& connect_raw = parts.scheme == "http" ? resolved : parts.host;
        const std::wstring connect_host = utf8_to_wide(
            connect_raw.find(':') != std::string::npos ? "[" + connect_raw + "]" : connect_raw);
        WinHttpHandle connection(
            ::WinHttpConnect(session.handle, connect_host.c_str(), parts.port_num, 0));
        if (!connection.handle) {
            throw Error(ErrorKind::RemoteUnavailable, "failed to open WinHTTP connection");
        }

        const std::wstring wide_target = utf8_to_wide(parts.path_query);
        const DWORD flags              = parts.scheme == "https" ? WINHTTP_FLAG_SECURE : 0;
        WinHttpRequestContext ctx;
        ctx.allow_private_network = policy.allow_private_network;
        AsyncRequestGuard req;
        req.ctx    = &ctx;
        req.handle = ::WinHttpOpenRequest(connection.handle, L"GET", wide_target.c_str(), nullptr,
                                          WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
        if (!req.handle) {
            throw Error(ErrorKind::RemoteUnavailable, "failed to create WinHTTP request");
        }
        if (parts.scheme == "http" && resolved != parts.host) {
            const std::string host_part =
                parts.host.find(':') != std::string::npos ? "[" + parts.host + "]" : parts.host;
            const std::string host_value =
                parts.port_num == 80 ? host_part : host_part + ":" + parts.port;
            const std::wstring wide_host = utf8_to_wide("Host: " + host_value + "\r\n");
            if (!::WinHttpAddRequestHeaders(
                    req.handle, wide_host.c_str(), static_cast<DWORD>(wide_host.size()),
                    WINHTTP_ADDREQ_FLAG_ADD | WINHTTP_ADDREQ_FLAG_REPLACE)) {
                throw Error(ErrorKind::RemoteUnavailable, "failed to set WinHTTP Host header");
            }
        }
        DWORD disable_redirects = WINHTTP_DISABLE_REDIRECTS;
        if (!::WinHttpSetOption(req.handle, WINHTTP_OPTION_DISABLE_FEATURE, &disable_redirects,
                                sizeof(disable_redirects))) {
            throw Error(ErrorKind::RemoteUnavailable, "failed to disable automatic redirects");
        }
        DWORD_PTR ctx_ptr = reinterpret_cast<DWORD_PTR>(&ctx);
        if (!::WinHttpSetOption(req.handle, WINHTTP_OPTION_CONTEXT_VALUE, &ctx_ptr,
                                sizeof(ctx_ptr))) {
            throw Error(ErrorKind::RemoteUnavailable, "failed to bind WinHTTP context");
        }
        constexpr DWORD kMask = WINHTTP_CALLBACK_FLAG_CONNECT_TO_SERVER |
                                WINHTTP_CALLBACK_FLAG_HANDLES |
                                WINHTTP_CALLBACK_FLAG_SENDREQUEST_COMPLETE |
                                WINHTTP_CALLBACK_FLAG_HEADERS_AVAILABLE |
                                WINHTTP_CALLBACK_FLAG_READ_COMPLETE |
                                WINHTTP_CALLBACK_FLAG_REQUEST_ERROR;
        if (::WinHttpSetStatusCallback(req.handle, winhttp_callback, kMask, 0) ==
            WINHTTP_INVALID_STATUS_CALLBACK) {
            throw Error(ErrorKind::RemoteUnavailable, "failed to set WinHTTP callback");
        }
        req.callback_installed = true;

        if (!::WinHttpSendRequest(req.handle, WINHTTP_NO_ADDITIONAL_HEADERS, 0, nullptr, 0, 0,
                                  ctx_ptr)) {
            const DWORD err = ::GetLastError();
            req.close_and_wait();
            if (ctx.blocked_private) {
                throw std::invalid_argument(
                    "media URL resolves only to disallowed network addresses");
            }
            throw_winhttp_failure(err, policy, "failed to send media URL request");
        }
        (void)wait_async_step(req, ctx, policy, op_deadline, "send",
                              [&] { return ctx.send_done || ctx.handle_closed; });

        if (!::WinHttpReceiveResponse(req.handle, nullptr)) {
            const DWORD err = ::GetLastError();
            req.close_and_wait();
            if (ctx.blocked_private) {
                throw std::invalid_argument(
                    "media URL resolves only to disallowed network addresses");
            }
            throw_winhttp_failure(err, policy, "failed to receive media URL response");
        }
        (void)wait_async_step(req, ctx, policy, op_deadline, "headers",
                              [&] { return ctx.headers_done || ctx.handle_closed; });

        DWORD status      = 0;
        DWORD status_size = sizeof(status);
        if (!::WinHttpQueryHeaders(req.handle, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                                   WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_size,
                                   WINHTTP_NO_HEADER_INDEX)) {
            throw Error(ErrorKind::RemoteUnavailable, "failed to inspect media URL HTTP status");
        }

        if (status >= 200 && status < 300) {
            const std::string content_length =
                query_header_string(req.handle, WINHTTP_QUERY_CONTENT_LENGTH);
            if (!content_length.empty()) {
                std::uint64_t declared = 0;
                bool valid_len         = true;
                for (unsigned char ch : content_length) {
                    if (!std::isdigit(ch)) {
                        valid_len = false;
                        break;
                    }
                    const auto d = static_cast<std::uint64_t>(ch - '0');
                    if (declared > (std::numeric_limits<std::uint64_t>::max() - d) / 10) {
                        declared = std::numeric_limits<std::uint64_t>::max();
                        break;
                    }
                    declared = declared * 10 + d;
                }
                if (valid_len && declared > policy.max_bytes) {
                    throw Error(ErrorKind::BudgetExceeded, "media URL exceeds byte limit");
                }
            }

            std::vector<std::uint8_t> bytes;
            std::array<std::uint8_t, 64 * 1024> scratch{};
            for (;;) {
                check_control(policy);
                {
                    std::lock_guard lock(ctx.mu);
                    ctx.read_done  = false;
                    ctx.bytes_read = 0;
                }
                if (!::WinHttpReadData(req.handle, scratch.data(),
                                       static_cast<DWORD>(scratch.size()), nullptr)) {
                    const DWORD err = ::GetLastError();
                    req.close_and_wait();
                    throw_winhttp_failure(err, policy, "failed to read media URL body");
                }
                const DWORD bytes_read =
                    wait_async_step(req, ctx, policy, op_deadline, "read",
                                    [&] { return ctx.read_done || ctx.handle_closed; });
                if (bytes_read == 0) { break; }
                if (bytes_read > policy.max_bytes - bytes.size()) {
                    throw Error(ErrorKind::BudgetExceeded, "media URL exceeds byte limit");
                }
                bytes.insert(bytes.end(), scratch.data(), scratch.data() + bytes_read);
            }
            req.close_and_wait();
            check_control(policy);
            return bytes;
        }

        if (!is_redirect_status(status) || redirect == policy.max_redirects) {
            throw Error(ErrorKind::RemoteUnavailable,
                        "media URL returned HTTP " + std::to_string(status));
        }
        const std::string location = query_header_string(req.handle, WINHTTP_QUERY_LOCATION);
        req.close_and_wait();
        url = resolve_redirect_url(parts, location);
    }
    throw Error(ErrorKind::RemoteUnavailable, "too many media URL redirects");
}
#endif
#endif

std::vector<std::uint8_t> read_path(const Source& source, const Policy& policy) {
    check_control(policy);
    std::error_code ec;
    std::filesystem::path path = std::filesystem::weakly_canonical(source.value, ec);
    if (ec || !std::filesystem::is_regular_file(path, ec)) {
        throw std::invalid_argument("media path is not a regular file: " + source.value);
    }
    if (!policy.media_root.empty()) {
        const std::filesystem::path root = std::filesystem::weakly_canonical(policy.media_root, ec);
        const auto relative              = std::filesystem::relative(path, root, ec);
        if (ec || relative.empty() || relative.generic_string().starts_with("..")) {
            throw std::invalid_argument("media path is outside configured media root");
        }
    }
    const std::uintmax_t size = std::filesystem::file_size(path, ec);
    check_control(policy);
    if (ec) { throw std::invalid_argument("failed to inspect media file: " + source.value); }
    if (size > policy.max_bytes) {
        throw Error(ErrorKind::BudgetExceeded, "media file exceeds byte limit");
    }

    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::invalid_argument("failed to open media path: " + path.string()); }
    stream.seekg(0, std::ios::end);
    const std::streamoff end = stream.tellg();
    if (end < 0 || static_cast<std::uintmax_t>(end) > policy.max_bytes) {
        throw Error(ErrorKind::BudgetExceeded, "media file exceeds byte limit");
    }
    stream.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    if (!bytes.empty()) {
        stream.read(reinterpret_cast<char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
        if (!stream) { throw std::invalid_argument("failed to read media path: " + path.string()); }
    }
    check_control(policy);
    return bytes;
}

} // namespace

std::vector<std::uint8_t> acquire_bytes(const Source& source, const Policy& policy) {
    if (policy.max_bytes == 0) { throw std::invalid_argument("media byte limit must be positive"); }
    check_control(policy);
    if (source.kind == SourceKind::Bytes) {
        if (source.bytes.empty()) { throw std::invalid_argument("media source is empty"); }
        if (source.bytes.size() > policy.max_bytes) {
            throw Error(ErrorKind::BudgetExceeded, "media bytes exceed byte limit");
        }
        std::vector<std::uint8_t> bytes = source.bytes;
        check_control(policy);
        return bytes;
    }
    if (source.value.empty()) { throw std::invalid_argument("media source is empty"); }

    if (source.kind == SourceKind::Url) {
#if defined(NINFER_HAVE_LIBCURL) || defined(_WIN32)
        std::vector<std::uint8_t> bytes = fetch_url(source.value, policy);
        if (bytes.empty()) { throw std::invalid_argument("media source contains no data"); }
        return bytes;
#else
        throw std::invalid_argument("ninfer compiled without curl. Remote URLs are unsupported.");
#endif
    }
    if (source.kind == SourceKind::Data) {
        const std::size_t comma = source.value.find(',');
        if (!source.value.starts_with("data:") || comma == std::string::npos ||
            source.value.substr(0, comma).find(";base64") == std::string::npos) {
            throw std::invalid_argument("media data source must be a base64 data URI");
        }
        const std::size_t encoded = source.value.size() - comma - 1;
        if (encoded > (policy.max_bytes / 3 + 1) * 4) {
            throw Error(ErrorKind::BudgetExceeded, "media data exceeds byte limit");
        }
        std::vector<std::uint8_t> bytes =
            decode_base64(std::string_view(source.value).substr(comma + 1));
        check_control(policy);
        if (bytes.size() > policy.max_bytes) {
            throw Error(ErrorKind::BudgetExceeded, "media data exceeds byte limit");
        }
        if (bytes.empty()) { throw std::invalid_argument("media source contains no data"); }
        return bytes;
    }
    return read_path(source, policy);
}

} // namespace ninfer::product::media_acquire
