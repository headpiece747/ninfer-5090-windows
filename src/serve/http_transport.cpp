#include "serve/http_transport.h"

#include "core/cpu/task_pool.h"
#include "serve/request_validation.h"
#include "text/json_input.h"

#if defined(__linux__)
#    include <netinet/tcp.h>
#    include <sys/socket.h>
#elif defined(_WIN32)
// winsock2.h first: mstcpip.h depends on its types, and including them in the other order breaks
// once anything pulls in windows.h ahead of this header.
#    include <winsock2.h>
#    include <mstcpip.h>
#endif

#include <stdexcept>
#include <limits>
#include <utility>

namespace ninfer::serve {
namespace {

class HttpTaskQueue final : public httplib::TaskQueue {
public:
    HttpTaskQueue(std::uint32_t workers, std::size_t capacity)
        : pool_({.threads = workers, .queue_capacity = capacity}) {}

    bool enqueue(std::function<void()> task) override {
        return pool_.try_enqueue(task) == EnqueueResult::Accepted;
    }

    void shutdown() override { pool_.shutdown(); }

private:
    TaskPool pool_;
};

#if defined(__linux__)
constexpr int kKeepAliveIdleSeconds                = 10;
constexpr int kKeepAliveIntervalSeconds            = 3;
constexpr int kKeepAliveProbeCount                 = 3;
constexpr unsigned int kTcpUserTimeoutMilliseconds = 15000;

template <class T>
void set_socket_option(socket_t socket, int level, int option, const T& value) noexcept {
    (void)::setsockopt(socket, level, option, &value, sizeof(value));
}
#elif defined(_WIN32)
constexpr DWORD kKeepAliveIdleSeconds     = 10;
constexpr DWORD kKeepAliveIntervalSeconds = 3;
constexpr DWORD kKeepAliveProbeCount      = 3;

// Winsock takes the option value as a char buffer, and the keepalive knobs are DWORDs rather than
// ints, so this overload is not interchangeable with the POSIX one.
template <class T>
void set_socket_option(socket_t socket, int level, int option, const T& value) noexcept {
    (void)::setsockopt(socket, level, option, reinterpret_cast<const char*>(&value), sizeof(value));
}
#endif

} // namespace

std::unique_ptr<httplib::TaskQueue> make_http_task_queue(std::size_t workers,
                                                         std::size_t queued_requests) {
    if (workers > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("HTTP worker count exceeds uint32");
    return std::make_unique<HttpTaskQueue>(static_cast<std::uint32_t>(workers), queued_requests);
}

RequestJson parse_json_body(const httplib::Request& request) {
    text::ParsedJsonNumbers parsed;
    try {
        parsed = text::parse_json_numbers(request.body);
    } catch (const std::exception&) {
        bad_request("request body is not valid JSON, or nests deeper than the 64 levels this server "
                    "accepts");
    }
    validate_schema_number_input(parsed);
    return std::move(parsed.value);
}

bool client_disconnected(const httplib::Request& request) { return request.is_connection_closed(); }

void prepare_sse_response(httplib::Response& response) {
    response.set_header("Cache-Control", "no-cache");
    response.set_header("X-Accel-Buffering", "no");
}

SseTransport::SseTransport(httplib::DataSink& sink, std::atomic<bool>& cancelled,
                           Clock::duration heartbeat_interval, Clock::time_point now)
    : sink_(sink), cancelled_(cancelled), heartbeat_interval_(heartbeat_interval),
      last_write_(now) {
    if (heartbeat_interval_ <= Clock::duration::zero()) {
        throw std::invalid_argument("SSE heartbeat interval must be positive");
    }
}

bool SseTransport::mark_cancelled() noexcept {
    cancelled_.store(true, std::memory_order_release);
    return true;
}

void SseTransport::write(std::string_view item, Clock::time_point now) {
    if (cancelled_.load(std::memory_order_acquire) || !sink_.write(item.data(), item.size())) {
        mark_cancelled();
        throw ClientDisconnected();
    }
    last_write_ = now;
}

void SseTransport::write(const std::vector<std::string>& items, Clock::time_point now) {
    for (const std::string& item : items) { write(item, now); }
}

bool SseTransport::poll(Clock::time_point now) {
    if (cancelled_.load(std::memory_order_acquire)) { return true; }
    if (sink_.is_writable && !sink_.is_writable()) { return mark_cancelled(); }
    if (now - last_write_ < heartbeat_interval_) { return false; }
    if (!sink_.write(kHeartbeatComment.data(), kHeartbeatComment.size())) {
        return mark_cancelled();
    }
    last_write_ = now;
    return false;
}

void configure_http_server_socket(socket_t socket) noexcept {
#if defined(__linux__)
    const int enabled = 1;
    // Reuse closed connections' addresses while keeping the active listener exclusive.
    set_socket_option(socket, SOL_SOCKET, SO_REUSEADDR, enabled);
    set_socket_option(socket, SOL_SOCKET, SO_KEEPALIVE, enabled);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPIDLE, kKeepAliveIdleSeconds);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPINTVL, kKeepAliveIntervalSeconds);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPCNT, kKeepAliveProbeCount);
    set_socket_option(socket, IPPROTO_TCP, TCP_USER_TIMEOUT, kTcpUserTimeoutMilliseconds);
#elif defined(_WIN32)
    httplib::default_socket_options(socket);
    // SO_EXCLUSIVEADDRUSE prevents another process from hijacking the port, and it only does so when
    // SO_REUSEADDR is not already set: default_socket_options above sets that on Windows, and a later
    // SO_EXCLUSIVEADDRUSE does not override it. Measured with a three-arm bind probe -- with
    // SO_REUSEADDR then SO_EXCLUSIVEADDRUSE a second process still binds (rc=0), with
    // SO_EXCLUSIVEADDRUSE alone it fails with WSAEACCES (10013), which is the protection this line
    // exists for. Microsoft's Winsock guidance: "All server applications must set
    // SO_EXCLUSIVEADDRUSE." The same page carries a restart caveat -- a port bound this way "cannot
    // always be reused immediately after socket closure" -- and that is measured rather than argued
    // here (2026-10-11): with a server-side TIME_WAIT present on the port at rebind time (verified
    // with netstat, alongside a FIN_WAIT_2 from the still-open client), a new listener bound it in
    // 0.5 s and answered a request. The caveat did not apply to this loopback bind; why is not
    // established, and the measurement is what the comment can honestly carry.
    // `tools/bench/probe_listener_restart.py` is the instrument, and it prints the port's state at
    // rebind time so a pass cannot be read against an empty port table.
    const BOOL enabled  = TRUE;
    const BOOL disabled = FALSE;
    set_socket_option(socket, SOL_SOCKET, SO_REUSEADDR, disabled);
    set_socket_option(socket, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, enabled);
    // A half-open connection otherwise holds a request slot until the client's own timeout fires.
    // TCP_KEEPIDLE/TCP_KEEPINTVL/TCP_KEEPCNT are only defined by newer SDKs, so each is guarded.
    set_socket_option(socket, SOL_SOCKET, SO_KEEPALIVE, enabled);
#    if defined(TCP_KEEPIDLE)
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPIDLE, kKeepAliveIdleSeconds);
#    endif
#    if defined(TCP_KEEPINTVL)
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPINTVL, kKeepAliveIntervalSeconds);
#    endif
#    if defined(TCP_KEEPCNT)
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPCNT, kKeepAliveProbeCount);
#    endif
#else
    httplib::default_socket_options(socket);
#endif
}

void set_owned_json_content(httplib::Response& response, std::string body,
                            std::shared_ptr<RequestLifetime> lifetime) {
    response.set_content(std::move(body), "application/json");
    response.user_data.set("ninfer.request_lifetime", std::move(lifetime));
}

} // namespace ninfer::serve
