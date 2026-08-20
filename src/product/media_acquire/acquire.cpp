#include "product/media_acquire/acquire.h"

#include <algorithm>
#include <array>
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
        throw std::invalid_argument("ninfer compiled without curl. Remote URLs are unsupported.");
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
