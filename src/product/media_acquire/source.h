#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::product::media_acquire {

enum class SourceKind {
    Path,
    Url,
    Data,
    Bytes,
};

struct Source {
    SourceKind kind = SourceKind::Path;
    std::string value;
    std::string media_type;
    std::vector<std::uint8_t> bytes;
};

// The prefix classification every wire endpoint needs: a data URI or an HTTP(S) URL.
//
// It returns nullopt for anything else, because what an unrecognised value means is the endpoint's
// decision -- the chat and Responses endpoints reject it, the CLI reads it as a path, and Anthropic
// dispatches on a declared type rather than on a prefix at all. Each caller also keeps its own
// message: that text is the endpoint's error contract, and the endpoints that share this rule word
// it three different ways.
[[nodiscard]] inline std::optional<SourceKind> classify_wire_source(std::string_view value) {
    if (value.starts_with("data:")) { return SourceKind::Data; }
    if (value.starts_with("http://") || value.starts_with("https://")) { return SourceKind::Url; }
    return std::nullopt;
}

} // namespace ninfer::product::media_acquire
