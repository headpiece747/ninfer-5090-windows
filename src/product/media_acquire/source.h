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

// The label a prepared source is recorded under.
//
// It was written twice, with different wording for the same case -- "inline-data" on the serving
// path and "inline data URI" in the CLI -- so one input was recorded under two names depending on
// which endpoint took it. The server's wording wins, because it distinguishes inline bytes from an
// inline data URI, which the CLI's version did not.
[[nodiscard]] inline std::string_view source_name(SourceKind kind, std::string_view value) {
    switch (kind) {
    case SourceKind::Data: return "inline-data";
    case SourceKind::Bytes: return "inline-bytes";
    case SourceKind::Path:
    case SourceKind::Url: break;
    }
    return value;
}

} // namespace ninfer::product::media_acquire
