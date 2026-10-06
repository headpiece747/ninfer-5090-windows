#pragma once

#include <string>
#include <string_view>

namespace ninfer::text {

// Validate the supported JSON Schema dialect, retaining property order and literal data.
// Returns normalized schema text for the vendor converter; errors carry a JSON Pointer.
std::string prepare_json_schema(std::string_view source);

} // namespace ninfer::text
