#include "text/json_schema.h"
#include "ninfer/types.h"
#include "json_string_grammar.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ninfer::text {
namespace {
using Json = nlohmann::ordered_json;
using Kind = RequestErrorKind;

[[noreturn]] void fail(Kind kind, const std::string& path, const std::string& message) {
    throw RequestError(kind, message + " at " + (path.empty() ? "/" : path), path);
}

std::string child(const std::string& path, std::string_view key) {
    std::string result = path + '/';
    for (char c : key) {
        if (c == '~')
            result += "~0";
        else if (c == '/')
            result += "~1";
        else
            result += c;
    }
    return result;
}

bool annotation(std::string_view key) {
    static const std::unordered_set<std::string_view> keys{
        "title",    "description", "default", "examples", "$comment", "deprecated",
        "readOnly", "writeOnly",   "$schema", "$id",      "$defs",    "definitions"};
    return keys.contains(key);
}

bool matches_type(const Json& value, std::string_view type) {
    if (type == "object") return value.is_object();
    if (type == "array") return value.is_array();
    if (type == "string") return value.is_string();
    if (type == "boolean") return value.is_boolean();
    if (type == "null") return value.is_null();
    if (type == "number") return value.is_number();
    if (type == "integer")
        return value.is_number_integer() ||
               (value.is_number_float() && std::floor(value.get<double>()) == value.get<double>());
    return false;
}

void check_literal(const Json& value, const std::string& path) {
    if ((value.is_number_unsigned() &&
         value.get<std::uint64_t>() > std::uint64_t(std::numeric_limits<std::int64_t>::max())) ||
        (value.is_number_float() && std::abs(value.get<double>()) >= 0x1p63))
        fail(Kind::UnsupportedJsonSchema, path,
             "const/enum integers must fit signed 64-bit integers");
    if (value.is_object() || value.is_array())
        for (const auto& [key, item] : value.items()) check_literal(item, child(path, key));
}

class Schema {
public:
    Json root;
    std::vector<std::pair<std::string, std::string>> references;
    std::unordered_set<std::string> locations;

    void check(Json& node, const std::string& path) {
        locations.insert(path);
        if (node.is_boolean()) return;
        if (!node.is_object())
            fail(Kind::InvalidJsonSchema, path, "schema must be an object or boolean");
        static const std::unordered_set<std::string_view> keywords{"type",
                                                                   "properties",
                                                                   "required",
                                                                   "additionalProperties",
                                                                   "items",
                                                                   "minItems",
                                                                   "maxItems",
                                                                   "minLength",
                                                                   "maxLength",
                                                                   "pattern",
                                                                   "minimum",
                                                                   "maximum",
                                                                   "exclusiveMinimum",
                                                                   "exclusiveMaximum",
                                                                   "const",
                                                                   "enum",
                                                                   "anyOf",
                                                                   "oneOf",
                                                                   "allOf",
                                                                   "$ref"};
        for (const auto& [key, value] : node.items()) {
            if (!annotation(key) && !keywords.contains(key))
                fail(Kind::UnsupportedJsonSchema, child(path, key),
                     "unsupported schema keyword: " + key);
        }
        if (node.contains("$schema")) {
            const auto& dialect = node["$schema"];
            if (!dialect.is_string())
                fail(Kind::InvalidJsonSchema, child(path, "$schema"), "dialect must be a string");
            const auto name = dialect.get<std::string>();
            if (name != "https://json-schema.org/draft/2020-12/schema" &&
                name != "https://json-schema.org/draft/2020-12/schema#" &&
                name != "http://json-schema.org/draft-07/schema#" &&
                name != "https://json-schema.org/draft-07/schema#")
                fail(Kind::UnsupportedJsonSchema, child(path, "$schema"),
                     "unsupported JSON Schema dialect");
        }
        if (node.contains("$id")) {
            if (!node["$id"].is_string())
                fail(Kind::InvalidJsonSchema, child(path, "$id"), "$id must be a string");
            if (!path.empty())
                fail(Kind::UnsupportedJsonSchema, child(path, "$id"),
                     "nested reference scopes are not supported");
        }
        for (const char* key : {"$defs", "definitions"}) {
            if (!node.contains(key)) continue;
            if (!node[key].is_object())
                fail(Kind::InvalidJsonSchema, child(path, key), "definitions must be an object");
            for (auto& [name, definition] : node[key].items())
                check(definition, child(child(path, key), name));
        }
        for (const char* key : {"$ref", "anyOf", "oneOf", "allOf"}) {
            if (!node.contains(key)) continue;
            for (const auto& [sibling, value] : node.items()) {
                if (sibling != key && !annotation(sibling))
                    fail(Kind::UnsupportedJsonSchema, child(path, sibling),
                         "assertion siblings of " + std::string(key) + " are not supported");
            }
            if (std::string_view(key) == "$ref") {
                if (!node[key].is_string())
                    fail(Kind::InvalidJsonSchema, child(path, key), "$ref must be a string");
                const auto ref = node[key].get<std::string>();
                if (ref != "#" && !ref.starts_with("#/"))
                    fail(Kind::UnsupportedJsonSchema, child(path, key),
                         "only local JSON Pointer references are supported");
                references.emplace_back(ref.substr(1), child(path, key));
            } else {
                if (!node[key].is_array() || node[key].empty())
                    fail(Kind::InvalidJsonSchema, child(path, key),
                         "schema alternatives must be a nonempty array");
                if (std::string_view(key) == "allOf" && node[key].size() != 1)
                    fail(Kind::UnsupportedJsonSchema, child(path, key),
                         "allOf requires a single branch");
                for (std::size_t i = 0; i < node[key].size(); ++i)
                    check(node[key][i], child(child(path, key), std::to_string(i)));
            }
            return;
        }
        std::set<std::string> types;
        if (node.contains("type")) {
            const auto add = [&](const Json& value) {
                if (!value.is_string())
                    fail(Kind::InvalidJsonSchema, child(path, "type"),
                         "type must name a JSON type");
                const auto type = value.get<std::string>();
                if (type != "object" && type != "array" && type != "string" && type != "integer" &&
                    type != "number" && type != "boolean" && type != "null")
                    fail(Kind::InvalidJsonSchema, child(path, "type"),
                         "unknown JSON type: " + type);
                if (!types.insert(type).second)
                    fail(Kind::InvalidJsonSchema, child(path, "type"), "duplicate JSON type");
            };
            if (node["type"].is_array()) {
                if (node["type"].empty())
                    fail(Kind::InvalidJsonSchema, child(path, "type"),
                         "type array must not be empty");
                for (const auto& type : node["type"]) add(type);
            } else
                add(node["type"]);
        }
        if (node.contains("const") || node.contains("enum")) {
            const bool constant = node.contains("const");
            const char* key     = constant ? "const" : "enum";
            for (const auto& [sibling, value] : node.items())
                if (sibling != key && sibling != "type" && !annotation(sibling))
                    fail(Kind::UnsupportedJsonSchema, child(path, sibling),
                         "finite values support only a type assertion alongside const/enum");
            if (!constant && (!node[key].is_array() || node[key].empty()))
                fail(Kind::InvalidJsonSchema, child(path, key), "enum must be a nonempty array");
            check_literal(node[key], child(path, key));
            const auto admitted = [&](const Json& value) {
                return types.empty() || std::any_of(types.begin(), types.end(), [&](const auto& t) {
                           return matches_type(value, t);
                       });
            };
            const auto impossible = [&] {
                // Definitions remain addressable even when their containing schema admits no value.
                for (auto it = node.begin(); it != node.end();) {
                    if (annotation(it.key()))
                        ++it;
                    else
                        it = node.erase(it);
                }
                node["allOf"] = Json::array({false});
            };
            if (constant) {
                if (!admitted(node[key])) impossible();
            } else {
                Json filtered = Json::array();
                for (const auto& value : node[key])
                    if (admitted(value)) filtered.push_back(value);
                if (filtered.empty())
                    impossible();
                else
                    node[key] = std::move(filtered);
            }
            return;
        }
        // Without type, typed assertions apply only to their own instance types. Preserve that
        // JSON Schema meaning instead of silently treating the whole schema as unrestricted.
        if (types.empty()) {
            bool assertions = false;
            for (const auto& [key, value] : node.items()) assertions |= !annotation(key);
            if (assertions) {
                node["type"] =
                    Json::array({"object", "array", "string", "number", "boolean", "null"});
                types = {"object", "array", "string", "number", "boolean", "null"};
            }
        }
        for (const char* key : {"minItems", "maxItems", "minLength", "maxLength"}) {
            if (!node.contains(key)) continue;
            const auto& value = node[key];
            if (!value.is_number_integer() || value < 0)
                fail(Kind::InvalidJsonSchema, child(path, key),
                     "bound must be a nonnegative integer");
            if (value > std::numeric_limits<std::int32_t>::max())
                fail(Kind::UnsupportedJsonSchema, child(path, key),
                     "length bound exceeds the supported range");
        }
        for (const char* key : {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}) {
            if (!node.contains(key)) continue;
            const auto& value = node[key];
            if (!value.is_number())
                fail(Kind::InvalidJsonSchema, child(path, key), "numeric bound must be a number");
            if (types.contains("number"))
                fail(Kind::UnsupportedJsonSchema, child(path, key),
                     "numeric ranges currently require type integer");
            if (types.contains("integer") &&
                (!value.is_number_integer() ||
                 (value.is_number_unsigned() &&
                  value.get<std::uint64_t>() >
                      std::uint64_t(std::numeric_limits<std::int64_t>::max()))))
                fail(Kind::UnsupportedJsonSchema, child(path, key),
                     "integer bounds must fit signed 64-bit integers");
            if (types.contains("integer") && ((std::string_view(key) == "exclusiveMinimum" &&
                                               value == std::numeric_limits<std::int64_t>::max()) ||
                                              (std::string_view(key) == "exclusiveMaximum" &&
                                               value == std::numeric_limits<std::int64_t>::min())))
                fail(Kind::UnsupportedJsonSchema, child(path, key),
                     "exclusive bound requires integers beyond the supported range");
        }
        if (node.contains("pattern")) {
            if (!node["pattern"].is_string())
                fail(Kind::InvalidJsonSchema, child(path, "pattern"), "pattern must be a string");
            if (node.contains("minLength") || node.contains("maxLength"))
                fail(Kind::UnsupportedJsonSchema, child(path, "pattern"),
                     "pattern combined with length bounds is not supported");
        }
        if (node.contains("pattern") && types.contains("string")) {
            try {
                (void)xgrammar::JSONStringPattern(node["pattern"].get<std::string>());
            } catch (const std::exception& error) {
                fail(Kind::UnsupportedJsonSchema, child(path, "pattern"), error.what());
            }
        }
        if (node.contains("properties")) {
            if (!node["properties"].is_object())
                fail(Kind::InvalidJsonSchema, child(path, "properties"),
                     "properties must be an object");
            for (auto& [key, value] : node["properties"].items())
                check(value, child(child(path, "properties"), key));
        }
        if (node.contains("required")) {
            if (!node["required"].is_array())
                fail(Kind::InvalidJsonSchema, child(path, "required"), "required must be an array");
            std::set<std::string> names;
            for (const auto& item : node["required"]) {
                if (!item.is_string() || !names.insert(item.get<std::string>()).second)
                    fail(Kind::InvalidJsonSchema, child(path, "required"),
                         "required must contain distinct property names");
                if (!node.contains("properties") ||
                    !node["properties"].contains(item.get<std::string>()))
                    fail(Kind::UnsupportedJsonSchema, child(path, "required"),
                         "required properties must be declared in properties");
            }
        }
        for (const char* key : {"items", "additionalProperties"})
            if (node.contains(key)) check(node[key], child(path, key));
    }
};
} // namespace

std::string prepare_json_schema(std::string_view source) {
    Schema schema;
    try {
        schema.root = Json::parse(source);
    } catch (const Json::exception& error) { fail(Kind::InvalidJsonSchema, {}, error.what()); }
    schema.check(schema.root, {});
    for (const auto& [pointer, path] : schema.references) {
        try {
            const auto& target = schema.root.at(Json::json_pointer(pointer));
            if (!schema.locations.contains(pointer) ||
                (!target.is_object() && !target.is_boolean()))
                fail(Kind::UnsupportedJsonSchema, path,
                     "reference target is not a schema location");
        } catch (const Json::exception&) {
            fail(Kind::InvalidJsonSchema, path, "unresolved local reference");
        }
    }
    if (schema.root.is_boolean() && !schema.root.get<bool>())
        fail(Kind::UnsatisfiableJsonSchema, {}, "schema cannot accept any value");
    return schema.root.dump();
}
} // namespace ninfer::text
