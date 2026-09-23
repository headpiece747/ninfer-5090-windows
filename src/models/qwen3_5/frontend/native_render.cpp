#include "models/qwen3_5/frontend/native_render.h"

#include "models/qwen3_5/frontend/digest.h"
#include "text/unicode.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::models::qwen3_5::frontend {

namespace {

// One entry per transcribed template, and the whole registry is the list: a template that differs by
// even one byte is not in it and takes the Jinja path.
struct Registration {
    Sha256Digest digest;
};

// `tools/chat_templates/qwen3_8.jinja`, the template the shipped artifacts embed and the one the
// launchers pass explicitly. Its bytes digest to this value; any other source is not transcribed and
// takes the Jinja path, which is what bounds the cost of a template changing under this port.
constexpr Sha256Digest kQwen38TemplateDigest{
    0xea, 0x10, 0x06, 0x6b, 0x7a, 0xd5, 0xec, 0xf8, 0xa6, 0xc8, 0x48, 0xfe, 0x9f, 0x18, 0xcb, 0xde,
    0xe3, 0xe4, 0x69, 0x3b, 0x2d, 0x85, 0xf7, 0xe1, 0xb8, 0xb6, 0x94, 0xa4, 0x20, 0x35, 0x5f, 0x3b};

constexpr std::array<Registration, 1> kRegistrations{{{kQwen38TemplateDigest}}};

using Json = nlohmann::ordered_json;

bool instruction(ChatRole role) { return role == ChatRole::System || role == ChatRole::Developer; }

// The literal strings, transcribed from the template. Every one is emitted verbatim by the Jinja
// route as well: the `-` modifiers on the template's tags strip the source's own whitespace, so the
// output is exactly the concatenation of these and the input text.
constexpr std::string_view kImStart       = "<|im_start|>";
constexpr std::string_view kImEnd         = "<|im_end|>";
constexpr std::string_view kThinkOpen         = "<" "think" ">\n";
constexpr std::string_view kThinkClose        = "\n</" "think" ">\n\n";
constexpr std::string_view kToolCallOpen  = "\n\n<tool_call>\n<function=";
constexpr std::string_view kToolCallOpenB = "<tool_call>\n<function=";
constexpr std::string_view kToolCallNext  = "\n<tool_call>\n<function=";
constexpr std::string_view kFunctionEnd   = "</function>\n</tool_call>";
constexpr std::string_view kParamOpen     = "<parameter=";
constexpr std::string_view kParamValueSep = ">\n";
constexpr std::string_view kParamClose    = "\n</parameter>\n";
constexpr std::string_view kToolResponseOpen  = "\n<tool_response>\n";
constexpr std::string_view kToolResponseClose = "\n</tool_response>";
constexpr std::string_view kToolResponseStart = "<tool_response>";
constexpr std::string_view kToolResponseEnd   = "</tool_response>";
constexpr std::string_view kGenPromptAssistant = "<|im_start|>assistant\n";
constexpr std::string_view kGenPromptThinkOff  = "<" "think" ">\n\n</" "think" ">\n\n";
constexpr std::string_view kGenPromptThink     = "<" "think" ">\n";
constexpr std::string_view kNoThink =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
    "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
    "clarity in the final answer.";
constexpr std::string_view kLowThink =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
    "conclusion without unnecessary elaboration.";
constexpr std::string_view kToolsBlock =
    "# Tools\n\nYou have access to the following functions:\n\n<tools>";
constexpr std::string_view kToolsBlockEnd = "\n</tools>";
constexpr std::string_view kToolsReminder =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
    "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
    "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n"
    "Reminder:\n- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your "
    "function call in natural language BEFORE the function call, but NOT after\n- If there is no "
    "function call available, answer the question like normal with your current knowledge and do "
    "not tell the user about function calls\n</IMPORTANT>";

// `|trim` is `strip(true, true)` over codepoints: runtime.cpp aliases `trim` to `strip`, value.cpp
// calls `string::strip(true, true)`, and that walks `unicode::characters` testing
// `unicode::whitespace(cp)`. Matching it byte-wise would be a silent difference on any input that
// starts or ends with a non-ASCII space.
bool unicode_whitespace(std::int32_t codepoint) noexcept {
    return (codepoint >= 0x1c && codepoint <= 0x1f) || codepoint == 0x85 ||
           text::unicode_internal::is_whitespace(codepoint);
}

std::string trim_whitespace(std::string_view text) {
    const auto spans = text::unicode_internal::utf8_codepoints(text, {});
    std::size_t begin = 0;
    std::size_t end   = spans.size();
    while (begin < end && unicode_whitespace(spans[begin].value)) { ++begin; }
    while (end > begin && unicode_whitespace(spans[end - 1].value)) { --end; }
    const std::size_t first =
        begin == spans.size() ? text.size() : spans[begin].offset;
    const std::size_t last =
        end == spans.size() ? text.size() : spans[end - 1].offset + spans[end - 1].length;
    return std::string(text.substr(first, last - first));
}

// A JSON string as the interpreter writes it: `Json(text).dump(-1, ' ', ensure_ascii)`.
std::string to_json_string(std::string_view text) {
    return nlohmann::ordered_json(text).dump();
}

// Jinja's `string`/`safe` filters serialize through the interpreter's own `tojson`, whose defaults
// are `json.dumps`': item separator ", ", key separator ": ", ensure_ascii false, no pretty-printing
// (`json.cpp:14`). Neither nlohmann spelling matches -- compact uses "," / ":", indent puts a newline
// after every brace -- so the separators are assembled here and only scalars go through nlohmann,
// where a string is escaped by `dump(-1, ' ', ensure_ascii)` exactly as the interpreter does it.
void write_json(std::string& out, const nlohmann::ordered_json& value) {
    if (value.is_object()) {
        out.push_back('{');
        bool first = true;
        for (auto item = value.begin(); item != value.end(); ++item) {
            if (!first) { out += ", "; }
            first = false;
            out += to_json_string(item.key());
            out += ": ";
            write_json(out, item.value());
        }
        out.push_back('}');
    } else if (value.is_array()) {
        out.push_back('[');
        bool first = true;
        for (const auto& item : value) {
            if (!first) { out += ", "; }
            first = false;
            write_json(out, item);
        }
        out.push_back(']');
    } else if (value.is_string()) {
        out += to_json_string(value.get_ref<const std::string&>());
    } else {
        out += value.dump();
    }
}

std::string to_json(const nlohmann::ordered_json& value) {
    std::string out;
    write_json(out, value);
    return out;
}

// The media placeholder a message part contributes, in template order.
struct MediaUse {
    Modality modality;
    std::size_t part_index = 0;
};

// What one message's `render_content` produced, plus the surfaces the boundary records need.
struct MessageRender {
    std::string text;
    std::string trimmed;             // the template's `render_content(..)|trim`
    std::size_t content_begin = 0;   // offset of the trimmed content inside `text`'s parent buffer
    std::size_t content_end   = 0;
    bool unsupported          = false; // the template would raise: a media part in an instruction turn
};

// A message's parts rendered to text. Mirrors `render_content`: text parts contribute their text,
// media parts contribute the vision wrapper, and `add_vision_id` prefixes a running count.
std::string render_content(const ChatMessage& message, bool is_instruction, bool add_vision_id,
                           std::size_t& image_count, std::size_t& video_count) {
    std::string out;
    for (const ChatPart& part : message.parts) {
        if (part.kind == ChatPartKind::Text) {
            out += part.text;
            continue;
        }
        const bool image = part.kind == ChatPartKind::Image;
        if (is_instruction) {
            // The template raises for a media part in a system or developer turn; the frontend
            // rejects that earlier, so reaching here is a programming error rather than input.
            throw std::invalid_argument("system and developer messages may contain only text");
        }
        if (image) {
            ++image_count;
            if (add_vision_id) { out += "Picture " + std::to_string(image_count) + ": "; }
            out += "<|vision_start|><|image_pad|><|vision_end|>";
        } else {
            ++video_count;
            if (add_vision_id) { out += "Video " + std::to_string(video_count) + ": "; }
            out += "<|vision_start|><|video_pad|><|vision_end|>";
        }
    }
    return out;
}

} // namespace

bool native_render_supported(const Sha256Digest& template_digest) {
    for (const Registration& registration : kRegistrations) {
        if (registration.digest == template_digest) { return true; }
    }
    return false;
}

RenderedChat render_native(const std::vector<ChatMessage>& messages,
                           const ChatRenderOptions& options) {
    if (messages.empty()) { throw std::invalid_argument("chat requires at least one message"); }
    const bool continuation = options.continuation == PromptContinuationMode::ContinueFinalAssistant;
    const bool thinking_enabled =
        !(options.enable_thinking.has_value() && !*options.enable_thinking);

    // `reasoning_instructions`, exactly as the template resolves it: only while thinking is enabled,
    // the typed option projected to its protocol name, then the client aliases the template applies,
    // then the rendered preset.
    std::string_view reasoning_instructions;
    if (thinking_enabled) {
        std::string effort = "xhigh";
        if (options.reasoning_effort.has_value()) {
            const std::string_view name = reasoning_effort_name(*options.reasoning_effort);
            if (!name.empty()) { effort = std::string(name); }
        }
        if (effort == "high" || effort == "max" || effort == "ultracode" || effort == "extreme") {
            effort = "xhigh";
        } else if (effort == "minimal") {
            effort = "low";
        }
        if (effort == "xhigh") {
            reasoning_instructions = kNoThink;
        } else if (effort == "low") {
            reasoning_instructions = kLowThink;
        } else if (effort != "medium") {
            throw std::invalid_argument("Unexpected reasoning effort " + effort +
                                        ". Supported types are xhigh (default), medium, and low.");
        }
    }

    // The template's own branch on `tools` decides the preamble shape.
    std::vector<Json> tool_definitions;
    tool_definitions.reserve(options.tool_jsons.size());
    for (const std::string& tool_json : options.tool_jsons) {
        tool_definitions.push_back(Json::parse(tool_json));
    }
    const bool has_tools = !tool_definitions.empty();

    std::size_t image_count = 0;
    std::size_t video_count = 0;
    // The counters are only observable through `add_vision_id`, so the count phases run on a copy of
    // their state and the output phases start from zero, exactly as the template's two passes do.
    std::size_t count_images = 0;
    std::size_t count_videos = 0;

    // Pass two's state: the index of the last user turn that is not a folded tool response, or of the
    // last tool turn when every user turn was one.
    std::optional<std::size_t> last_query_index;
    std::optional<std::size_t> last_tool_index;
    for (std::size_t i = messages.size(); i-- > 0;) {
        if (messages[i].role == ChatRole::Tool) { last_tool_index = i; break; }
    }
    for (std::size_t i = messages.size(); i-- > 0;) {
        if (messages[i].role != ChatRole::User) { continue; }
        const std::string raw =
            render_content(messages[i], false, false, count_images, count_videos);
        if (!(raw.starts_with(kToolResponseStart) && raw.ends_with(kToolResponseEnd))) {
            last_query_index = i;
            break;
        }
    }
    if (!last_query_index && last_tool_index) { last_query_index = last_tool_index; }

    RenderedChat result;
    std::string& text = result.text;
    // Every literal this renderer emits is template bytes, so a literal span is the complement of the
    // ranges an input contributed. Collected as input ranges and inverted at the end.
    std::vector<text::ByteSpan> input_spans;
    const auto append_input = [&](std::string_view bytes) {
        if (bytes.empty()) { return; }
        input_spans.push_back({text.size(), text.size() + bytes.size()});
        text += bytes;
    };
    result.message_boundaries.resize(messages.size() + 1);

    if (has_tools) {
        text += kImStart;
        text += "system\n";
        if (!reasoning_instructions.empty()) { text += reasoning_instructions; text += "\n\n"; }
        text += kToolsBlock;
        for (const Json& definition : tool_definitions) {
            text += "\n";
            text += to_json(definition);
        }
        text += kToolsBlockEnd;
        text += kToolsReminder;
        if (instruction(messages.front().role)) {
            const std::string content = trim_whitespace(
                render_content(messages.front(), true, false, image_count, video_count));
            if (!content.empty()) { text += "\n\n"; append_input(content); }
        }
        text += kImEnd;
        text += "\n";
    } else if (instruction(messages.front().role)) {
        const std::string_view content =
            trim_whitespace(render_content(messages.front(), true, false, image_count, video_count));
        if (!content.empty()) {
            text += kImStart;
            text += "system\n";
            if (!reasoning_instructions.empty()) { text += reasoning_instructions; text += "\n\n"; }
            append_input(content);
            text += kImEnd;
            text += "\n";
        } else if (!reasoning_instructions.empty()) {
            text += kImStart;
            text += "system\n";
            text += reasoning_instructions;
            text += kImEnd;
            text += "\n";
        }
    } else if (!reasoning_instructions.empty()) {
        text += kImStart;
        text += "system\n";
        text += reasoning_instructions;
        text += kImEnd;
        text += "\n";
    }

    for (std::size_t i = 0; i < messages.size(); ++i) {
        const ChatMessage& message = messages[i];
        const bool first = i == 0;
        const bool last  = i + 1 == messages.size();
        const std::string content = trim_whitespace(render_content(
            message, instruction(message.role), options.add_vision_id, image_count, video_count));
        if (instruction(message.role)) {
            if (!first) {
                text += kImStart;
                text += "system\n";
                append_input(content);
                text += kImEnd;
                text += "\n";
            }
        } else if (message.role == ChatRole::User) {
            text += kImStart;
            append_input("user");
            text += "\n";
            append_input(content);
            text += kImEnd;
            text += "\n";
        } else if (message.role == ChatRole::Assistant) {
            const bool continuing = continuation && last;
            text += kImStart;
            append_input("assistant");
            text += "\n";
            const bool emit_reasoning =
                !continuing &&
                (!options.preserve_thinking.has_value() || *options.preserve_thinking ||
                 i > last_query_index.value_or(messages.size() - 1));
            if (emit_reasoning) {
                text += kThinkOpen;
                if (!message.reasoning_content.empty()) {
                    append_input(trim_whitespace(message.reasoning_content));
                }
                text += kThinkClose;
            }
            append_input(content);
            if (!message.tool_calls.empty()) {
                for (std::size_t call_index = 0; call_index < message.tool_calls.size();
                     ++call_index) {
                    const ToolCall& call = message.tool_calls[call_index];
                    if (call_index == 0) {
                        text += content.empty() ? kToolCallOpenB : kToolCallOpen;
                    } else {
                        text += kToolCallNext;
                    }
                    append_input(call.name);
                    text += kParamValueSep;
                    if (!call.arguments_json.empty()) {
                        const Json arguments = Json::parse(call.arguments_json);
                        for (auto item = arguments.begin(); item != arguments.end(); ++item) {
                            text += kParamOpen;
                            append_input(item.key());
                            text += kParamValueSep;
                            if (item.value().is_string()) {
                                append_input(item.value().get_ref<const std::string&>());
                            } else {
                                text += to_json(item.value());
                            }
                            text += kParamClose;
                        }
                    }
                    text += kFunctionEnd;
                }
            }
            if (!continuing) {
                text += kImEnd;
                text += "\n";
            }
        } else if (message.role == ChatRole::Tool) {
            if (first || messages[i - 1].role != ChatRole::Tool) {
                text += kImStart;
                append_input("user");
            }
            text += kToolResponseOpen;
            append_input(content);
            text += kToolResponseClose;
            if (last || messages[i + 1].role != ChatRole::Tool) {
                text += kImEnd;
                text += "\n";
            }
        } else {
            throw std::invalid_argument("Unexpected message role.");
        }
        result.message_boundaries[i + 1] = text.size();
    }

    if (options.add_generation_prompt) {
        text += kGenPromptAssistant;
        if (options.enable_thinking.has_value() && !*options.enable_thinking) {
            text += kGenPromptThinkOff;
        } else {
            text += kGenPromptThink;
        }
    }

    // Literal spans are the complement of the input ranges, which is what the Jinja route's input
    // marking produces: template bytes are literal, input bytes are not.
    std::size_t cursor = 0;
    for (const text::ByteSpan& span : input_spans) {
        if (span.begin > cursor) { result.literal_spans.push_back({cursor, span.begin}); }
        cursor = std::max(cursor, span.end);
    }
    if (cursor < text.size()) { result.literal_spans.push_back({cursor, text.size()}); }
    return result;
}

} // namespace ninfer::models::qwen3_5::frontend
