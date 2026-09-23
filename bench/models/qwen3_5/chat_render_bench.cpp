// Host-only: what one prompt preparation costs, by conversation length.
//
// The frontend renders a chat template more than once per request. `CompiledChatTemplate::render`
// runs the output render, and then, to prove which byte offsets are safe prefix-cache boundaries, it
// runs `prefix(...)` probes that each re-render a variant of the whole conversation
// (src/models/qwen3_5/frontend/chat_template.cpp:252, :294, :394). Every one of those renders walks
// the message list twice through the Jinja interpreter, because the template scans it in reverse to
// find `last_query_index` and then forward to emit the turns.
//
// This bench measures the curve and separates the probes by turning the conditions that trigger
// them on and off, so the cost of each is read rather than inferred:
//
//   --generation-prompt on|off   `prefix(messages.size())` fires only when on
//   --tail-assistant on|off      the checkpoint probe fires only when the last real user turn is
//                                followed by an assistant message
//
// It needs no artifact and no GPU: the template is the only input.
//
// usage: ninfer_qwen3_5_chat_render_bench [--template <path>] [--sweep 32,64,128,229]
//          [--chars <chars per message>] [--tools <n>] [--warmup <n>] [--reps <n>]
//          [--generation-prompt on|off] [--tail-assistant on|off] [--quiet]

#include "models/qwen3_5/frontend/chat_template.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fi    = ninfer::models::qwen3_5::frontend;
using Clock     = std::chrono::steady_clock;

struct Options {
    std::filesystem::path template_path = "tools/chat_templates/qwen3_8.jinja";
    std::vector<std::size_t> sweep      = {32, 64, 128, 229};
    std::size_t chars_per_message       = 3000;
    std::size_t call_args_chars         = 45;
    std::size_t tool_count              = 5;
    int warmup                          = 1;
    int repetitions                     = 3;
    bool generation_prompt              = true;
    bool tail_assistant                 = true;
    // Install a cancellation callback so the interpreter's per-statement checkpoint runs, as it does
    // in production where the callback is an httplib socket probe. The arm reports how often it fires.
    bool cancel_probe                   = false;
    bool quiet                          = false;
};

void print_usage(const char* executable) {
    std::cout << "usage: " << executable
              << " [--template <path>] [--sweep <n,n,...>] [--chars <n>] [--call-args <n>]"
                 " [--tools <n>] [--warmup <n>] [--reps <n>] [--generation-prompt on|off]"
                 " [--tail-assistant on|off] [--cancel-probe] [--quiet]\n";
}

bool parse_bool(std::string_view text) {
    if (text == "on" || text == "true" || text == "1") { return true; }
    if (text == "off" || text == "false" || text == "0") { return false; }
    throw std::invalid_argument("expected on or off, got " + std::string(text));
}

std::size_t parse_size(const char* text, const char* label) {
    std::size_t consumed = 0;
    const auto value     = std::stoull(text, &consumed);
    if (text[consumed] != '\0') { throw std::invalid_argument(std::string(label) + " is not a count"); }
    return static_cast<std::size_t>(value);
}

std::vector<std::size_t> parse_sweep(const char* text) {
    std::vector<std::size_t> counts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) { continue; }
        counts.push_back(parse_size(item.c_str(), "sweep entry"));
    }
    if (counts.empty()) { throw std::invalid_argument("sweep is empty"); }
    return counts;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        const auto next = [&](const char* label) -> const char* {
            if (i + 1 >= argc) { throw std::invalid_argument(std::string("missing value for ") + label); }
            return argv[++i];
        };
        if (flag == "--template") {
            options.template_path = next("--template");
        } else if (flag == "--sweep") {
            options.sweep = parse_sweep(next("--sweep"));
        } else if (flag == "--chars") {
            options.chars_per_message = parse_size(next("--chars"), "--chars");
        } else if (flag == "--call-args") {
            options.call_args_chars = parse_size(next("--call-args"), "--call-args");
        } else if (flag == "--cancel-probe") {
            options.cancel_probe = true;
        } else if (flag == "--tools") {
            options.tool_count = parse_size(next("--tools"), "--tools");
        } else if (flag == "--warmup") {
            options.warmup = static_cast<int>(parse_size(next("--warmup"), "--warmup"));
        } else if (flag == "--reps") {
            options.repetitions = static_cast<int>(parse_size(next("--reps"), "--reps"));
        } else if (flag == "--generation-prompt") {
            options.generation_prompt = parse_bool(next("--generation-prompt"));
        } else if (flag == "--tail-assistant") {
            options.tail_assistant = parse_bool(next("--tail-assistant"));
        } else if (flag == "--quiet") {
            options.quiet = true;
        } else if (flag == "--help" || flag == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option " + std::string(flag));
        }
    }
    return options;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { throw std::invalid_argument("cannot read " + path.string()); }
    const auto size = file.tellg();
    std::string source(static_cast<std::size_t>(size), '\0');
    file.seekg(0);
    if (!file.read(source.data(), size)) { throw std::invalid_argument("cannot read " + path.string()); }
    return source;
}

// Deterministic filler. The render cost follows bytes, not tokens, so a character budget is the
// honest parameter.
std::string filler(std::size_t target_chars, std::size_t seed) {
    static constexpr std::string_view kWords[] = {
        "the",   "resolver", "reads",   "each",  "pending", "worklist", "entry",  "and",
        "writes", "back",    "a",       "dirty", "flag",    "before",   "yield",  "to",
        "the",   "scheduler", "which",  "then",  "coalesces", "adjacent", "runs", "in",
        "one",   "pass",     "over",    "the",   "arena",   "without",  "copying"};
    constexpr std::size_t kWordCount = sizeof(kWords) / sizeof(kWords[0]);
    std::string out;
    out.reserve(target_chars + 64);
    std::size_t index = seed;
    while (out.size() < target_chars) {
        out.append(kWords[index % kWordCount]);
        out.push_back(' ');
        ++index;
    }
    return out;
}

std::string tool_json(std::size_t index, std::size_t fill_chars) {
    return std::string("{\"type\":\"function\",\"function\":{\"name\":\"tool_") +
           std::to_string(index) +
           "\",\"description\":\"" + filler(fill_chars, index) +
           "\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},"
           "\"limit\":{\"type\":\"integer\"}},\"required\":[\"path\"]}}}";
}

// A coding agent's tool-call arguments are not small: a write or an edit carries file content, and
// the render parses `arguments_json` for every call on every render.
std::string call_arguments(std::size_t fill_chars, std::size_t seed) {
    return "{\"path\":\"/src/arena.cpp\",\"content\":\"" + filler(fill_chars, seed) + "\"}";
}

fi::ChatMessage text_message(ninfer::ChatRole role, std::string content) {
    fi::ChatMessage message;
    message.role = role;
    message.parts.push_back(fi::ChatPart::text_part(std::move(content)));
    return message;
}

// The agent-loop shape: a system turn, then repeating user / assistant-with-tool-call / tool result /
// assistant-answer groups. Assistant turns carry reasoning content because preservation is on for
// this workload and the template's `<think>` emission depends on it.
std::vector<fi::ChatMessage> build_conversation(const Options& options, std::size_t message_count) {
    std::vector<fi::ChatMessage> messages;
    messages.reserve(message_count);
    messages.push_back(text_message(ninfer::ChatRole::System,
                                    "You are a coding agent.\n" + filler(options.chars_per_message, 0)));
    std::size_t group = 0;
    while (messages.size() < message_count) {
        messages.push_back(
            text_message(ninfer::ChatRole::User, filler(options.chars_per_message, 1 + group)));
        if (messages.size() >= message_count) { break; }

        fi::ChatMessage assistant = text_message(ninfer::ChatRole::Assistant, std::string{});
        assistant.reasoning_content = filler(options.chars_per_message / 2, 2 + group);
        fi::ToolCall call;
        call.id              = "call_" + std::to_string(group);
        call.name            = "write_file";
        call.arguments_json  = call_arguments(options.call_args_chars, 6 + group);
        assistant.tool_calls.push_back(std::move(call));
        messages.push_back(std::move(assistant));
        if (messages.size() >= message_count) { break; }

        fi::ChatMessage tool = text_message(
            ninfer::ChatRole::Tool, "<tool_response>\n" + filler(options.chars_per_message, 3 + group) +
                                        "\n</tool_response>");
        tool.tool_call_id = "call_" + std::to_string(group);
        messages.push_back(std::move(tool));
        if (messages.size() >= message_count) { break; }

        fi::ChatMessage answer        = text_message(ninfer::ChatRole::Assistant,
                                                     filler(options.chars_per_message, 4 + group));
        answer.reasoning_content      = filler(options.chars_per_message / 2, 5 + group);
        messages.push_back(std::move(answer));
        ++group;
    }
    messages.resize(message_count);
    if (!options.tail_assistant) {
        // The checkpoint probe is gated on a closed assistant message after the last real user turn.
        // Ending on a user turn removes that condition without shortening the conversation.
        messages.push_back(text_message(ninfer::ChatRole::User, "status?"));
    }
    return messages;
}

double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

std::size_t resolved_boundaries(const fi::RenderedChat& rendered) {
    return static_cast<std::size_t>(std::count_if(rendered.message_boundaries.begin(),
                                                  rendered.message_boundaries.end(),
                                                  [](const auto& value) { return value.has_value(); }));
}

std::size_t total_bytes(const std::vector<fi::ChatMessage>& messages) {
    std::size_t bytes = 0;
    for (const auto& message : messages) {
        for (const auto& part : message.parts) { bytes += part.text.size(); }
        bytes += message.reasoning_content.size();
    }
    return bytes;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse_options(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        print_usage(argv[0]);
        return 2;
    }

    try {
        const std::string source = read_file(options.template_path);
        const fi::CompiledChatTemplate compiled =
            fi::CompiledChatTemplate::resolve(source, options.template_path.string());

        std::vector<std::string> tool_jsons;
        for (std::size_t i = 0; i < options.tool_count; ++i) {
            tool_jsons.push_back(tool_json(i, 400));
        }

        fi::ChatRenderOptions render_options;
        render_options.add_generation_prompt = options.generation_prompt;
        render_options.enable_thinking       = false;
        render_options.preserve_thinking     = true;
        render_options.tool_jsons            = tool_jsons;

        // The interpreter calls its checkpoint once per executed statement. In production that
        // callback ends in an httplib socket probe, so this arm counts how often it would fire.
        std::size_t checkpoints = 0;
        ninfer::PreparationControl control;
        if (options.cancel_probe) {
            control.cancellation =
                ninfer::CancellationView([&checkpoints] { ++checkpoints; return false; });
        }

        std::cout << "template            " << options.template_path.string() << "\n";
        std::cout << "chars per message   " << options.chars_per_message << "\n";
        std::cout << "tool-call args      " << options.call_args_chars << " chars\n";
        std::cout << "tools               " << options.tool_count << "\n";
        std::cout << "generation prompt   " << (options.generation_prompt ? "on" : "off") << "\n";
        std::cout << "tail assistant      " << (options.tail_assistant ? "on" : "off") << "\n";
        std::cout << "cancel probe        " << (options.cancel_probe ? "on" : "off") << "\n";
        std::cout << "\n";
        std::cout << "messages   bytes   render ms   ms/message   boundaries   text bytes   checkpoints\n";

        for (const std::size_t count : options.sweep) {
            const std::vector<fi::ChatMessage> messages = build_conversation(options, count);

            fi::RenderedChat rendered;
            for (int i = 0; i < options.warmup; ++i) {
                rendered = compiled.render(messages, render_options, control);
            }

            std::vector<double> samples;
            std::vector<std::size_t> probes;
            samples.reserve(static_cast<std::size_t>(options.repetitions));
            for (int i = 0; i < options.repetitions; ++i) {
                checkpoints        = 0;
                const auto started = Clock::now();
                rendered           = compiled.render(messages, render_options, control);
                samples.push_back(milliseconds(Clock::now() - started));
                probes.push_back(checkpoints);
            }
            const double best = *std::min_element(samples.begin(), samples.end());
            const std::size_t probe_count = *std::max_element(probes.begin(), probes.end());

            std::printf("%8zu %7zu %11.2f %12.4f %12zu %12zu %13zu\n", messages.size(),
                        total_bytes(messages), best,
                        best / static_cast<double>(messages.size()), resolved_boundaries(rendered),
                        rendered.text.size(), probe_count);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "chat render bench failed: " << error.what() << "\n";
        return 1;
    }
}
