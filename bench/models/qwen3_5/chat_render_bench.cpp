// Host-only: what one prompt preparation costs, by conversation length.
//
// The frontend renders a chat template more than once per request. `CompiledChatTemplate::render`
// runs the output render, and then, to prove which byte offsets are safe prefix-cache boundaries,
// it runs `prefix(...)` probes that each re-render a variant of the whole conversation
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
//          [--chars <chars per message>] [--call-args <n>] [--tools <n>] [--warmup <n>] [--reps
//          <n>]
//          [--generation-prompt on|off] [--tail-assistant on|off] [--special-tokens]
//          [--thinking-default] [--cancel-probe] [--from <body.json>] [--native] [--quiet]
//
// `--from` replays a recorded chat-completions body, which is how a reading can be taken of the
// same conversation a server rendered rather than of a conversation that merely looks like it.

#include "models/qwen3_5/frontend/chat_template.h"
#include "models/qwen3_5/frontend/native_render.h"

#include <algorithm>
#include <atomic>
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
#include <thread>
#include <vector>

namespace {

namespace fi = ninfer::models::qwen3_5::frontend;
using Clock  = std::chrono::steady_clock;

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
    // The server resolves the template with the artifact's special tokens, which makes those names
    // control variables rather than ordinary inputs; the bench passes none unless asked.
    bool special_tokens = false;
    // The server does not send enable_thinking, so the template's own default applies.
    bool thinking_default = false;
    // Install a cancellation callback so the interpreter's per-statement checkpoint runs, as it
    // does in production where the callback is an httplib socket probe. The arm reports how often
    // it fires.
    bool cancel_probe = false;
    // Replay a recorded request body instead of synthesizing a conversation. The point is to render
    // the conversation a server has already rendered, so the two readings are of the same input.
    std::filesystem::path from_path;
    // Background threads that allocate and free, to test whether the process's global heap is what
    // makes an otherwise identical measurement slow: the NT heap is process-wide and serialized.
    std::size_t noise_threads = 0;
    // Report a neutral allocation loop next to the render, so the process's own allocation cost is
    // visible. This is the loop that showed the serving process ~40x slower than a fresh one.
    bool calibrate = false;
    // Render the conversation through the native renderer as well and report the first byte that
    // differs. This is the differential loop ADR-0012 is built against; the control it prints first
    // validates the instrument before any comparison means anything.
    bool native = false;
    bool quiet     = false;};

void print_usage(const char* executable) {
    std::cout << "usage: " << executable
              << " [--template <path>] [--sweep <n,n,...>] [--chars <n>] [--call-args <n>]"
                 " [--tools <n>] [--warmup <n>] [--reps <n>] [--generation-prompt on|off]"
                 " [--tail-assistant on|off] [--cancel-probe] [--special-tokens]"
                 " [--thinking-default] [--from <body.json>] [--noise-threads <n>] [--calibrate]"
                 " [--native] [--quiet]\n";
}

bool parse_bool(std::string_view text) {
    if (text == "on" || text == "true" || text == "1") { return true; }
    if (text == "off" || text == "false" || text == "0") { return false; }
    throw std::invalid_argument("expected on or off, got " + std::string(text));
}

std::size_t parse_size(const char* text, const char* label) {
    std::size_t consumed = 0;
    const auto value     = std::stoull(text, &consumed);
    if (text[consumed] != '\0') {
        throw std::invalid_argument(std::string(label) + " is not a count");
    }
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
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + label);
            }
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
        } else if (flag == "--from") {
            options.from_path = next("--from");
        } else if (flag == "--noise-threads") {
            options.noise_threads = parse_size(next("--noise-threads"), "--noise-threads");
        } else if (flag == "--calibrate") {
            options.calibrate = true;
        } else if (flag == "--special-tokens") {
            options.special_tokens = true;
        } else if (flag == "--thinking-default") {
            options.thinking_default = true;
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
        } else if (flag == "--native") {
            options.native = true;
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
    if (!file.read(source.data(), size)) {
        throw std::invalid_argument("cannot read " + path.string());
    }
    return source;
}

// Deterministic filler. The render cost follows bytes, not tokens, so a character budget is the
// honest parameter.
std::string filler(std::size_t target_chars, std::size_t seed) {
    static constexpr std::string_view kWords[] = {
        "the",    "resolver",  "reads", "each",  "pending",   "worklist", "entry",  "and",
        "writes", "back",      "a",     "dirty", "flag",      "before",   "yield",  "to",
        "the",    "scheduler", "which", "then",  "coalesces", "adjacent", "runs",   "in",
        "one",    "pass",      "over",  "the",   "arena",     "without",  "copying"};
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
           std::to_string(index) + "\",\"description\":\"" + filler(fill_chars, index) +
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

// The agent-loop shape: a system turn, then repeating user / assistant-with-tool-call / tool result
// / assistant-answer groups. Assistant turns carry reasoning content because preservation is on for
// this workload and the template's `<think>` emission depends on it.
std::vector<fi::ChatMessage> build_conversation(const Options& options, std::size_t message_count) {
    std::vector<fi::ChatMessage> messages;
    messages.reserve(message_count);
    messages.push_back(
        text_message(ninfer::ChatRole::System,
                     "You are a coding agent.\n" + filler(options.chars_per_message, 0)));
    std::size_t group = 0;
    while (messages.size() < message_count) {
        messages.push_back(
            text_message(ninfer::ChatRole::User, filler(options.chars_per_message, 1 + group)));
        if (messages.size() >= message_count) { break; }

        fi::ChatMessage assistant   = text_message(ninfer::ChatRole::Assistant, std::string{});
        assistant.reasoning_content = filler(options.chars_per_message / 2, 2 + group);
        fi::ToolCall call;
        call.id             = "call_" + std::to_string(group);
        call.name           = "write_file";
        call.arguments_json = call_arguments(options.call_args_chars, 6 + group);
        assistant.tool_calls.push_back(std::move(call));
        messages.push_back(std::move(assistant));
        if (messages.size() >= message_count) { break; }

        fi::ChatMessage tool =
            text_message(ninfer::ChatRole::Tool, "<tool_response>\n" +
                                                     filler(options.chars_per_message, 3 + group) +
                                                     "\n</tool_response>");
        tool.tool_call_id = "call_" + std::to_string(group);
        messages.push_back(std::move(tool));
        if (messages.size() >= message_count) { break; }

        fi::ChatMessage answer =
            text_message(ninfer::ChatRole::Assistant, filler(options.chars_per_message, 4 + group));
        answer.reasoning_content = filler(options.chars_per_message / 2, 5 + group);
        messages.push_back(std::move(answer));
        ++group;
    }
    messages.resize(message_count);
    if (!options.tail_assistant) {
        // The checkpoint probe is gated on a closed assistant message after the last real user
        // turn. Ending on a user turn removes that condition without shortening the conversation.
        messages.push_back(text_message(ninfer::ChatRole::User, "status?"));
    }
    return messages;
}

double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

std::size_t resolved_boundaries(const fi::RenderedChat& rendered) {
    return static_cast<std::size_t>(
        std::count_if(rendered.message_boundaries.begin(), rendered.message_boundaries.end(),
                      [](const auto& value) { return value.has_value(); }));
}

// Replay a recorded OpenAI chat-completions body. The point is to render the conversation the
// server has already rendered, so the two measurements are of the same input rather than of two
// conversations that merely look alike.
struct RecordedRequest {
    std::vector<fi::ChatMessage> messages;
    std::vector<std::string> tool_jsons;
};

ninfer::ChatRole role_from(std::string_view name) {
    if (name == "system") { return ninfer::ChatRole::System; }
    if (name == "developer") { return ninfer::ChatRole::Developer; }
    if (name == "user") { return ninfer::ChatRole::User; }
    if (name == "assistant") { return ninfer::ChatRole::Assistant; }
    if (name == "tool") { return ninfer::ChatRole::Tool; }
    throw std::invalid_argument("unknown role " + std::string(name));
}

RecordedRequest load_recorded_request(const std::filesystem::path& path) {
    const nlohmann::json body = nlohmann::json::parse(read_file(path));
    RecordedRequest recorded;
    for (const auto& entry : body.at("messages")) {
        fi::ChatMessage message;
        message.role = role_from(entry.at("role").get<std::string>());
        if (entry.contains("content")) {
            const auto& content = entry.at("content");
            if (content.is_string()) {
                message.parts.push_back(fi::ChatPart::text_part(content.get<std::string>()));
            } else if (content.is_array()) {
                for (const auto& part : content) {
                    if (part.contains("text")) {
                        message.parts.push_back(
                            fi::ChatPart::text_part(part.at("text").get<std::string>()));
                    }
                }
            }
        }
        if (entry.contains("reasoning_content") && entry.at("reasoning_content").is_string()) {
            message.reasoning_content = entry.at("reasoning_content").get<std::string>();
        }
        if (entry.contains("tool_call_id") && entry.at("tool_call_id").is_string()) {
            message.tool_call_id = entry.at("tool_call_id").get<std::string>();
        }
        if (entry.contains("tool_calls")) {
            for (const auto& call : entry.at("tool_calls")) {
                fi::ToolCall parsed;
                parsed.id             = call.value("id", std::string{});
                parsed.name           = call.at("function").value("name", std::string{});
                const auto& arguments = call.at("function").at("arguments");
                parsed.arguments_json =
                    arguments.is_string() ? arguments.get<std::string>() : arguments.dump();
                message.tool_calls.push_back(std::move(parsed));
            }
        }
        recorded.messages.push_back(std::move(message));
    }
    if (body.contains("tools")) {
        for (const auto& tool : body.at("tools")) { recorded.tool_jsons.push_back(tool.dump()); }
    }
    return recorded;
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

// The differential loop ADR-0012 is built against. Its control -- the Jinja render compared with
// itself -- validates the instrument before any comparison means anything, and what it reports is the
// first byte that differs, so a native renderer can be built against an exact target rather than a
// reading of the template.
void report_native_comparison(const fi::CompiledChatTemplate& compiled,
                              const std::vector<fi::ChatMessage>& messages,
                              const fi::ChatRenderOptions& render_options,
                              const ninfer::PreparationControl& control, std::string_view source) {
    const auto first_difference = [](std::string_view lhs, std::string_view rhs) {
        const std::size_t limit = std::min(lhs.size(), rhs.size());
        for (std::size_t i = 0; i < limit; ++i) {
            if (lhs[i] != rhs[i]) { return std::optional<std::size_t>(i); }
        }
        return lhs.size() == rhs.size() ? std::nullopt : std::optional<std::size_t>(limit);
    };
    const fi::RenderedChat jinja = compiled.render(messages, render_options, control);
    const auto self              = first_difference(jinja.text, jinja.text);
    std::cout << "native control      : "
              << (self ? "differs at byte " + std::to_string(*self) : std::string("identical"))
              << " (jinja against itself)\n";

    const fi::Sha256Digest digest = fi::sha256(source);
    const bool registered         = fi::native_render_supported(digest);
    std::cout << "native registered   : " << (registered ? "yes" : "no") << " (template digest "
              << fi::sha256_hex(digest) << ")\n";
    if (!registered) { return; }

    const fi::RenderedChat native = fi::render_native(messages, render_options);
    const auto offset             = first_difference(jinja.text, native.text);
    std::cout << "native compare      : "
              << (offset ? "differs at byte " + std::to_string(*offset)
                         : std::string("identical"))
              << " (jinja " << jinja.text.size() << " bytes, native " << native.text.size()
              << " bytes)\n";
}

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
        // compile_chat_template builds this from the artifact's tokenizer_config.json over exactly
        // these keys; the values only have to be present for the control-variable mode to match.
        nlohmann::ordered_json special_tokens = nlohmann::ordered_json::object();
        if (options.special_tokens) {
            for (const char* key : {"bos_token", "eos_token", "pad_token", "unk_token", "sep_token",
                                    "cls_token", "mask_token"}) {
                special_tokens[key] = "<|endoftext|>";
            }
            special_tokens["additional_special_tokens"] = nlohmann::ordered_json::array(
                {"<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
                 "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>"});
        }
        const fi::CompiledChatTemplate compiled = fi::CompiledChatTemplate::resolve(
            source, options.template_path.string(), std::move(special_tokens));

        std::vector<std::string> tool_jsons;
        for (std::size_t i = 0; i < options.tool_count; ++i) {
            tool_jsons.push_back(tool_json(i, 400));
        }

        std::vector<fi::ChatMessage> recorded_messages;
        if (!options.from_path.empty()) {
            RecordedRequest recorded = load_recorded_request(options.from_path);
            recorded_messages        = std::move(recorded.messages);
            tool_jsons               = std::move(recorded.tool_jsons);
            options.sweep            = {recorded_messages.size()};
        }

        // Allocate and free on other threads for the duration of the measurement. The NT heap is
        // process-wide and every malloc/free takes its lock, so if that is what a long-lived
        // multi-threaded process pays, this reproduces it in a process that has nothing else in it.
        std::atomic<bool> noise_stop{false};
        std::vector<std::thread> noise;
        for (std::size_t i = 0; i < options.noise_threads; ++i) {
            noise.emplace_back([&noise_stop] {
                std::vector<std::string> keep;
                while (!noise_stop.load(std::memory_order_relaxed)) {
                    keep.clear();
                    for (int j = 0; j < 2000; ++j) { keep.emplace_back(64, 'x'); }
                }
            });
        }
        const auto stop_noise = [&noise_stop, &noise] {
            noise_stop.store(true, std::memory_order_relaxed);
            for (std::thread& thread : noise) { thread.join(); }
        };

        // A neutral allocation loop, run while the noise threads are running, so the process's
        // allocation cost is reported next to the render's under the same contention. The checksum
        // is printed so a loop that was optimised away or mis-sized is visible rather than fast.
        if (options.calibrate) {
            const auto started = Clock::now();
            std::size_t sum    = 0;
            for (int i = 0; i < 20000; ++i) {
                std::string text(64, 'x');
                text += std::to_string(i);
                sum += text.size();
            }
            const double ms =
                std::chrono::duration<double, std::milli>(Clock::now() - started).count();
            std::printf("calibration %.3f ms (checksum %zu)\n", ms, sum);
        }

        fi::ChatRenderOptions render_options;
        render_options.add_generation_prompt = options.generation_prompt;
        if (!options.thinking_default) { render_options.enable_thinking = false; }
        render_options.preserve_thinking = true;
        render_options.tool_jsons        = tool_jsons;

        // The interpreter calls its checkpoint once per executed statement. In production that
        // callback ends in an httplib socket probe, so this arm counts how often it would fire.
        std::size_t checkpoints = 0;
        ninfer::PreparationControl control;
        if (options.cancel_probe) {
            control.cancellation = ninfer::CancellationView([&checkpoints] {
                ++checkpoints;
                return false;
            });
        }

        std::cout << "template            " << options.template_path.string() << "\n";
        std::cout << "chars per message   " << options.chars_per_message << "\n";
        std::cout << "tool-call args      " << options.call_args_chars << " chars\n";
        std::cout << "tools               " << options.tool_count << "\n";
        std::cout << "generation prompt   " << (options.generation_prompt ? "on" : "off") << "\n";
        std::cout << "tail assistant      " << (options.tail_assistant ? "on" : "off") << "\n";
        std::cout << "special tokens      " << (options.special_tokens ? "on" : "off") << "\n";
        std::cout << "enable_thinking     "
                  << (options.thinking_default ? "template default" : "off") << "\n";
        std::cout << "cancel probe        " << (options.cancel_probe ? "on" : "off") << "\n";
        std::cout << "noise threads       " << options.noise_threads << "\n";
        if (options.native) {
            report_native_comparison(compiled, build_conversation(options, options.sweep.front()),
                                     render_options, control, source);
        }
        std::cout << "\n";
        std::cout << "messages   bytes   render ms   ms/message   boundaries   text bytes   "
                     "checkpoints\n";

        for (const std::size_t count : options.sweep) {
            const std::vector<fi::ChatMessage> messages =
                recorded_messages.empty() ? build_conversation(options, count) : recorded_messages;

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
            const double best             = *std::min_element(samples.begin(), samples.end());
            const std::size_t probe_count = *std::max_element(probes.begin(), probes.end());

            std::printf("%8zu %7zu %11.2f %12.4f %12zu %12zu %13zu\n", messages.size(),
                        total_bytes(messages), best, best / static_cast<double>(messages.size()),
                        resolved_boundaries(rendered), rendered.text.size(), probe_count);
        }
        stop_noise();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "chat render bench failed: " << error.what() << "\n";
        return 1;
    }
}
