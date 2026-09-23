#include "models/qwen3_5/frontend/native_render.h"

#include <array>
#include <stdexcept>

namespace ninfer::models::qwen3_5::frontend {

namespace {

// One entry per transcribed template, and the whole registry is the list: a template that differs by
// even one byte is not in it and takes the Jinja path.
struct Registration {
    Sha256Digest digest;
};

constexpr std::array<Registration, 0> kRegistrations{};

} // namespace

bool native_render_supported(const Sha256Digest& template_digest) {
    for (const Registration& registration : kRegistrations) {
        if (registration.digest == template_digest) { return true; }
    }
    return false;
}

RenderedChat render_native(const std::vector<ChatMessage>& messages,
                           const ChatRenderOptions& options) {
    // Unreachable while the registry is empty, and guarded rather than silent so that a caller which
    // skips `native_render_supported` fails loudly instead of receiving a partial render.
    (void)messages;
    (void)options;
    throw std::logic_error("no native renderer is registered for this template digest");
}

} // namespace ninfer::models::qwen3_5::frontend
