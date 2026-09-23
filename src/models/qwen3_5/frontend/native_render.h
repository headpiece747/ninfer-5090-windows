#pragma once

#include "models/qwen3_5/frontend/chat_template.h"
#include "models/qwen3_5/frontend/digest.h"

#include <vector>

namespace ninfer::models::qwen3_5::frontend {

// The registered template's serialization, written out in C++ instead of interpreted. See
// `docs/adr/0012-native-render-for-the-registered-template.md`.
//
// A native renderer is registered for one template *source* digest. `native_render_supported` is the
// gate: on any other source the answer is false and the Jinja path renders exactly as it always has,
// so a template this port has not transcribed costs speed and never correctness.
[[nodiscard]] bool native_render_supported(const Sha256Digest& template_digest);

// Renders the registered template. Preconditions: `native_render_supported` is true for the digest of
// the source this template was resolved from, and `messages` is non-empty.
//
// Returns the same `RenderedChat` the Jinja path returns for the same inputs, including every
// boundary. Those are known here by construction rather than derived: this path knows which bytes it
// took from which message, where each message's serialization ends, and where the generation prompt
// begins. The artifact's special tokens are not an input, because the registered template's output
// does not depend on them -- the differential loop renders the Jinja side with them and this side
// without, so an accidental dependency appears as a difference rather than passing unnoticed.
[[nodiscard]] RenderedChat render_native(const std::vector<ChatMessage>& messages,
                                         const ChatRenderOptions& options);

} // namespace ninfer::models::qwen3_5::frontend
