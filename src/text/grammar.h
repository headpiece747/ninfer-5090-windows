#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::text {

// One immutable vocabulary/compiler per model; one transactional matcher per request.
class GrammarSession {
public:
    ~GrammarSession();
    GrammarSession(GrammarSession&&) noexcept;
    GrammarSession& operator=(GrammarSession&&) noexcept;
    [[nodiscard]] std::uint32_t masks(std::span<const std::int32_t> drafts,
                                      std::span<std::uint32_t> words);
    void accept(std::int32_t token);
    void confirm() noexcept;
    void discard();
    [[nodiscard]] std::size_t mask_words() const noexcept;

private:
    class Impl;
    explicit GrammarSession(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class GrammarCompiler;
};

class GrammarCompiler {
public:
    // Empty entries explicitly identify forbidden/special IDs. EOS is supplied separately.
    GrammarCompiler(std::vector<std::string> vocabulary, std::vector<std::int32_t> eos,
                    std::size_t cache_bytes);
    ~GrammarCompiler();
    [[nodiscard]] std::unique_ptr<GrammarSession> compile(const OutputConstraint& constraint,
                                                          std::string_view reasoning_close,
                                                          std::string_view continuation);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::text
