#include "text/grammar.h"
#include "text/json_schema.h"

#include <xgrammar/xgrammar.h>
#include "grammar_impl.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::text {
namespace {

std::string quoted(unsigned char c) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result   = "\"\\x00\"";
    result[3]            = hex[c >> 4];
    result[4]            = hex[c & 15];
    return result;
}

// A small DFA consumes through the first complete delimiter, including delimiters split across
// tokens. Concatenation keeps the user's rule names in their own namespace.
xgrammar::Grammar reasoning_prefix(std::string_view delimiter) {
    std::string alphabet(delimiter);
    std::sort(alphabet.begin(), alphabet.end());
    alphabet.erase(std::unique(alphabet.begin(), alphabet.end()), alphabet.end());
    std::string source = "root ::= s0\n";
    for (std::size_t state = 0; state < delimiter.size(); ++state) {
        source += "s" + std::to_string(state) + " ::= ";
        for (unsigned char c : alphabet) {
            std::string next(delimiter.substr(0, state));
            next += static_cast<char>(c);
            std::size_t matched = std::min(next.size(), delimiter.size());
            while (matched && !next.ends_with(delimiter.substr(0, matched))) { --matched; }
            source += quoted(c);
            if (matched != delimiter.size()) { source += " s" + std::to_string(matched); }
            source += " | ";
        }
        source += "[^";
        for (unsigned char c : alphabet) {
            const auto escaped = quoted(c);
            source += escaped.substr(1, escaped.size() - 2);
        }
        source += "] s0\n";
    }
    return xgrammar::Grammar::FromEBNF(source);
}

void validate(const xgrammar::Grammar& grammar, const std::vector<std::string>& vocab,
              const std::vector<std::int32_t>& eos) {
    for (int i = 0; i < grammar->NumRules(); ++i) {
        const auto& rule = grammar->GetRule(i);
        if (rule.max_tokens >= 0 || rule.max_chars >= 0 || !rule.capture_name.empty() ||
            rule.is_lazy || rule.temperature || grammar->GetSuffixStopInfo(i)) {
            throw std::invalid_argument(
                "GBNF rules cannot set budgets, captures, lazy matching or temperature");
        }
    }
    using Type = xgrammar::Grammar::Impl::GrammarExprType;
    for (int i = 0; i < grammar->NumGrammarExprs(); ++i) {
        const auto expression = grammar->GetGrammarExpr(i);
        if (expression.type == Type::kTagDispatch || expression.type == Type::kTokenTagDispatch ||
            expression.type == Type::kRegex || expression.type == Type::kSubstring) {
            throw std::invalid_argument("grammar uses an extension outside the GBNF contract");
        }
        if (expression.type == Type::kToken || expression.type == Type::kExcludeToken) {
            for (int id : expression) {
                if (id < 0 || static_cast<std::size_t>(id) >= vocab.size() || vocab[id].empty() ||
                    std::find(eos.begin(), eos.end(), id) != eos.end()) {
                    throw std::invalid_argument(
                        "GBNF token references must name ordinary vocabulary tokens");
                }
            }
        }
    }
}
} // namespace

class GrammarCompiler::Impl {
public:
    Impl(std::vector<std::string> vocab_, std::vector<std::int32_t> eos_, std::size_t bytes)
        : vocab(std::move(vocab_)), eos(std::move(eos_)),
          tokenizer(vocab, xgrammar::VocabType::RAW, static_cast<int>(vocab.size()), eos),
          compiler(tokenizer, 1, true, static_cast<std::int64_t>(bytes)) {}

    std::vector<std::string> vocab;
    std::vector<std::int32_t> eos;
    xgrammar::TokenizerInfo tokenizer;
    xgrammar::GrammarCompiler compiler;
};

class GrammarSession::Impl {
public:
    Impl(const xgrammar::CompiledGrammar& grammar, int vocab)
        : matcher(grammar), vocabulary(vocab), words((vocab + 31) / 32) {}

    xgrammar::GrammarMatcher matcher;
    int vocabulary;
    std::size_t words;
    int tentative = 0;

    void fill(std::span<std::uint32_t> target) {
        if (matcher.IsTerminated()) {
            std::fill(target.begin(), target.end(), 0);
            return;
        }
        std::int64_t shape[] = {1, static_cast<std::int64_t>(words)};
        DLTensor tensor{target.data(), {kDLCPU, 0}, 2, {kDLInt, 32, 1}, shape, nullptr, 0};
        matcher.FillNextTokenBitmask(&tensor);
        if (vocabulary % 32) { target.back() &= (1u << (vocabulary % 32)) - 1u; }
    }
};

GrammarCompiler::GrammarCompiler(std::vector<std::string> vocab, std::vector<std::int32_t> eos,
                                 std::size_t bytes)
    : impl_(std::make_unique<Impl>(std::move(vocab), std::move(eos), bytes)) {}

GrammarCompiler::~GrammarCompiler() = default;

std::unique_ptr<GrammarSession> GrammarCompiler::compile(const OutputConstraint& constraint,
                                                         std::string_view close,
                                                         std::string_view continuation) {
    try {
        const std::string key = std::to_string(static_cast<int>(constraint.kind)) + ":" +
                                std::to_string(close.size()) + ":" + std::string(close) +
                                constraint.source;
        auto compiled = impl_->compiler.CompileCachedGrammar(key, [&] {
            auto grammar = [&]() -> xgrammar::Grammar {
                if (constraint.kind == OutputConstraintKind::Grammar) {
                    auto parsed = xgrammar::Grammar::FromEBNF(constraint.source, "root");
                    validate(parsed, impl_->vocab, impl_->eos);
                    return parsed;
                } else {
                    std::string source;
                    if (constraint.kind == OutputConstraintKind::JsonObject) {
                        if (!constraint.source.empty())
                            throw RequestError(RequestErrorKind::InvalidJsonSchema,
                                               "JSON object mode has no source payload");
                        source = R"({"type":"object"})";
                    } else if (constraint.kind == OutputConstraintKind::JsonSchema) {
                        source = prepare_json_schema(constraint.source);
                    } else {
                        throw RequestError(RequestErrorKind::InvalidJsonSchema,
                                           "unknown output constraint kind");
                    }
                    return xgrammar::Grammar::FromJSONSchema(
                        source, false, std::nullopt, std::pair<std::string, std::string>{",", ":"},
                        false, std::nullopt, false, false);
                }
            }();
            if (!close.empty())
                grammar = xgrammar::Grammar::Concat({reasoning_prefix(close), grammar});
            return grammar;
        });
        auto session =
            std::make_unique<GrammarSession::Impl>(compiled, static_cast<int>(impl_->vocab.size()));
        if (!continuation.empty() && !session->matcher.AcceptString(std::string(continuation))) {
            throw std::invalid_argument("assistant continuation is not a prefix of the grammar");
        }
        return std::unique_ptr<GrammarSession>(new GrammarSession(std::move(session)));
    } catch (const xgrammar::JSONSchemaCompileError& error) {
        RequestErrorKind kind = RequestErrorKind::InvalidJsonSchema;
        if (error.kind == xgrammar::SchemaErrorType::kUnsupportedSchema)
            kind = RequestErrorKind::UnsupportedJsonSchema;
        else if (error.kind == xgrammar::SchemaErrorType::kUnsatisfiableSchema)
            kind = RequestErrorKind::UnsatisfiableJsonSchema;
        throw RequestError(kind,
                           std::string(error.what()) + " at " +
                               (error.pointer.empty() ? "/" : error.pointer),
                           error.pointer);
    } catch (const xgrammar::LogFatalError& error) {
        if (constraint.kind == OutputConstraintKind::Grammar)
            throw std::invalid_argument(error.what());
        throw RequestError(RequestErrorKind::InvalidJsonSchema, error.what());
    }
}

GrammarSession::GrammarSession(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

GrammarSession::~GrammarSession()                                    = default;
GrammarSession::GrammarSession(GrammarSession&&) noexcept            = default;
GrammarSession& GrammarSession::operator=(GrammarSession&&) noexcept = default;

std::size_t GrammarSession::mask_words() const noexcept { return impl_->words; }

std::uint32_t GrammarSession::masks(std::span<const std::int32_t> drafts,
                                    std::span<std::uint32_t> words) {
    if (drafts.size() >= 32 || impl_->tentative ||
        words.size() != (drafts.size() + 1) * impl_->words) {
        throw std::logic_error("invalid grammar lookahead transaction");
    }
    int advanced       = 0;
    bool reachable     = true;
    std::uint32_t dead = 0;
    try {
        for (std::size_t position = 0; position <= drafts.size(); ++position) {
            auto mask = words.subspan(position * impl_->words, impl_->words);
            if (reachable && !impl_->matcher.IsTerminated()) {
                impl_->fill(mask);
                if (std::all_of(mask.begin(), mask.end(), [](auto word) { return word == 0; })) {
                    dead |= 1u << position;
                    mask[0] = 1;
                }
            } else {
                std::fill(mask.begin(), mask.end(), 0);
                mask[0] = 1;
            }
            if (position == drafts.size()) { break; }
            if (reachable && !impl_->matcher.IsTerminated() &&
                impl_->matcher.AcceptToken(drafts[position])) {
                ++advanced;
            } else {
                reachable = false;
            }
        }
    } catch (...) {
        if (advanced) { impl_->matcher.Rollback(advanced); }
        throw;
    }
    if (advanced) { impl_->matcher.Rollback(advanced); }
    return dead;
}

void GrammarSession::accept(std::int32_t token) {
    if (impl_->matcher.IsTerminated() || !impl_->matcher.AcceptToken(token)) {
        throw std::logic_error("selected token violates the request grammar");
    }
    ++impl_->tentative;
}

void GrammarSession::confirm() noexcept { impl_->tentative = 0; }

void GrammarSession::discard() {
    if (impl_->tentative) { impl_->matcher.Rollback(impl_->tentative); }
    impl_->tentative = 0;
}
} // namespace ninfer::text
