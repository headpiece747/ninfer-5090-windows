#include "json_string_grammar.h"

#include <picojson.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "grammar_functor.h"
#include "support/encoding.h"
#include "support/json_parse.h"

namespace xgrammar {
namespace {
std::string encode(const std::string& value) {
  auto json = picojson::value(value).serialize(false);
  return json.substr(1, json.size() - 2);
}

class JSONEncoder : public GrammarMutator {
  int32_t VisitByteString(const GrammarExpr& expr) final {
    std::string value;
    for (int byte : expr) value += static_cast<char>(byte);
    return builder_->AddByteString(encode(value));
  }

  int32_t VisitCharacterClass(const GrammarExpr& expr) final {
    std::vector<std::pair<int32_t, int32_t>> ranges;
    for (int i = 1; i < expr.size(); i += 2) ranges.emplace_back(expr[i], expr[i + 1]);
    if (expr[0]) {
      std::sort(ranges.begin(), ranges.end());
      std::vector<std::pair<int32_t, int32_t>> complement;
      int next = 0;
      for (const auto& [lo, hi] : ranges) {
        if (next < lo) complement.emplace_back(next, lo - 1);
        next = std::max(next, hi + 1);
      }
      if (next <= 0x10ffff) complement.emplace_back(next, 0x10ffff);
      ranges = std::move(complement);
    }
    std::vector<GrammarBuilder::CharacterClassElement> raw;
    std::vector<int32_t> choices;
    // ASCII control bytes, quote and backslash need JSON escapes. Surrogates are not scalar values.
    const std::pair<int, int> safe[] = {
        {0x20, 0x21}, {0x23, 0x5b}, {0x5d, 0xd7ff}, {0xe000, 0x10ffff}};
    for (const auto& [lo, hi] : ranges) {
      for (const auto& [a, b] : safe)
        if (std::max(lo, a) <= std::min(hi, b)) raw.emplace_back(std::max(lo, a), std::min(hi, b));
      for (int c = std::max(lo, 0); c <= std::min(hi, 0x5c); ++c) {
        if (c < 0x20 || c == '"' || c == '\\')
          choices.push_back(builder_->AddByteString(encode(std::string(1, static_cast<char>(c)))));
      }
    }
    if (!raw.empty()) choices.push_back(builder_->AddCharacterClass(raw));
    if (choices.empty()) return builder_->AddCharacterClass({{0, 0x10ffff}}, true);
    return choices.size() == 1 ? choices[0] : builder_->AddChoices(choices);
  }

  int32_t VisitCharacterClassStar(const GrammarExpr& expr) final {
    return builder_->AddRepeatFromExpr("json_char", VisitCharacterClass(expr), 0, -1);
  }
};

// JSON Schema uses ECMAScript character classes. Normalize the few classes whose language
// differs from the vendor regex dialect, and reject escapes it would only warn about.
std::string schema_characters(const std::string& pattern) {
  const std::string space =
      R"(\u0009-\u000d\u0020\u00a0\u1680\u2000-\u200a\u2028-\u2029\u202f\u205f\u3000\ufeff)";
  const std::string nonspace =
      R"(\u0000-\u0008\u000e-\u001f\u0021-\u009f\u00a1-\u167f\u1681-\u1fff\u200b-\u2027\u202a-\u202e\u2030-\u205e\u2060-\u2fff\u3001-\ufefe\uff00-\u{10ffff})";
  std::string result;
  bool in_class = false;
  for (size_t i = 0; i < pattern.size(); ++i) {
    char c = pattern[i];
    if (c == '\\') {
      if (++i == pattern.size()) throw std::invalid_argument("unfinished pattern escape");
      c = pattern[i];
      if (c == 's' || c == 'S') {
        result += (in_class ? "" : "[") + (c == 's' ? space : nonspace) + (in_class ? "" : "]");
      } else if (in_class && (c == '^' || c == '[' || c == '-' || c == 'b')) {
        result += c == '^' ? R"(\x5e)" : c == '[' ? R"(\x5b)" : c == '-' ? R"(\x2d)" : R"(\x08)";
      } else {
        if (std::string_view("dDwWfnrtvuxc^$.*+?\\()[]{}|/-").find(c) == std::string_view::npos)
          throw std::invalid_argument("unsupported pattern escape: \\" + std::string(1, c));
        if (c == 'x' || c == 'u') {
          const size_t digits = c == 'x' ? 2 : 4;
          if (i + digits >= pattern.size())
            throw std::invalid_argument("unfinished pattern escape");
          for (size_t j = 1; j <= digits; ++j)
            if (HexCharToInt(pattern[i + j]) < 0)
              throw std::invalid_argument("invalid hexadecimal pattern escape");
        }
        if (c == 'c' &&
            (i + 1 == pattern.size() || !((pattern[i + 1] >= 'A' && pattern[i + 1] <= 'Z') ||
                                          (pattern[i + 1] >= 'a' && pattern[i + 1] <= 'z'))))
          throw std::invalid_argument("invalid control character pattern escape");
        if (c == 'u' && i + 4 < pattern.size() &&
            (pattern[i + 1] == 'd' || pattern[i + 1] == 'D') && HexCharToInt(pattern[i + 2]) >= 8)
          throw std::invalid_argument(
              "surrogate escapes in patterns are not supported; use the Unicode character");
        result += '\\';
        result += c;
      }
    } else if (c == '.' && !in_class) {
      result += R"([^\n\r\u2028\u2029])";
    } else {
      result += c;
      if (c == '[')
        in_class = true;
      else if (c == ']')
        in_class = false;
    }
  }
  return result;
}

// Anchors are supported at the ends of each top-level alternative. Other zero-width assertions
// require a different regex execution model and are rejected rather than erased by conversion.
std::string search_pattern(const std::string& pattern) {
  std::vector<std::string> branches;
  bool escaped = false, in_class = false;
  int depth = 0;
  std::size_t begin = 0;
  for (std::size_t i = 0; i <= pattern.size(); ++i) {
    if (i == pattern.size() || (!escaped && !in_class && depth == 0 && pattern[i] == '|')) {
      branches.push_back(pattern.substr(begin, i - begin));
      begin = i + 1;
      continue;
    }
    const char c = pattern[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (c == '\\') {
      escaped = true;
      continue;
    }
    if (c == '[' && !in_class)
      in_class = true;
    else if (c == ']' && in_class)
      in_class = false;
    else if (!in_class && c == '(')
      ++depth;
    else if (!in_class && c == ')')
      --depth;
  }
  std::string result;
  for (auto branch : branches) {
    bool start = !branch.empty() && branch.front() == '^';
    if (start) branch.erase(0, 1);
    std::size_t slashes = 0;
    if (branch.size() > 1)
      for (std::size_t i = branch.size() - 1; i > 0 && branch[i - 1] == '\\'; --i) ++slashes;
    bool end = !branch.empty() && branch.back() == '$' && slashes % 2 == 0;
    if (end) branch.pop_back();
    escaped = false;
    in_class = false;
    for (char c : branch) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (c == '\\') {
        escaped = true;
        continue;
      }
      if (c == '[' && !in_class)
        in_class = true;
      else if (c == ']' && in_class)
        in_class = false;
      else if (!in_class && (c == '^' || c == '$'))
        throw std::invalid_argument("pattern anchors must bound a top-level alternative");
    }
    if (!result.empty()) result += "|";
    const std::string any = R"([^\uD800-\uDFFF]*)";
    result += (start ? "" : any) + std::string("(?:") + branch + ")" + (end ? "" : any);
  }
  return result;
}
}  // namespace

std::string SchemaStringPattern(const std::string& pattern) {
  return search_pattern(schema_characters(pattern));
}

Grammar JSONStringPattern(const std::string& pattern) {
  return GrammarNormalizer::Apply(
      JSONEncoder().Apply(Grammar::FromRegex(SchemaStringPattern(pattern))));
}

Grammar JSONStringLength(int minimum, int maximum) {
  std::string count = "{" + std::to_string(minimum) + ",";
  if (maximum >= 0) count += std::to_string(maximum);
  count += "}";
  return GrammarNormalizer::Apply(JSONEncoder().Apply(Grammar::FromEBNF("root ::= [^]" + count)));
}
Grammar JSONStringExcept(const std::vector<std::string>& excluded) {
  struct Node {
    bool terminal = false;
    std::map<int, Node> next;
  } root;
  for (const auto& key : excluded) {
    Node* node = &root;
    for (size_t offset = 0; offset < key.size();) {
      const auto [cp, length] = ParseNextUTF8(key.c_str() + offset);
      XGRAMMAR_CHECK(length > 0) << "invalid property name UTF-8";
      node = &node->next[cp];
      offset += length;
    }
    node->terminal = true;
  }
  GrammarBuilder builder;
  const auto any_char = builder.AddCharacterClass({{0, 0x10ffff}});
  const auto any = builder.AddRepeatFromExpr("tail", any_char, 0, -1);
  const std::function<int32_t(const Node&)> build = [&](const Node& node) {
    std::vector<int32_t> alternatives;
    if (!node.terminal) alternatives.push_back(builder.AddEmptyStr());
    std::vector<GrammarBuilder::CharacterClassElement> excluded_chars;
    for (const auto& [cp, next] : node.next) excluded_chars.push_back({cp, cp});
    alternatives.push_back(
        builder.AddSequence({builder.AddCharacterClass(excluded_chars, true), any}));
    for (const auto& [cp, next] : node.next) {
      auto ref = builder.AddRuleRef(builder.AddRuleWithHint("key", build(next)));
      alternatives.push_back(builder.AddSequence({builder.AddByteString(CharToUTF8(cp)), ref}));
    }
    return builder.AddChoices(alternatives);
  };
  const auto root_id = builder.AddRule("root", build(root));
  return GrammarNormalizer::Apply(JSONEncoder().Apply(builder.Get(root_id)));
}
}  // namespace xgrammar
