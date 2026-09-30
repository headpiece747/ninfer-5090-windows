// The top-k scoring record's two load-bearing properties, checked against something outside itself.
//
// 1. The token digest must equal hashlib's. tools/release/per_domain_kl.py computes it with
//    hashlib.sha256 over `struct.pack("<i", token)`, and the reader refuses a comparison whose
//    streams' digests differ -- which is the guard that stops a re-tokenized stream from reading as
//    a result. So a C++ digest that disagrees with Python's does not produce a wrong number, it
//    produces a refusal, and the instrument is simply dead. Every expected value below was produced
//    by hashlib, so this is a cross-language check and not a self-agreement; `docstring` in
//    per_domain_kl.py carries the same construction.
//
// 2. The record must match the reader's byte layout exactly. The writer emits integers
//    little-endian by explicit shifts rather than by struct overlay so that this holds without
//    either side assuming the host's byte order, and nothing but taking a written file apart proves
//    the two layouts still agree. The header's stream and position counters are patched on close,
//    so the writer has to be out of scope before the bytes are read -- which is itself part of what
//    this checks, because a reader that ran first would see zeroed counters.
#include "topk_record.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

int require(bool condition, const char* label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

std::string to_hex(const std::array<std::uint8_t, 32>& digest) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (const std::uint8_t byte : digest) {
        out.push_back(kHex[byte >> 4]);
        out.push_back(kHex[byte & 0x0Fu]);
    }
    return out;
}

std::vector<std::uint8_t> slurp(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input),
                                     std::istreambuf_iterator<char>());
}

std::uint32_t read_u32(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

std::uint64_t read_u64(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(bytes[offset + i]) << (8 * i);
    }
    return value;
}

int check_digest_against_hashlib() {
    int failures = 0;
    // The empty message is SHA-256's own published vector, so it pins the padding path for a
    // zero-length input as well as the cross-language agreement.
    failures += require(to_hex(ninfer::perplexity::token_digest({})) ==
                            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                        "empty token sequence matches hashlib");
    failures += require(to_hex(ninfer::perplexity::token_digest({0})) ==
                            "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119",
                        "one token matches hashlib");
    failures += require(to_hex(ninfer::perplexity::token_digest({0, 1})) ==
                            "01acecb507abfe1a354aa8064f4af5d3f1acd019e37db3c11c97523b71c76e9d",
                        "two tokens match hashlib");
    // 20 tokens is 80 packed bytes, so it crosses the 64-byte block boundary -- where a padding or
    // block-ordering bug hides, and where a single-block implementation would still pass the
    // shorter cases above.
    std::vector<ninfer::TokenId> twenty(20);
    for (std::size_t i = 0; i < twenty.size(); ++i) {
        twenty[i] = static_cast<ninfer::TokenId>(i * 7 + 3);
    }
    failures += require(to_hex(ninfer::perplexity::token_digest(twenty)) ==
                            "c7db3524532af46b316187b019496cc38dfa28dc25b330e661f570fee06f4db0",
                        "multi-block token sequence matches hashlib");
    // A negative token id: TokenId is a signed int32 and a corpus can carry one, so the packing must
    // not reinterpret it as unsigned. struct.pack("<i", -1) is ff ff ff ff.
    failures += require(to_hex(ninfer::perplexity::token_digest({-1, 248319})) ==
                            "5a9679dff9c9660d738c8a450b27931e08cc5c6ec688916e9c1e156f842ce717",
                        "negative and max token ids match hashlib");
    // The digest must depend on the contents and not on the vector's capacity, or a caller reusing a
    // buffer would get a different answer for the same stream.
    std::vector<ninfer::TokenId> roomy = twenty;
    roomy.reserve(4096);
    failures += require(to_hex(ninfer::perplexity::token_digest(roomy)) ==
                            to_hex(ninfer::perplexity::token_digest(twenty)),
                        "digest ignores spare capacity");
    return failures;
}

int check_record_layout() {
    int failures = 0;
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "ninfer_topk_record_layout.bin";
    std::filesystem::remove(path);
    constexpr std::int32_t k = 4;

    {
        // A 64-character signature, which is what every real artifact carries: the prefill
        // signature is a SHA-256 hex digest in a 64-byte field, so it fills the field exactly. A
        // writer that rejected the exact fit would pass a short placeholder and then refuse every
        // real artifact at record time, so the test uses the real width.
        const std::string signature(64, 'a');
        ninfer::perplexity::TopkRecordWriter writer(path, k, "unit-corpus", 4096, 2048,
                                                     "fp8-e4m3-r256", signature);
        writer.begin_stream("stream-0", "ninfer_code",
                            ninfer::perplexity::token_digest(std::vector<ninfer::TokenId>{7, 8, 9}));
        ninfer::CausalTopk first;
        first.k        = k;
        first.indices  = {1, 2, 3, 4, 5, 6, 7, 8};
        first.logprobs = {-0.1f, -1.1f, -2.1f, -3.1f, -0.2f, -1.2f, -2.2f, -3.2f};
        writer.append(first, 0);
        ninfer::CausalTopk second;
        second.k        = k;
        second.indices  = {9, 10, 11, 12};
        second.logprobs = {-0.3f, -1.3f, -2.3f, -3.3f};
        writer.append(second, 2);
        writer.end_stream();
    }

    const std::vector<std::uint8_t> bytes = slurp(path);
    failures += require(!bytes.empty(), "record file was written");
    if (bytes.empty()) {
        std::filesystem::remove(path);
        return failures;
    }
    // Explicit offsets rather than a cursor advanced as the checks walk. An earlier revision of
    // this test advanced by one int32 where it meant one position's k entries, and the three
    // failures that produced all pointed downstream of the same mistake -- which is the argument
    // for naming each field's offset instead of tracking a position implicitly.
    constexpr std::size_t kOffMagic     = 0;
    constexpr std::size_t kOffK         = 8;
    constexpr std::size_t kOffStreams   = 12;
    constexpr std::size_t kOffPositions = 16;
    constexpr std::size_t kOffCorpus    = 24;
    constexpr std::size_t kOffContext   = 56;
    constexpr std::size_t kOffStride    = 60;
    constexpr std::size_t kOffKv        = 64;
    constexpr std::size_t kOffSignature = 96;
    constexpr std::size_t kHeaderSize   = 160;
    // These three field lengths are written out rather than spelled inline at each comparison,
    // because an inline length is where the previous revision's off-by-one lived: "unit-corpus" is
    // 11 bytes and the check read 10, so it reported a layout fault against a correct writer.
    const std::string corpus_id    = "unit-corpus";
    const std::string kv_dtype     = "fp8-e4m3-r256";
    const std::string signature(64, 'a');
    const auto text_at              = [&](std::size_t offset, std::size_t length) {
        return std::string(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                           bytes.begin() + static_cast<std::ptrdiff_t>(offset + length));
    };
    failures += require(text_at(kOffMagic, 8) == "NINFKL01", "magic is the reader's");
    failures += require(read_u32(bytes, kOffK) == static_cast<std::uint32_t>(k), "k is recorded");
    failures += require(read_u32(bytes, kOffStreams) == 1, "stream count is patched on close");
    failures += require(read_u64(bytes, kOffPositions) == 3,
                        "position count is patched on close");
    failures += require(text_at(kOffCorpus, corpus_id.size()) == corpus_id,
                        "corpus id is at the reader's offset");
    failures += require(read_u32(bytes, kOffContext) == 4096, "context is at the reader's offset");
    failures += require(read_u32(bytes, kOffStride) == 2048, "stride is at the reader's offset");
    // kv_dtype is a 32-byte field, not 8: the engine's own name is 13 bytes, and an 8-byte field
    // would have thrown rather than truncating -- which is the behaviour being pinned here.
    failures += require(text_at(kOffKv, kv_dtype.size()) == kv_dtype,
                        "kv dtype carries the engine's own name untruncated");
    failures += require(text_at(kOffSignature, signature.size()) == signature,
                        "prefill signature is at the reader's offset");

    // stream section: id_len u16 | id | domain_len u16 | domain | digest_len u16 | 32B | count u32
    const auto u16 = [&](std::size_t offset) {
        return static_cast<std::size_t>(bytes[offset]) |
               (static_cast<std::size_t>(bytes[offset + 1]) << 8);
    };
    std::size_t at = kHeaderSize;
    const std::string stream_id = "stream-0";
    const std::string domain    = "ninfer_code";
    failures += require(u16(at) == stream_id.size(), "stream id length");
    at += 2;
    failures += require(text_at(at, stream_id.size()) == stream_id, "stream id");
    at += stream_id.size();
    failures += require(u16(at) == domain.size(), "stream domain length");
    at += 2;
    failures += require(text_at(at, domain.size()) == domain, "stream domain");
    at += domain.size();
    failures += require(u16(at) == 32, "token digest length is the reader's");
    at += 2;
    const auto expected_digest =
        ninfer::perplexity::token_digest(std::vector<ninfer::TokenId>{7, 8, 9});
    failures += require(std::equal(expected_digest.begin(), expected_digest.end(),
                                   bytes.begin() + static_cast<std::ptrdiff_t>(at)),
                        "token digest is written verbatim");
    at += 32;
    // The count precedes the positions. This is the ordering the first draft of the writer got
    // wrong, and the reader is what would have failed.
    failures += require(read_u32(bytes, at) == 3, "stream position count precedes the positions");
    at += 4;
    // Per position: k indices then k log-probabilities, interleaved. An earlier revision of the
    // writer emitted the tile's whole index block and then its whole log-probability block, which
    // the reader does not expect; this offset arithmetic is what makes that disagreement visible.
    // A position is `block` bytes of indices PLUS `block` of log-probabilities, so consecutive
    // positions are 2*block apart -- advancing by one block reads position 2's slot as position
    // 1's log-probabilities, which is how an earlier revision of this test read -0.2f where it meant
    // to read an index. The offsets are unaligned on purpose: the stream id and domain are
    // length-prefixed, so the position data does not start on a 4-byte boundary, and both the reader
    // and this test assemble bytes rather than casting.
    const std::size_t block = static_cast<std::size_t>(k) * 4;
    const std::size_t p0    = at;
    const std::size_t p1    = p0 + 2 * block;
    const std::size_t p2    = p1 + 2 * block;
    const auto f32          = [&](std::size_t offset) {
        const std::uint32_t bits = read_u32(bytes, offset);
        float value              = 0.0f;
        static_assert(sizeof(value) == sizeof(bits), "float must be 32 bits");
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    };
    failures += require(read_u32(bytes, p0) == 1, "position 0 index 0 is 1");
    failures += require(read_u32(bytes, p0 + 4) == 2, "position 0 index 1 is 2");
    failures += require(f32(p0 + block) == -0.1f, "position 0 log-probability 0 round-trips");
    failures += require(f32(p0 + block + 4) == -1.1f, "position 0 log-probability 1 round-trips");
    failures += require(read_u32(bytes, p1) == 5, "position 1 index 0 is 5");
    failures += require(f32(p1 + block) == -0.2f, "position 1 log-probability 0 round-trips");
    failures += require(read_u32(bytes, p2) == 9, "position 2 index 0 is 9");
    failures += require(f32(p2 + block) == -0.3f, "position 2 log-probability 0 round-trips");
    failures += require(p2 + 2 * block == bytes.size(),
                        "record ends exactly after the last log-probability");

    std::filesystem::remove(path);
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += check_digest_against_hashlib();
    failures += check_record_layout();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "PASS: top-k record digest agrees with hashlib and its layout is the reader's\n";
    return 0;
}
