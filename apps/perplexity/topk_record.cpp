#include "topk_record.h"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace ninfer::perplexity {
namespace {

// Every integer is emitted little-endian by explicit shifts rather than by memcpy of a native
// integer, so the file is byte-identical whatever the writer's host byte order. The reader in
// tools/release/per_domain_kl.py unpacks with an explicit "<" format, so the two agree by
// construction rather than by both machines happening to be little-endian.
void put_u16(std::ofstream& out, std::uint16_t value) {
    const std::array<char, 2> bytes = {static_cast<char>(value & 0xFFu),
                                       static_cast<char>((value >> 8) & 0xFFu)};
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void put_u32(std::ofstream& out, std::uint32_t value) {
    const std::array<char, 4> bytes = {
        static_cast<char>(value & 0xFFu), static_cast<char>((value >> 8) & 0xFFu),
        static_cast<char>((value >> 16) & 0xFFu), static_cast<char>((value >> 24) & 0xFFu)};
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void put_u64(std::ofstream& out, std::uint64_t value) {
    std::array<char, 8> bytes{};
    for (int i = 0; i < 8; ++i) {
        bytes[static_cast<std::size_t>(i)] = static_cast<char>((value >> (8 * i)) & 0xFFu);
    }
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void put_i32_array(std::ofstream& out, const TokenId* values, std::size_t count) {
    std::array<char, 4> bytes{};
    for (std::size_t i = 0; i < count; ++i) {
        const auto value = static_cast<std::uint32_t>(values[i]);
        for (int b = 0; b < 4; ++b) {
            bytes[static_cast<std::size_t>(b)] = static_cast<char>((value >> (8 * b)) & 0xFFu);
        }
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
}

// FP32 is written as its IEEE-754 bit pattern. The reduction is already documented as an FP32
// approximation, so preserving the device's own value rather than re-rounding through double is
// what makes the record a record of what the Op produced.
void put_f32_array(std::ofstream& out, const float* values, std::size_t count) {
    std::array<char, 4> bytes{};
    for (std::size_t i = 0; i < count; ++i) {
        std::uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(values[i]), "float must be 32 bits");
        std::memcpy(&bits, &values[i], sizeof(bits));
        for (int b = 0; b < 4; ++b) {
            bytes[static_cast<std::size_t>(b)] = static_cast<char>((bits >> (8 * b)) & 0xFFu);
        }
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
}

void put_padded(std::ofstream& out, const std::string& text, std::size_t width,
                const char* label) {
    // `<=`, not `<`: the prefill signature is a 64-character hex digest written into a 64-byte
    // field, so it fills it exactly with no room for a terminator. Rejecting the exact fit would
    // have made the writer refuse every real artifact while the unit test -- which used a seven
    // character placeholder -- passed. The reader strips at the first NUL, so a field with no
    // terminator reads back whole.
    if (text.size() > width) {
        throw std::runtime_error(std::string("top-k record ") + label + " is " +
                                 std::to_string(text.size()) + " bytes, which does not fit the " +
                                 std::to_string(width) + "-byte field");
    }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    const std::string padding(width - text.size(), '\0');
    out.write(padding.data(), static_cast<std::streamsize>(padding.size()));
}

void put_string(std::ofstream& out, std::string_view text) {
    if (text.size() > 0xFFFFu) {
        throw std::runtime_error("top-k record string exceeds 65535 bytes");
    }
    put_u16(out, static_cast<std::uint16_t>(text.size()));
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

// --- SHA-256, FIPS 180-4 -------------------------------------------------------------
// Present so this app can produce a digest the Python reader reproduces exactly. It is the
// standard construction; `test_topk_record.cpp` pins it against the published vectors, including
// the multi-block case, so a transcription error here is a failing test rather than a digest that
// silently differs from hashlib's.

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
    0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
    0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
    0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
    0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
    0xc67178f2u};

constexpr std::uint32_t rotr(std::uint32_t value, unsigned bits) {
    return (value >> bits) | (value << (32u - bits));
}

void sha256_compress(std::array<std::uint32_t, 8>& state, const std::uint8_t* block) {
    std::array<std::uint32_t, 64> w{};
    for (std::size_t i = 0; i < 16; ++i) {
        w[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24) |
               (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16) |
               (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8) |
               static_cast<std::uint32_t>(block[i * 4 + 3]);
    }
    for (std::size_t i = 16; i < 64; ++i) {
        const std::uint32_t s0 =
            rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i]                = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::array<std::uint32_t, 8> v = state;
    for (std::size_t i = 0; i < 64; ++i) {
        const std::uint32_t s1    = rotr(v[4], 6) ^ rotr(v[4], 11) ^ rotr(v[4], 25);
        const std::uint32_t ch    = (v[4] & v[5]) ^ (~v[4] & v[6]);
        const std::uint32_t temp1 = v[7] + s1 + ch + kRoundConstants[i] + w[i];
        const std::uint32_t s0    = rotr(v[0], 2) ^ rotr(v[0], 13) ^ rotr(v[0], 22);
        const std::uint32_t maj   = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
        const std::uint32_t temp2 = s0 + maj;
        v[7] = v[6];
        v[6] = v[5];
        v[5] = v[4];
        v[4] = v[3] + temp1;
        v[3] = v[2];
        v[2] = v[1];
        v[1] = v[0];
        v[0] = temp1 + temp2;
    }
    for (std::size_t i = 0; i < 8; ++i) { state[i] += v[i]; }
}

std::array<std::uint8_t, 32> sha256(const std::vector<std::uint8_t>& message) {
    std::array<std::uint32_t, 8> state = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                          0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    const std::size_t full_blocks = message.size() / 64;
    for (std::size_t b = 0; b < full_blocks; ++b) {
        sha256_compress(state, message.data() + b * 64);
    }
    // Padding: 0x80, zeros to 56 mod 64, then the length in bits as a 64-bit big-endian count.
    std::vector<std::uint8_t> tail(message.begin() + static_cast<std::ptrdiff_t>(full_blocks * 64),
                                   message.end());
    tail.push_back(0x80u);
    while (tail.size() % 64 != 56) { tail.push_back(0x00u); }
    const std::uint64_t bits = static_cast<std::uint64_t>(message.size()) * 8u;
    for (int i = 7; i >= 0; --i) {
        tail.push_back(static_cast<std::uint8_t>((bits >> (8 * i)) & 0xFFu));
    }
    for (std::size_t b = 0; b < tail.size(); b += 64) { sha256_compress(state, tail.data() + b); }
    std::array<std::uint8_t, 32> digest{};
    for (std::size_t i = 0; i < 8; ++i) {
        digest[i * 4]     = static_cast<std::uint8_t>((state[i] >> 24) & 0xFFu);
        digest[i * 4 + 1] = static_cast<std::uint8_t>((state[i] >> 16) & 0xFFu);
        digest[i * 4 + 2] = static_cast<std::uint8_t>((state[i] >> 8) & 0xFFu);
        digest[i * 4 + 3] = static_cast<std::uint8_t>(state[i] & 0xFFu);
    }
    return digest;
}

} // namespace

std::array<std::uint8_t, 32> token_digest(const std::vector<TokenId>& tokens) {
    std::vector<std::uint8_t> message;
    message.reserve(tokens.size() * 4);
    for (const TokenId token : tokens) {
        const auto value = static_cast<std::uint32_t>(token);
        for (int b = 0; b < 4; ++b) {
            message.push_back(static_cast<std::uint8_t>((value >> (8 * b)) & 0xFFu));
        }
    }
    return sha256(message);
}

TopkRecordWriter::TopkRecordWriter(const std::filesystem::path& path, std::int32_t k,
                                   std::string corpus_id, std::uint32_t context,
                                   std::uint32_t stride, std::string kv_dtype,
                                   std::string prefill_signature)
    : path_(path),
      k_(k),
      corpus_id_(std::move(corpus_id)),
      prefill_signature_(std::move(prefill_signature)) {
    if (k_ < 1) { throw std::invalid_argument("top-k record k must be positive"); }
    if (path_.has_parent_path()) {
        std::filesystem::create_directories(path_.parent_path());
    }
    out_.open(path_, std::ios::binary | std::ios::trunc);
    if (!out_) { throw std::runtime_error("could not open top-k record " + path_.string()); }
    out_.write(kTopkRecordMagic.data(), static_cast<std::streamsize>(kTopkRecordMagic.size()));
    put_u32(out_, static_cast<std::uint32_t>(k_));
    header_count_at_ = out_.tellp();
    put_u32(out_, 0);
    header_positions_at_ = out_.tellp();
    put_u64(out_, 0);
    put_padded(out_, corpus_id_, 32, "corpus id");
    put_u32(out_, context);
    put_u32(out_, stride);
    // 32 bytes, not 8: the engine's own KV names are longer than that -- "fp8-e4m3-r256" is 14 --
    // and truncating the field would make two different KV representations compare equal.
    put_padded(out_, kv_dtype, 32, "kv dtype");
    put_padded(out_, prefill_signature_, 64, "prefill signature");
    out_.flush();
}

void TopkRecordWriter::begin_stream(std::string_view id, std::string_view domain,
                                    const std::array<std::uint8_t, 32>& digest) {
    if (stream_open_) { throw std::logic_error("top-k record stream already open"); }
    put_string(out_, id);
    put_string(out_, domain);
    put_u16(out_, 32);
    out_.write(reinterpret_cast<const char*>(digest.data()), 32);
    // The position count precedes the positions, so it is reserved here and back-patched by
    // end_stream once the stream's windows have all been tiled. Writing it at the end instead would
    // put it after the position block, which is not where the reader looks for it.
    stream_count_at_ = out_.tellp();
    put_u32(out_, 0);
    stream_positions_ = 0;
    stream_open_      = true;
}

void TopkRecordWriter::append(const CausalTopk& tile, std::size_t first_position) {
    if (!stream_open_) { throw std::logic_error("top-k record append outside a stream"); }
    if (tile.k != k_) {
        throw std::logic_error("top-k record append carries k=" + std::to_string(tile.k) +
                               " but the record was opened with k=" + std::to_string(k_));
    }
    if (first_position != stream_positions_) {
        // The reader matches positions by ordinal, so a gap or overlap here would silently shift
        // every later position of the stream against its counterpart in the other record.
        throw std::logic_error("top-k record append at position " + std::to_string(first_position) +
                               " but the stream holds " + std::to_string(stream_positions_));
    }
    const std::size_t entries = tile.positions();
    if (entries == 0) { return; }
    if (tile.indices.size() != entries * static_cast<std::size_t>(k_) ||
        tile.logprobs.size() != entries * static_cast<std::size_t>(k_)) {
        throw std::logic_error("top-k record tile has an inconsistent shape");
    }
    // Per POSITION, not per tile: each position's k indices are immediately followed by that
    // position's k log-probabilities. The first draft of this wrote the tile's whole index block
    // and then its whole log-probability block, which is a different file from the one
    // tools/release/per_domain_kl.py reads -- it interleaves per position -- and the two disagreed
    // silently, because a file that parses as a stream of ints and floats of the right total size
    // still yields wrong values. tests/test_topk_record.cpp takes the file apart to catch it.
    // Per position is also the order a streaming reader wants: it can consume a position without
    // knowing how many follow.
    for (std::size_t position = 0; position < entries; ++position) {
        const std::size_t base = position * static_cast<std::size_t>(k_);
        put_i32_array(out_, tile.indices.data() + base, static_cast<std::size_t>(k_));
        put_f32_array(out_, tile.logprobs.data() + base, static_cast<std::size_t>(k_));
    }
    stream_positions_ += static_cast<std::uint32_t>(entries);
    positions_ += entries;
}

void TopkRecordWriter::end_stream() {
    if (!stream_open_) { throw std::logic_error("top-k record stream is not open"); }
    const std::streampos here = out_.tellp();
    out_.seekp(stream_count_at_);
    put_u32(out_, stream_positions_);
    out_.flush();
    out_.seekp(here);
    if (!out_) { throw std::runtime_error("top-k record write failed for " + path_.string()); }
    stream_open_ = false;
    ++stream_count_;
}

TopkRecordWriter::~TopkRecordWriter() {
    if (stream_open_) {
        // A stream left open means the caller threw mid-stream. Finish it so the file is at least
        // structurally valid, and let the reader's count mismatch surface the truncation.
        try {
            end_stream();
        } catch (...) {}
    }
    if (out_.is_open()) {
        out_.flush();
        // Patch the header's two counters where they were reserved. A crash before this leaves
        // them zero, which the reader reports as a stream-count mismatch rather than as data.
        const std::streampos here = out_.tellp();
        out_.seekp(header_count_at_);
        put_u32(out_, stream_count_);
        out_.seekp(header_positions_at_);
        put_u64(out_, positions_);
        out_.flush();
        out_.seekp(here);
        out_.close();
    }
}

} // namespace ninfer::perplexity
