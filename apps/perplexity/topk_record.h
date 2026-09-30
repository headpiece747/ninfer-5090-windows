#pragma once

// The on-disk form of one artifact's top-k causal-scoring record, and its SHA-256 token digest.
//
// `tools/release/per_domain_kl.py` is the reader and the reduction. The format is specified here
// because the two sides are written by different toolchains -- this one by MSVC, the reader by
// CPython -- and a divergence between them would not fail loudly: it would produce a record whose
// positions did not line up, and a per-position comparison of two unrelated contexts still returns
// a number that looks like a result. The token digest exists for the same reason: it is what lets
// the reader refuse that case instead of reporting it.
//
// Layout, little-endian throughout, written byte by byte rather than by struct overlay, because the
// writer's byte order and the reader's are not assumed to agree and a struct overlay would make that
// assumption invisible:
//
//   header   magic "NINFKL01" | k u32 | stream_count u32 | positions u64
//            corpus_id 32B NUL-padded | context u32 | stride u32
//            kv_dtype 8B NUL-padded | prefill_signature 64B NUL-padded
//   stream   id_len u16 | id | domain_len u16 | domain | digest_len u16 (=32)
//            token_digest 32B | position_count u32
//   position k * i32 indices (descending log-probability), then k * f32 log-probabilities
//
// Positions are interleaved one at a time -- each position's indices then its log-probabilities --
// not written as two blocks. A streaming reader can then consume a position without knowing how
// many follow, and the layout matches what ops::topk_logprobs produces (two tensors reduced
// together).
//
// A stream's position count is not known when its section opens, so it is written on
// `end_stream`; a file cut short therefore carries a count that does not match the positions
// actually present, which the reader refuses rather than scoring.
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "ninfer/types.h"

namespace ninfer::perplexity {

// A fixed-width magic, so a file that is not a record is rejected on its first eight bytes rather
// than by a field that happens to be zero.
inline constexpr std::array<char, 8> kTopkRecordMagic = {'N', 'I', 'N', 'F', 'K', 'L', '0', '1'};

// SHA-256 over a token-id sequence, byte-for-byte equal to
// hashlib.sha256(b"".join(struct.pack("<i", t) for t in tokens)).digest().
//
// Implemented here rather than taken from the frontend's digest because an app does not include
// private model headers in this tree, and because the packing has to be explicitly little-endian to
// match the reader: hashing the tokens' native representation would tie the format to the host's
// byte order, which is the assumption this file exists to avoid. A local test pins it against
// published SHA-256 vectors, so the two implementations agreeing is checked rather than assumed.
[[nodiscard]] std::array<std::uint8_t, 32> token_digest(const std::vector<TokenId>& tokens);

class TopkRecordWriter {
public:
    TopkRecordWriter(const std::filesystem::path& path, std::int32_t k, std::string corpus_id,
                     std::uint32_t context, std::uint32_t stride, std::string kv_dtype,
                     std::string prefill_signature);

    TopkRecordWriter(const TopkRecordWriter&)            = delete;
    TopkRecordWriter& operator=(const TopkRecordWriter&) = delete;
    ~TopkRecordWriter();

    void begin_stream(std::string_view id, std::string_view domain,
                      const std::array<std::uint8_t, 32>& digest);

    // `first_position` is the tile's offset within this stream, carried explicitly because a tile is
    // a slice of the scored range and the writer has no other way to know where it starts.
    void append(const CausalTopk& tile, std::size_t first_position);

    void end_stream();

    [[nodiscard]] std::uint64_t positions() const noexcept { return positions_; }
    [[nodiscard]] std::uint32_t streams() const noexcept { return stream_count_; }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
    std::int32_t k_         = 0;
    std::string corpus_id_;
    std::string prefill_signature_;
    std::uint64_t positions_    = 0;
    std::uint32_t stream_count_ = 0;
    // The header's two counters are reserved in the constructor and back-patched on close, because
    // neither is known until every stream has been written. Both are fixed-width so the reader can
    // seek past them. `stream_count_at_` is the *per-stream* position count, reserved in
    // begin_stream and back-patched by end_stream; it is a different slot from the header's.
    std::streampos header_count_at_    = 0;
    std::streampos header_positions_at_ = 0;
    std::streampos stream_count_at_    = 0;
    std::uint32_t stream_positions_    = 0;
    bool stream_open_                  = false;
    std::ofstream out_;
};

} // namespace ninfer::perplexity
