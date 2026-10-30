// Token-exact per-token log probabilities for a supplied token-id file.
//
// This exists because the question it answers cannot be asked any other way on this engine. To judge
// whether a speculative route's divergence from non-speculative decoding is a near-tie or a real
// disagreement, the model's log probability at the diverging position has to be read at the *same
// token index* on both sides. The perplexity tool's --text front door re-tokenises, and re-tokenising
// decoded text does not reproduce the ids that produced it -- measured at 153 generated ids
// re-tokenising to 265 scored rows -- so an index-aligned read through it silently inspects a
// different position. Engine::score_tokens already takes the ids directly; this is that path with a
// file front door, so the alignment is exact by construction rather than by assumption.
//
// It reports the log probability of each supplied token under a teacher-forced forward. For a token
// produced by a greedy path, that value is minus the gap to the runner-up, and so upper-bounds the
// top-2 gap at that position: a value near zero means a genuine near-tie, a large magnitude means the
// model had a clear preference there and something overrode it.
//
// Why this exists, with the numbers that motivated it. Measured on this product (NVFP4 27B, 8192
// context, greedy, 256 new tokens) by generating on the non-speculative route and on DFlash2 and
// comparing the two id sequences, then scoring each sequence and reading the index where they first
// differ -- the prefixes up to that index are identical, so it is one context and two candidates:
//
//   kv-dtype   first differing token   logprob A   logprob B   gap (nats)
//   fp8        1                      -6.540      -9.681      3.141
//   fp8        26                     -8.206      -6.331      1.875
//   fp8        29                     -2.204      -1.204      1.000
//   bf16       58                     -0.499      -0.999      0.500
//   bf16       24                     -1.385      -1.760      0.375
//   bf16       29                     -1.135      -2.260      1.125
//
// The masked-draft accept that produced those sequences is textbook and was read to confirm it
// (src/ops/kernel/speculative_round.cuh): `a` is the first column whose target argmax differs from
// the draft, and the committed token at that column is the target's own argmax there. There is no
// cross-column leakage, which is also why both routes agree on the very first generated token.
//
// The 0.375-1.125 nat spread under lossless KV is consistent with bf16 error accumulated across the
// target's 64 layers (48 of them GatedDeltaNet recurrences, which accumulate along the context), not
// with a single rounding step -- one ulp at a logit of magnitude ~50 is ~0.1, and a random walk over
// 64 layers lands near the observed range. That is an order-of-magnitude consistency argument, not a
// proof: the logit magnitude is not measured here. fp8 KV roughly triples the gap and can move the
// first divergence from token 58 to token 1, so KV quantisation is a real amplifier rather than the
// whole explanation.
//
// Two limits on reading these numbers. The scorer resolves log probabilities to 1/16 nat, so a gap
// below about 0.06 is not resolvable and every value here is a multiple of 0.0625. And the reference
// is a teacher-forced forward over the whole sequence, which is neither of the two decode shapes, so
// this measures how far a third computation sits from the candidates -- not directly how far the two
// decode forwards disagree with each other.
//
// Traps this tool exists to avoid, each of which produced a wrong or empty answer first:
//   * ninfer-perplexity --text re-tokenises, and re-tokenising decoded text does not reproduce the
//     ids that produced it (153 generated ids re-tokenised to 265 scored rows), so an index-aligned
//     read through it silently inspects a different position.
//   * ninfer --print-token-ids writes the ids to STDERR, not stdout (apps/cli/main.cpp:308), on a
//     line reading "tokens  <pad>  generated ids  <pad>  9419 0 2500 ...". The ids FOLLOW the label.
//     Sending stderr to /dev/null silently yields the answer text instead, and every index is then a
//     word index rather than a token index.
//   * Start-Process -ArgumentList splits on whitespace unless each element is quoted, so a prompt
//     containing spaces arrives as several argv entries and the CLI prints usage.
// Assert the shape before reporting: N ids scored from index 1 must yield exactly N-1 rows.
#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void usage(const char* executable) {
    std::cerr << "usage: " << executable
              << " <model.ninfer> --ids <tokens.txt> [--first-target N] [--context N]"
                 " [--kv-dtype bf16|int8|fp8|nvfp4|k8v4]\n"
                 "  --ids accepts whitespace-separated integer token ids, one stream, '#' comments.\n"
                 "  Prints 'index,logprob' to stdout.\n";
}

std::vector<ninfer::TokenId> read_ids(const std::string& path) {
    std::ifstream input(path);
    if (!input) { throw std::runtime_error("could not open --ids " + path); }
    std::vector<ninfer::TokenId> ids;
    std::string token;
    while (input >> token) {
        if (token.empty() || token.front() == '#') { continue; }
        std::istringstream parsed(token);
        long long value = 0;
        if (!(parsed >> value)) { continue; }
        ids.push_back(static_cast<ninfer::TokenId>(value));
    }
    if (ids.empty()) { throw std::runtime_error("--ids contained no token ids"); }
    return ids;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            usage(argc > 0 ? argv[0] : "score_token_logprobs");
            return 2;
        }
        const std::string artifact = argv[1];
        std::string ids_path;
        std::uint32_t first_target = 1;
        std::uint32_t context      = 2048;
        ninfer::KvCacheStorage kv   = ninfer::KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
        for (int i = 2; i < argc; ++i) {
            const std::string option = argv[i];
            const auto value = [&]() -> std::string {
                if (i + 1 >= argc) { throw std::invalid_argument(option + " needs a value"); }
                return argv[++i];
            };
            if (option == "--ids") {
                ids_path = value();
            } else if (option == "--first-target") {
                first_target = static_cast<std::uint32_t>(std::stoul(value()));
            } else if (option == "--context") {
                context = static_cast<std::uint32_t>(std::stoul(value()));
            } else if (option == "--kv-dtype") {
                const std::string mode = value();
                if (mode == "bf16") {
                    kv = ninfer::KvCacheStorage::BFloat16;
                } else if (mode == "int8") {
                    kv = ninfer::KvCacheStorage::Int8Group64;
                } else if (mode == "fp8") {
                    kv = ninfer::KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
                } else if (mode == "nvfp4") {
                    kv = ninfer::KvCacheStorage::Nvfp4Group16;
                } else if (mode == "k8v4") {
                    kv = ninfer::KvCacheStorage::Fp8KeyNvfp4Value;
                } else {
                    throw std::invalid_argument("--kv-dtype must be bf16, int8, fp8, nvfp4, or k8v4");
                }
            } else {
                usage(argv[0]);
                return 2;
            }
        }
        if (ids_path.empty()) {
            usage(argv[0]);
            return 2;
        }

        ninfer::EngineOptions options;
        options.artifact_path        = artifact;
        options.purpose              = ninfer::EnginePurpose::CausalScoring;
        options.max_context          = context;
        options.kv_cache             = kv;
        options.prefill_chunk        = 1024;
        options.context_cache.enabled = false;
        ninfer::Engine engine(options);

        const std::vector<ninfer::TokenId> ids = read_ids(ids_path);
        if (first_target == 0 || first_target >= ids.size()) {
            throw std::invalid_argument("--first-target must be in [1, id count)");
        }
        const std::vector<float> logprobs = engine.score_tokens(ids, first_target);
        const std::size_t expected = ids.size() - first_target;
        if (logprobs.size() != expected) {
            throw std::runtime_error("scoring returned " + std::to_string(logprobs.size()) +
                                     " values for " + std::to_string(expected) + " targets");
        }
        std::cout << "index,logprob\n";
        for (std::size_t i = 0; i < logprobs.size(); ++i) {
            std::cout << (first_target + i) << ',' << logprobs[i] << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "score_token_logprobs: " << error.what() << '\n';
        return 1;
    }
}
