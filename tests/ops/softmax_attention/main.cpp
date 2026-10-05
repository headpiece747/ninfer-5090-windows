#include "kv_cache_storage.h"
#include <iostream>
#include <optional>
#include <string_view>

int run_softmax_attention_causal_cache_tests(std::optional<ninfer::KvCacheStorage> storage);
int run_softmax_attention_plain_and_packed_tests();
int run_softmax_attention_context_tests();

int main(int argc, char** argv) {
    bool causal_only = false;
    bool skip_causal = false;
    std::optional<ninfer::KvCacheStorage> storage;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string_view argument(argv[i]);
            if (argument == "--causal-only")
                causal_only = true;
            else if (argument == "--no-causal")
                skip_causal = true;
            else if (argument == "--kv-dtype" && i + 1 < argc) {
                const std::string_view name(argv[++i]);
                storage     = name == "all" ? std::nullopt
                                            : std::optional(ninfer::test::parse_kv_cache_storage(name));
                causal_only = true;
            } else
                throw std::invalid_argument("invalid attention test option");
        }
        if (skip_causal && causal_only) {
            throw std::invalid_argument("--no-causal contradicts --causal-only or --kv-dtype");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what()
                  << "\nusage: ninfer_softmax_attention_test [--causal-only] "
                     "[--kv-dtype bf16|int8|fp8|nvfp4|k8v4|all]\n";
        return 2;
    }
    // --no-causal exists so the causal sweep can be registered as one ctest entry per KV format
    // while this binary still covers the other two scenarios in an entry of its own. Without it
    // every --kv-dtype path sets causal_only and returns before them, so a per-format split would
    // either duplicate 553 s of work or silently drop plain_and_packed and context.
    int causal = 0;
    if (!skip_causal) {
        causal = run_softmax_attention_causal_cache_tests(storage);
        if (causal == 77) return 77;
        if (causal_only) return causal;
    }

    const int plain_and_packed = run_softmax_attention_plain_and_packed_tests();
    if (plain_and_packed == 77) return 77;

    const int context = run_softmax_attention_context_tests();
    if (context == 77) return 77;

    const int failures = causal + plain_and_packed + context;
    std::cout << (failures == 0 ? "softmax_attention: PASS\n" : "softmax_attention: FAIL\n");
    return failures == 0 ? 0 : 1;
}
