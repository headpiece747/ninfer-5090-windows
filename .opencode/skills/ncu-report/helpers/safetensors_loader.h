// safetensors_loader.h — header-only, no-dependencies safetensors reader.
//
// The safetensors format is:
//   [u64 header_len]
//   [JSON header (header_len bytes)]
//   [raw tensor bytes]
//
// The JSON header is a flat map: tensor_name -> { dtype, shape, data_offsets: [start, end] }
// (plus an optional "__metadata__" key).
//
// This loader parses the JSON with a small tailored scanner — no external
// dependency — and exposes:
//   SafetensorsFile::load(path)    -> loads file into memory
//   .entry(name)                   -> metadata (dtype string, shape, offsets)
//   .tensor_bytes(name)            -> pointer to raw tensor bytes
//
// The entire tensor payload is kept in a std::vector<uint8_t>. Workload
// safetensors files are < 100 MB, so this is fine for profiling harnesses.
//
// Usage:
//   #include "safetensors_loader.h"
//   SafetensorsFile st = SafetensorsFile::load("path/to/workload.safetensors");
//   const auto& q_entry = st.entry("q");  // shape, dtype, offsets
//   std::memcpy(h_q.data(), st.tensor_bytes("q"), q_bytes);
//
// Limitations:
//   - No support for sharded safetensors.
//   - No validation that the dtype matches what you expect — caller must check.
//   - Loads entire file into memory (mmap would be nicer but adds complexity).

#pragma once

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

struct StEntry {
    std::string dtype;           // "F32", "BF16", "F16", "I64", "I32", ...
    std::vector<int64_t> shape;
    int64_t offset_start = 0;
    int64_t offset_end = 0;
};

struct SafetensorsFile {
    std::unordered_map<std::string, StEntry> entries;
    std::vector<uint8_t> payload;  // tensor data region (after header)

    static SafetensorsFile load(const std::string& path);

    const uint8_t* tensor_bytes(const std::string& name, size_t* nbytes = nullptr) const {
        auto it = entries.find(name);
        if (it == entries.end()) {
            std::fprintf(stderr, "[safetensors] tensor '%s' not found\n", name.c_str());
            std::exit(6);
        }
        size_t sz = (size_t)(it->second.offset_end - it->second.offset_start);
        if (nbytes) *nbytes = sz;
        return payload.data() + it->second.offset_start;
    }

    const StEntry& entry(const std::string& name) const {
        auto it = entries.find(name);
        if (it == entries.end()) {
            std::fprintf(stderr, "[safetensors] tensor '%s' not found\n", name.c_str());
            std::exit(6);
        }
        return it->second;
    }
};

namespace safetensors_detail {

inline std::vector<int64_t> parse_i64_list(const std::string& s) {
    std::vector<int64_t> out;
    std::string buf;
    for (char c : s) {
        if (c == '[' || c == ']' || c == ',' || std::isspace((unsigned char)c)) {
            if (!buf.empty()) { out.push_back(std::stoll(buf)); buf.clear(); }
        } else {
            buf.push_back(c);
        }
    }
    if (!buf.empty()) out.push_back(std::stoll(buf));
    return out;
}

inline std::unordered_map<std::string, StEntry> parse_header(const std::string& text) {
    std::unordered_map<std::string, StEntry> out;
    size_t i = 0, n = text.size();

    auto skip_ws = [&]() { while (i < n && std::isspace((unsigned char)text[i])) ++i; };
    auto expect = [&](char c) {
        skip_ws();
        if (i >= n || text[i] != c) {
            std::fprintf(stderr, "[safetensors] expected '%c' at %zu\n", c, i);
            std::exit(4);
        }
        ++i;
    };
    auto read_string = [&]() -> std::string {
        skip_ws();
        if (i >= n || text[i] != '"') {
            std::fprintf(stderr, "[safetensors] expected \" at %zu\n", i);
            std::exit(4);
        }
        ++i;
        std::string s;
        while (i < n && text[i] != '"') { s.push_back(text[i++]); }
        if (i < n) ++i;
        return s;
    };
    auto parse_value = [&]() -> std::string {
        // returns raw substring of the next JSON value; skips it
        skip_ws();
        if (i >= n) return {};
        char c = text[i];
        if (c == '"') return read_string();
        if (c == '[' || c == '{') {
            size_t start = i;
            char open = c, close = (c == '[') ? ']' : '}';
            int depth = 0;
            while (i < n) {
                if (text[i] == open) ++depth;
                else if (text[i] == close) { --depth; if (depth == 0) { ++i; break; } }
                ++i;
            }
            return text.substr(start, i - start);
        }
        // number / bool / null
        size_t start = i;
        while (i < n && text[i] != ',' && text[i] != '}' && !std::isspace((unsigned char)text[i])) ++i;
        return text.substr(start, i - start);
    };

    expect('{');
    skip_ws();
    bool first = true;
    while (i < n && text[i] != '}') {
        if (!first) expect(',');
        first = false;
        skip_ws();
        std::string key = read_string();
        expect(':');
        skip_ws();
        if (text[i] != '{') {
            // Skip non-object entries like "__metadata__"
            parse_value();
            skip_ws();
            continue;
        }
        expect('{');
        StEntry e;
        skip_ws();
        bool fi = true;
        while (i < n && text[i] != '}') {
            if (!fi) expect(',');
            fi = false;
            skip_ws();
            std::string k = read_string();
            expect(':');
            skip_ws();
            std::string val = parse_value();
            if (k == "dtype") e.dtype = val;
            else if (k == "shape") e.shape = parse_i64_list(val);
            else if (k == "data_offsets") {
                auto off = parse_i64_list(val);
                if (off.size() >= 2) { e.offset_start = off[0]; e.offset_end = off[1]; }
            }
            skip_ws();
        }
        expect('}');
        skip_ws();
        out[key] = std::move(e);
    }
    return out;
}

} // namespace safetensors_detail

inline SafetensorsFile SafetensorsFile::load(const std::string& path) {
    SafetensorsFile s;
    std::ifstream fin(path, std::ios::binary);
    if (!fin.good()) {
        std::fprintf(stderr, "[safetensors] failed to open %s\n", path.c_str());
        std::exit(5);
    }
    uint64_t hlen = 0;
    fin.read(reinterpret_cast<char*>(&hlen), 8);
    std::string header(hlen, '\0');
    fin.read(header.data(), hlen);
    s.entries = safetensors_detail::parse_header(header);

    fin.seekg(0, std::ios::end);
    size_t total = fin.tellg();
    size_t payload_size = total - 8 - hlen;
    s.payload.resize(payload_size);
    fin.seekg(8 + hlen, std::ios::beg);
    fin.read(reinterpret_cast<char*>(s.payload.data()), payload_size);
    return s;
}
