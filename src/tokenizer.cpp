#include "tokenizer.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <sstream>

// Simple GGUF reader for tokenizer data only
static std::string read_string(FILE* f) {
    uint64_t len;
    fread(&len, 8, 1, f);
    std::string s(len, '\0');
    fread(&s[0], 1, len, f);
    return s;
}

struct GGUFValue {
    uint32_t type;
    uint32_t u32_val;
    float f32_val;
    std::string str_val;
    std::vector<std::string> str_arr;
    std::vector<int32_t> i32_arr;
};

static GGUFValue read_value(FILE* f, uint32_t vtype) {
    GGUFValue v;
    v.type = vtype;
    switch (vtype) {
        case 0: { uint8_t x; fread(&x, 1, 1, f); v.u32_val = x; break; }
        case 1: { int8_t x; fread(&x, 1, 1, f); v.u32_val = x; break; }
        case 2: { uint16_t x; fread(&x, 2, 1, f); v.u32_val = x; break; }
        case 3: { int16_t x; fread(&x, 2, 1, f); v.u32_val = x; break; }
        case 4: fread(&v.u32_val, 4, 1, f); break;
        case 5: { int32_t x; fread(&x, 4, 1, f); v.u32_val = (uint32_t)x; break; }
        case 6: fread(&v.f32_val, 4, 1, f); break;
        case 7: { uint8_t x; fread(&x, 1, 1, f); v.u32_val = x; break; }
        case 8: v.str_val = read_string(f); break;
        case 9: {
            uint32_t arr_type;
            uint64_t arr_len;
            fread(&arr_type, 4, 1, f);
            fread(&arr_len, 8, 1, f);
            if (arr_type == 8) { // string array
                v.str_arr.resize(arr_len);
                for (uint64_t i = 0; i < arr_len; i++) {
                    v.str_arr[i] = read_string(f);
                }
            } else if (arr_type == 5) { // int32 array
                v.i32_arr.resize(arr_len);
                fread(v.i32_arr.data(), 4, arr_len, f);
            } else {
                // Skip other array types
                for (uint64_t i = 0; i < arr_len; i++) {
                    GGUFValue dummy = read_value(f, arr_type);
                    (void)dummy;
                }
            }
            break;
        }
        case 10: { uint64_t x; fread(&x, 8, 1, f); v.u32_val = (uint32_t)x; break; }
        case 11: { int64_t x; fread(&x, 8, 1, f); v.u32_val = (uint32_t)x; break; }
        case 12: { double x; fread(&x, 8, 1, f); v.f32_val = (float)x; break; }
    }
    return v;
}

bool Tokenizer::load(const std::string& gguf_path) {
    FILE* f = fopen(gguf_path.c_str(), "rb");
    if (!f) return false;

    // Skip GGUF header
    fseek(f, 4, SEEK_CUR); // magic
    uint32_t version;
    fread(&version, 4, 1, f);
    uint64_t n_tensors, n_kv;
    fread(&n_tensors, 8, 1, f);
    fread(&n_kv, 8, 1, f);

    // Read KV pairs, extract tokenizer data
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = read_string(f);
        uint32_t vtype;
        fread(&vtype, 4, 1, f);
        GGUFValue val = read_value(f, vtype);

        if (key == "tokenizer.ggml.tokens") {
            id_to_token_ = std::move(val.str_arr);
        } else if (key == "tokenizer.ggml.merges") {
            merges_.clear();
            for (auto& m : val.str_arr) {
                size_t space = m.find(' ');
                if (space != std::string::npos) {
                    merges_.emplace_back(m.substr(0, space), m.substr(space + 1));
                }
            }
        } else if (key == "tokenizer.ggml.eos_token_id") {
            eos_id_ = (int)val.u32_val;
        }
    }
    fclose(f);

    // Build token -> id map
    for (size_t i = 0; i < id_to_token_.size(); i++) {
        token_to_id_[id_to_token_[i]] = (int)i;
    }

    // Build merge rank map
    for (size_t i = 0; i < merges_.size(); i++) {
        std::string pair_key = merges_[i].first + " " + merges_[i].second;
        merge_rank_[pair_key] = (int)i;
    }

    printf("Tokenizer loaded: %zu tokens, %zu merges\n", id_to_token_.size(), merges_.size());
    return true;
}

// GPT-2 byte-to-unicode mapping
// Bytes 33-126, 161-172, 174-255 map to themselves (as unicode)
// Other bytes (0-32, 127-160, 173) map to U+0100 + sequential offset
static std::string byte_to_unicode(unsigned char b) {
    // Printable ranges that map to themselves
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
        return std::string(1, (char)b);
    }
    // Non-printable bytes map to U+0100+ range
    // Build the mapping: iterate through 0-255, count non-printable bytes
    static std::string table[256];
    static bool initialized = false;
    if (!initialized) {
        int offset = 0;
        for (int i = 0; i < 256; i++) {
            if ((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174)) {
                table[i] = std::string(1, (char)i);
            } else {
                int cp = 256 + offset;
                offset++;
                // Encode code point as UTF-8
                if (cp < 0x80) {
                    table[i] = std::string(1, (char)cp);
                } else if (cp < 0x800) {
                    char utf8[3];
                    utf8[0] = (char)(0xC0 | (cp >> 6));
                    utf8[1] = (char)(0x80 | (cp & 0x3F));
                    utf8[2] = 0;
                    table[i] = std::string(utf8);
                }
            }
        }
        initialized = true;
    }
    return table[b];
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    if (text.empty()) return {};

    // Step 1: Convert each byte to its GPT-2 unicode representation
    std::string unicode_text;
    for (unsigned char c : text) {
        unicode_text += byte_to_unicode(c);
    }

    // Step 2: Try to find longest matching tokens greedily, then apply BPE
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < unicode_text.size()) {
        // Find longest matching token from vocabulary
        bool found = false;
        size_t max_len = std::min(unicode_text.size() - i, (size_t)64);
        for (size_t len = max_len; len >= 1; len--) {
            std::string sub = unicode_text.substr(i, len);
            if (token_to_id_.count(sub)) {
                tokens.push_back(sub);
                i += len;
                found = true;
                break;
            }
        }

        if (!found) {
            // Single character token
            tokens.push_back(unicode_text.substr(i, 1));
            i++;
        }
    }

    // Step 3: Apply BPE merges
    tokens = bpe(tokens);

    // Step 4: Convert to IDs
    std::vector<int> ids;
    for (const auto& tok : tokens) {
        auto it = token_to_id_.find(tok);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            fprintf(stderr, "Warning: unknown token '%s' (len=%zu)\n", tok.c_str(), tok.size());
        }
    }

    return ids;
}

std::vector<std::string> Tokenizer::bpe(const std::vector<std::string>& input_tokens) const {
    if (input_tokens.size() <= 1) return input_tokens;

    std::vector<std::string> tokens = input_tokens;

    while (tokens.size() > 1) {
        // Find the highest-priority merge (lowest rank)
        int best_rank = INT32_MAX;
        int best_pos = -1;

        for (size_t i = 0; i + 1 < tokens.size(); i++) {
            std::string pair_key = tokens[i] + " " + tokens[i + 1];
            auto it = merge_rank_.find(pair_key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = (int)i;
            }
        }

        if (best_pos < 0) break; // No more merges possible

        // Apply the merge
        std::vector<std::string> new_tokens;
        for (size_t i = 0; i < tokens.size(); i++) {
            if ((int)i == best_pos) {
                new_tokens.push_back(tokens[i] + tokens[i + 1]);
                i++; // skip next
            } else {
                new_tokens.push_back(tokens[i]);
            }
        }
        tokens = std::move(new_tokens);
    }

    return tokens;
}

std::string Tokenizer::decode(int token_id) const {
    if (token_id < 0 || token_id >= (int)id_to_token_.size()) return "";
    const std::string& tok = id_to_token_[token_id];

    // Decode GPT-2 byte-level encoding back to UTF-8 bytes
    std::string result;
    size_t i = 0;
    while (i < tok.size()) {
        unsigned char c = tok[i];
        if (c == 0xC4 && i + 1 < tok.size() && (unsigned char)tok[i + 1] == 0xA0) {
            // Ġ -> space
            result += ' ';
            i += 2;
        } else if (c >= 0xC4 && c <= 0xC5 && i + 1 < tok.size()) {
            // Check if this is a byte-mapped char (U+0100 to U+01FF)
            unsigned char c2 = tok[i + 1];
            int cp = ((c & 0x1F) << 6) | (c2 & 0x3F);
            if (cp >= 0x100 && cp <= 0x1FF) {
                result += (char)(cp - 0x100);
                i += 2;
            } else {
                result += tok[i];
                i++;
            }
        } else {
            result += tok[i];
            i++;
        }
    }
    return result;
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) const {
    std::string result;
    for (int id : token_ids) {
        result += decode(id);
    }
    return result;
}
