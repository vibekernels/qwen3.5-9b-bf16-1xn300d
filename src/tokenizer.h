#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

class Tokenizer {
public:
    bool load(const std::string& gguf_path);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(int token_id) const;
    std::string decode(const std::vector<int>& token_ids) const;

    int eos_token_id() const { return eos_id_; }
    int bos_token_id() const { return bos_id_; }
    int vocab_size() const { return (int)id_to_token_.size(); }

private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::pair<std::string, std::string>> merges_;

    int eos_id_ = 248046;  // <|im_end|>
    int bos_id_ = -1;      // Qwen doesn't use BOS

    // BPE merge ranking (pair -> priority, lower = higher priority)
    std::unordered_map<std::string, int> merge_rank_;

    // Apply BPE merges to a list of tokens
    std::vector<std::string> bpe(const std::vector<std::string>& tokens) const;
};
