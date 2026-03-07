// Tests for the tokenizer: encode/decode roundtrip, special tokens,
// emoji handling, and edge cases with long inputs.
// Requires a GGUF model file (set MODEL_PATH env var, or auto-detects).

#include "../src/tokenizer.h"
#include "../src/download.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

static int g_failures = 0;
static int g_tests = 0;

#define EXPECT_EQ(a, b) do { \
    g_tests++; \
    auto _a = (a); auto _b = (b); \
    if (_a != _b) { \
        fprintf(stderr, "FAIL %s:%d: %s != %s\n", __FILE__, __LINE__, #a, #b); \
        g_failures++; \
    } \
} while(0)

#define EXPECT_TRUE(x) do { \
    g_tests++; \
    if (!(x)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
        g_failures++; \
    } \
} while(0)

static const char* DEFAULT_MODEL_SPEC = "unsloth/Qwen3.5-9B-GGUF:BF16";

static std::string resolve_test_model() {
    std::string spec = DEFAULT_MODEL_SPEC;
    if (const char* p = getenv("MODEL_PATH")) {
        spec = p;
    }

    printf("Resolving model: %s\n", spec.c_str());
    std::string path = resolve_model(spec, "",
        [](int64_t downloaded, int64_t total) {
            if (total > 0) {
                fprintf(stderr, "\rDownloading: %.1f%%", downloaded * 100.0 / total);
            } else {
                fprintf(stderr, "\rDownloading: %.1f MB", downloaded / 1e6);
            }
        });

    if (!path.empty()) {
        fprintf(stderr, "\n");
    }
    return path;
}

static void test_encode_decode_roundtrip(Tokenizer& tok) {
    printf("  test_encode_decode_roundtrip...\n");

    std::vector<std::string> inputs = {
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "1234567890",
        "x",
        "",
    };

    for (const auto& text : inputs) {
        auto ids = tok.encode(text);
        std::string decoded = tok.decode(ids);
        EXPECT_EQ(decoded, text);
    }
}

static void test_special_tokens(Tokenizer& tok) {
    printf("  test_special_tokens...\n");

    std::string text = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
    auto ids = tok.encode(text);

    // Should contain special token IDs, not byte-level encoding of "<|im_start|>"
    EXPECT_TRUE(ids.size() > 0);
    EXPECT_TRUE(ids.size() < 30); // special tokens compress it significantly

    // First token should be <|im_start|> (not 'H' or '<')
    std::string first_decoded = tok.decode(ids[0]);
    EXPECT_EQ(first_decoded, "<|im_start|>");
}

static void test_emoji_roundtrip(Tokenizer& tok) {
    printf("  test_emoji_roundtrip...\n");

    std::vector<std::string> emoji_texts = {
        "Hello 😀",
        "🎉🔥🚀",
        "emoji: 👍 done",
        "中文测试",
        "Mixed: café résumé naïve",
        "Flags: 🇺🇸 🇯🇵",
        "Family: 👨‍👩‍👧‍👦",
        "Math: ∑∫∂ ≤≥≠",
    };

    for (const auto& text : emoji_texts) {
        auto ids = tok.encode(text);
        EXPECT_TRUE(ids.size() > 0);
        std::string decoded = tok.decode(ids);
        if (decoded != text) {
            fprintf(stderr, "    roundtrip mismatch for: %s\n", text.c_str());
            fprintf(stderr, "    got:                    %s\n", decoded.c_str());
            fprintf(stderr, "    tokens: %zu\n", ids.size());
        }
        EXPECT_EQ(decoded, text);
    }
}

static void test_single_token_decode_reassembly(Tokenizer& tok) {
    printf("  test_single_token_decode_reassembly...\n");

    // Decode token-by-token and verify concatenation matches bulk decode
    std::string text = "Hello 😀 world! 🎉 中文";
    auto ids = tok.encode(text);

    std::string bulk = tok.decode(ids);
    std::string piecewise;
    for (int id : ids) {
        piecewise += tok.decode(id);
    }

    EXPECT_EQ(piecewise, bulk);
    EXPECT_EQ(piecewise, text);
}

static void test_long_prompt_encode(Tokenizer& tok) {
    printf("  test_long_prompt_encode...\n");

    // Generate a long prompt (~100KB of text)
    std::string long_text;
    for (int i = 0; i < 10000; i++) {
        long_text += "The quick brown fox jumps over the lazy dog. ";
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    auto ids = tok.encode(long_text);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("    %zu chars -> %zu tokens in %.1f ms\n", long_text.size(), ids.size(), ms);

    EXPECT_TRUE(ids.size() > 1000);

    // Roundtrip
    std::string decoded = tok.decode(ids);
    EXPECT_EQ(decoded, long_text);

    // Performance: encoding 450KB shouldn't take more than 30s
    EXPECT_TRUE(ms < 30000.0);
}

static void test_long_emoji_prompt(Tokenizer& tok) {
    printf("  test_long_emoji_prompt...\n");

    // Stress test with repeated emoji
    std::string emoji_text;
    for (int i = 0; i < 1000; i++) {
        emoji_text += "😀🎉🔥🚀👍 ";
    }

    auto ids = tok.encode(emoji_text);
    EXPECT_TRUE(ids.size() > 0);

    std::string decoded = tok.decode(ids);
    EXPECT_EQ(decoded, emoji_text);
}

static void test_eos_token(Tokenizer& tok) {
    printf("  test_eos_token...\n");

    int eos = tok.eos_token_id();
    EXPECT_TRUE(eos >= 0);
    EXPECT_TRUE(eos < tok.vocab_size());

    // EOS token should decode to the special token string
    std::string eos_str = tok.decode(eos);
    EXPECT_TRUE(eos_str.find("|>") != std::string::npos || !eos_str.empty());
}

static void test_empty_input(Tokenizer& tok) {
    printf("  test_empty_input...\n");
    auto ids = tok.encode("");
    EXPECT_EQ(ids.size(), (size_t)0);
    EXPECT_EQ(tok.decode(ids), std::string(""));
}

static void test_whitespace_only(Tokenizer& tok) {
    printf("  test_whitespace_only...\n");
    std::vector<std::string> ws = {" ", "  ", "\n", "\t", "\n\n\n"};
    for (const auto& text : ws) {
        auto ids = tok.encode(text);
        EXPECT_TRUE(ids.size() > 0);
        EXPECT_EQ(tok.decode(ids), text);
    }
}

static void test_out_of_range_decode(Tokenizer& tok) {
    printf("  test_out_of_range_decode...\n");
    // Should not crash, just return empty
    EXPECT_EQ(tok.decode(-1), std::string(""));
    EXPECT_EQ(tok.decode(tok.vocab_size() + 100), std::string(""));
    EXPECT_EQ(tok.decode(999999999), std::string(""));
}

int main() {
    std::string model_path = resolve_test_model();
    if (model_path.empty()) {
        fprintf(stderr, "Failed to resolve model. Set MODEL_PATH to a local .gguf or HF tag.\n");
        return 1;
    }

    printf("=== Tokenizer Tests ===\n");
    printf("Model: %s\n", model_path.c_str());

    Tokenizer tok;
    if (!tok.load(model_path)) {
        fprintf(stderr, "Failed to load tokenizer from %s\n", model_path.c_str());
        return 1;
    }

    test_encode_decode_roundtrip(tok);
    test_special_tokens(tok);
    test_emoji_roundtrip(tok);
    test_single_token_decode_reassembly(tok);
    test_long_prompt_encode(tok);
    test_long_emoji_prompt(tok);
    test_eos_token(tok);
    test_empty_input(tok);
    test_whitespace_only(tok);
    test_out_of_range_decode(tok);

    printf("\n%d/%d tests passed\n", g_tests - g_failures, g_tests);
    return g_failures > 0 ? 1 : 0;
}
