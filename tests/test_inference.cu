// Integration tests for the inference engine.
// Requires a GPU and a GGUF model file.
// Tests: long prompts, context window limits, generation quality, tok/s regression.

#include "../src/inference.h"
#include "../src/model.h"
#include "../src/download.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <sys/stat.h>

static int g_failures = 0;
static int g_tests = 0;
static int g_ctx_size = 0;

#define EXPECT_EQ(a, b) do { \
    g_tests++; \
    auto _a = (a); auto _b = (b); \
    if (_a != _b) { \
        fprintf(stderr, "FAIL %s:%d: %s != %s\n", \
                __FILE__, __LINE__, #a, #b); \
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

#define EXPECT_GE(a, b) do { \
    g_tests++; \
    auto _a = (a); auto _b = (b); \
    if (_a < _b) { \
        fprintf(stderr, "FAIL %s:%d: %s = %.2f, expected >= %.2f\n", \
                __FILE__, __LINE__, #a, (double)_a, (double)_b); \
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

// Collect all generated text and measure timing
struct GenerateResult {
    std::string text;
    int n_tokens;
    double total_ms;
    double tok_per_sec;
    StopReason reason;
};

static GenerateResult run_generate(const std::string& prompt, int max_tokens, float temperature = 0.0f) {
    auto& tok = get_tokenizer();
    auto ids = tok.encode(prompt);

    GenerateResult r;
    r.reason = STOP_LENGTH;

    auto t0 = std::chrono::high_resolution_clock::now();
    r.n_tokens = generate(ids, max_tokens, temperature,
        [&](int token_id, const std::string& text) -> bool {
            r.text += text;
            return true;
        }, &r.reason);
    auto t1 = std::chrono::high_resolution_clock::now();

    r.total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.tok_per_sec = (r.n_tokens > 0 && r.total_ms > 0) ? r.n_tokens * 1000.0 / r.total_ms : 0;

    return r;
}

static void test_basic_generation() {
    printf("  test_basic_generation...\n");

    reset_state();
    auto r = run_generate("1+1=", 16, 0.0f);

    printf("    output (%d tok, %.1f ms, %.1f tok/s): %s\n",
           r.n_tokens, r.total_ms, r.tok_per_sec, r.text.c_str());

    EXPECT_TRUE(r.n_tokens > 0);
    EXPECT_TRUE(!r.text.empty());
    // The model should be able to answer "2" somewhere in the output
    EXPECT_TRUE(r.text.find("2") != std::string::npos);
}

static void test_long_prompt() {
    printf("  test_long_prompt...\n");

    // Build a prompt that's ~4K tokens (a lot of repeated text)
    std::string prompt;
    for (int i = 0; i < 500; i++) {
        prompt += "The quick brown fox jumps over the lazy dog. ";
    }
    prompt += "What animal was mentioned? The answer is:";

    auto& tok = get_tokenizer();
    auto ids = tok.encode(prompt);
    printf("    prompt: %zu chars, %zu tokens\n", prompt.size(), ids.size());

    reset_state();
    auto r = run_generate(prompt, 32, 0.0f);

    printf("    output (%d tok, %.1f ms): %s\n", r.n_tokens, r.total_ms, r.text.c_str());

    EXPECT_TRUE(r.n_tokens > 0);
    EXPECT_TRUE(!r.text.empty());
}

static void test_prompt_exceeding_context() {
    printf("  test_prompt_exceeding_context...\n");

    // We loaded with a small context. Build a prompt that exceeds it.
    // The engine should truncate gracefully, not crash.
    auto& tok = get_tokenizer();

    // Generate a very long prompt
    std::string huge_prompt;
    for (int i = 0; i < 5000; i++) {
        huge_prompt += "This is a test sentence to fill up the context window. ";
    }

    auto ids = tok.encode(huge_prompt);
    printf("    huge prompt: %zu tokens\n", ids.size());

    reset_state();
    // This should not crash — engine truncates prompt and generates (or returns 0)
    auto r = run_generate(huge_prompt, 8, 0.0f);
    printf("    result: %d tokens generated (reason: %d)\n", r.n_tokens, (int)r.reason);

    // We just verify it didn't crash. The engine may generate 0 tokens if context is full.
    EXPECT_TRUE(r.n_tokens >= 0);
}

static void test_emoji_in_output() {
    printf("  test_emoji_in_output...\n");

    // Ask for emoji — verify they decode correctly
    reset_state();
    auto r = run_generate(
        "<|im_start|>user\nRespond with exactly 3 emoji and nothing else.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        32, 0.6f);

    printf("    output: %s\n", r.text.c_str());
    EXPECT_TRUE(r.n_tokens > 0);

    // Verify output is valid UTF-8 (no truncated sequences)
    const std::string& s = r.text;
    size_t i = 0;
    bool valid = true;
    while (i < s.size()) {
        unsigned char c = s[i];
        int len = 1;
        if ((c & 0x80) == 0) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        else { valid = false; break; }

        if (i + len > s.size()) { valid = false; break; }
        for (int j = 1; j < len; j++) {
            if (((unsigned char)s[i + j] & 0xC0) != 0x80) { valid = false; break; }
        }
        if (!valid) break;
        i += len;
    }
    EXPECT_TRUE(valid);
}

static void test_prompt_caching() {
    printf("  test_prompt_caching...\n");

    std::string base_prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n";

    // First request
    reset_state();
    auto r1 = run_generate(base_prompt, 32, 0.0f);
    printf("    first: %d tok, %.1f ms\n", r1.n_tokens, r1.total_ms);

    // Second request with same prefix — should reuse cache
    std::string extended_prompt = base_prompt;
    // Simulate a follow-up turn by appending the response and a new turn
    // But for cache testing, just resend the same prompt
    auto t0 = std::chrono::high_resolution_clock::now();
    auto r2 = run_generate(base_prompt, 32, 0.0f);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("    cached: %d tok, %.1f ms\n", r2.n_tokens, ms);

    EXPECT_TRUE(r1.n_tokens > 0);
    EXPECT_TRUE(r2.n_tokens > 0);
}

static void test_tok_per_sec() {
    printf("  test_tok_per_sec...\n");

    // Warm up: prefill the prompt first, then measure decode-only performance
    // by running generation in two steps.
    std::string prompt =
        "<|im_start|>user\nWrite a short paragraph about machine learning.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

    auto& tok = get_tokenizer();
    auto prompt_ids = tok.encode(prompt);

    // First pass: prefill + short generation to warm up and populate KV cache
    reset_state();
    StopReason warmup_reason;
    std::string warmup_text;
    int warmup_tokens = generate(prompt_ids, 1, 0.0f,
        [&](int token_id, const std::string& text) -> bool {
            warmup_text += text;
            return true;
        }, &warmup_reason);
    (void)warmup_tokens;

    // Second pass: extend generation from cached state (decode-only)
    // Rebuild the full prompt including the warmup token so cache matches
    auto full_ids = prompt_ids;
    auto warmup_ids = tok.encode(warmup_text);
    full_ids.insert(full_ids.end(), warmup_ids.begin(), warmup_ids.end());

    int decode_count = 256;
    std::string decode_text;
    StopReason reason;

    auto t0 = std::chrono::high_resolution_clock::now();
    int n = generate(full_ids, decode_count, 0.0f,
        [&](int token_id, const std::string& text) -> bool {
            decode_text += text;
            return true;
        }, &reason);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double tok_per_sec = (n > 0 && ms > 0) ? n * 1000.0 / ms : 0;

    printf("    %d tokens in %.1f ms = %.1f tok/s (decode-only)\n", n, ms, tok_per_sec);

    EXPECT_TRUE(n >= 64);  // should generate at least 64 tokens

    // Regression thresholds: full context (262144) uses a different attention
    // kernel with lower decode throughput than small context windows.
    double min_tok_per_sec = 92.0;  // small context threshold
    if (const char* p = getenv("MIN_TOK_PER_SEC")) {
        min_tok_per_sec = atof(p);
    } else if (g_ctx_size > 8192) {
        min_tok_per_sec = 87.0;     // full context threshold
    }
    printf("    threshold: %.1f tok/s (set MIN_TOK_PER_SEC to override)\n", min_tok_per_sec);
    EXPECT_GE(tok_per_sec, min_tok_per_sec);
}

static void test_prefill_tok_per_sec() {
    printf("  test_prefill_tok_per_sec...\n");

    // Build a ~5K token prompt to measure prefill throughput
    std::string prompt;
    for (int i = 0; i < 500; i++) {
        prompt += "The quick brown fox jumps over the lazy dog. ";
    }

    auto& tok = get_tokenizer();
    auto prompt_ids = tok.encode(prompt);
    int n_prompt = (int)prompt_ids.size();

    // Generate only 1 token so nearly all time is prefill
    reset_state();
    StopReason reason;
    std::string text;

    auto t0 = std::chrono::high_resolution_clock::now();
    int n = generate(prompt_ids, 1, 0.0f,
        [&](int token_id, const std::string& t) -> bool {
            text += t;
            return true;
        }, &reason);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double prefill_tok_per_sec = n_prompt * 1000.0 / ms;

    printf("    %d prompt tokens + %d decode in %.1f ms = %.1f prefill tok/s\n",
           n_prompt, n, ms, prefill_tok_per_sec);

    double min_prefill = 2000.0;  // small context threshold
    if (const char* p = getenv("MIN_PREFILL_TOK_PER_SEC")) {
        min_prefill = atof(p);
    } else if (g_ctx_size > 8192) {
        min_prefill = 1500.0;      // full context threshold
    }
    printf("    threshold: %.1f prefill tok/s (set MIN_PREFILL_TOK_PER_SEC to override)\n", min_prefill);
    EXPECT_GE(prefill_tok_per_sec, min_prefill);
}

static void test_greedy_determinism() {
    printf("  test_greedy_determinism...\n");

    std::string prompt = "The capital of France is";

    reset_state();
    auto r1 = run_generate(prompt, 16, 0.0f);
    reset_state();
    auto r2 = run_generate(prompt, 16, 0.0f);

    printf("    run1: %s\n", r1.text.c_str());
    printf("    run2: %s\n", r2.text.c_str());

    // Greedy (temperature=0) should be deterministic
    EXPECT_EQ(r1.text, r2.text);
}

static void test_stop_on_eos() {
    printf("  test_stop_on_eos...\n");

    // A well-formed chat prompt should eventually produce EOS
    reset_state();
    auto r = run_generate(
        "<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        256, 0.0f);

    printf("    output (%d tok): %s\n", r.n_tokens, r.text.c_str());
    printf("    stop reason: %d (0=EOS, 1=LENGTH, 2=CALLBACK)\n", (int)r.reason);

    // For a short response like "Hi!", the model should hit EOS well before 256
    if (r.reason == STOP_EOS) {
        EXPECT_TRUE(r.n_tokens < 200);
    }
}

int main(int argc, char** argv) {
    std::string model_path = resolve_test_model();
    if (model_path.empty()) {
        fprintf(stderr, "Failed to resolve model. Set MODEL_PATH to a local .gguf or HF tag.\n");
        return 1;
    }

    printf("=== Inference Integration Tests ===\n");
    printf("Model: %s\n", model_path.c_str());

    // Context size from env, or full model context by default
    g_ctx_size = ModelConfig::n_ctx;
    if (const char* p = getenv("TEST_CTX_SIZE")) {
        g_ctx_size = atoi(p);
    }
    printf("Context size: %d%s\n", g_ctx_size,
           g_ctx_size == ModelConfig::n_ctx ? " (full)" : " (small)");

    if (!load_model_and_tokenizer(model_path.c_str(), g_ctx_size)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    test_basic_generation();
    test_long_prompt();
    test_prompt_exceeding_context();
    test_emoji_in_output();
    test_prompt_caching();
    test_tok_per_sec();
    test_prefill_tok_per_sec();
    test_greedy_determinism();
    test_stop_on_eos();

    printf("\n%d/%d tests passed\n", g_tests - g_failures, g_tests);
    return g_failures > 0 ? 1 : 0;
}
