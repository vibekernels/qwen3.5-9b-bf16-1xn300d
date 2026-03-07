#include "inference.h"
#include "download.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s -m <model_or_hf_tag> -p <prompt> [-n <max_tokens>] [-t <temperature>] [--model-dir <dir>]\n", prog);
    fprintf(stderr, "\n  -m   Local .gguf file path, or HuggingFace tag like org/repo:quant\n");
    fprintf(stderr, "       Example: -m unsloth/Qwen3.5-9B-GGUF:BF16\n");
    fprintf(stderr, "  --model-dir  Local cache directory (default: ~/.cache/qwen-models)\n");
}

int main(int argc, char** argv) {
    std::string model_spec;
    std::string model_dir;
    std::string prompt;
    int max_gen_tokens = 128;
    float temperature = 0.8f;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_spec = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_gen_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        }
    }

    if (model_spec.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    if (prompt.empty()) {
        prompt = "Hello, world!";
    }

    // Resolve model (downloads from HF if needed)
    std::string model_path = resolve_model(model_spec, model_dir);
    if (model_path.empty()) {
        fprintf(stderr, "Failed to resolve model: %s\n", model_spec.c_str());
        return 1;
    }

    printf("Model: %s\n", model_path.c_str());
    printf("Prompt: %s\n", prompt.c_str());
    printf("Max tokens: %d\n", max_gen_tokens);
    printf("Temperature: %.2f\n\n", temperature);

    int max_ctx = (int)prompt.size() + max_gen_tokens + 256;
    if (max_ctx < 4096) max_ctx = 4096;

    if (!load_model_and_tokenizer(model_path.c_str(), max_ctx)) {
        return 1;
    }

    auto& tokenizer = get_tokenizer();

    // Tokenize prompt
    std::vector<int> prompt_tokens = tokenizer.encode(prompt);
    printf("Prompt tokens (%zu): ", prompt_tokens.size());
    for (int t : prompt_tokens) printf("%d ", t);
    printf("\n\n");

    printf("Generating...\n");
    printf("%s", prompt.c_str());
    fflush(stdout);

    auto t_start = std::chrono::high_resolution_clock::now();

    int n_generated = generate(prompt_tokens, max_gen_tokens, temperature,
        [](int token_id, const std::string& text) -> bool {
            printf("%s", text.c_str());
            fflush(stdout);
            return true;
        });

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    printf("\n\n--- Performance ---\n");
    printf("Prompt tokens: %zu\n", prompt_tokens.size());
    printf("Generated tokens: %d (%.1f ms, %.1f tok/s)\n",
        n_generated, total_ms, n_generated * 1000.0 / total_ms);

    return 0;
}
