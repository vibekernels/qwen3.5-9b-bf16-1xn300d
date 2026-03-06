// Quick hack to dump ggml's hidden states per layer
// Compile: g++ -std=c++17 -I/tmp/llama.cpp/include -I/tmp/llama.cpp/common -L/tmp/llama.cpp/build/src -L/tmp/llama.cpp/build/common -o dump_ggml_hidden dump_ggml_hidden.cpp -lllama -lcommon -Wl,-rpath,/tmp/llama.cpp/build/src:/tmp/llama.cpp/build/common
#include "llama.h"
#include "llama-cpp.h"
#include "common.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>

struct dump_data {
    std::vector<uint8_t> buf;
};

static bool dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true;
    auto * dd = (dump_data *)user_data;
    const char * name = t->name;

    // We want "l_out-N" tensors (hidden state after each layer)
    // and "result_output" (final logits)
    bool is_result = (strcmp(name, "result_output") == 0);
    // Match "ffn_out-N" exactly (not "ffn_out-N (reshaped)" etc.)
    bool is_ffn_out = (strstr(name, "ffn_out-") != nullptr && strstr(name, "(") == nullptr);
    bool is_linear_attn = (strstr(name, "linear_attn_out-") != nullptr && strstr(name, "(") == nullptr);
    bool is_attn_output = (strstr(name, "attn_output-") != nullptr && strstr(name, "(") == nullptr);
    bool is_final_output = (strstr(name, "final_output-") != nullptr && strstr(name, "(") == nullptr);
    bool is_z = (strstr(name, "z-") != nullptr && strstr(name, "(") == nullptr);
    bool is_q_conv = (strstr(name, "q_conv_predelta-") != nullptr && strstr(name, "(") == nullptr);
    bool is_k_conv = (strstr(name, "k_conv_predelta-") != nullptr && strstr(name, "(") == nullptr);
    bool is_v_conv = (strstr(name, "v_conv_predelta-") != nullptr && strstr(name, "(") == nullptr);

    if (!is_result && !is_ffn_out && !is_linear_attn && !is_attn_output && !is_final_output && !is_z && !is_q_conv && !is_k_conv && !is_v_conv) return true;

    size_t nbytes = ggml_nbytes(t);
    dd->buf.resize(nbytes);
    ggml_backend_tensor_get(t, dd->buf.data(), 0, nbytes);

    // Get last token's data (position n_tokens-1)
    int64_t n_embd = t->ne[0];
    int64_t n_tokens = t->ne[1];

    // Determine element size
    size_t elem_size = ggml_type_size(t->type);
    int64_t row_bytes = n_embd * elem_size;

    // Get last token's hidden state
    int last_tok = n_tokens - 1;

    // Convert to f32 if needed
    std::vector<float> f32_data(n_embd);
    uint8_t * row_ptr = dd->buf.data() + last_tok * row_bytes;

    if (t->type == GGML_TYPE_F32) {
        memcpy(f32_data.data(), row_ptr, n_embd * sizeof(float));
    } else if (t->type == GGML_TYPE_BF16) {
        uint16_t * bf16 = (uint16_t *)row_ptr;
        for (int64_t i = 0; i < n_embd; i++) {
            uint32_t tmp = (uint32_t)bf16[i] << 16;
            memcpy(&f32_data[i], &tmp, 4);
        }
    } else {
        fprintf(stderr, "Unsupported type for %s: %d\n", name, t->type);
        return true;
    }

    // Compute stats
    double sum = 0, l2 = 0;
    for (int64_t i = 0; i < n_embd; i++) {
        sum += f32_data[i];
        l2 += (double)f32_data[i] * f32_data[i];
    }
    printf("ggml %s (tok=%lld): sum=%.4f L2=%.4f ne=[%lld,%lld] type=%s\n",
           name, (long long)last_tok, sum, sqrt(l2),
           (long long)n_embd, (long long)n_tokens, ggml_type_name(t->type));

    // Dump all tokens' data for position comparison
    for (int tok = 0; tok < n_tokens; tok++) {
        uint8_t * rp = dd->buf.data() + tok * row_bytes;
        std::vector<float> td(n_embd);
        if (t->type == GGML_TYPE_F32) {
            memcpy(td.data(), rp, n_embd * sizeof(float));
        } else if (t->type == GGML_TYPE_BF16) {
            uint16_t * bf16 = (uint16_t *)rp;
            for (int64_t i = 0; i < n_embd; i++) {
                uint32_t tmp = (uint32_t)bf16[i] << 16;
                memcpy(&td[i], &tmp, 4);
            }
        }

        char fname[256];
        snprintf(fname, sizeof(fname), "/tmp/ggml_%s_pos%d.bin", name, tok);
        FILE * fp = fopen(fname, "wb");
        if (fp) {
            fwrite(td.data(), sizeof(float), n_embd, fp);
            fclose(fp);
        }
    }

    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    params.prompt = "is";
    params.n_predict = 0;
    params.warmup = false;
    // params.no_conversation = true;

    // Parse -m flag
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc) {
            params.model.path = argv[++i];
        }
    }
    if (params.model.path.empty()) {
        params.model.path = "/home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-BF16.gguf";
    }

    dump_data dd;
    params.cb_eval = dump_cb;
    params.cb_eval_user_data = &dd;

    common_init();
    llama_backend_init();

    auto llama_init = common_init_from_params(params);
    if (!llama_init) {
        fprintf(stderr, "Failed to init\n");
        return 1;
    }

    auto * ctx = llama_init->context();

    // Tokenize
    const auto * vocab = llama_model_get_vocab(llama_init->model());
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, false);

    printf("Tokens (%zu):", tokens.size());
    for (auto t : tokens) printf(" %d", t);
    printf("\n");

    // Eval
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        fprintf(stderr, "Failed to eval\n");
        return 1;
    }

    printf("Done!\n");
    return 0;
}
