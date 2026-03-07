// SPDX-License-Identifier: Apache-2.0
// Qwen3.5-9B inference engine for Tenstorrent N300 via tt-metal.
//
// Weights stored on device DRAM, read back per-layer during inference.
// All computation done on host in f32 for correctness.

#include "engine.h"
#include "model_config.h"
#include "gguf_loader.h"
#include "tokenizer.h"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cfloat>

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using MC = ModelConfig;

// ============================================================================
// Global state
// ============================================================================
static std::shared_ptr<MeshDevice> g_mesh;
static ModelBuffers g_model;
static Tokenizer g_tokenizer;
static bool g_loaded = false;
static int g_max_ctx = 0;
static int g_pos = 0;

// Host-side f32 hidden state
static std::vector<float> g_hidden_f32(MC::n_embd);

// KV cache for attention layers [8 layers][max_ctx * kv_dim]
static constexpr int kv_dim = MC::n_head_kv * MC::head_dim;  // 1024
static std::vector<float> g_k_cache[8];
static std::vector<float> g_v_cache[8];

// SSM recurrent state [24 layers]
static constexpr int ssm_n_v_heads = MC::ssm_dt_rank;       // 32
static constexpr int ssm_head_k_dim = MC::ssm_d_state;      // 128
static constexpr int ssm_head_v_dim_c = MC::ssm_head_v_dim; // 128
static std::vector<float> g_ssm_state[24];

// Conv1d state [24 layers]
static constexpr int conv_state_len = MC::ssm_conv_kernel - 1;  // 3
static std::vector<float> g_conv_state[24];

// ============================================================================
// Helper: read bf16 weights from device MeshBuffer → host bf16 vector (flat)
// ============================================================================
static std::vector<bfloat16> read_tiled_bf16(MeshCommandQueue& cq,
    std::shared_ptr<MeshBuffer> buf) {
    std::vector<bfloat16> tiled;
    EnqueueReadMeshBuffer(cq, tiled, buf, true);
    return tiled;
}

// Untilize 1D weight stored as [1, len] tiled → f32 vector
static std::vector<float> read_1d_f32(MeshCommandQueue& cq,
    std::shared_ptr<MeshBuffer> buf, uint32_t len) {
    auto tiled = read_tiled_bf16(cq, buf);
    uint32_t cp = ((len + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    auto flat = untilize_nfaces(tiled, TILE_HEIGHT, cp);
    std::vector<float> result(len);
    for (uint32_t i = 0; i < len; i++)
        result[i] = static_cast<float>(flat[i]);
    return result;
}

// Untilize 2D weight and do GEMV in one pass to avoid storing full f32 matrix
// y[M] += W_bf16[M, K] @ x[K]  (W read from tiled device buffer)
static void gemv_from_device(MeshCommandQueue& cq,
    std::shared_ptr<MeshBuffer> buf,
    uint32_t M, uint32_t K,
    const float* x, float* y)
{
    auto tiled = read_tiled_bf16(cq, buf);
    uint32_t Mp = ((M + TILE_HEIGHT - 1) / TILE_HEIGHT) * TILE_HEIGHT;
    uint32_t Kp = ((K + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    auto flat = untilize_nfaces(tiled, Mp, Kp);

    for (uint32_t r = 0; r < M; r++) {
        float sum = 0;
        const bfloat16* row = flat.data() + r * Kp;
        for (uint32_t c = 0; c < K; c++)
            sum += static_cast<float>(row[c]) * x[c];
        y[r] = sum;
    }
}

// GEMV with bf16 host-side weight (uint16_t array, row-major)
static void gemv_bf16_host(const uint16_t* W, const float* x, float* y, int M, int K) {
    for (int i = 0; i < M; i++) {
        float sum = 0;
        const uint16_t* row = W + (size_t)i * K;
        for (int j = 0; j < K; j++) {
            bfloat16 b;
            memcpy(&b, &row[j], 2);
            sum += static_cast<float>(b) * x[j];
        }
        y[i] = sum;
    }
}

// ============================================================================
// RMSNorm: out[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]
// ============================================================================
static void rmsnorm(const float* x, const float* w, float* out, int dim) {
    float sum_sq = 0;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float rms = 1.0f / sqrtf(sum_sq / dim + MC::rms_norm_eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * rms * w[i];
}

// ============================================================================
// RoPE: apply rotary embeddings
// ============================================================================
static void apply_rope(float* head, int pos) {
    for (int p = 0; p < MC::rope_dim / 2; p++) {
        float freq = 1.0f / powf(MC::rope_freq_base, 2.0f * p / MC::rope_dim);
        float theta = pos * freq;
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);
        int i0 = p;
        int i1 = p + MC::rope_dim / 2;
        float x0 = head[i0], x1 = head[i1];
        head[i0] = x0 * cos_t - x1 * sin_t;
        head[i1] = x1 * cos_t + x0 * sin_t;
    }
}

// ============================================================================
// Cached small weights (norms, SSM params — tiny, keep permanently)
// ============================================================================
struct SmallAttnWeights {
    std::vector<float> q_norm;
    std::vector<float> k_norm;
};
static SmallAttnWeights g_attn_small[8];

struct LayerNorms {
    std::vector<float> attn_norm;
    std::vector<float> post_norm;
};
static LayerNorms g_layer_norms[32];
static std::vector<float> g_output_norm;

static void cache_small_weights(MeshCommandQueue& cq) {
    g_output_norm = read_1d_f32(cq, g_model.output_norm, MC::n_embd);

    int attn_idx = 0, ssm_idx = 0;
    for (int layer = 0; layer < MC::n_layers; layer++) {
        auto& ln = g_layer_norms[layer];

        if (MC::is_recurrent(layer)) {
            auto& lw = g_model.ssm_layers[ssm_idx];
            ln.attn_norm = read_1d_f32(cq, lw.attn_norm, MC::n_embd);
            ln.post_norm = read_1d_f32(cq, lw.post_attn_norm, MC::n_embd);
            // SSM small weights already on host as f32
            ssm_idx++;
        } else {
            auto& lw = g_model.attn_layers[attn_idx];
            ln.attn_norm = read_1d_f32(cq, lw.attn_norm, MC::n_embd);
            ln.post_norm = read_1d_f32(cq, lw.post_attn_norm, MC::n_embd);

            auto& aw = g_attn_small[attn_idx];
            aw.q_norm = read_1d_f32(cq, lw.attn_q_norm, MC::head_dim);
            aw.k_norm = read_1d_f32(cq, lw.attn_k_norm, MC::head_dim);
            attn_idx++;
        }
    }
}

// ============================================================================
// Forward pass: single decode token (all host-side f32)
// Large weight matrices read from device per-layer to avoid OOM
// ============================================================================
static std::vector<float> forward_decode(MeshCommandQueue& cq) {
    int pos = g_pos;
    std::vector<float> norm_out(MC::n_embd);
    std::vector<float> residual(MC::n_embd);

    int attn_idx = 0, ssm_idx = 0;
    for (int layer = 0; layer < MC::n_layers; layer++) {
        auto& ln = g_layer_norms[layer];

        // Save residual
        memcpy(residual.data(), g_hidden_f32.data(), MC::n_embd * sizeof(float));

        // Pre-norm
        rmsnorm(g_hidden_f32.data(), ln.attn_norm.data(), norm_out.data(), MC::n_embd);

        if (MC::is_recurrent(layer)) {
            // ======== SSM (Delta-Net) Layer ========
            auto& lw = g_model.ssm_layers[ssm_idx];

            // 1. Combined projection via device read
            int combined_rows = MC::ssm_conv_channels + MC::ssm_d_inner
                              + MC::ssm_dt_rank + MC::ssm_dt_rank;
            std::vector<float> proj(combined_rows);
            gemv_from_device(cq, lw.w_combined, combined_rows, MC::n_embd,
                           norm_out.data(), proj.data());

            float* qkv_raw = proj.data();
            float* z_raw = proj.data() + MC::ssm_conv_channels;
            float* alpha_raw = z_raw + MC::ssm_d_inner;
            float* beta_raw = alpha_raw + MC::ssm_dt_rank;

            // 2. Conv1d + SiLU
            auto& cs = g_conv_state[ssm_idx];
            std::vector<float> conv_out(MC::ssm_conv_channels);
            for (int ch = 0; ch < MC::ssm_conv_channels; ch++) {
                float sum = 0;
                for (int k = 0; k < MC::ssm_conv_kernel; k++) {
                    float val;
                    if (k < conv_state_len)
                        val = cs[k * MC::ssm_conv_channels + ch];
                    else
                        val = qkv_raw[ch];
                    sum += val * lw.ssm_conv1d_host[ch * MC::ssm_conv_kernel + k];
                }
                conv_out[ch] = sum / (1.0f + expf(-sum));  // SiLU

                // Shift state
                for (int i = 0; i < conv_state_len - 1; i++)
                    cs[i * MC::ssm_conv_channels + ch] = cs[(i + 1) * MC::ssm_conv_channels + ch];
                cs[(conv_state_len - 1) * MC::ssm_conv_channels + ch] = qkv_raw[ch];
            }

            // 3. Split conv output: Q[2048] | K[2048] | V[4096]
            constexpr int num_k_heads = MC::ssm_n_group;
            constexpr int head_k = MC::ssm_d_state;
            constexpr int num_v = ssm_n_v_heads;
            constexpr int head_v = ssm_head_v_dim_c;

            float* conv_q = conv_out.data();
            float* conv_k = conv_out.data() + num_k_heads * head_k;
            float* conv_v = conv_out.data() + 2 * num_k_heads * head_k;

            // 4. Delta-net recurrence
            auto& state = g_ssm_state[ssm_idx];
            std::vector<float> delta_out(MC::ssm_d_inner);
            constexpr float ssm_scale = 1.0f / 11.3137f;  // 1/sqrt(128)

            for (int vh = 0; vh < num_v; vh++) {
                int kh = vh % num_k_heads;
                float q[head_k], k[head_k], v[head_v];
                memcpy(q, conv_q + kh * head_k, head_k * sizeof(float));
                memcpy(k, conv_k + kh * head_k, head_k * sizeof(float));
                memcpy(v, conv_v + vh * head_v, head_v * sizeof(float));

                // L2 normalize
                float qn = 0, kn = 0;
                for (int d = 0; d < head_k; d++) { qn += q[d]*q[d]; kn += k[d]*k[d]; }
                qn = 1.0f / sqrtf(qn + MC::rms_norm_eps);
                kn = 1.0f / sqrtf(kn + MC::rms_norm_eps);
                for (int d = 0; d < head_k; d++) { q[d] *= qn; k[d] *= kn; }

                // Gate
                float biased = alpha_raw[vh] + lw.ssm_dt_bias_host[vh];
                float sp = (biased > 20.0f) ? biased : logf(1.0f + expf(biased));
                float gate = sp * lw.ssm_a_host[vh];
                float decay = expf(gate);

                // Beta
                float beta = 1.0f / (1.0f + expf(-beta_raw[vh]));

                float* sh = state.data() + vh * head_k * head_v;

                // Decay
                for (int j = 0; j < head_k * head_v; j++) sh[j] *= decay;

                // Delta update + output
                for (int i = 0; i < head_v; i++) {
                    float sk = 0;
                    for (int j = 0; j < head_k; j++) sk += sh[j * head_v + i] * k[j];
                    float d = beta * (v[i] - sk);
                    for (int j = 0; j < head_k; j++) sh[j * head_v + i] += k[j] * d;
                    float out = 0;
                    for (int j = 0; j < head_k; j++) out += sh[j * head_v + i] * q[j];
                    delta_out[vh * head_v + i] = out * ssm_scale;
                }
            }

            // 5. Gated RMSNorm (ssm_norm is [128], broadcast across all v_heads)
            std::vector<float> ssm_proj_in(MC::ssm_d_inner);
            for (int vh = 0; vh < num_v; vh++) {
                float sum_sq = 0;
                for (int d = 0; d < head_v; d++) {
                    float val = delta_out[vh * head_v + d];
                    sum_sq += val * val;
                }
                float rms = 1.0f / sqrtf(sum_sq / head_v + MC::rms_norm_eps);
                for (int d = 0; d < head_v; d++) {
                    int idx = vh * head_v + d;
                    float normalized = delta_out[idx] * rms * lw.ssm_norm_host[d];  // [d] not [idx]
                    float z = z_raw[idx];
                    float silu_z = z / (1.0f + expf(-z));
                    ssm_proj_in[idx] = normalized * silu_z;
                }
            }

            // 6. Output projection (host bf16)
            std::vector<float> layer_out(MC::n_embd);
            gemv_bf16_host(lw.ssm_out_host.data(), ssm_proj_in.data(),
                          layer_out.data(), MC::n_embd, MC::ssm_d_inner);

            // 7. Residual
            for (int i = 0; i < MC::n_embd; i++)
                g_hidden_f32[i] = residual[i] + layer_out[i];

            memcpy(residual.data(), g_hidden_f32.data(), MC::n_embd * sizeof(float));

            // 8. Post-norm + FFN
            rmsnorm(g_hidden_f32.data(), ln.post_norm.data(), norm_out.data(), MC::n_embd);

            std::vector<float> ffn_buf(2 * MC::n_ff);
            gemv_from_device(cq, lw.ffn_gate_up, 2 * MC::n_ff, MC::n_embd,
                           norm_out.data(), ffn_buf.data());

            std::vector<float> ffn_act(MC::n_ff);
            for (int i = 0; i < MC::n_ff; i++) {
                float g = ffn_buf[i];
                float u = ffn_buf[MC::n_ff + i];
                ffn_act[i] = (g / (1.0f + expf(-g))) * u;
            }

            std::vector<float> ffn_out(MC::n_embd);
            gemv_from_device(cq, lw.ffn_down, MC::n_embd, MC::n_ff,
                           ffn_act.data(), ffn_out.data());

            for (int i = 0; i < MC::n_embd; i++)
                g_hidden_f32[i] = residual[i] + ffn_out[i];

            ssm_idx++;
        } else {
            // ======== Full Attention Layer ========
            auto& lw = g_model.attn_layers[attn_idx];
            auto& aw = g_attn_small[attn_idx];

            // 1. QKV projection
            int q_dim = MC::n_head * MC::head_dim * 2;
            int kv_dim_one = MC::n_head_kv * MC::head_dim;
            int total_rows = q_dim + 2 * kv_dim_one;
            std::vector<float> qkv(total_rows);
            gemv_from_device(cq, lw.wqkv, total_rows, MC::n_embd,
                           norm_out.data(), qkv.data());

            // 2. Deinterleave Q and gate
            std::vector<float> q_heads(MC::n_head * MC::head_dim);
            std::vector<float> gate_heads(MC::n_head * MC::head_dim);
            for (int h = 0; h < MC::n_head; h++) {
                for (int d = 0; d < MC::head_dim; d++) {
                    q_heads[h * MC::head_dim + d] = qkv[h * MC::head_dim * 2 + d];
                    gate_heads[h * MC::head_dim + d] = qkv[h * MC::head_dim * 2 + MC::head_dim + d];
                }
            }
            float* k_proj = qkv.data() + q_dim;
            float* v_proj = k_proj + kv_dim_one;

            // 3. Per-head Q/K RMSNorm
            for (int h = 0; h < MC::n_head; h++) {
                float* qh = q_heads.data() + h * MC::head_dim;
                float ss = 0;
                for (int d = 0; d < MC::head_dim; d++) ss += qh[d] * qh[d];
                float rms = 1.0f / sqrtf(ss / MC::head_dim + MC::rms_norm_eps);
                for (int d = 0; d < MC::head_dim; d++)
                    qh[d] = qh[d] * rms * aw.q_norm[d];
            }
            for (int h = 0; h < MC::n_head_kv; h++) {
                float* kh = k_proj + h * MC::head_dim;
                float ss = 0;
                for (int d = 0; d < MC::head_dim; d++) ss += kh[d] * kh[d];
                float rms = 1.0f / sqrtf(ss / MC::head_dim + MC::rms_norm_eps);
                for (int d = 0; d < MC::head_dim; d++)
                    kh[d] = kh[d] * rms * aw.k_norm[d];
            }

            // 4. RoPE
            for (int h = 0; h < MC::n_head; h++)
                apply_rope(q_heads.data() + h * MC::head_dim, pos);
            for (int h = 0; h < MC::n_head_kv; h++)
                apply_rope(k_proj + h * MC::head_dim, pos);

            // 5. KV cache
            memcpy(g_k_cache[attn_idx].data() + (size_t)pos * kv_dim,
                   k_proj, kv_dim * sizeof(float));
            memcpy(g_v_cache[attn_idx].data() + (size_t)pos * kv_dim,
                   v_proj, kv_dim * sizeof(float));
            int kv_len = pos + 1;

            // 6. Attention (online softmax)
            std::vector<float> attn_out(MC::n_head * MC::head_dim, 0.0f);
            for (int h = 0; h < MC::n_head; h++) {
                int kv_h = h / (MC::n_head / MC::n_head_kv);
                float* qh = q_heads.data() + h * MC::head_dim;
                float* out = attn_out.data() + h * MC::head_dim;
                float max_score = -FLT_MAX, sum_exp = 0;
                std::vector<float> acc(MC::head_dim, 0.0f);

                for (int kp = 0; kp < kv_len; kp++) {
                    float* kh = g_k_cache[attn_idx].data() + (size_t)kp * kv_dim + kv_h * MC::head_dim;
                    float dot = 0;
                    for (int d = 0; d < MC::head_dim; d++) dot += qh[d] * kh[d];
                    float score = dot * MC::attn_scale;

                    float new_max = std::max(max_score, score);
                    float exp_s = expf(score - new_max);
                    float corr = expf(max_score - new_max);
                    sum_exp = sum_exp * corr + exp_s;

                    float* vh = g_v_cache[attn_idx].data() + (size_t)kp * kv_dim + kv_h * MC::head_dim;
                    for (int d = 0; d < MC::head_dim; d++)
                        acc[d] = acc[d] * corr + exp_s * vh[d];
                    max_score = new_max;
                }
                for (int d = 0; d < MC::head_dim; d++) out[d] = acc[d] / sum_exp;
            }

            // 7. Sigmoid gating
            for (int i = 0; i < MC::n_head * MC::head_dim; i++)
                attn_out[i] *= 1.0f / (1.0f + expf(-gate_heads[i]));

            // 8. Output projection (host bf16)
            std::vector<float> layer_out(MC::n_embd);
            gemv_bf16_host(lw.wo_host.data(), attn_out.data(),
                          layer_out.data(), MC::n_embd, MC::n_head * MC::head_dim);

            // 9. Residual
            for (int i = 0; i < MC::n_embd; i++)
                g_hidden_f32[i] = residual[i] + layer_out[i];

            memcpy(residual.data(), g_hidden_f32.data(), MC::n_embd * sizeof(float));

            // 10. Post-norm + FFN
            rmsnorm(g_hidden_f32.data(), ln.post_norm.data(), norm_out.data(), MC::n_embd);

            std::vector<float> ffn_buf(2 * MC::n_ff);
            gemv_from_device(cq, lw.ffn_gate_up, 2 * MC::n_ff, MC::n_embd,
                           norm_out.data(), ffn_buf.data());

            std::vector<float> ffn_act(MC::n_ff);
            for (int i = 0; i < MC::n_ff; i++) {
                float g = ffn_buf[i];
                float u = ffn_buf[MC::n_ff + i];
                ffn_act[i] = (g / (1.0f + expf(-g))) * u;
            }

            std::vector<float> ffn_out(MC::n_embd);
            gemv_from_device(cq, lw.ffn_down, MC::n_embd, MC::n_ff,
                           ffn_act.data(), ffn_out.data());

            for (int i = 0; i < MC::n_embd; i++)
                g_hidden_f32[i] = residual[i] + ffn_out[i];

            attn_idx++;
        }

        if (layer % 8 == 7)
            printf("  [layer %d/%d done]\n", layer + 1, MC::n_layers);
    }

    // Final norm
    rmsnorm(g_hidden_f32.data(), g_output_norm.data(), norm_out.data(), MC::n_embd);

    // LM head (host bf16)
    std::vector<float> logits(MC::n_vocab);
    gemv_bf16_host(g_model.output_host.data(), norm_out.data(),
                  logits.data(), MC::n_vocab, MC::n_embd);

    return logits;
}

// ============================================================================
// Public API
// ============================================================================

bool load_model_and_tokenizer(const char* model_path, int max_ctx) {
    printf("Loading model from %s (max_ctx=%d)...\n", model_path, max_ctx);

    g_mesh = MeshDevice::create_unit_mesh(0);
    auto grid = g_mesh->compute_with_storage_grid_size();
    printf("Device opened: compute grid %dx%d (%d cores)\n",
           grid.x, grid.y, grid.x * grid.y);

    MeshCommandQueue& cq = g_mesh->mesh_command_queue();
    g_max_ctx = max_ctx;

    if (!g_tokenizer.load(model_path)) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return false;
    }

    if (!load_gguf_weights(model_path, g_model, g_mesh.get(), cq)) {
        fprintf(stderr, "Failed to load weights\n");
        return false;
    }
    Finish(cq);

    // Allocate KV caches
    for (int i = 0; i < 8; i++) {
        g_k_cache[i].resize((size_t)max_ctx * kv_dim, 0.0f);
        g_v_cache[i].resize((size_t)max_ctx * kv_dim, 0.0f);
    }
    for (int i = 0; i < 24; i++) {
        g_ssm_state[i].resize(ssm_n_v_heads * ssm_head_k_dim * ssm_head_v_dim_c, 0.0f);
        g_conv_state[i].resize(conv_state_len * MC::ssm_conv_channels, 0.0f);
    }

    // Cache small weights (norms, SSM params) — tiny, < 50 MB total
    printf("Caching small weights...\n");
    cache_small_weights(cq);
    printf("Ready.\n");

    g_loaded = true;
    g_pos = 0;
    return true;
}

int generate(const std::vector<int>& prompt_tokens, int max_tokens,
             float temperature, TokenCallback cb, StopReason* stop_reason) {
    if (!g_loaded) {
        fprintf(stderr, "Model not loaded\n");
        return 0;
    }

    MeshCommandQueue& cq = g_mesh->mesh_command_queue();
    int total_generated = 0;
    int next_token = -1;

    // Process all prompt tokens
    for (int i = 0; i < (int)prompt_tokens.size(); i++) {
        int token = prompt_tokens[i];
        printf("  [prefill token %d/%d: %d]\n", i + 1, (int)prompt_tokens.size(), token);

        const bfloat16* emb = reinterpret_cast<const bfloat16*>(
            g_model.tok_embd_host.data() + (size_t)token * MC::n_embd);
        for (int j = 0; j < MC::n_embd; j++)
            g_hidden_f32[j] = static_cast<float>(emb[j]);

        auto logits = forward_decode(cq);
        g_pos++;

        // After last prompt token, sample first output
        if (i == (int)prompt_tokens.size() - 1) {
            float max_l = -FLT_MAX;
            for (int v = 0; v < MC::n_vocab; v++) {
                if (logits[v] > max_l) { max_l = logits[v]; next_token = v; }
            }
        }
    }

    // Generate tokens
    while (total_generated < max_tokens) {
        total_generated++;
        if (cb) {
            std::string text = g_tokenizer.decode(next_token);
            if (!cb(next_token, text)) {
                if (stop_reason) *stop_reason = STOP_CALLBACK;
                return total_generated;
            }
        }
        if (next_token == g_tokenizer.eos_token_id()) {
            if (stop_reason) *stop_reason = STOP_EOS;
            return total_generated;
        }

        // Forward pass with generated token
        const bfloat16* emb = reinterpret_cast<const bfloat16*>(
            g_model.tok_embd_host.data() + (size_t)next_token * MC::n_embd);
        for (int j = 0; j < MC::n_embd; j++)
            g_hidden_f32[j] = static_cast<float>(emb[j]);

        auto logits = forward_decode(cq);
        g_pos++;

        float max_l = -FLT_MAX;
        int best = 0;
        for (int v = 0; v < MC::n_vocab; v++) {
            if (logits[v] > max_l) { max_l = logits[v]; best = v; }
        }
        next_token = best;
    }

    if (stop_reason) *stop_reason = STOP_LENGTH;
    return total_generated;
}

void reset_state() {
    g_pos = 0;
    std::fill(g_hidden_f32.begin(), g_hidden_f32.end(), 0.0f);
    for (int i = 0; i < 8; i++) {
        std::fill(g_k_cache[i].begin(), g_k_cache[i].end(), 0.0f);
        std::fill(g_v_cache[i].begin(), g_v_cache[i].end(), 0.0f);
    }
    for (int i = 0; i < 24; i++) {
        std::fill(g_ssm_state[i].begin(), g_ssm_state[i].end(), 0.0f);
        std::fill(g_conv_state[i].begin(), g_conv_state[i].end(), 0.0f);
    }
}

const Tokenizer& get_tokenizer() {
    return g_tokenizer;
}

void shutdown() {
    if (!g_loaded) return;
    g_loaded = false;

    g_model.output_norm.reset();
    for (auto& l : g_model.attn_layers) {
        l.attn_norm.reset(); l.wqkv.reset(); l.attn_q_norm.reset();
        l.attn_k_norm.reset(); l.post_attn_norm.reset();
        l.ffn_gate_up.reset(); l.ffn_down.reset();
    }
    for (auto& l : g_model.ssm_layers) {
        l.attn_norm.reset(); l.w_combined.reset();
        l.post_attn_norm.reset(); l.ffn_gate_up.reset(); l.ffn_down.reset();
    }

    if (g_mesh) {
        g_mesh->close();
        g_mesh.reset();
    }
}
