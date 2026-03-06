#include "model.h"
#include "utils.h"
#include "gguf_loader.h"
#include "tokenizer.h"
#include "sampling.h"
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>

// Forward declarations for kernel launchers
// (embedding_to_f32 is defined inline in this file)
void launch_rmsnorm(__nv_bfloat16* output, const __nv_bfloat16* input,
    const float* weight, int n_tokens, int dim, float eps, cudaStream_t stream);
void launch_rmsnorm_f32in(__nv_bfloat16* output, const float* input,
    const float* weight, int n_tokens, int dim, float eps, cudaStream_t stream);
void launch_rmsnorm_head(__nv_bfloat16* output, const __nv_bfloat16* input,
    const float* weight, int n_tokens, int n_heads, int head_dim,
    float eps, cudaStream_t stream);
void launch_rope(__nv_bfloat16* qk, const int* positions, int n_tokens,
    int n_heads, int head_dim, int rope_dim, float freq_base, cudaStream_t stream);
void launch_swiglu(__nv_bfloat16* output, const __nv_bfloat16* gate,
    const __nv_bfloat16* up, int n_tokens, int n_ff, cudaStream_t stream);
void launch_sigmoid_mul(__nv_bfloat16* output, const __nv_bfloat16* attn_out,
    const __nv_bfloat16* gate, int n_elements, cudaStream_t stream);
void launch_kv_cache_append(__nv_bfloat16* cache, const __nv_bfloat16* new_kv,
    int kv_pos, int n_new_tokens, int kv_dim, cudaStream_t stream);
void launch_attention_decode(__nv_bfloat16* output, const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    int kv_len, int n_head, int n_head_kv, int head_dim, float scale, cudaStream_t stream);
void launch_compute_gate(float* gate_out, const float* alpha,
    const float* dt_bias, const float* ssm_a, int n_tokens, int num_v_heads, cudaStream_t stream);
void launch_sigmoid(float* output, const float* input, int n, cudaStream_t stream);
void launch_conv1d_silu(float* output, const float* input,
    const float* conv_state, const float* conv_weight,
    int n_tokens, int channels, int conv_kernel_size, cudaStream_t stream);
void launch_update_conv_state(float* new_state, const float* input,
    const float* old_state, int n_tokens, int channels, int conv_kernel_size, cudaStream_t stream);
void launch_l2_norm(float* output, const float* input,
    int n_vectors, int dim, float eps, cudaStream_t stream);
void launch_delta_net_decode(float* output, float* state,
    const float* q, const float* k, const float* v,
    const float* gate, const float* beta, int num_v_heads, int head_dim,
    float scale, cudaStream_t stream);
void launch_gated_rmsnorm(__nv_bfloat16* output, const float* input,
    const float* weight, const float* gate,
    int num_heads, int head_dim, float eps, cudaStream_t stream);
void launch_repeat_heads(float* output, const float* input,
    int num_k_heads, int num_v_heads, int head_dim, cudaStream_t stream);

using MC = ModelConfig;

// cuBLAS GEMM wrapper: C = A @ B^T  (row-major), bf16 output
// A: [M, K] bf16, B: [N, K] bf16, C: [M, N] bf16
static void gemm_bf16(
    cublasHandle_t handle,
    __nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B,
    int M, int N, int K
) {
    float alpha = 1.0f, beta_val = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, K,
        A, CUDA_R_16BF, K,
        &beta_val,
        C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// bf16 -> f32 cast kernel (matching ggml's to_fp32_cuda)
__global__ void gemm_bf16_to_f32_kernel(
    float* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __bfloat162float(input[idx]);
    }
}

// cuBLAS GEMM wrapper: C = A @ B^T, f32 output matching ggml precision
// ggml does: bf16×bf16→bf16 (cuBLAS) → to_fp32. We must match this truncation.
// A: [M, K] bf16, B: [N, K] bf16, C: [M, N] f32
static __nv_bfloat16* g_gemm_bf16_tmp = nullptr;
static int g_gemm_bf16_tmp_size = 0;

static void gemm_bf16_f32out(
    cublasHandle_t handle,
    float* C, const __nv_bfloat16* A, const __nv_bfloat16* B,
    int M, int N, int K
) {
    int needed = M * N;
    if (needed > g_gemm_bf16_tmp_size) {
        if (g_gemm_bf16_tmp) cudaFree(g_gemm_bf16_tmp);
        g_gemm_bf16_tmp = cuda_alloc<__nv_bfloat16>(needed);
        g_gemm_bf16_tmp_size = needed;
    }

    // Step 1: GEMM to bf16 (matching ggml)
    float alpha = 1.0f, beta_val = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, K,
        A, CUDA_R_16BF, K,
        &beta_val,
        g_gemm_bf16_tmp, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    // Step 2: Cast bf16 → f32 (matching ggml's to_fp32_cuda)
    gemm_bf16_to_f32_kernel<<<cdiv(needed, 256), 256>>>(C, g_gemm_bf16_tmp, needed);
}

// GEMM: C(f32) = A(f32) @ B(bf16)^T  — matches ggml precision (f32 activations × bf16 weights)
static void gemm_f32_bf16_f32out(
    cublasHandle_t handle,
    float* C, const float* A, const __nv_bfloat16* B,
    int M, int N, int K
) {
    float alpha = 1.0f, beta_val = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, K,
        A, CUDA_R_32F, K,
        &beta_val,
        C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// GEMM: C(bf16) = A(f32) @ B(bf16)^T
static void gemm_f32_bf16_bf16out(
    cublasHandle_t handle,
    __nv_bfloat16* C, const float* A, const __nv_bfloat16* B,
    int M, int N, int K
) {
    float alpha = 1.0f, beta_val = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, K,
        A, CUDA_R_32F, K,
        &beta_val,
        C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// f32 -> bf16 cast kernel
__global__ void f32_to_bf16_kernel(
    __nv_bfloat16* __restrict__ output,
    const float* __restrict__ input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2bfloat16(input[idx]);
    }
}

static void cast_f32_to_bf16(__nv_bfloat16* output, const float* input, int n, cudaStream_t stream = 0) {
    f32_to_bf16_kernel<<<cdiv(n, 256), 256, 0, stream>>>(output, input, n);
}

static int g_position = 0; // set externally for debug dumps

static bool should_dump() {
    return getenv("DUMP_SSM_INTERMEDIATES") || getenv("DUMP_ATTN_INTERMEDIATES");
}

// Debug: dump f32 GPU buffer to file
static void dump_f32(const char* name, const float* gpu, int n, int pos, int layer) {
    if (!should_dump()) return;
    cudaDeviceSynchronize();
    std::vector<float> h(n);
    cuda_download(h.data(), gpu, n);
    char fname[256];
    snprintf(fname, sizeof(fname), "/tmp/ssm_%s_pos%d_layer%d.bin", name, pos, layer);
    FILE* f = fopen(fname, "wb");
    if (f) { fwrite(h.data(), 4, n, f); fclose(f); }
    // Print first 8 values + L2 norm
    double l2 = 0;
    for (int i = 0; i < n; i++) l2 += (double)h[i] * h[i];
    printf("  [DUMP] %s: L2=%.6f first8=[", name, sqrt(l2));
    for (int i = 0; i < 8 && i < n; i++) printf("%.6f%s", h[i], i<7?", ":"");
    printf("]\n");
}

static void dump_bf16(const char* name, const __nv_bfloat16* gpu, int n, int pos, int layer) {
    if (!should_dump()) return;
    cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> hbf(n);
    cuda_download(hbf.data(), gpu, n);
    std::vector<float> h(n);
    for (int i = 0; i < n; i++) h[i] = __bfloat162float(hbf[i]);
    char fname[256];
    snprintf(fname, sizeof(fname), "/tmp/ssm_%s_pos%d_layer%d.bin", name, pos, layer);
    FILE* f = fopen(fname, "wb");
    if (f) { fwrite(h.data(), 4, n, f); fclose(f); }
    double l2 = 0;
    for (int i = 0; i < n; i++) l2 += (double)h[i] * h[i];
    printf("  [DUMP] %s: L2=%.6f first8=[", name, sqrt(l2));
    for (int i = 0; i < 8 && i < n; i++) printf("%.6f%s", h[i], i<7?", ":"");
    printf("]\n");
}

// Allocate inference buffers
static void allocate_buffers(Model& model, int max_tokens, int max_kv_len) {
    model.max_tokens = max_tokens;
    model.max_kv_len = max_kv_len;
    model.kv_len = 0;

    // Hidden state in f32 for residual stream precision (matching ggml behavior)
    model.hidden_state = cuda_alloc<float>(max_tokens * MC::n_embd);
    model.hidden_bf16  = cuda_alloc<__nv_bfloat16>(max_tokens * MC::n_embd);
    model.norm_out     = cuda_alloc<__nv_bfloat16>(max_tokens * MC::n_embd);
    model.norm_out_f32 = cuda_alloc<float>(max_tokens * MC::n_embd);
    model.attn_out     = cuda_alloc<__nv_bfloat16>(max_tokens * MC::n_embd);
    int gemm_max = MC::n_ff > MC::n_embd ? MC::n_ff : MC::n_embd;
    gemm_max = gemm_max > MC::n_vocab ? gemm_max : MC::n_vocab;
    model.gemm_out     = cuda_alloc<float>(max_tokens * gemm_max);
    model.gemm_out2    = cuda_alloc<float>(max_tokens * gemm_max);
    model.ffn_buf      = cuda_alloc<__nv_bfloat16>(max_tokens * MC::n_ff);
    model.ffn_buf2     = cuda_alloc<__nv_bfloat16>(max_tokens * MC::n_ff);

    // QKV temp buffer
    model.qkv_buf = cuda_alloc<__nv_bfloat16>(max_tokens * MC::ssm_conv_channels);
    // SSM f32 projection buffer for conv state precision
    model.ssm_proj_f32 = cuda_alloc<float>(max_tokens * MC::ssm_conv_channels);

    model.logits_f32 = cuda_alloc<float>(max_tokens * MC::n_vocab);

    // KV caches for attention layers
    int kv_dim = MC::n_head_kv * MC::head_dim;
    for (int i = 0; i < 8; i++) {
        model.k_cache[i] = cuda_alloc<__nv_bfloat16>(max_kv_len * kv_dim);
        model.v_cache[i] = cuda_alloc<__nv_bfloat16>(max_kv_len * kv_dim);
    }

    // SSM states
    int conv_state_size = (MC::ssm_conv_kernel - 1) * MC::ssm_conv_channels;
    int recurrent_state_size = MC::ssm_dt_rank * MC::ssm_head_v_dim * MC::ssm_head_v_dim;
    for (int i = 0; i < 24; i++) {
        model.ssm_conv_state[i] = cuda_alloc<float>(conv_state_size);
        CUDA_CHECK(cudaMemset(model.ssm_conv_state[i], 0, conv_state_size * sizeof(float)));
        model.ssm_recurrent_state[i] = cuda_alloc<float>(recurrent_state_size);
        CUDA_CHECK(cudaMemset(model.ssm_recurrent_state[i], 0, recurrent_state_size * sizeof(float)));
    }
}

// Residual add: f32 output = f32 a + f32 b
__global__ void residual_add_f32_kernel(float* output, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}

static void residual_add_f32(float* output, const float* a, const float* b, int n, cudaStream_t stream = 0) {
    residual_add_f32_kernel<<<cdiv(n, 256), 256, 0, stream>>>(output, a, b, n);
}

// Forward pass for one full-attention layer (decode mode, n_tokens=1)
// hidden is f32 (residual stream). Internal computation uses bf16 for GEMMs.
static void forward_attention_layer(Model& model, int layer_idx, int attn_idx,
    float* hidden, int n_tokens, int* positions_d) {
    auto& lw = model.attn_layers[attn_idx];
    auto handle = model.cublas_handle;
    bool do_dump = (attn_idx == 0 && getenv("DUMP_ATTN_INTERMEDIATES"));
    int dpos = g_position;

    // 1. RMSNorm (f32 in, bf16 out)
    launch_rmsnorm_f32in(model.norm_out, hidden, lw.attn_norm, n_tokens, MC::n_embd, MC::rms_norm_eps, 0);

    if (do_dump) { dump_f32("attn_hidden_in", hidden, MC::n_embd, dpos, layer_idx); dump_bf16("attn_normed", model.norm_out, MC::n_embd, dpos, layer_idx); }

    // 2. Q+Gate projection: [n_tokens, 4096] -> [n_tokens, 8192]
    gemm_bf16(handle, model.qkv_buf, model.norm_out, lw.wq, n_tokens, MC::n_head * MC::head_dim * 2, MC::n_embd);

    // 3. K projection: [n_tokens, 4096] -> [n_tokens, 1024]
    __nv_bfloat16* k_proj = model.attn_out; // reuse buffer
    gemm_bf16(handle, k_proj, model.norm_out, lw.wk, n_tokens, MC::n_head_kv * MC::head_dim, MC::n_embd);

    // 4. V projection: [n_tokens, 4096] -> [n_tokens, 1024]
    __nv_bfloat16* v_proj = model.ffn_buf; // reuse buffer
    gemm_bf16(handle, v_proj, model.norm_out, lw.wv, n_tokens, MC::n_head_kv * MC::head_dim, MC::n_embd);

    // 5. Extract Q and Gate from interleaved qkv_buf
    __nv_bfloat16* q_contiguous = model.norm_out;  // reuse
    __nv_bfloat16* gate_buf = model.hidden_bf16;    // reuse (done with it)

    for (int h = 0; h < MC::n_head; h++) {
        CUDA_CHECK(cudaMemcpy(
            q_contiguous + h * MC::head_dim,
            model.qkv_buf + h * MC::head_dim * 2,
            MC::head_dim * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(
            gate_buf + h * MC::head_dim,
            model.qkv_buf + h * MC::head_dim * 2 + MC::head_dim,
            MC::head_dim * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice));
    }

    if (do_dump) { dump_bf16("attn_q_raw", q_contiguous, MC::n_head * MC::head_dim, dpos, layer_idx); dump_bf16("attn_gate", gate_buf, MC::n_head * MC::head_dim, dpos, layer_idx); }

    // Q/K norm
    launch_rmsnorm_head(q_contiguous, q_contiguous, lw.attn_q_norm,
        n_tokens, MC::n_head, MC::head_dim, MC::rms_norm_eps, 0);
    launch_rmsnorm_head(k_proj, k_proj, lw.attn_k_norm,
        n_tokens, MC::n_head_kv, MC::head_dim, MC::rms_norm_eps, 0);

    if (do_dump) { dump_bf16("attn_q_normed", q_contiguous, MC::n_head * MC::head_dim, dpos, layer_idx); dump_bf16("attn_k_normed", k_proj, MC::n_head_kv * MC::head_dim, dpos, layer_idx); }

    // 6. RoPE on Q and K
    launch_rope(q_contiguous, positions_d, n_tokens, MC::n_head, MC::head_dim,
        MC::rope_dim, MC::rope_freq_base, 0);
    launch_rope(k_proj, positions_d, n_tokens, MC::n_head_kv, MC::head_dim,
        MC::rope_dim, MC::rope_freq_base, 0);

    if (do_dump) { dump_bf16("attn_q_rope", q_contiguous, MC::n_head * MC::head_dim, dpos, layer_idx); dump_bf16("attn_k_rope", k_proj, MC::n_head_kv * MC::head_dim, dpos, layer_idx); dump_bf16("attn_v", v_proj, MC::n_head_kv * MC::head_dim, dpos, layer_idx); }

    // 7. Append K, V to cache
    int kv_dim = MC::n_head_kv * MC::head_dim;
    launch_kv_cache_append(model.k_cache[attn_idx], k_proj, model.kv_len, n_tokens, kv_dim, 0);
    launch_kv_cache_append(model.v_cache[attn_idx], v_proj, model.kv_len, n_tokens, kv_dim, 0);

    // 8. Attention
    int total_kv_len = model.kv_len + n_tokens;
    launch_attention_decode(model.attn_out, q_contiguous,
        model.k_cache[attn_idx], model.v_cache[attn_idx],
        total_kv_len, MC::n_head, MC::n_head_kv, MC::head_dim, MC::attn_scale, 0);

    if (do_dump) dump_bf16("attn_out", model.attn_out, MC::n_head * MC::head_dim, dpos, layer_idx);

    // 9. Sigmoid gate
    launch_sigmoid_mul(model.attn_out, model.attn_out, gate_buf,
        n_tokens * MC::n_head * MC::head_dim, 0);

    if (do_dump) dump_bf16("attn_gated", model.attn_out, MC::n_head * MC::head_dim, dpos, layer_idx);

    // 10. Output projection -> f32 output for residual
    gemm_bf16_f32out(handle, model.gemm_out, model.attn_out, lw.wo, n_tokens, MC::n_embd, MC::n_head * MC::head_dim);

    if (do_dump) dump_f32("attn_proj", model.gemm_out, MC::n_embd, dpos, layer_idx);

    // 11. Residual connection (f32 + f32 -> f32)
    residual_add_f32(hidden, hidden, model.gemm_out, n_tokens * MC::n_embd);

    // 12. Post-attention norm + FFN (f32 in, bf16 out)
    launch_rmsnorm_f32in(model.norm_out, hidden, lw.post_attn_norm, n_tokens, MC::n_embd, MC::rms_norm_eps, 0);

    // FFN: SwiGLU
    gemm_bf16(handle, model.ffn_buf, model.norm_out, lw.ffn_gate, n_tokens, MC::n_ff, MC::n_embd);
    gemm_bf16(handle, model.ffn_buf2, model.norm_out, lw.ffn_up, n_tokens, MC::n_ff, MC::n_embd);
    launch_swiglu(model.ffn_buf, model.ffn_buf, model.ffn_buf2, n_tokens, MC::n_ff, 0);

    // down_proj -> f32 for residual
    gemm_bf16_f32out(handle, model.gemm_out, model.ffn_buf, lw.ffn_down, n_tokens, MC::n_embd, MC::n_ff);

    // FFN residual (f32 + f32 -> f32)
    residual_add_f32(hidden, hidden, model.gemm_out, n_tokens * MC::n_embd);
}

// Forward pass for one SSM (delta-net) layer (decode mode, n_tokens=1)
// hidden is f32 (residual stream)
static void forward_ssm_layer(Model& model, int layer_idx, int ssm_idx,
    float* hidden, int n_tokens) {
    auto& lw = model.ssm_layers[ssm_idx];
    auto handle = model.cublas_handle;

    bool do_dump = (layer_idx == 0 && getenv("DUMP_SSM_INTERMEDIATES"));
    int dpos = g_position;

    // 1. RMSNorm (f32 in, bf16 out)
    launch_rmsnorm_f32in(model.norm_out, hidden, lw.attn_norm, n_tokens, MC::n_embd, MC::rms_norm_eps, 0);

    if (do_dump) { dump_f32("hidden_in", hidden, MC::n_embd, dpos, 0); dump_bf16("normed", model.norm_out, MC::n_embd, dpos, 0); }

    // 2. QKV mixed projection: [n_tokens, 4096] -> [n_tokens, 8192] (f32 output for conv precision)
    gemm_bf16_f32out(handle, model.ssm_proj_f32, model.norm_out, lw.wqkv, n_tokens, MC::ssm_conv_channels, MC::n_embd);

    if (do_dump) dump_f32("qkv_proj", model.ssm_proj_f32, MC::ssm_conv_channels, dpos, 0);

    // 3. Gate Z projection: [n_tokens, 4096] -> [n_tokens, 4096] (f32 output to match ggml precision)
    float* z_buf = model.norm_out_f32;  // reuse f32 buffer for z gate
    gemm_bf16_f32out(handle, z_buf, model.norm_out, lw.wqkv_gate, n_tokens, MC::ssm_d_inner, MC::n_embd);

    if (do_dump) dump_f32("z_gate", z_buf, MC::ssm_d_inner, dpos, 0);

    // 4. Alpha projection: [n_tokens, 4096] -> [n_tokens, 32] (f32 output)
    float* alpha_f32 = model.gemm_out;  // reuse large buffer, only need 32 elements
    gemm_bf16_f32out(handle, alpha_f32, model.norm_out, lw.ssm_alpha, n_tokens, MC::ssm_dt_rank, MC::n_embd);

    if (do_dump) dump_f32("alpha", alpha_f32, MC::ssm_dt_rank, dpos, 0);

    // 5. Beta projection: [n_tokens, 4096] -> [n_tokens, 32] (f32 output)
    float* beta_raw_f32 = model.gemm_out2;  // reuse
    gemm_bf16_f32out(handle, beta_raw_f32, model.norm_out, lw.ssm_beta, n_tokens, MC::ssm_dt_rank, MC::n_embd);

    if (do_dump) dump_f32("beta_raw", beta_raw_f32, MC::ssm_dt_rank, dpos, 0);

    // 6. Compute gate = softplus(alpha + dt_bias) * ssm_a (all f32)
    float* gate_f32 = cuda_alloc<float>(n_tokens * MC::ssm_dt_rank);
    launch_compute_gate(gate_f32, alpha_f32, lw.ssm_dt_bias, lw.ssm_a,
        n_tokens, MC::ssm_dt_rank, 0);

    if (do_dump) dump_f32("gate", gate_f32, MC::ssm_dt_rank, dpos, 0);

    // 7. Beta sigmoid (f32 in, f32 out)
    float* beta_f32 = cuda_alloc<float>(n_tokens * MC::ssm_dt_rank);
    launch_sigmoid(beta_f32, beta_raw_f32, n_tokens * MC::ssm_dt_rank, 0);

    if (do_dump) dump_f32("beta", beta_f32, MC::ssm_dt_rank, dpos, 0);

    // 8. Conv1d + SiLU on QKV mixed (f32 in, f32 out — matches ggml)
    if (do_dump) dump_f32("conv_state_before", model.ssm_conv_state[ssm_idx], (MC::ssm_conv_kernel-1)*MC::ssm_conv_channels, dpos, 0);

    float* conv_out_f32 = cuda_alloc<float>(n_tokens * MC::ssm_conv_channels);
    launch_conv1d_silu(conv_out_f32, model.ssm_proj_f32, model.ssm_conv_state[ssm_idx],
        lw.ssm_conv1d, n_tokens, MC::ssm_conv_channels, MC::ssm_conv_kernel, 0);

    if (do_dump) dump_f32("conv_out", conv_out_f32, MC::ssm_conv_channels, dpos, 0);

    // Update conv state (f32 input)
    launch_update_conv_state(model.ssm_conv_state[ssm_idx], model.ssm_proj_f32,
        model.ssm_conv_state[ssm_idx], n_tokens, MC::ssm_conv_channels, MC::ssm_conv_kernel, 0);

    // 9. Split conv output into Q, K, V (all f32)
    int qk_size = MC::ssm_d_state * MC::ssm_n_group;  // 128 * 16 = 2048

    float* q_ssm = conv_out_f32;
    float* k_ssm = conv_out_f32 + qk_size;
    float* v_ssm = conv_out_f32 + 2 * qk_size;

    // 10. L2 normalize Q and K (f32)
    launch_l2_norm(q_ssm, q_ssm, MC::ssm_n_group, MC::ssm_d_state, MC::rms_norm_eps, 0);
    launch_l2_norm(k_ssm, k_ssm, MC::ssm_n_group, MC::ssm_d_state, MC::rms_norm_eps, 0);

    if (do_dump) { dump_f32("q_norm", q_ssm, qk_size, dpos, 0); dump_f32("k_norm", k_ssm, qk_size, dpos, 0); dump_f32("v", v_ssm, MC::ssm_d_inner, dpos, 0); }

    // 11. Repeat Q and K from num_k_heads=16 to num_v_heads=32 (f32)
    float* q_repeated = cuda_alloc<float>(MC::ssm_dt_rank * MC::ssm_d_state);
    float* k_repeated = cuda_alloc<float>(MC::ssm_dt_rank * MC::ssm_d_state);
    launch_repeat_heads(q_repeated, q_ssm, MC::ssm_n_group, MC::ssm_dt_rank, MC::ssm_d_state, 0);
    launch_repeat_heads(k_repeated, k_ssm, MC::ssm_n_group, MC::ssm_dt_rank, MC::ssm_d_state, 0);

    // 12. Delta-net autoregressive step (all f32)
    float scale = 1.0f / sqrtf((float)MC::ssm_d_state);
    float* delta_out_f32 = cuda_alloc<float>(MC::ssm_dt_rank * MC::ssm_head_v_dim);
    launch_delta_net_decode(delta_out_f32, model.ssm_recurrent_state[ssm_idx],
        q_repeated, k_repeated, v_ssm, gate_f32, beta_f32,
        MC::ssm_dt_rank, MC::ssm_head_v_dim, scale, 0);

    if (do_dump) dump_f32("delta_out", delta_out_f32, MC::ssm_dt_rank * MC::ssm_head_v_dim, dpos, 0);

    // 13. Gated RMSNorm (f32 input, bf16 output for output projection GEMM)
    __nv_bfloat16* gated_out = model.norm_out;
    launch_gated_rmsnorm(gated_out, delta_out_f32, lw.ssm_norm, z_buf,
        MC::ssm_dt_rank, MC::ssm_head_v_dim, MC::rms_norm_eps, 0);

    if (do_dump) dump_bf16("gated_out", gated_out, MC::ssm_d_inner, dpos, 0);

    // 14. Output projection -> f32 for residual
    gemm_bf16_f32out(handle, model.gemm_out, gated_out, lw.ssm_out, n_tokens, MC::n_embd, MC::ssm_d_inner);

    if (do_dump) dump_f32("ssm_proj", model.gemm_out, MC::n_embd, dpos, 0);

    // 15. Residual connection (f32 + f32 -> f32)
    residual_add_f32(hidden, hidden, model.gemm_out, n_tokens * MC::n_embd);

    if (do_dump) dump_f32("after_ssm_residual", hidden, MC::n_embd, dpos, 0);

    // 16. Post-attention norm + FFN (f32 in, bf16 out)
    launch_rmsnorm_f32in(model.norm_out, hidden, lw.post_attn_norm, n_tokens, MC::n_embd, MC::rms_norm_eps, 0);

    gemm_bf16(handle, model.ffn_buf, model.norm_out, lw.ffn_gate, n_tokens, MC::n_ff, MC::n_embd);
    gemm_bf16(handle, model.ffn_buf2, model.norm_out, lw.ffn_up, n_tokens, MC::n_ff, MC::n_embd);
    launch_swiglu(model.ffn_buf, model.ffn_buf, model.ffn_buf2, n_tokens, MC::n_ff, 0);

    // down_proj -> f32 for residual
    gemm_bf16_f32out(handle, model.gemm_out, model.ffn_buf, lw.ffn_down, n_tokens, MC::n_embd, MC::n_ff);

    if (do_dump) dump_f32("ffn_out", model.gemm_out, MC::n_embd, dpos, 0);

    // FFN residual (f32 + f32 -> f32)
    residual_add_f32(hidden, hidden, model.gemm_out, n_tokens * MC::n_embd);

    // Cleanup
    cudaFree(gate_f32);
    cudaFree(beta_f32);
    cudaFree(conv_out_f32);
    cudaFree(q_repeated);
    cudaFree(k_repeated);
    cudaFree(delta_out_f32);
}

// Global temperature (set from command line)
static float g_temperature = 0.8f;

// Embedding lookup to f32: look up bf16 embedding, convert to f32
__global__ void embedding_to_f32_kernel(
    float* __restrict__ output,
    const __nv_bfloat16* __restrict__ embed_table,
    const int* __restrict__ token_ids,
    int dim
) {
    const int token_idx = blockIdx.x;
    const int token_id = token_ids[token_idx];
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const __nv_bfloat16* src = embed_table + (int64_t)token_id * dim;
    float* dst = output + (int64_t)token_idx * dim;
    for (int i = tid; i < dim; i += stride) {
        dst[i] = __bfloat162float(src[i]);
    }
}

// Full forward pass for n_tokens=1 (decode step)
static int forward_decode(Model& model, int token_id, int position) {
    int n_tokens = 1;

    // Upload token ID
    int* token_d = cuda_alloc<int>(1);
    cuda_upload(token_d, &token_id, 1);

    // Positions
    int* pos_d = cuda_alloc<int>(1);
    cuda_upload(pos_d, &position, 1);

    // Embedding lookup -> f32 hidden state
    {
        int threads = 1024;
        embedding_to_f32_kernel<<<n_tokens, threads, 0, 0>>>(
            model.hidden_state, model.tok_embd, token_d, MC::n_embd);
    }

    // Debug: print first few hidden state values after embedding
    if (position == 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
        float dbg[8];
        cuda_download(dbg, model.hidden_state, 8);
        printf("[DEBUG] After embedding (pos=%d, tok=%d): ", position, token_id);
        for (int i = 0; i < 8; i++) printf("%.4f ", dbg[i]);
        printf("...\n");
    }

    // Process all layers
    g_position = position;
    int n_layers_to_run = MC::n_layers;
    if (const char* env = getenv("N_LAYERS")) n_layers_to_run = atoi(env);
    for (int il = 0; il < n_layers_to_run; il++) {
        if (MC::is_recurrent(il)) {
            forward_ssm_layer(model, il, model.layer_subidx[il], model.hidden_state, n_tokens);
        } else {
            forward_attention_layer(model, il, model.layer_subidx[il], model.hidden_state, n_tokens, pos_d);
        }

        // Debug: print after first few layers
        if (position == 0 && il < 4) {
            CUDA_CHECK(cudaDeviceSynchronize());
            float dbg[8];
            cuda_download(dbg, model.hidden_state, 8);
            printf("[DEBUG] After layer %d: ", il);
            for (int i = 0; i < 8; i++) printf("%.4f ", dbg[i]);
            printf("...\n");
        }
        // Dump f32 hidden state for debugging
        if (getenv("DUMP_ALL_LAYERS")) {
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<float> dump(MC::n_embd);
            cuda_download(dump.data(), model.hidden_state, MC::n_embd);
            char fname[128];
            snprintf(fname, sizeof(fname), "/tmp/hidden_f32_pos%d_layer%d.bin", position, il);
            FILE* df = fopen(fname, "wb");
            if (df) { fwrite(dump.data(), 4, MC::n_embd, df); fclose(df); }
        }
    }

    // Update KV cache position
    model.kv_len += n_tokens;

    // Final norm: f32 in, bf16 out
    launch_rmsnorm_f32in(model.norm_out, model.hidden_state, model.output_norm,
        n_tokens, MC::n_embd, MC::rms_norm_eps, 0);

    // LM head: [1, 4096] -> [1, 248320] -> f32 logits
    gemm_bf16_f32out(model.cublas_handle, model.logits_f32, model.norm_out, model.output,
        n_tokens, MC::n_vocab, MC::n_embd);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Debug: show top-5 logits
    {
        std::vector<float> logits_host(MC::n_vocab);
        cuda_download(logits_host.data(), model.logits_f32, MC::n_vocab);
        std::vector<int> top_idx(MC::n_vocab);
        std::iota(top_idx.begin(), top_idx.end(), 0);
        std::partial_sort(top_idx.begin(), top_idx.begin() + 5, top_idx.end(),
            [&](int a, int b) { return logits_host[a] > logits_host[b]; });
        printf("[DEBUG] Top-5 logits: ");
        for (int i = 0; i < 5; i++) {
            printf("%d(%.2f) ", top_idx[i], logits_host[top_idx[i]]);
        }
        printf("\n");
        int greedy = 0;
        for (int i = 1; i < MC::n_vocab; i++) {
            if (logits_host[i] > logits_host[greedy]) greedy = i;
        }
        printf("[DEBUG] Greedy pick: %d (%.4f)\n", greedy, logits_host[greedy]);
        printf("[DEBUG] Paris tokens: 11751(%.4f) 57590(%.4f)\n", logits_host[11751], logits_host[57590]);
        printf("[DEBUG] Ref tokens: 314(%.4f) 279(%.4f) 369(%.4f) 3177(%.4f)\n",
            logits_host[314], logits_host[279], logits_host[369], logits_host[3177]);
    }

    // Sample
    int next_token = sample_token(model.logits_f32, MC::n_vocab, g_temperature);

    cudaFree(token_d);
    cudaFree(pos_d);

    return next_token;
}

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s -m <model_path> -p <prompt> [-n <max_tokens>] [-t <temperature>]\n", prog);
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string prompt;
    int max_gen_tokens = 128;
    float temperature = 0.8f;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_gen_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            temperature = atof(argv[++i]);
            g_temperature = temperature;
        }
    }

    if (model_path.empty()) {
        model_path = "/home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-BF16.gguf";
    }
    if (prompt.empty()) {
        prompt = "Hello, world!";
    }

    printf("Model: %s\n", model_path.c_str());
    printf("Prompt: %s\n", prompt.c_str());
    printf("Max tokens: %d\n", max_gen_tokens);
    printf("Temperature: %.2f\n\n", temperature);

    // Load tokenizer
    Tokenizer tokenizer;
    if (!tokenizer.load(model_path)) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return 1;
    }

    // Tokenize prompt
    std::vector<int> prompt_tokens = tokenizer.encode(prompt);
    printf("Prompt tokens (%zu): ", prompt_tokens.size());
    for (int t : prompt_tokens) printf("%d ", t);
    printf("\n\n");

    // Load model
    Model model;
    memset(&model, 0, sizeof(model));
    if (!load_model(model_path, model)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Allocate buffers
    int max_kv_len = (int)prompt_tokens.size() + max_gen_tokens + 16;
    allocate_buffers(model, 1, max_kv_len);

    printf("\nGenerating...\n");

    // Process prompt tokens one by one (simple decode-only path)
    auto t_start = std::chrono::high_resolution_clock::now();

    int next_token = -1;
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        next_token = forward_decode(model, prompt_tokens[i], (int)i);
    }

    auto t_prompt = std::chrono::high_resolution_clock::now();
    double prompt_ms = std::chrono::duration<double, std::milli>(t_prompt - t_start).count();

    // Print prompt
    printf("%s", prompt.c_str());
    fflush(stdout);

    // Generate tokens
    std::vector<int> generated;
    auto t_gen_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < max_gen_tokens; i++) {
        if (next_token == tokenizer.eos_token_id()) break;

        generated.push_back(next_token);
        std::string tok_str = tokenizer.decode(next_token);
        fprintf(stderr, "[tok %d = %s] ", next_token, tok_str.c_str());
        printf("%s", tok_str.c_str());
        fflush(stdout);
        fflush(stderr);

        int pos = (int)prompt_tokens.size() + i;
        next_token = forward_decode(model, next_token, pos);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t_end - t_gen_start).count();

    printf("\n\n--- Performance ---\n");
    printf("Prompt tokens: %zu (%.1f ms, %.1f tok/s)\n",
        prompt_tokens.size(), prompt_ms,
        prompt_tokens.size() * 1000.0 / prompt_ms);
    printf("Generated tokens: %zu (%.1f ms, %.1f tok/s)\n",
        generated.size(), gen_ms,
        generated.size() * 1000.0 / gen_ms);

    free_model(model);
    return 0;
}
