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
void launch_fused_residual_rmsnorm(__nv_bfloat16* norm_output, float* hidden,
    const float* residual, const float* weight, int n_tokens, int dim, float eps, cudaStream_t stream);
void launch_fused_bf16_residual_rmsnorm(__nv_bfloat16* norm_output, float* hidden,
    const __nv_bfloat16* residual_bf16, const float* weight, int n_tokens, int dim, float eps, cudaStream_t stream);
void launch_bf16_residual_add(float* hidden, const __nv_bfloat16* residual_bf16, int n, cudaStream_t stream);
void launch_rmsnorm_head(__nv_bfloat16* output, const __nv_bfloat16* input,
    const float* weight, int n_tokens, int n_heads, int head_dim,
    float eps, cudaStream_t stream);
void launch_rope(__nv_bfloat16* qk, const int* positions, int n_tokens,
    int n_heads, int head_dim, int rope_dim, float freq_base, cudaStream_t stream);
void launch_swiglu(__nv_bfloat16* output, const __nv_bfloat16* gate,
    const __nv_bfloat16* up, int n_tokens, int n_ff, cudaStream_t stream);
void launch_swiglu_packed(__nv_bfloat16* output, const __nv_bfloat16* packed,
    int n_tokens, int n_ff, cudaStream_t stream);
void launch_sigmoid_mul(__nv_bfloat16* output, const __nv_bfloat16* attn_out,
    const __nv_bfloat16* gate, int n_elements, cudaStream_t stream);
void launch_kv_cache_append(__nv_bfloat16* cache, const __nv_bfloat16* new_kv,
    const int* d_kv_pos, int n_new_tokens, int kv_dim, cudaStream_t stream);
void launch_attention_decode(__nv_bfloat16* output, const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    const int* d_kv_len, int max_kv_len, int n_head, int n_head_kv, int head_dim, float scale, cudaStream_t stream);
void launch_attention_prefill(__nv_bfloat16* output, const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    int n_tokens, int kv_start, int n_head, int n_head_kv, int head_dim, float scale, cudaStream_t stream);
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
void launch_fused_ssm_step(__nv_bfloat16* output, float* state,
    const float* conv_out, const float* alpha, const float* dt_bias,
    const float* ssm_a, const float* beta_raw, const float* z,
    const float* norm_weight,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    float scale, float l2_eps, float rms_eps, cudaStream_t stream);
void launch_fused_ssm_step_batched(__nv_bfloat16* output, float* state,
    const float* conv_out, const float* alpha, const float* dt_bias,
    const float* ssm_a, const float* beta_raw, const float* z,
    const float* norm_weight, int n_tokens,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int conv_channels, int d_inner,
    float scale, float l2_eps, float rms_eps, cudaStream_t stream);

using MC = ModelConfig;

// Vectorized GEMV: y[N] = W[N,K] @ x[K], bf16 weights × bf16 input → bf16/f32 output
// Uses shared memory for x vector, 128-bit vectorized loads for weights.
// Each warp processes one row. 8 warps/block = 8 rows/block.
// Requires K divisible by 8 (true for all our GEMM sizes).
template<bool F32_OUTPUT>
__global__ void fast_gemv_kernel(
    void* __restrict__ y,
    const __nv_bfloat16* __restrict__ W,  // [N, K] row-major
    const __nv_bfloat16* __restrict__ x,  // [K]
    int N, int K
) {
    extern __shared__ __nv_bfloat16 sx[];  // [K] in shared memory

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int row = blockIdx.x * 8 + warp_id;  // 8 warps = 8 rows per block

    // Cooperatively load x into shared memory using 128-bit loads
    const int K8 = K / 8;  // number of uint4 chunks
    uint4* sx_v = reinterpret_cast<uint4*>(sx);
    const uint4* x_v = reinterpret_cast<const uint4*>(x);
    for (int i = tid; i < K8; i += 256) {
        sx_v[i] = x_v[i];
    }
    __syncthreads();

    if (row >= N) return;

    // Vectorized dot product: each thread handles K8/32 chunks (128-bit each)
    const uint4* W_v = reinterpret_cast<const uint4*>(&W[(int64_t)row * K]);
    float acc = 0.0f;

    for (int i = lane_id; i < K8; i += 32) {
        uint4 wc = W_v[i];
        uint4 xc = sx_v[i];
        const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wc);
        const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xc);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            acc += __bfloat162float(wb[j]) * __bfloat162float(xb[j]);
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane_id == 0) {
        if constexpr (F32_OUTPUT) {
            reinterpret_cast<float*>(y)[row] = acc;
        } else {
            reinterpret_cast<__nv_bfloat16*>(y)[row] = __float2bfloat16(acc);
        }
    }
}

static void launch_gemv_bf16(__nv_bfloat16* y, const __nv_bfloat16* W, const __nv_bfloat16* x,
    int N, int K, cudaStream_t stream) {
    int blocks = cdiv(N, 8);
    size_t smem = K * sizeof(__nv_bfloat16);
    fast_gemv_kernel<false><<<blocks, 256, smem, stream>>>(y, W, x, N, K);
}

static void launch_gemv_bf16_f32out(float* y, const __nv_bfloat16* W, const __nv_bfloat16* x,
    int N, int K, cudaStream_t stream) {
    int blocks = cdiv(N, 8);
    size_t smem = K * sizeof(__nv_bfloat16);
    fast_gemv_kernel<true><<<blocks, 256, smem, stream>>>(y, W, x, N, K);
}

// Flag to enable/disable custom GEMV (set via env var CUSTOM_GEMV=1)
static bool g_use_custom_gemv = false;

// Pre-allocated decode buffers (avoid per-token cudaMalloc)
// Packed as [token_id, position, kv_len, kv_len+1] for single memcpy
static int* g_decode_params_d = nullptr;
static int* g_token_d = nullptr;   // points to g_decode_params_d[0]
static int* g_pos_d = nullptr;     // points to g_decode_params_d[1]

// cuBLAS GEMM wrapper: C = A @ B^T  (row-major), bf16 output
// A: [M, K] bf16, B: [N, K] bf16, C: [M, N] bf16
static cudaStream_t g_compute_stream = 0;  // set from model.compute_stream

static void gemm_bf16(
    cublasHandle_t handle,
    __nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B,
    int M, int N, int K
) {
    if (M == 1 && g_use_custom_gemv) {
        launch_gemv_bf16(C, B, A, N, K, g_compute_stream);
        return;
    }
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

// cuBLAS GEMM wrapper: C = A @ B^T, f32 output
// For small M (decode): direct f32 output from cuBLAS
// For large M (batched): bf16 GEMM + cast (cuBLAS has better bf16→bf16 kernels for large M)
static __nv_bfloat16* g_gemm_bf16_tmp = nullptr;
static int g_gemm_bf16_tmp_size = 0;

static void gemm_bf16_f32out(
    cublasHandle_t handle,
    float* C, const __nv_bfloat16* A, const __nv_bfloat16* B,
    int M, int N, int K
) {
    if (M == 1 && g_use_custom_gemv) {
        launch_gemv_bf16_f32out(C, B, A, N, K, g_compute_stream);
        return;
    }
    float alpha = 1.0f, beta_val = 0.0f;

    if (M <= 4) {
        // Small M: direct f32 output saves a cast kernel
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            B, CUDA_R_16BF, K,
            A, CUDA_R_16BF, K,
            &beta_val,
            C, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    } else {
        // Large M: bf16 GEMM + cast is faster
        int needed = M * N;
        if (needed > g_gemm_bf16_tmp_size) {
            if (g_gemm_bf16_tmp) cudaFree(g_gemm_bf16_tmp);
            g_gemm_bf16_tmp = cuda_alloc<__nv_bfloat16>(needed);
            g_gemm_bf16_tmp_size = needed;
        }
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
        gemm_bf16_to_f32_kernel<<<cdiv(needed, 256), 256>>>(C, g_gemm_bf16_tmp, needed);
    }
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

// Deinterleave combined SSM projection from bf16 [n_tokens, combined_n] -> 4 separate f32 buffers
__global__ void deinterleave_ssm_proj_bf16_kernel(
    float* __restrict__ qkv,      // [n_tokens, conv_channels]
    float* __restrict__ z,         // [n_tokens, d_inner]
    float* __restrict__ alpha,     // [n_tokens, dt_rank]
    float* __restrict__ beta,      // [n_tokens, dt_rank]
    const __nv_bfloat16* __restrict__ combined,  // [n_tokens, combined_n]
    int n_tokens, int conv_channels, int d_inner, int dt_rank, int combined_n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * combined_n;
    if (tid >= total) return;
    int t = tid / combined_n;
    int c = tid % combined_n;
    float val = __bfloat162float(combined[tid]);
    if (c < conv_channels) {
        qkv[t * conv_channels + c] = val;
    } else if (c < conv_channels + d_inner) {
        z[t * d_inner + (c - conv_channels)] = val;
    } else if (c < conv_channels + d_inner + dt_rank) {
        alpha[t * dt_rank + (c - conv_channels - d_inner)] = val;
    } else {
        beta[t * dt_rank + (c - conv_channels - d_inner - dt_rank)] = val;
    }
}

// Deinterleave Q and Gate from packed [n_heads, head_dim*2] -> Q [n_heads, head_dim] + Gate [n_heads, head_dim]
__global__ void deinterleave_qg_kernel(
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ packed,
    int n_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads * head_dim;
    if (idx >= total) return;
    int head = idx / head_dim;
    int d = idx % head_dim;
    q_out[idx] = packed[head * head_dim * 2 + d];
    gate_out[idx] = packed[head * head_dim * 2 + head_dim + d];
}

// Allocate inference buffers
static void allocate_buffers(Model& model, int max_batch, int max_kv_len) {
    // max_batch = max prompt length for batched processing
    model.max_tokens = max_batch;
    model.max_kv_len = max_kv_len;
    model.kv_len = 0;

    // Hidden state in f32 for residual stream precision (matching ggml behavior)
    model.hidden_state = cuda_alloc<float>(max_batch * MC::n_embd);
    model.hidden_bf16  = cuda_alloc<__nv_bfloat16>(max_batch * MC::n_embd);
    model.norm_out     = cuda_alloc<__nv_bfloat16>(max_batch * MC::n_embd);
    model.norm_out_f32 = cuda_alloc<float>(max_batch * MC::n_embd);
    model.attn_out     = cuda_alloc<__nv_bfloat16>(max_batch * MC::n_embd);
    int gemm_max = MC::n_ff > MC::n_embd ? MC::n_ff : MC::n_embd;
    gemm_max = gemm_max > MC::n_vocab ? gemm_max : MC::n_vocab;
    // n_head * head_dim * 2 = 8192, ssm_conv_channels = 8192 - use max
    int qkv_max = MC::n_head * MC::head_dim * 2;
    if (MC::ssm_conv_channels > qkv_max) qkv_max = MC::ssm_conv_channels;
    model.gemm_out     = cuda_alloc<float>(max_batch * gemm_max);
    model.gemm_out2    = cuda_alloc<float>(max_batch * gemm_max);
    model.ffn_buf      = cuda_alloc<__nv_bfloat16>(max_batch * 2 * MC::n_ff);  // sized for packed gate+up
    model.ffn_buf2     = cuda_alloc<__nv_bfloat16>(max_batch * MC::n_ff);

    // QKV temp buffer (sized for batched attention)
    model.qkv_buf = cuda_alloc<__nv_bfloat16>(max_batch * qkv_max);
    // SSM f32 projection buffer for conv state precision
    model.ssm_proj_f32 = cuda_alloc<float>(max_batch * MC::ssm_conv_channels);

    model.logits_f32 = cuda_alloc<float>(max_batch * MC::n_vocab);

    // KV caches for attention layers
    int kv_dim = MC::n_head_kv * MC::head_dim;
    for (int i = 0; i < 8; i++) {
        model.k_cache[i] = cuda_alloc<__nv_bfloat16>(max_kv_len * kv_dim);
        model.v_cache[i] = cuda_alloc<__nv_bfloat16>(max_kv_len * kv_dim);
    }

    // Pre-allocated SSM temp buffers (sized for batched prompt)
    model.ssm_gate_buf     = cuda_alloc<float>(max_batch * MC::ssm_dt_rank);
    model.ssm_beta_buf     = cuda_alloc<float>(max_batch * MC::ssm_dt_rank);
    model.ssm_conv_out_buf = cuda_alloc<float>(max_batch * MC::ssm_conv_channels);
    model.ssm_q_rep_buf    = cuda_alloc<float>(max_batch * MC::ssm_dt_rank * MC::ssm_d_state);
    model.ssm_k_rep_buf    = cuda_alloc<float>(max_batch * MC::ssm_dt_rank * MC::ssm_d_state);
    model.ssm_delta_out_buf = cuda_alloc<float>(max_batch * MC::ssm_dt_rank * MC::ssm_head_v_dim);

    // SSM states
    int conv_state_size = (MC::ssm_conv_kernel - 1) * MC::ssm_conv_channels;
    int recurrent_state_size = MC::ssm_dt_rank * MC::ssm_head_v_dim * MC::ssm_head_v_dim;
    for (int i = 0; i < 24; i++) {
        model.ssm_conv_state[i] = cuda_alloc<float>(conv_state_size);
        CUDA_CHECK(cudaMemset(model.ssm_conv_state[i], 0, conv_state_size * sizeof(float)));
        model.ssm_recurrent_state[i] = cuda_alloc<float>(recurrent_state_size);
        CUDA_CHECK(cudaMemset(model.ssm_recurrent_state[i], 0, recurrent_state_size * sizeof(float)));
    }

    // CUDA graph support
    // Packed decode params: [token_id, position, kv_len, kv_len+1] — single memcpy per decode
    g_decode_params_d = cuda_alloc<int>(4);
    g_token_d = g_decode_params_d;
    g_pos_d = g_decode_params_d + 1;
    model.d_kv_len = g_decode_params_d + 2;
    CUDA_CHECK(cudaStreamCreate(&model.compute_stream));
    cublasSetStream(model.cublas_handle, model.compute_stream);
    g_compute_stream = model.compute_stream;
    g_use_custom_gemv = (getenv("CUSTOM_GEMV") != nullptr);
    model.decode_graph = nullptr;
    model.decode_graph_exec = nullptr;
    model.decode_graph_captured = false;
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

// Forward pass for one full-attention layer
// hidden is f32 (residual stream). Internal computation uses bf16 for GEMMs.
static void forward_attention_layer(Model& model, int layer_idx, int attn_idx,
    float* hidden, int n_tokens, int* positions_d) {
    auto& lw = model.attn_layers[attn_idx];
    auto handle = model.cublas_handle;
    cudaStream_t s = model.compute_stream;

    // 1. RMSNorm (f32 in, bf16 out)
    launch_rmsnorm_f32in(model.norm_out, hidden, lw.attn_norm, n_tokens, MC::n_embd, MC::rms_norm_eps, s);

    // 2. Q+Gate projection: [n_tokens, 4096] -> [n_tokens, 8192]
    gemm_bf16(handle, model.qkv_buf, model.norm_out, lw.wq, n_tokens, MC::n_head * MC::head_dim * 2, MC::n_embd);

    int kv_dim = MC::n_head_kv * MC::head_dim;
    __nv_bfloat16* k_proj;
    __nv_bfloat16* v_proj;

    if (n_tokens == 1) {
        // Decode: packed K+V GEMM saves 1 cuBLAS launch
        __nv_bfloat16* kv_packed = model.attn_out;
        gemm_bf16(handle, kv_packed, model.norm_out, lw.wkv, 1, 2 * kv_dim, MC::n_embd);
        k_proj = kv_packed;
        v_proj = kv_packed + kv_dim;
    } else {
        // Batched: separate GEMMs (avoids deinterleave overhead)
        k_proj = model.attn_out;
        gemm_bf16(handle, k_proj, model.norm_out, lw.wk, n_tokens, kv_dim, MC::n_embd);
        v_proj = model.ffn_buf2;
        gemm_bf16(handle, v_proj, model.norm_out, lw.wv, n_tokens, kv_dim, MC::n_embd);
    }

    // 5. Deinterleave Q and Gate with a single kernel
    __nv_bfloat16* q_contiguous = model.norm_out;
    __nv_bfloat16* gate_buf = model.hidden_bf16;
    {
        int total = n_tokens * MC::n_head * MC::head_dim;
        deinterleave_qg_kernel<<<cdiv(total, 256), 256, 0, s>>>(
            q_contiguous, gate_buf, model.qkv_buf, n_tokens * MC::n_head, MC::head_dim);
    }

    // Q/K norm
    launch_rmsnorm_head(q_contiguous, q_contiguous, lw.attn_q_norm,
        n_tokens, MC::n_head, MC::head_dim, MC::rms_norm_eps, s);
    launch_rmsnorm_head(k_proj, k_proj, lw.attn_k_norm,
        n_tokens, MC::n_head_kv, MC::head_dim, MC::rms_norm_eps, s);

    // 6. RoPE on Q and K
    launch_rope(q_contiguous, positions_d, n_tokens, MC::n_head, MC::head_dim,
        MC::rope_dim, MC::rope_freq_base, s);
    launch_rope(k_proj, positions_d, n_tokens, MC::n_head_kv, MC::head_dim,
        MC::rope_dim, MC::rope_freq_base, s);

    // 7. Append K, V to cache (d_kv_len has current kv_len value)
    launch_kv_cache_append(model.k_cache[attn_idx], k_proj, model.d_kv_len, n_tokens, kv_dim, s);
    launch_kv_cache_append(model.v_cache[attn_idx], v_proj, model.d_kv_len, n_tokens, kv_dim, s);

    // 8. Attention (use prefill kernel for n_tokens > 1)
    if (n_tokens > 1) {
        launch_attention_prefill(model.attn_out, q_contiguous,
            model.k_cache[attn_idx], model.v_cache[attn_idx],
            n_tokens, model.kv_len, MC::n_head, MC::n_head_kv, MC::head_dim, MC::attn_scale, s);
    } else {
        // d_kv_len[1] has kv_len + n_tokens (total entries after append)
        launch_attention_decode(model.attn_out, q_contiguous,
            model.k_cache[attn_idx], model.v_cache[attn_idx],
            model.d_kv_len + 1, model.max_kv_len, MC::n_head, MC::n_head_kv, MC::head_dim, MC::attn_scale, s);
    }

    // 9. Sigmoid gate
    launch_sigmoid_mul(model.attn_out, model.attn_out, gate_buf,
        n_tokens * MC::n_head * MC::head_dim, s);

    // 10. Output projection -> bf16
    gemm_bf16(handle, model.attn_out, model.attn_out, lw.wo, n_tokens, MC::n_embd, MC::n_head * MC::head_dim);

    // 11. Fused bf16 cast + residual + post-attention norm (saves cast kernel)
    launch_fused_bf16_residual_rmsnorm(model.norm_out, hidden, model.attn_out,
        lw.post_attn_norm, n_tokens, MC::n_embd, MC::rms_norm_eps, s);

    // FFN: fused gate+up GEMM, then SwiGLU from packed layout
    gemm_bf16(handle, model.ffn_buf, model.norm_out, lw.ffn_gate_up, n_tokens, 2 * MC::n_ff, MC::n_embd);
    launch_swiglu_packed(model.ffn_buf, model.ffn_buf, n_tokens, MC::n_ff, s);

    // down_proj -> bf16, then fused bf16 residual add
    gemm_bf16(handle, model.attn_out, model.ffn_buf, lw.ffn_down, n_tokens, MC::n_embd, MC::n_ff);
    launch_bf16_residual_add(hidden, model.attn_out, n_tokens * MC::n_embd, s);
}

// Forward pass for one SSM (delta-net) layer
// hidden is f32 (residual stream), n_tokens can be > 1 for prompt batching
static void forward_ssm_layer(Model& model, int layer_idx, int ssm_idx,
    float* hidden, int n_tokens) {
    auto& lw = model.ssm_layers[ssm_idx];
    auto handle = model.cublas_handle;
    cudaStream_t s = model.compute_stream;

    // 1. RMSNorm (f32 in, bf16 out)
    launch_rmsnorm_f32in(model.norm_out, hidden, lw.attn_norm, n_tokens, MC::n_embd, MC::rms_norm_eps, s);

    // Pointers to SSM projection outputs
    float* qkv_proj;     // [n_tokens, ssm_conv_channels=8192]
    float* z_buf;        // [n_tokens, ssm_d_inner=4096]
    float* alpha_f32;    // [n_tokens, ssm_dt_rank=32]
    float* beta_raw_f32; // [n_tokens, ssm_dt_rank=32]

    if (n_tokens == 1) {
        // Decode: single combined GEMM, use pointer offsets (no copies!)
        static constexpr int combined_n = MC::ssm_conv_channels + MC::ssm_d_inner + MC::ssm_dt_rank + MC::ssm_dt_rank;
        gemm_bf16_f32out(handle, model.gemm_out, model.norm_out, lw.w_combined, 1, combined_n, MC::n_embd);

        qkv_proj    = model.gemm_out;
        z_buf       = model.gemm_out + MC::ssm_conv_channels;
        alpha_f32   = model.gemm_out + MC::ssm_conv_channels + MC::ssm_d_inner;
        beta_raw_f32= model.gemm_out + MC::ssm_conv_channels + MC::ssm_d_inner + MC::ssm_dt_rank;
    } else {
        // Batched: bf16 combined GEMM + fused deinterleave+cast
        static constexpr int combined_n = MC::ssm_conv_channels + MC::ssm_d_inner + MC::ssm_dt_rank + MC::ssm_dt_rank;
        qkv_proj = model.ssm_proj_f32;
        z_buf = model.norm_out_f32;
        alpha_f32 = model.gemm_out2;
        beta_raw_f32 = model.gemm_out2 + n_tokens * MC::ssm_dt_rank;

        // bf16 GEMM into bf16 temp buffer (reuse ffn_buf which is max_batch * 2*n_ff bf16 — large enough)
        __nv_bfloat16* combined_bf16 = model.ffn_buf;
        gemm_bf16(handle, combined_bf16, model.norm_out, lw.w_combined, n_tokens, combined_n, MC::n_embd);
        // Fused deinterleave + bf16→f32 cast
        int total = n_tokens * combined_n;
        deinterleave_ssm_proj_bf16_kernel<<<cdiv(total, 256), 256, 0, s>>>(
            qkv_proj, z_buf, alpha_f32, beta_raw_f32,
            combined_bf16, n_tokens, MC::ssm_conv_channels, MC::ssm_d_inner, MC::ssm_dt_rank, combined_n);
    }

    // 6. Conv1d + SiLU on QKV mixed (f32 in, f32 out, batched)
    launch_conv1d_silu(model.ssm_conv_out_buf, qkv_proj, model.ssm_conv_state[ssm_idx],
        lw.ssm_conv1d, n_tokens, MC::ssm_conv_channels, MC::ssm_conv_kernel, s);

    // Update conv state (f32 input)
    launch_update_conv_state(model.ssm_conv_state[ssm_idx], qkv_proj,
        model.ssm_conv_state[ssm_idx], n_tokens, MC::ssm_conv_channels, MC::ssm_conv_kernel, s);

    // 7-13. Fused SSM step
    float scale = 1.0f / sqrtf((float)MC::ssm_d_state);

    if (n_tokens == 1) {
        launch_fused_ssm_step(
            model.norm_out,
            model.ssm_recurrent_state[ssm_idx],
            model.ssm_conv_out_buf,
            alpha_f32, lw.ssm_dt_bias, lw.ssm_a, beta_raw_f32,
            z_buf, lw.ssm_norm,
            MC::ssm_n_group, MC::ssm_dt_rank,
            MC::ssm_d_state, MC::ssm_head_v_dim,
            scale, MC::rms_norm_eps, MC::rms_norm_eps, s);
    } else {
        launch_fused_ssm_step_batched(
            model.norm_out,
            model.ssm_recurrent_state[ssm_idx],
            model.ssm_conv_out_buf,
            alpha_f32, lw.ssm_dt_bias, lw.ssm_a, beta_raw_f32,
            z_buf, lw.ssm_norm, n_tokens,
            MC::ssm_n_group, MC::ssm_dt_rank,
            MC::ssm_d_state, MC::ssm_head_v_dim,
            MC::ssm_conv_channels, MC::ssm_d_inner,
            scale, MC::rms_norm_eps, MC::rms_norm_eps, s);
    }

    // 14. Output projection -> bf16
    gemm_bf16(handle, model.attn_out, model.norm_out, lw.ssm_out, n_tokens, MC::n_embd, MC::ssm_d_inner);

    // 15. Fused bf16 cast + residual + post-attention norm
    launch_fused_bf16_residual_rmsnorm(model.norm_out, hidden, model.attn_out,
        lw.post_attn_norm, n_tokens, MC::n_embd, MC::rms_norm_eps, s);

    // FFN: fused gate+up GEMM, then SwiGLU from packed layout
    gemm_bf16(handle, model.ffn_buf, model.norm_out, lw.ffn_gate_up, n_tokens, 2 * MC::n_ff, MC::n_embd);
    launch_swiglu_packed(model.ffn_buf, model.ffn_buf, n_tokens, MC::n_ff, s);

    // down_proj -> bf16, then fused bf16 residual add
    gemm_bf16(handle, model.attn_out, model.ffn_buf, lw.ffn_down, n_tokens, MC::n_embd, MC::n_ff);
    launch_bf16_residual_add(hidden, model.attn_out, n_tokens * MC::n_embd, s);
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

static void ensure_decode_bufs() {
    // g_decode_params_d is allocated during model init
}

// Execute decode forward pass body (embedding through LM head)
// All operations go to model.compute_stream
static void forward_decode_body(Model& model) {
    cudaStream_t s = model.compute_stream;

    // Embedding lookup -> f32 hidden state
    embedding_to_f32_kernel<<<1, 1024, 0, s>>>(
        model.hidden_state, model.tok_embd, g_token_d, MC::n_embd);

    // Process all layers
    for (int il = 0; il < MC::n_layers; il++) {
        if (MC::is_recurrent(il)) {
            forward_ssm_layer(model, il, model.layer_subidx[il], model.hidden_state, 1);
        } else {
            forward_attention_layer(model, il, model.layer_subidx[il], model.hidden_state, 1, g_pos_d);
        }
    }

    // Final norm: f32 in, bf16 out
    launch_rmsnorm_f32in(model.norm_out, model.hidden_state, model.output_norm,
        1, MC::n_embd, MC::rms_norm_eps, s);

    // LM head: [1, 4096] -> [1, 248320] -> f32 logits
    gemm_bf16_f32out(model.cublas_handle, model.logits_f32, model.norm_out, model.output,
        1, MC::n_vocab, MC::n_embd);
}

// Full forward pass for n_tokens=1 (decode step) with CUDA graph acceleration
static int forward_decode(Model& model, int token_id, int position) {
    ensure_decode_bufs();
    cudaStream_t s = model.compute_stream;

    // Upload all decode parameters in a single memcpy (token_id, position, kv_len, kv_len+1)
    int decode_params[4] = { token_id, position, model.kv_len, model.kv_len + 1 };
    CUDA_CHECK(cudaMemcpyAsync(g_decode_params_d, decode_params, 4 * sizeof(int), cudaMemcpyHostToDevice, s));

    if (getenv("NO_GRAPH")) {
        forward_decode_body(model);
    } else if (!model.decode_graph_captured) {
        // First decode: capture the compute graph
        CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        forward_decode_body(model);
        CUDA_CHECK(cudaStreamEndCapture(s, &model.decode_graph));
        CUDA_CHECK(cudaGraphInstantiate(&model.decode_graph_exec, model.decode_graph, nullptr, nullptr, 0));
        model.decode_graph_captured = true;
        CUDA_CHECK(cudaGraphLaunch(model.decode_graph_exec, s));
    } else {
        // Subsequent decodes: replay the captured graph
        CUDA_CHECK(cudaGraphLaunch(model.decode_graph_exec, s));
    }

    model.kv_len += 1;

    // Sample: for greedy, launch argmax on compute_stream (avoids sync gap)
    if (g_temperature <= 0.0f) {
        return gpu_argmax_on_stream(model.logits_f32, MC::n_vocab, s);
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
    return sample_token(model.logits_f32, MC::n_vocab, g_temperature);
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

    // Allocate buffers (max_batch = prompt size for batched prefill)
    int max_batch = (int)prompt_tokens.size();
    if (max_batch < 1) max_batch = 1;
    int max_kv_len = (int)prompt_tokens.size() + max_gen_tokens + 16;
    allocate_buffers(model, max_batch, max_kv_len);

    // Warm up cuBLAS on compute stream (first call allocates workspace)
    {
        __nv_bfloat16* dummy_a = model.norm_out;
        __nv_bfloat16* dummy_b = model.norm_out;
        __nv_bfloat16* dummy_c = model.attn_out;
        gemm_bf16(model.cublas_handle, dummy_c, dummy_a, dummy_b, 1, 64, 64);
        // Also warm up the specific GEMM sizes used during decode (cuBLAS caches kernel plans)
        static constexpr int combined_n = MC::ssm_conv_channels + MC::ssm_d_inner + MC::ssm_dt_rank + MC::ssm_dt_rank;
        gemm_bf16_f32out(model.cublas_handle, model.gemm_out, dummy_a, model.ssm_layers[0].w_combined, 1, combined_n, MC::n_embd);
        gemm_bf16(model.cublas_handle, dummy_c, dummy_a, model.ssm_layers[0].ssm_out, 1, MC::n_embd, MC::ssm_d_inner);
        gemm_bf16(model.cublas_handle, model.ffn_buf, dummy_c, model.ssm_layers[0].ffn_gate_up, 1, 2 * MC::n_ff, MC::n_embd);
        gemm_bf16(model.cublas_handle, dummy_c, model.ffn_buf, model.ssm_layers[0].ffn_down, 1, MC::n_embd, MC::n_ff);
        gemm_bf16_f32out(model.cublas_handle, model.logits_f32, dummy_a, model.output, 1, MC::n_vocab, MC::n_embd);
        CUDA_CHECK(cudaStreamSynchronize(model.compute_stream));
    }

    printf("\nGenerating...\n");

    // Batched prefill: process all prompt tokens through each layer together
    auto t_start = std::chrono::high_resolution_clock::now();

    int n_prompt = (int)prompt_tokens.size();
    int next_token = -1;
    {
        cudaStream_t s = model.compute_stream;

        // Upload all token IDs and positions
        int* tokens_d = cuda_alloc<int>(n_prompt);
        int* pos_d = cuda_alloc<int>(n_prompt);
        cuda_upload(tokens_d, prompt_tokens.data(), n_prompt);
        std::vector<int> positions(n_prompt);
        for (int i = 0; i < n_prompt; i++) positions[i] = i;
        cuda_upload(pos_d, positions.data(), n_prompt);

        // Upload kv_len for attention layers (kv_len=0 at start)
        int kv_params[2] = { model.kv_len, model.kv_len + n_prompt };
        cuda_upload(model.d_kv_len, kv_params, 2);

        // Embedding lookup for all prompt tokens -> f32 hidden state
        embedding_to_f32_kernel<<<n_prompt, 1024, 0, s>>>(
            model.hidden_state, model.tok_embd, tokens_d, MC::n_embd);

        // Process all layers with batched tokens
        for (int il = 0; il < MC::n_layers; il++) {
            if (MC::is_recurrent(il)) {
                forward_ssm_layer(model, il, model.layer_subidx[il], model.hidden_state, n_prompt);
            } else {
                forward_attention_layer(model, il, model.layer_subidx[il], model.hidden_state, n_prompt, pos_d);
            }
        }

        // Update KV cache position
        model.kv_len += n_prompt;

        // Final norm on last token only
        float* last_hidden = model.hidden_state + (n_prompt - 1) * MC::n_embd;
        launch_rmsnorm_f32in(model.norm_out, last_hidden, model.output_norm,
            1, MC::n_embd, MC::rms_norm_eps, s);

        // LM head on last token
        gemm_bf16_f32out(model.cublas_handle, model.logits_f32, model.norm_out, model.output,
            1, MC::n_vocab, MC::n_embd);

        CUDA_CHECK(cudaStreamSynchronize(s));
        next_token = sample_token(model.logits_f32, MC::n_vocab, g_temperature);

        cudaFree(tokens_d);
        cudaFree(pos_d);
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
        printf("%s", tok_str.c_str());
        fflush(stdout);

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
