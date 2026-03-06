#include "../utils.h"
#include "../model.h"
#include <cuda_bf16.h>
#include <cstdint>
#include <cfloat>

// ============================================================
// Delta-Net Linear Attention (for SSM/recurrent layers)
//
// Architecture per layer:
//   1. attn_norm(x)
//   2. qkv_mixed = wqkv(x)        [n_tokens, ssm_conv_channels=8192]
//   3. z = wqkv_gate(x)            [n_tokens, ssm_d_inner=4096]
//   4. alpha = ssm_alpha(x)        [n_tokens, ssm_dt_rank=32]
//   5. beta = sigmoid(ssm_beta(x)) [n_tokens, ssm_dt_rank=32]
//   6. gate = softplus(alpha + dt_bias) * ssm_a  [n_tokens, ssm_dt_rank=32]
//   7. conv_out = silu(conv1d(concat(conv_state, qkv_mixed)))
//   8. Split conv_out -> q, k, v
//   9. q, k = l2_norm(q), l2_norm(k)
//  10. delta-net attention (autoregressive for decode, chunked for prefill)
//  11. output = gated_rmsnorm(attn_out, z) using ssm_norm
//  12. final = ssm_out(output)
//
// Key dimensions:
//   ssm_d_inner = 4096, ssm_d_state = 128, ssm_n_group = 16, ssm_dt_rank = 32
//   num_k_heads = 16, num_v_heads = 32
//   head_k_dim = 128, head_v_dim = 128
//   conv_channels = 8192 (= d_inner + 2 * n_group * d_state)
//   State shape: [num_v_heads, head_v_dim, head_v_dim] = [32, 128, 128]
// ============================================================

// Softplus: log(1 + exp(x))
__device__ __forceinline__ float softplus(float x) {
    if (x > 20.0f) return x; // numerical stability
    return logf(1.0f + expf(x));
}

// Compute gate = softplus(alpha + dt_bias) * ssm_a
// alpha: [n_tokens, num_v_heads]
// dt_bias: [num_v_heads]
// ssm_a: [num_v_heads]
// gate_out: [n_tokens, num_v_heads]
__global__ void compute_gate_kernel(
    float* __restrict__ gate_out,
    const float* __restrict__ alpha,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    int n_tokens,
    int num_v_heads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tokens * num_v_heads) return;

    int head = idx % num_v_heads;
    float a = alpha[idx];
    float biased = a + dt_bias[head];
    float sp = softplus(biased);
    gate_out[idx] = sp * ssm_a[head]; // ssm_a is typically negative (for decay)
}

void launch_compute_gate(
    float* gate_out,
    const float* alpha,
    const float* dt_bias,
    const float* ssm_a,
    int n_tokens,
    int num_v_heads,
    cudaStream_t stream
) {
    int n = n_tokens * num_v_heads;
    compute_gate_kernel<<<cdiv(n, 256), 256, 0, stream>>>(
        gate_out, alpha, dt_bias, ssm_a, n_tokens, num_v_heads);
}

// Sigmoid kernel for beta (f32 input)
__global__ void sigmoid_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = input[idx];
    output[idx] = 1.0f / (1.0f + expf(-x));
}

void launch_sigmoid(
    float* output,
    const float* input,
    int n,
    cudaStream_t stream
) {
    sigmoid_kernel<<<cdiv(n, 256), 256, 0, stream>>>(output, input, n);
}

// 1D convolution (causal, no bias)
// Input: conv_state [conv_kernel-1, channels] + new_input [n_tokens, channels]
// → concatenated as [conv_kernel-1 + n_tokens, channels]
// conv_kernel: [conv_kernel_size, channels] (depthwise)
// Output: [n_tokens, channels]
// Then apply SiLU
__global__ void conv1d_silu_kernel(
    float* __restrict__ output,               // [n_tokens, channels] f32 output
    const float* __restrict__ input,          // [n_tokens, channels] new input (f32)
    const float* __restrict__ conv_state,     // [(conv_kernel-1) * channels] previous state
    const float* __restrict__ conv_weight,    // [conv_kernel, channels]
    int n_tokens,
    int channels,
    int conv_kernel_size
) {
    int token = blockIdx.x;
    int ch = blockIdx.y * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    float sum = 0.0f;
    int state_len = conv_kernel_size - 1;

    for (int k = 0; k < conv_kernel_size; k++) {
        int pos = token + k;

        float val;
        if (pos < state_len) {
            val = conv_state[(int64_t)pos * channels + ch];
        } else {
            int input_pos = pos - state_len;
            val = input[(int64_t)input_pos * channels + ch];
        }
        sum += val * conv_weight[(int64_t)ch * conv_kernel_size + k];
    }

    // SiLU activation
    float silu = sum / (1.0f + expf(-sum));
    output[(int64_t)token * channels + ch] = silu;
}

void launch_conv1d_silu(
    float* output,
    const float* input,
    const float* conv_state,
    const float* conv_weight,
    int n_tokens,
    int channels,
    int conv_kernel_size,
    cudaStream_t stream
) {
    dim3 threads(256);
    dim3 blocks(n_tokens, cdiv(channels, 256));
    conv1d_silu_kernel<<<blocks, threads, 0, stream>>>(
        output, input, conv_state, conv_weight, n_tokens, channels, conv_kernel_size);
}

// Update conv state: save last (conv_kernel-1) inputs as new state
// new_state[i] = input at position (n_tokens - conv_kernel + 1 + i)
__global__ void update_conv_state_kernel(
    float* __restrict__ new_state,            // [(conv_kernel-1) * channels]
    const float* __restrict__ input,          // [n_tokens, channels] (f32)
    const float* __restrict__ old_state,      // [(conv_kernel-1) * channels]
    int n_tokens,
    int channels,
    int conv_kernel_size
) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    int state_len = conv_kernel_size - 1;

    for (int i = 0; i < state_len; i++) {
        int src_pos = n_tokens + i;
        float val;
        if (src_pos < state_len) {
            val = old_state[(int64_t)src_pos * channels + ch];
        } else {
            val = input[(int64_t)(src_pos - state_len) * channels + ch];
        }
        new_state[(int64_t)i * channels + ch] = val;
    }
}

void launch_update_conv_state(
    float* new_state,
    const float* input,
    const float* old_state,
    int n_tokens,
    int channels,
    int conv_kernel_size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = cdiv(channels, threads);
    update_conv_state_kernel<<<blocks, threads, 0, stream>>>(
        new_state, input, old_state, n_tokens, channels, conv_kernel_size);
}

// L2 normalization per vector
// input: [n, dim], normalize each row (f32 in/out)
__global__ void l2_norm_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int dim,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const float* x = input + row * dim;
    float* y = output + row * dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += stride) {
        float val = x[i];
        sum_sq += val * val;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float shared[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < (stride / 32)) ? shared[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }

    __shared__ float s_inv_norm;
    if (tid == 0) {
        s_inv_norm = rsqrtf(fmaxf(sum_sq, eps * eps));
    }
    __syncthreads();

    for (int i = tid; i < dim; i += stride) {
        y[i] = x[i] * s_inv_norm;
    }
}

void launch_l2_norm(
    float* output,
    const float* input,
    int n_vectors,
    int dim,
    float eps,
    cudaStream_t stream
) {
    int threads = (dim < 256) ? dim : 256;
    threads = ((threads + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    l2_norm_kernel<<<n_vectors, threads, 0, stream>>>(output, input, dim, eps);
}

// Delta-net autoregressive step (for decode, n_tokens=1)
//
// For each v_head h:
//   g_exp = exp(gate[h])
//   state[h] = state[h] * g_exp           // decay
//   sk = sum(state[h]^T * k[kh], dim=-1)  // [head_v_dim]
//   d = beta[h] * (v[h] - sk)             // delta
//   state[h] += k[kh] outer d             // update state
//   output[h] = sum(state[h]^T * q[kh], dim=-1)  // query
//
// Where kh = h * num_k_heads / num_v_heads (GQA-like mapping for SSM)
//
// state: [num_v_heads, head_v_dim, head_v_dim] = [32, 128, 128]
// q, k: [num_k_heads, head_k_dim] = [16, 128]  (repeated to [32, 128])
// v: [num_v_heads, head_v_dim] = [32, 128]
// gate: [num_v_heads] (scalar per head)
// beta: [num_v_heads] (scalar per head)
__global__ void delta_net_decode_kernel(
    float* __restrict__ output,              // [num_v_heads * head_v_dim] f32 output
    float* __restrict__ state,               // [num_v_heads, head_v_dim, head_v_dim] IN/OUT
    const float* __restrict__ q,             // [num_v_heads, head_k_dim] (already repeated, f32)
    const float* __restrict__ k,             // [num_v_heads, head_k_dim] (f32)
    const float* __restrict__ v,             // [num_v_heads, head_v_dim] (f32)
    const float* __restrict__ gate,          // [num_v_heads]
    const float* __restrict__ beta,          // [num_v_heads]
    int head_dim,                            // head_k_dim = head_v_dim = 128
    float scale                              // 1/sqrt(head_k_dim)
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    float* s = state + (int64_t)head * head_dim * head_dim;
    float g = expf(gate[head]);
    float b = beta[head];

    // Step 1: Decay state
    for (int i = tid; i < head_dim * head_dim; i += stride) {
        s[i] *= g;
    }
    __syncthreads();

    __shared__ float s_k[128];
    __shared__ float s_v[128];
    __shared__ float s_q[128];

    for (int i = tid; i < head_dim; i += stride) {
        s_k[i] = k[head * head_dim + i];
        s_v[i] = v[head * head_dim + i];
        s_q[i] = q[head * head_dim + i] * scale;
    }
    __syncthreads();

    // Compute sk[i] = sum_j s[j*head_dim + i] * k[j]
    // Then d[i] = beta * (v[i] - sk[i])
    // Then update: s[j*head_dim + i] += k[j] * d[i]
    // Then output[i] = sum_j s[j*head_dim + i] * q[j]
    // We process per column i
    for (int i = tid; i < head_dim; i += stride) {
        // sk[i]
        float sk = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            sk += s[j * head_dim + i] * s_k[j];
        }

        float d = b * (s_v[i] - sk);

        // Update state column i
        for (int j = 0; j < head_dim; j++) {
            s[j * head_dim + i] += s_k[j] * d;
        }

        // Query output
        float out = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            out += s[j * head_dim + i] * s_q[j];
        }

        output[head * head_dim + i] = out;
    }
}

void launch_delta_net_decode(
    float* output,
    float* state,
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    int num_v_heads,
    int head_dim,
    float scale,
    cudaStream_t stream
) {
    delta_net_decode_kernel<<<num_v_heads, 128, 0, stream>>>(
        output, state, q, k, v, gate, beta, head_dim, scale);
}

// Gated RMSNorm: output = rmsnorm(input, weight) * silu(gate)
// input: [num_v_heads, head_v_dim]
// weight: [head_v_dim]  (ssm_norm, shared across heads)
// gate: [num_v_heads, head_v_dim]  (z from wqkv_gate)
__global__ void gated_rmsnorm_kernel(
    __nv_bfloat16* __restrict__ output,
    const float* __restrict__ input,          // f32 delta-net output
    const float* __restrict__ weight,
    const float* __restrict__ gate,           // f32 gate (z) — matches ggml precision
    int head_dim,
    float eps
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const float* x = input + head * head_dim;
    const float* z = gate + head * head_dim;
    __nv_bfloat16* y = output + head * head_dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < head_dim; i += stride) {
        float val = x[i];
        sum_sq += val * val;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float shared[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < (stride / 32)) ? shared[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }

    __shared__ float s_rms;
    if (tid == 0) {
        s_rms = rsqrtf(sum_sq / head_dim + eps);
    }
    __syncthreads();

    for (int i = tid; i < head_dim; i += stride) {
        float val = x[i];
        float w = weight[i];
        float normalized = val * s_rms * w;

        float g = z[i];
        float silu_g = g / (1.0f + expf(-g));

        y[i] = __float2bfloat16(normalized * silu_g);
    }
}

void launch_gated_rmsnorm(
    __nv_bfloat16* output,
    const float* input,
    const float* weight,
    const float* gate,
    int num_heads,
    int head_dim,
    float eps,
    cudaStream_t stream
) {
    int threads = (head_dim < 128) ? head_dim : 128;
    threads = ((threads + 31) / 32) * 32;
    gated_rmsnorm_kernel<<<num_heads, threads, 0, stream>>>(
        output, input, weight, gate, head_dim, eps);
}

// Fused SSM step: l2_norm + repeat + delta_net_decode + gated_rmsnorm
// One block per v_head (32 blocks). Does everything for one token.
__global__ void fused_ssm_step_kernel(
    __nv_bfloat16* __restrict__ output,      // [ssm_d_inner] bf16 output
    float* __restrict__ state,               // [num_v_heads, head_dim, head_dim] IN/OUT
    const float* __restrict__ conv_out,      // [conv_channels] (q+k+v concatenated)
    const float* __restrict__ alpha,         // [num_v_heads] raw alpha values
    const float* __restrict__ dt_bias,       // [num_v_heads] dt bias
    const float* __restrict__ ssm_a,         // [num_v_heads] ssm_a values
    const float* __restrict__ beta_raw,      // [num_v_heads] raw beta (pre-sigmoid)
    const float* __restrict__ z,             // [ssm_d_inner] z gate values
    const float* __restrict__ norm_weight,   // [head_dim] ssm_norm weight
    int num_k_heads,                         // 16
    int num_v_heads,                         // 32
    int head_k_dim,                          // 128 (d_state)
    int head_v_dim,                          // 128
    float scale,                             // 1/sqrt(head_k_dim)
    float l2_eps,                            // eps for l2 norm
    float rms_eps                            // eps for rmsnorm
) {
    const int v_head = blockIdx.x;
    const int k_head = v_head % num_k_heads;  // tile mapping
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    int qk_size = head_k_dim * num_k_heads; // 2048

    // Pointers into conv_out for this head's Q, K, V
    const float* q_raw = conv_out + k_head * head_k_dim;
    const float* k_raw = conv_out + qk_size + k_head * head_k_dim;
    const float* v_ptr = conv_out + 2 * qk_size + v_head * head_v_dim;

    // Shared memory for q, k, v, and temporaries
    extern __shared__ float smem[];
    float* s_q = smem;                    // [head_k_dim]
    float* s_k = s_q + head_k_dim;       // [head_k_dim]
    float* s_v = s_k + head_k_dim;       // [head_v_dim]

    // Load Q and K, compute L2 norms
    float q_sq = 0.0f, k_sq = 0.0f;
    for (int i = tid; i < head_k_dim; i += stride) {
        float qv = q_raw[i];
        float kv = k_raw[i];
        s_q[i] = qv;
        s_k[i] = kv;
        q_sq += qv * qv;
        k_sq += kv * kv;
    }

    // Reduce L2 norms
    for (int offset = 16; offset > 0; offset >>= 1) {
        q_sq += __shfl_down_sync(0xffffffff, q_sq, offset);
        k_sq += __shfl_down_sync(0xffffffff, k_sq, offset);
    }
    __shared__ float sq_shared[32], sk_shared[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) { sq_shared[warp_id] = q_sq; sk_shared[warp_id] = k_sq; }
    __syncthreads();
    if (warp_id == 0) {
        q_sq = (lane_id < (stride / 32)) ? sq_shared[lane_id] : 0.0f;
        k_sq = (lane_id < (stride / 32)) ? sk_shared[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            q_sq += __shfl_down_sync(0xffffffff, q_sq, offset);
            k_sq += __shfl_down_sync(0xffffffff, k_sq, offset);
        }
    }
    __shared__ float s_q_inv, s_k_inv;
    if (tid == 0) {
        s_q_inv = rsqrtf(fmaxf(q_sq, l2_eps * l2_eps));
        s_k_inv = rsqrtf(fmaxf(k_sq, l2_eps * l2_eps));
    }
    __syncthreads();

    // Apply L2 norm and scale Q
    for (int i = tid; i < head_k_dim; i += stride) {
        s_q[i] = s_q[i] * s_q_inv * scale;  // pre-multiply scale into Q
        s_k[i] = s_k[i] * s_k_inv;
    }

    // Load V
    for (int i = tid; i < head_v_dim; i += stride) {
        s_v[i] = v_ptr[i];
    }
    __syncthreads();

    // Compute gate and beta inline (saves 2 kernel launches)
    float* s = state + (int64_t)v_head * head_v_dim * head_v_dim;
    float alpha_val = alpha[v_head];
    float biased = alpha_val + dt_bias[v_head];
    // softplus
    float sp = (biased > 20.0f) ? biased : logf(1.0f + expf(biased));
    float gate_val = sp * ssm_a[v_head];
    float g = expf(gate_val);
    float b = 1.0f / (1.0f + expf(-beta_raw[v_head]));

    // Decay state
    for (int i = tid; i < head_v_dim * head_v_dim; i += stride) {
        s[i] *= g;
    }
    __syncthreads();

    // Per-column: sk, delta update, query
    for (int i = tid; i < head_v_dim; i += stride) {
        float sk = 0.0f;
        for (int j = 0; j < head_k_dim; j++) {
            sk += s[j * head_v_dim + i] * s_k[j];
        }
        float d = b * (s_v[i] - sk);
        for (int j = 0; j < head_k_dim; j++) {
            s[j * head_v_dim + i] += s_k[j] * d;
        }
        float out = 0.0f;
        for (int j = 0; j < head_k_dim; j++) {
            out += s[j * head_v_dim + i] * s_q[j];
        }
        s_v[i] = out;  // reuse s_v for delta output
    }
    __syncthreads();

    // Gated RMSNorm: rmsnorm(delta_out) * silu(z)
    float sum_sq = 0.0f;
    for (int i = tid; i < head_v_dim; i += stride) {
        float val = s_v[i];
        sum_sq += val * val;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    if (lane_id == 0) sq_shared[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane_id < (stride / 32)) ? sq_shared[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }
    __shared__ float s_rms;
    if (tid == 0) {
        s_rms = rsqrtf(sum_sq / head_v_dim + rms_eps);
    }
    __syncthreads();

    // Apply normalization, weight, and gate
    const float* z_head = z + v_head * head_v_dim;
    __nv_bfloat16* out_head = output + v_head * head_v_dim;
    for (int i = tid; i < head_v_dim; i += stride) {
        float val = s_v[i] * s_rms * norm_weight[i];
        float zv = z_head[i];
        float silu_z = zv / (1.0f + expf(-zv));
        out_head[i] = __float2bfloat16(val * silu_z);
    }
}

void launch_fused_ssm_step(
    __nv_bfloat16* output,
    float* state,
    const float* conv_out,
    const float* alpha,
    const float* dt_bias,
    const float* ssm_a,
    const float* beta_raw,
    const float* z,
    const float* norm_weight,
    int num_k_heads,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    float scale,
    float l2_eps,
    float rms_eps,
    cudaStream_t stream
) {
    int threads = 128;
    size_t smem = (head_k_dim + head_k_dim + head_v_dim) * sizeof(float);
    fused_ssm_step_kernel<<<num_v_heads, threads, smem, stream>>>(
        output, state, conv_out, alpha, dt_bias, ssm_a, beta_raw, z, norm_weight,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, scale, l2_eps, rms_eps);
}

// Batched fused SSM step with state in shared memory
// Each block handles one v_head, looping over tokens sequentially
// State (128×128 = 64KB) lives in shared memory for fast access
__global__ void __launch_bounds__(128, 1)
fused_ssm_step_batched_kernel(
    __nv_bfloat16* __restrict__ output,      // [n_tokens, ssm_d_inner]
    float* __restrict__ state,               // [num_v_heads, head_dim, head_dim] IN/OUT
    const float* __restrict__ conv_out,      // [n_tokens, conv_channels]
    const float* __restrict__ alpha,         // [n_tokens, num_v_heads]
    const float* __restrict__ dt_bias,       // [num_v_heads]
    const float* __restrict__ ssm_a,         // [num_v_heads]
    const float* __restrict__ beta_raw,      // [n_tokens, num_v_heads]
    const float* __restrict__ z,             // [n_tokens, ssm_d_inner]
    const float* __restrict__ norm_weight,   // [head_dim]
    int n_tokens,
    int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim,
    int conv_channels, int d_inner,
    float scale, float l2_eps, float rms_eps
) {
    const int v_head = blockIdx.x;
    const int k_head = v_head % num_k_heads;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int qk_size = head_k_dim * num_k_heads;

    // Shared memory layout:
    // s_state: [head_k_dim * head_v_dim] = 128*128 = 16384 floats = 64KB
    // s_q: [head_k_dim] = 128 floats
    // s_k: [head_k_dim] = 128 floats
    // s_v: [head_v_dim] = 128 floats
    extern __shared__ float smem[];
    float* s_state = smem;                                     // 64KB
    float* s_q = s_state + head_k_dim * head_v_dim;           // 512B
    float* s_k = s_q + head_k_dim;                            // 512B
    float* s_v = s_k + head_k_dim;                            // 512B

    __shared__ float sq_shared[32], sk_shared[32];
    __shared__ float s_q_inv, s_k_inv, s_rms;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Load state from global to shared memory
    float* g_state = state + (int64_t)v_head * head_v_dim * head_v_dim;
    for (int i = tid; i < head_k_dim * head_v_dim; i += stride) {
        s_state[i] = g_state[i];
    }
    __syncthreads();

    for (int t = 0; t < n_tokens; t++) {
        const float* conv_t = conv_out + (int64_t)t * conv_channels;
        const float* q_raw = conv_t + k_head * head_k_dim;
        const float* k_raw = conv_t + qk_size + k_head * head_k_dim;
        const float* v_ptr = conv_t + 2 * qk_size + v_head * head_v_dim;

        // Load Q/K, compute L2 norms
        float q_sq = 0.0f, k_sq = 0.0f;
        for (int i = tid; i < head_k_dim; i += stride) {
            float qv = q_raw[i];
            float kv = k_raw[i];
            s_q[i] = qv;
            s_k[i] = kv;
            q_sq += qv * qv;
            k_sq += kv * kv;
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            q_sq += __shfl_down_sync(0xffffffff, q_sq, offset);
            k_sq += __shfl_down_sync(0xffffffff, k_sq, offset);
        }
        if (lane_id == 0) { sq_shared[warp_id] = q_sq; sk_shared[warp_id] = k_sq; }
        __syncthreads();
        if (warp_id == 0) {
            q_sq = (lane_id < (stride / 32)) ? sq_shared[lane_id] : 0.0f;
            k_sq = (lane_id < (stride / 32)) ? sk_shared[lane_id] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                q_sq += __shfl_down_sync(0xffffffff, q_sq, offset);
                k_sq += __shfl_down_sync(0xffffffff, k_sq, offset);
            }
        }
        if (tid == 0) {
            s_q_inv = rsqrtf(fmaxf(q_sq, l2_eps * l2_eps));
            s_k_inv = rsqrtf(fmaxf(k_sq, l2_eps * l2_eps));
        }
        __syncthreads();

        for (int i = tid; i < head_k_dim; i += stride) {
            s_q[i] = s_q[i] * s_q_inv * scale;
            s_k[i] = s_k[i] * s_k_inv;
        }
        for (int i = tid; i < head_v_dim; i += stride) {
            s_v[i] = v_ptr[i];
        }
        __syncthreads();

        // Gate and beta
        float alpha_val = alpha[t * num_v_heads + v_head];
        float biased = alpha_val + dt_bias[v_head];
        float sp = (biased > 20.0f) ? biased : logf(1.0f + expf(biased));
        float gate_val = sp * ssm_a[v_head];
        float g = expf(gate_val);
        float b = 1.0f / (1.0f + expf(-beta_raw[t * num_v_heads + v_head]));

        // Decay state (in shared memory!)
        for (int i = tid; i < head_v_dim * head_v_dim; i += stride) {
            s_state[i] *= g;
        }
        __syncthreads();

        // Delta update + query (all in shared memory)
        for (int i = tid; i < head_v_dim; i += stride) {
            float sk = 0.0f;
            for (int j = 0; j < head_k_dim; j++) {
                sk += s_state[j * head_v_dim + i] * s_k[j];
            }
            float d = b * (s_v[i] - sk);
            for (int j = 0; j < head_k_dim; j++) {
                s_state[j * head_v_dim + i] += s_k[j] * d;
            }
            float out = 0.0f;
            for (int j = 0; j < head_k_dim; j++) {
                out += s_state[j * head_v_dim + i] * s_q[j];
            }
            s_v[i] = out;
        }
        __syncthreads();

        // Gated RMSNorm
        float sum_sq = 0.0f;
        for (int i = tid; i < head_v_dim; i += stride) {
            sum_sq += s_v[i] * s_v[i];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
        if (lane_id == 0) sq_shared[warp_id] = sum_sq;
        __syncthreads();
        if (warp_id == 0) {
            sum_sq = (lane_id < (stride / 32)) ? sq_shared[lane_id] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
            }
        }
        if (tid == 0) {
            s_rms = rsqrtf(sum_sq / head_v_dim + rms_eps);
        }
        __syncthreads();

        const float* z_head = z + (int64_t)t * d_inner + v_head * head_v_dim;
        __nv_bfloat16* out_head = output + (int64_t)t * d_inner + v_head * head_v_dim;
        for (int i = tid; i < head_v_dim; i += stride) {
            float val = s_v[i] * s_rms * norm_weight[i];
            float zv = z_head[i];
            float silu_z = zv / (1.0f + expf(-zv));
            out_head[i] = __float2bfloat16(val * silu_z);
        }
        __syncthreads();
    }

    // Write state back to global memory
    for (int i = tid; i < head_k_dim * head_v_dim; i += stride) {
        g_state[i] = s_state[i];
    }
}

void launch_fused_ssm_step_batched(
    __nv_bfloat16* output,
    float* state,
    const float* conv_out,
    const float* alpha,
    const float* dt_bias,
    const float* ssm_a,
    const float* beta_raw,
    const float* z,
    const float* norm_weight,
    int n_tokens,
    int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim,
    int conv_channels, int d_inner,
    float scale, float l2_eps, float rms_eps,
    cudaStream_t stream
) {
    int threads = 128;
    // State (128*128) + q (128) + k (128) + v (128) = 16768 floats = 67KB
    size_t smem = (head_k_dim * head_v_dim + head_k_dim + head_k_dim + head_v_dim) * sizeof(float);

    // Request extended shared memory (>48KB requires explicit opt-in)
    static bool smem_configured = false;
    if (!smem_configured) {
        cudaFuncSetAttribute(fused_ssm_step_batched_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        smem_configured = true;
    }

    fused_ssm_step_batched_kernel<<<num_v_heads, threads, smem, stream>>>(
        output, state, conv_out, alpha, dt_bias, ssm_a, beta_raw, z, norm_weight,
        n_tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_channels, d_inner, scale, l2_eps, rms_eps);
}

// Repeat Q/K heads: duplicate from num_k_heads to num_v_heads
// input: [num_k_heads, head_dim], output: [num_v_heads, head_dim]
// repeat_factor = num_v_heads / num_k_heads
__global__ void repeat_heads_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int num_k_heads,
    int num_v_heads,
    int head_dim
) {
    int v_head = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    int k_head = v_head % num_k_heads;

    for (int i = tid; i < head_dim; i += stride) {
        output[v_head * head_dim + i] = input[k_head * head_dim + i];
    }
}

void launch_repeat_heads(
    float* output,
    const float* input,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    cudaStream_t stream
) {
    int threads = (head_dim < 128) ? head_dim : 128;
    threads = ((threads + 31) / 32) * 32;
    repeat_heads_kernel<<<num_v_heads, threads, 0, stream>>>(
        output, input, num_k_heads, num_v_heads, head_dim);
}
