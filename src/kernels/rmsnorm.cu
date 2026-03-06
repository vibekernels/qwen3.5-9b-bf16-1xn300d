#include "../utils.h"
#include <cuda_bf16.h>
#include <cstdint>

// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
// Each block handles one row (one token's hidden state)

__global__ void rmsnorm_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ weight,
    int dim,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const __nv_bfloat16* x = input + row * dim;
    __nv_bfloat16* y = output + row * dim;

    // Compute sum of squares using f32 accumulation
    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += stride) {
        float val = __bfloat162float(x[i]);
        sum_sq += val * val;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Block-level reduction using shared memory
    __shared__ float shared[32]; // one per warp
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

    __shared__ float s_rms_scale;
    if (tid == 0) {
        s_rms_scale = rsqrtf(sum_sq / dim + eps);
    }
    __syncthreads();

    float rms_scale = s_rms_scale;

    // Apply normalization and weight (f32 weight for full precision)
    for (int i = tid; i < dim; i += stride) {
        float val = __bfloat162float(x[i]);
        float w = weight[i];
        y[i] = __float2bfloat16(val * rms_scale * w);
    }
}

void launch_rmsnorm(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const float* weight,
    int n_tokens,
    int dim,
    float eps,
    cudaStream_t stream
) {
    int threads = (dim < 1024) ? dim : 1024;
    // Round up to multiple of 32 (warp size)
    threads = ((threads + 31) / 32) * 32;
    rmsnorm_kernel<<<n_tokens, threads, 0, stream>>>(output, input, weight, dim, eps);
}

// RMSNorm with f32 input, bf16 output: y(bf16) = x(f32) * rsqrt(mean(x^2) + eps) * weight(bf16)
// Reads f32 hidden state directly (no bf16 truncation before normalization)

__global__ void rmsnorm_f32in_kernel(
    __nv_bfloat16* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int dim,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const float* x = input + row * dim;
    __nv_bfloat16* y = output + row * dim;

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

    __shared__ float s_rms_scale;
    if (tid == 0) {
        s_rms_scale = rsqrtf(sum_sq / dim + eps);
    }
    __syncthreads();

    float rms_scale = s_rms_scale;

    for (int i = tid; i < dim; i += stride) {
        float val = x[i];
        float w = weight[i];
        y[i] = __float2bfloat16(val * rms_scale * w);
    }
}

void launch_rmsnorm_f32in(
    __nv_bfloat16* output,
    const float* input,
    const float* weight,
    int n_tokens,
    int dim,
    float eps,
    cudaStream_t stream
) {
    int threads = (dim < 1024) ? dim : 1024;
    threads = ((threads + 31) / 32) * 32;
    rmsnorm_f32in_kernel<<<n_tokens, threads, 0, stream>>>(output, input, weight, dim, eps);
}

// RMSNorm applied per-head (for Q/K normalization)
// Input shape: [n_tokens, n_heads, head_dim]
// Weight shape: [head_dim]
__global__ void rmsnorm_head_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ weight,
    int head_dim,
    int n_heads,
    float eps
) {
    // Each block handles one (token, head) pair
    const int idx = blockIdx.x;  // flattened (token * n_heads + head)
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const __nv_bfloat16* x = input + idx * head_dim;
    __nv_bfloat16* y = output + idx * head_dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < head_dim; i += stride) {
        float val = __bfloat162float(x[i]);
        sum_sq += val * val;
    }

    // Warp reduction
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

    __shared__ float s_rms_scale;
    if (tid == 0) {
        s_rms_scale = rsqrtf(sum_sq / head_dim + eps);
    }
    __syncthreads();

    float rms_scale = s_rms_scale;

    for (int i = tid; i < head_dim; i += stride) {
        float val = __bfloat162float(x[i]);
        float w = weight[i];
        y[i] = __float2bfloat16(val * rms_scale * w);
    }
}

void launch_rmsnorm_head(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const float* weight,
    int n_tokens,
    int n_heads,
    int head_dim,
    float eps,
    cudaStream_t stream
) {
    int threads = (head_dim < 256) ? head_dim : 256;
    threads = ((threads + 31) / 32) * 32;
    rmsnorm_head_kernel<<<n_tokens * n_heads, threads, 0, stream>>>(
        output, input, weight, head_dim, n_heads, eps);
}
