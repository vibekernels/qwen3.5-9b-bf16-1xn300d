#include "../utils.h"
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

// Multi-section RoPE (MRoPE) for Qwen3.5
//
// rope_sections = [11, 11, 10, 0], total = 32 pairs = 64 dims
// For each pair (i, i+32) in the head dimension, apply rotation with frequency:
//   freq = 1 / (base ^ (2*section_local_idx / rope_dim))
//
// The sections divide the 32 pairs into groups that each get independent position indices.
// For text-only inference (no vision), all sections use the same position, so this
// simplifies to standard RoPE with the 64-dim subset.
//
// Only the first rope_dim=64 dimensions of each head get rotated.
// The remaining head_dim - rope_dim = 192 dimensions pass through unchanged.

__global__ void rope_kernel(
    __nv_bfloat16* __restrict__ qk,   // [n_tokens, n_heads, head_dim] (contiguous)
    const int* __restrict__ positions, // [n_tokens]
    int n_tokens,
    int n_heads,
    int head_dim,
    int rope_dim,       // 64
    float freq_base     // 10,000,000
) {
    // Each thread handles one (token, head, pair) combination
    const int pair_idx = threadIdx.x;        // which rotation pair [0, rope_dim/2)
    const int head = blockIdx.y;
    const int token = blockIdx.x;

    if (pair_idx >= rope_dim / 2) return;

    int pos = positions[token];

    // Compute frequency for this pair
    // theta = pos / (base ^ (2 * pair_idx / rope_dim))
    float freq = 1.0f / powf(freq_base, (2.0f * pair_idx) / rope_dim);
    float theta = pos * freq;
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    // Get pointers to the two elements being rotated
    int offset = (int64_t)token * n_heads * head_dim + head * head_dim;
    int i0 = offset + pair_idx;
    int i1 = offset + pair_idx + rope_dim / 2;

    float x0 = __bfloat162float(qk[i0]);
    float x1 = __bfloat162float(qk[i1]);

    // Apply rotation
    qk[i0] = __float2bfloat16(x0 * cos_theta - x1 * sin_theta);
    qk[i1] = __float2bfloat16(x1 * cos_theta + x0 * sin_theta);
}

void launch_rope(
    __nv_bfloat16* qk,
    const int* positions,
    int n_tokens,
    int n_heads,
    int head_dim,
    int rope_dim,
    float freq_base,
    cudaStream_t stream
) {
    // rope_dim/2 = 32 threads per block
    dim3 threads(rope_dim / 2);
    dim3 blocks(n_tokens, n_heads);
    rope_kernel<<<blocks, threads, 0, stream>>>(
        qk, positions, n_tokens, n_heads, head_dim, rope_dim, freq_base);
}
