#include "../utils.h"
#include <cuda_bf16.h>
#include <cstdint>

// Embedding lookup: output[i] = embedding_table[token_ids[i]]
// Each block handles one token
__global__ void embedding_lookup_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ embed_table,
    const int* __restrict__ token_ids,
    int dim
) {
    const int token_idx = blockIdx.x;
    const int token_id = token_ids[token_idx];
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const __nv_bfloat16* src = embed_table + (int64_t)token_id * dim;
    __nv_bfloat16* dst = output + (int64_t)token_idx * dim;

    // Copy embedding vector
    for (int i = tid; i < dim; i += stride) {
        dst[i] = src[i];
    }
}

void launch_embedding_lookup(
    __nv_bfloat16* output,
    const __nv_bfloat16* embed_table,
    const int* token_ids,   // device pointer
    int n_tokens,
    int dim,
    cudaStream_t stream
) {
    int threads = (dim < 1024) ? dim : 1024;
    threads = ((threads + 31) / 32) * 32;
    embedding_lookup_kernel<<<n_tokens, threads, 0, stream>>>(
        output, embed_table, token_ids, dim);
}
