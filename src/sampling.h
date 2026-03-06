#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Convert bf16 logits to f32 for sampling
void launch_bf16_to_f32(float* output, const __nv_bfloat16* input, int n, cudaStream_t stream = 0);

// Sample a token from logits using temperature + top-k + top-p
// Returns the sampled token ID
int sample_token(float* logits_device, int vocab_size, float temperature = 0.8f,
                 int top_k = 40, float top_p = 0.95f);
