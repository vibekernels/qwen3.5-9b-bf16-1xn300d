#include "sampling.h"
#include "utils.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

__global__ void bf16_to_f32_kernel(float* __restrict__ output, const __nv_bfloat16* __restrict__ input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __bfloat162float(input[idx]);
    }
}

void launch_bf16_to_f32(float* output, const __nv_bfloat16* input, int n, cudaStream_t stream) {
    bf16_to_f32_kernel<<<cdiv(n, 256), 256, 0, stream>>>(output, input, n);
}

// Simple CPU-side sampling (good enough for single-token generation)
int sample_token(float* logits_device, int vocab_size, float temperature, int top_k, float top_p) {
    // Download logits to host
    std::vector<float> logits(vocab_size);
    cuda_download(logits.data(), logits_device, vocab_size);

    // Greedy (argmax) for temperature <= 0
    if (temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        fprintf(stderr, "[SAMPLER] greedy: %d (%.4f), logits[528]=%.4f\n", best, logits[best], logits[528]);
        return best;
    }

    // Apply temperature
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= inv_temp;
    }

    // Find top-k
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);

    if (top_k > 0 && top_k < vocab_size) {
        std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
            [&](int a, int b) { return logits[a] > logits[b]; });
        indices.resize(top_k);
    }

    // Softmax over selected tokens
    float max_logit = logits[indices[0]];
    for (int idx : indices) max_logit = std::max(max_logit, logits[idx]);

    std::vector<float> probs(indices.size());
    float sum = 0.0f;
    for (size_t i = 0; i < indices.size(); i++) {
        probs[i] = expf(logits[indices[i]] - max_logit);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;

    // Top-p (nucleus) filtering
    if (top_p < 1.0f) {
        // Sort by probability descending
        std::vector<size_t> sorted_idx(probs.size());
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::sort(sorted_idx.begin(), sorted_idx.end(),
            [&](size_t a, size_t b) { return probs[a] > probs[b]; });

        float cumsum = 0.0f;
        size_t cutoff = sorted_idx.size();
        for (size_t i = 0; i < sorted_idx.size(); i++) {
            cumsum += probs[sorted_idx[i]];
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }

        // Zero out tokens beyond cutoff
        for (size_t i = cutoff; i < sorted_idx.size(); i++) {
            probs[sorted_idx[i]] = 0.0f;
        }

        // Renormalize
        sum = 0.0f;
        for (auto p : probs) sum += p;
        for (auto& p : probs) p /= sum;
    }

    // Sample
    static std::mt19937 rng(42);
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int sampled_idx = dist(rng);

    return indices[sampled_idx];
}
