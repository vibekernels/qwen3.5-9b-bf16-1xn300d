# CUDA Migration Progress: Qwen3.5-9B Inference

## Status: Phase 2 Complete (Optimized — Faster than llama.cpp)

Custom CUDA inference engine for Qwen3.5-9B (BF16) that **beats llama.cpp** on both prompt eval and generation.

## Performance (RTX 5090, BF16)

| Metric | Our Implementation | llama.cpp | Speedup |
|--------|-------------------|-----------|---------|
| Prompt eval (111 tok) | **1480 tok/s** | 651 tok/s | **2.27×** |
| Prompt eval (211 tok) | **1099 tok/s** | 1041 tok/s | **1.06×** |
| Generation | **83.5 tok/s** | 75.3 tok/s | **1.11×** |

## Phase 2 Optimizations Applied

### Weight Packing (Reduced GEMM launches)
- **Attention K+V packed**: Single GEMM for K+V during decode (saves 1 cuBLAS call per layer)
- **FFN gate+up packed**: Single GEMM for gate+up with `swiglu_packed` kernel (saves 1 per layer)
- **SSM combined projection**: Packed QKV+Z+alpha+beta into single GEMM with fused bf16 deinterleave (4→1 GEMM for batched SSM)

### Fused Kernels (Reduced kernel launches)
- **Fused residual + RMSNorm**: Combines residual add + norm in one kernel (2 launches → 1)
- **Fused bf16 residual + RMSNorm**: Cast bf16→f32 + residual + norm in one kernel (3 launches → 1)
- **Fused bf16 residual add**: Cast + add for FFN down output (2 launches → 1)
- **Fused SSM step**: Combines gate compute, beta sigmoid, L2 norm, repeat heads, delta-net decode, gated RMSNorm (7 ops → 1 kernel)
- **SwiGLU packed**: Works directly on interleaved gate+up GEMM output

### SSM Recurrence Optimization
- **Batched SSM kernel**: All prompt tokens processed in a single kernel launch (n_tokens launches → 1)
- **Shared memory state**: 64KB state matrix (128×128) kept in shared memory during batched recurrence (4.8× faster — 93ms→19ms for 24 SSM layers on 111 tokens)

### Adaptive GEMM Strategy
- **M≤4**: Direct bf16→f32 output (decode path, saves cast)
- **M>4**: bf16→bf16 GEMM + fused bf16→f32 cast in downstream kernel (faster cuBLAS kernels)
- **cuBLAS warm-up**: Pre-warm cuBLAS workspace before timing

### Other
- GPU argmax sampling (avoids downloading 248K float logits)
- Pre-allocated decode buffers (avoids per-token cudaMalloc)
- Batched prompt processing through all layers

## Architecture

Qwen3.5-9B is a **hybrid Mamba-Attention** model (delta-net linear attention):

- 32 layers: 24 SSM (delta-net) + 8 full attention (at positions 3,7,11,...,31)
- Hidden dim: 4096, FFN dim: 12288, vocab: 248,320
- Attention: 16 Q heads, 4 KV heads (GQA 4:1), head_dim=256
- SSM: d_state=128, n_group=16, dt_rank=32, conv_kernel=4

## Build & Run

```sh
make -j
./qwen-inference /path/to/qwen3.5-9b-bf16.gguf -p "Your prompt here" -n 128 -t 0
```
