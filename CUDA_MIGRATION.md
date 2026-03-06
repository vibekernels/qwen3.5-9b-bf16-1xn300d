# CUDA Migration Progress: Qwen3.5-9B Inference

## Status: Phase 1 Complete (Correct Forward Pass)

The custom CUDA inference engine produces correct output for Qwen3.5-9B (BF16). The model generates coherent, accurate text matching the llama.cpp reference implementation.

**Example**: Prompt "The capital of France is" → generates "Paris" (matching llama.cpp output).

## Performance (RTX 5090, BF16, batch-1)

| Metric | Value |
|---|---|
| Prompt eval | ~25 tok/s |
| Generation | ~69 tok/s |

These are unoptimized Phase 1 numbers. Phase 2 optimizations (FlashAttention, fused kernels, CUDA graphs) have not been applied yet.

## Architecture

Qwen3.5-9B is a **hybrid Mamba-Attention** model (delta-net linear attention, not Mamba2):

- 32 layers total
- SSM (delta-net) layers: positions where `(i+1) % 4 != 0` (layers 0-2, 4-6, 8-10, ...)
- Full attention layers: positions 3, 7, 11, 15, 19, 23, 27, 31
- Hidden dim: 4096, FFN dim: 12288, vocab: 248,320
- Attention: 16 Q heads, 4 KV heads (GQA 4:1), head_dim=256
- SSM: d_state=128, n_group=16, dt_rank=32, conv_kernel=4

## Codebase (~2,765 lines)

```
src/
  main.cu             (756)  - Entry point, inference loop, forward pass
  model.h             (161)  - Model config, weight structs, buffer allocation
  gguf_loader.h/cpp   (320)  - GGUF V3 parser, BF16 weight loading to GPU
  tokenizer.h/cpp     (312)  - BPE tokenizer (encode/decode)
  sampling.h/cu       (111)  - Top-k/p sampling with temperature
  utils.h              (47)  - CUDA error checking macros
  kernels/
    mamba.cu           (466)  - Delta-net SSM: conv1d, selective scan, gated RMSNorm
    rmsnorm.cu         (222)  - RMSNorm (bf16, f32-input, per-head variants)
    attention.cu       (221)  - GQA attention with KV cache
    rope.cu             (72)  - RoPE with dimension_sections
    embedding.cu        (39)  - Token embedding lookup
    ffn.cu              (38)  - SwiGLU FFN
```

All linear projections use cuBLAS `cublasGemmEx` with `CUBLAS_COMPUTE_32F` for BF16 GEMM.

## Bugs Found and Fixed

### 1. repeat_heads mapping (CRITICAL - wrong output)
**File**: `src/kernels/mamba.cu:447`

The delta-net SSM has 16 groups but 32 value heads. Q and K must be repeated from 16 → 32 heads to match. The mapping was wrong:

- **Bug**: `k_head = v_head * num_k_heads / num_v_heads` — interleaved mapping `[g0,g0,g1,g1,...,g15,g15]`
- **Fix**: `k_head = v_head % num_k_heads` — tiled mapping `[g0,g1,...,g15,g0,g1,...,g15]` matching ggml's `ggml_repeat_4d` semantics

This was the root cause of garbled output. Only head 0 happened to map correctly under both schemes, which is why per-head analysis showed cos_sim=1.0 for head 0 but wrong magnitudes for others.

### 2. Z gate BF16 precision loss
**File**: `src/kernels/mamba.cu`

The z gate values in the gated RMSNorm were being truncated to BF16 before the SiLU activation, losing precision on small values. Fixed by keeping z in f32 through the gate computation.

### 3. L2 norm formula
**File**: `src/kernels/mamba.cu`

The L2 normalization for Q and K vectors used an incorrect formula. Fixed to match the reference implementation's `l2_norm` behavior.

## Debugging Methodology

Systematic per-layer comparison between CUDA and ggml reference:

1. **Built custom ggml dump tool** (`dump_ggml_hidden.cpp`) — hooks into llama.cpp's eval callback to dump intermediate tensors (linear_attn_out, ffn_out, attn_output, z, conv outputs) per token position
2. **Added CUDA dump infrastructure** — environment variables (`DUMP_SSM_INTERMEDIATES`, `DUMP_ATTN_INTERMEDIATES`, `DUMP_ALL_LAYERS`) trigger binary dumps of hidden states
3. **Python comparison script** (`compare_layers.py`) — reconstructs cumulative hidden states from incremental outputs and computes max_diff/cosine_similarity per layer
4. **Narrowed divergence** — embedding matched exactly (max_diff=0.0), SSM layer 0 output had cos_sim=0.993 but wrong magnitude, traced to Q/K head mapping

Key insight: cos_sim=1.0 per head with wrong magnitudes meant V (determining direction) was correct but Q*K dot products (determining magnitude) were wrong — pointed to head mapping rather than computation bugs.

## What's Next (Phase 2)

- FlashAttention or cuDNN fused attention for the 8 attention layers
- Fused kernels: RMSNorm+residual, SwiGLU activation
- CUDA graphs for single-token decode
- Memory pool pre-allocation
- Specialized GEMV kernels for batch-1 decode
- Profiling with Nsight Compute/Systems
