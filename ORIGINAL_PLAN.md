# CUDA Inference Rewrite Plan: Qwen3.5-9B (BF16)

## Target Environment
- **GPU**: RTX 5090 (Blackwell, SM 12.0), 32 GB VRAM
- **CUDA**: 12.8 with cuBLAS, cuDNN 9.8
- **Model**: unsloth/Qwen3.5-9B-GGUF:BF16 (~16.68 GiB, 427 tensors)

## Qwen3.5-9B Architecture (Hybrid Mamba-Attention)

| Parameter | Value |
|---|---|
| Layers | 32 (every 4th = full attention, rest = Mamba SSM) |
| Hidden dim | 4096 |
| FFN dim | 12288 |
| Attention heads | 16 (4 KV heads, GQA) |
| Head dim | 256 |
| SSM state size | 128, 16 groups, conv kernel 4 |
| SSM inner size | 4096 |
| SSM time step rank | 32 |
| RoPE freq base | 10,000,000 |
| RoPE dim count | 64 |
| RoPE dimension sections | [11, 11, 10, 0] |
| Vocab | 248,320 |
| Context length | 262,144 |
| Weights format | BF16, GGUF V3 |

## File Structure

```
llama-rewrite/
├── src/
│   ├── main.cu            # Entry point, inference loop
│   ├── model.h            # Model struct, layer configs
│   ├── gguf_loader.h      # GGUF file parser header
│   ├── gguf_loader.cpp    # GGUF file parser + weight loading to GPU
│   ├── tokenizer.h        # BPE tokenizer header
│   ├── tokenizer.cpp      # BPE tokenizer implementation
│   ├── sampling.h         # Sampling header
│   ├── sampling.cu        # Top-k/p sampling
│   ├── kernels/
│   │   ├── rmsnorm.cu     # RMSNorm kernel
│   │   ├── rope.cu        # Rotary position embeddings
│   │   ├── attention.cu   # GQA attention (later: FlashAttention)
│   │   ├── mamba.cu       # Mamba2 SSM: conv1d + selective scan
│   │   ├── ffn.cu         # SwiGLU FFN
│   │   └── embedding.cu   # Token embedding + LM head
│   └── utils.h            # CUDA error checking, helpers
├── Makefile
├── CLAUDE.md
└── ORIGINAL_PLAN.md
```

## Phase 0: Scaffolding & Baseline

### 0.1 GGUF Loader
- Parse GGUF V3 header, metadata key-value pairs, and tensor info
- Map tensor names to model components (attention, SSM, FFN per layer)
- Load BF16 tensors directly to GPU via cudaMemcpy

### 0.2 Model Struct
- Define layer types: AttentionLayer vs MambaLayer
- Allocate all weight pointers and intermediate buffers
- Handle the hybrid layout (attention at layers 0,4,8,...,28; SSM elsewhere)

### 0.3 Tokenizer
- Implement BPE tokenizer compatible with Qwen3.5's vocabulary (GPT-2 style BPE, 248K tokens)
- Support encode (text -> tokens) and decode (tokens -> text)

### 0.4 Baseline Benchmark
- Measure llama.cpp prompt eval and generation tok/s for comparison

## Phase 1: Core Ops (Correct Forward Pass)

### 1.1 Embedding Lookup
- Kernel to index into BF16 embedding table (248320 x 4096)
- Output: BF16 hidden states

### 1.2 RMSNorm
- Custom kernel: y = x * rsqrt(mean(x^2) + eps) * weight
- eps = 1e-6
- Fuse with residual connection later

### 1.3 Linear Projections (BF16 GEMM)
- Use cuBLAS cublasGemmEx with CUBLAS_COMPUTE_32F
- All Q/K/V/O projections for attention layers
- All in_proj/out_proj/etc for Mamba layers
- All gate/up/down projections for FFN

### 1.4 RoPE (Rotary Position Embeddings)
- Custom kernel with freq_base = 10,000,000
- Handle dimension_sections = [11, 11, 10, 0] (non-standard split)
- dim_count = 64, head_dim = 256

### 1.5 GQA Attention (layers 0, 4, 8, 12, 16, 20, 24, 28)
- Q: 16 heads, K/V: 4 heads (4:1 GQA ratio)
- head_dim = 256
- Causal mask
- KV cache for autoregressive generation
- Start with naive softmax attention, upgrade later

### 1.6 Mamba2 SSM (layers 1,2,3, 5,6,7, ..., 29,30,31)
- 1D causal convolution (kernel=4)
- Selective scan: state_size=128, groups=16, time_step_rank=32
- inner_size = 4096
- This is the most complex kernel

### 1.7 SwiGLU FFN
- gate = gate_proj(x)
- up = up_proj(x)
- output = down_proj(SiLU(gate) * up)
- Dimensions: 4096 -> 12288 -> 4096

### 1.8 LM Head
- Final RMSNorm + linear projection to vocab (4096 -> 248320)

### 1.9 End-to-End Validation
- Compare output logits against llama.cpp for the same prompt
- Ensure generation matches token-by-token

## Phase 2: Make It Fast

### 2.1 FlashAttention
- Implement or integrate FlashAttention-2/3 for the attention layers
- Alternative: use cuDNN's fused scaled dot-product attention
- Especially important for long sequences (ctx up to 262K)

### 2.2 Fused Kernels
- RMSNorm + residual addition in one kernel
- SwiGLU: fuse SiLU activation with element-wise multiply
- Fuse embedding lookup with position encoding where possible

### 2.3 Mamba2 Kernel Optimization
- Reference: official Mamba CUDA kernels
- Optimize selective scan for SM 12.0
- Shared memory tiling for the recurrence

### 2.4 KV Cache Optimization
- Contiguous or paged allocation
- Optimize for single-token decode (the common case)

### 2.5 CUDA Graphs
- Capture the single-token decode step as a CUDA graph
- Eliminates kernel launch overhead (~5-10us per launch)

### 2.6 Memory Pools
- Use cudaMallocAsync / CUDA memory pools to avoid allocation overhead
- Pre-allocate all intermediate buffers at init time

## Phase 3: Profile & Specialize for RTX 5090

### 3.1 Profiling
- Use Nsight Compute (ncu) to profile individual kernels
- Use Nsight Systems (nsys) for end-to-end timeline
- Identify top hotspots (expected: GEMM, Mamba scan, attention)

### 3.2 RTX 5090 / Blackwell Tuning
- Native BF16 throughput optimization
- Explore native FP4 support (BLACKWELL_NATIVE_FP4 = 1) for weight quantization
- Tune thread block sizes, shared memory, occupancy
- Leverage 12.0 compute capability features

### 3.3 Batch-1 Decode Optimization
- During generation, all linear ops become GEMV (matrix-vector)
- Use specialized GEMV kernels instead of cuBLAS GEMM for decode
- Optimize memory bandwidth utilization (decode is memory-bound)

### 3.4 Overlap & Pipelining
- Use multiple CUDA streams to overlap compute with memory ops
- Pipeline prefill across layers where possible

## Implementation Order

1. `utils.h` - CUDA helpers, error checking macros
2. `model.h` - Model config and weight structs
3. `gguf_loader.cpp` - Load model to GPU
4. `kernels/embedding.cu` - Embedding lookup
5. `kernels/rmsnorm.cu` - RMSNorm
6. `kernels/rope.cu` - RoPE
7. `kernels/attention.cu` - GQA attention + KV cache
8. `kernels/mamba.cu` - Mamba2 SSM
9. `kernels/ffn.cu` - SwiGLU FFN
10. `sampling.cu` - Top-k/p sampling
11. `tokenizer.cpp` - BPE tokenizer
12. `main.cu` - Wire everything together
13. Validate against llama.cpp
14. Profile and optimize (Phase 2 & 3)
