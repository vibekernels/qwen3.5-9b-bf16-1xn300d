#!/usr/bin/env python3
"""Compare ggml vs CUDA per-layer hidden states to find divergence point."""
import numpy as np
import os

# ggml layer mapping:
# SSM layers: linear_attn_out-N (N=0..31 but only for SSM positions)
# Attention layers: linear_attn_out-N for attention positions
# FFN output: ffn_out-N for all layers
# Our CUDA dumps: hidden_f32_posM_layerN.bin = cumulative hidden state after layer N

n_embd = 4096
pos = 4  # last token position

# First, let's see what ggml files we have
ggml_files = sorted([f for f in os.listdir('/tmp') if f.startswith('ggml_') and f.endswith(f'_pos{pos}.bin')])
print(f"ggml files for pos {pos}: {len(ggml_files)}")
for f in ggml_files:
    data = np.fromfile(f'/tmp/{f}', dtype=np.float32)
    print(f"  {f}: shape={data.shape}, sum={data.sum():.4f}, L2={np.linalg.norm(data):.4f}")

print()

# Load CUDA hidden state dumps
cuda_files = sorted([f for f in os.listdir('/tmp') if f.startswith(f'hidden_f32_pos{pos}_layer') and f.endswith('.bin')])
print(f"CUDA hidden state files for pos {pos}: {len(cuda_files)}")
for f in cuda_files:
    data = np.fromfile(f'/tmp/{f}', dtype=np.float32)
    print(f"  {f}: shape={data.shape}, sum={data.sum():.4f}, L2={np.linalg.norm(data):.4f}")

print("\n" + "="*80)
print("Reconstructing ggml cumulative hidden states and comparing with CUDA")
print("="*80)

# ggml's tensors are incremental outputs:
# - linear_attn_out-N = output of SSM/attention block for layer N (before residual add)
# - ffn_out-N = output of FFN block for layer N (before residual add)
# Cumulative: hidden[N] = hidden[N-1] + linear_attn_out-N + ffn_out-N

# We need the embedding to start
# Check if we have it
emb_file = f'/tmp/ggml_emb_pos{pos}.bin'
if not os.path.exists(emb_file):
    print("No embedding dump available, trying to reconstruct from layer 0...")
    # hidden_after_layer0 = emb + linear_attn_out-0 + ffn_out-0
    # So emb = hidden_after_layer0 - linear_attn_out-0 - ffn_out-0
    # But we don't have ggml's hidden_after_layer0 directly...
    # Let's just compare the incremental outputs instead

print("\nDirect comparison of incremental outputs:")
print("(comparing ggml's per-layer outputs with CUDA's cumulative states)")
print()

# Actually, let's compare cumulative hidden states.
# We can reconstruct ggml cumulative by summing up all incremental outputs + embedding.
# But we don't have the embedding dump from ggml.
#
# Alternative approach: compare CUDA layer N hidden - CUDA layer N-1 hidden
# against ggml's linear_attn_out-N + ffn_out-N

# SSM layers: (i+1)%4 != 0 → 0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22, 24,25,26, 28,29,30
# Attention layers: 3,7,11,15,19,23,27,31

for layer in range(32):
    is_attn = ((layer + 1) % 4 == 0)
    layer_type = "attn" if is_attn else "ssm"

    # ggml incremental outputs
    attn_file = f'/tmp/ggml_linear_attn_out-{layer}_pos{pos}.bin'
    ffn_file = f'/tmp/ggml_ffn_out-{layer}_pos{pos}.bin'

    has_attn = os.path.exists(attn_file)
    has_ffn = os.path.exists(ffn_file)

    if has_attn:
        ggml_attn = np.fromfile(attn_file, dtype=np.float32)
    if has_ffn:
        ggml_ffn = np.fromfile(ffn_file, dtype=np.float32)

    # CUDA cumulative hidden state
    cuda_file = f'/tmp/hidden_f32_pos{pos}_layer{layer}.bin'
    has_cuda = os.path.exists(cuda_file)

    cuda_prev_file = f'/tmp/hidden_f32_pos{pos}_layer{layer-1}.bin' if layer > 0 else None
    has_cuda_prev = cuda_prev_file and os.path.exists(cuda_prev_file)

    if has_cuda:
        cuda_hidden = np.fromfile(cuda_file, dtype=np.float32)

    # Compute CUDA incremental = cuda_hidden[layer] - cuda_hidden[layer-1]
    if has_cuda and has_cuda_prev:
        cuda_prev = np.fromfile(cuda_prev_file, dtype=np.float32)
        cuda_incr = cuda_hidden - cuda_prev
    elif has_cuda and layer == 0:
        cuda_incr = None  # Can't compute without embedding
    else:
        cuda_incr = None

    # Compare incremental outputs
    if has_attn and has_ffn and cuda_incr is not None:
        ggml_incr = ggml_attn + ggml_ffn
        diff = cuda_incr - ggml_incr
        max_diff = np.abs(diff).max()
        cos_sim = np.dot(cuda_incr, ggml_incr) / (np.linalg.norm(cuda_incr) * np.linalg.norm(ggml_incr) + 1e-10)
        print(f"L{layer:2d} ({layer_type}): incr max_diff={max_diff:.6f}, cos_sim={cos_sim:.6f}, "
              f"cuda_incr_L2={np.linalg.norm(cuda_incr):.4f}, ggml_incr_L2={np.linalg.norm(ggml_incr):.4f}")
    elif has_attn and has_ffn and has_cuda and layer == 0:
        # For layer 0, compare cumulative: cuda_hidden = emb + attn + ffn
        # We can back-compute emb = cuda_hidden - attn - ffn
        ggml_incr = ggml_attn + ggml_ffn
        implied_emb = cuda_hidden - ggml_incr
        print(f"L{layer:2d} ({layer_type}): ggml_incr_L2={np.linalg.norm(ggml_incr):.4f}, "
              f"cuda_hidden_L2={np.linalg.norm(cuda_hidden):.4f}, implied_emb_L2={np.linalg.norm(implied_emb):.4f}")
    else:
        parts = []
        if has_attn:
            parts.append(f"ggml_attn_L2={np.linalg.norm(ggml_attn):.4f}")
        if has_ffn:
            parts.append(f"ggml_ffn_L2={np.linalg.norm(ggml_ffn):.4f}")
        if has_cuda:
            parts.append(f"cuda_hidden_L2={np.linalg.norm(cuda_hidden):.4f}")
        print(f"L{layer:2d} ({layer_type}): {', '.join(parts) if parts else 'NO DATA'}")

# Also compare the final output logits if available
print("\n" + "="*80)
print("Final output comparison")
ggml_result = f'/tmp/ggml_result_output_pos{pos}.bin'
if os.path.exists(ggml_result):
    ggml_logits = np.fromfile(ggml_result, dtype=np.float32)
    print(f"ggml result_output: shape={ggml_logits.shape}, sum={ggml_logits.sum():.4f}, L2={np.linalg.norm(ggml_logits):.4f}")
    top5 = np.argsort(ggml_logits)[-5:][::-1]
    print(f"ggml top5 tokens: {top5} with logits {ggml_logits[top5]}")
