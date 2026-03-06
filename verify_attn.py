#!/usr/bin/env python3
"""Verify attention layer (layer 3) computation."""
import struct
import numpy as np
import torch

GGUF_PATH = "/home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-BF16.gguf"

N_EMBD = 4096
N_HEAD = 16
N_HEAD_KV = 4
HEAD_DIM = 256
N_FF = 12288
RMS_NORM_EPS = 1e-6
ROPE_DIM = 64
ROPE_FREQ_BASE = 10000000.0
ATTN_SCALE = 1.0 / 16.0  # 1/sqrt(256)

def read_gguf_tensors(path):
    f = open(path, 'rb')
    f.read(4); struct.unpack('<I', f.read(4))
    n_tensors = struct.unpack('<Q', f.read(8))[0]; n_kv = struct.unpack('<Q', f.read(8))[0]
    def read_string():
        l = struct.unpack('<Q', f.read(8))[0]; return f.read(l).decode('utf-8')
    def skip_value(vtype):
        sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 8:0, 10:8, 11:8, 12:8}
        if vtype == 8: read_string()
        elif vtype == 9:
            arr_type = struct.unpack('<I', f.read(4))[0]; arr_len = struct.unpack('<Q', f.read(8))[0]
            for i in range(arr_len): skip_value(arr_type)
        else: f.read(sizes[vtype])
    for i in range(n_kv):
        read_string(); vtype = struct.unpack('<I', f.read(4))[0]; skip_value(vtype)
    tensors = {}
    for i in range(n_tensors):
        name = read_string()
        n_dims = struct.unpack('<I', f.read(4))[0]
        ne = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        dtype = struct.unpack('<I', f.read(4))[0]; offset = struct.unpack('<Q', f.read(8))[0]
        tensors[name] = (ne, dtype, offset)
    data_start = f.tell()
    data_start = (data_start + 31) & ~31
    return f, tensors, data_start

def load_tensor(f, tensors, data_start, name):
    ne, dtype, offset = tensors[name]
    n_elems = 1
    for d in ne: n_elems *= d
    f.seek(data_start + offset)
    if dtype == 30:  # BF16
        raw = np.frombuffer(f.read(n_elems * 2), dtype=np.uint16).copy()
        # Vectorized bf16->f32
        f32 = np.empty(n_elems, dtype=np.float32)
        raw32 = raw.astype(np.uint32) << 16
        f32 = np.frombuffer(raw32.tobytes(), dtype=np.float32).copy()
        return f32.reshape(ne[::-1]) if len(ne) > 1 else f32
    elif dtype == 0:  # F32
        arr = np.frombuffer(f.read(n_elems * 4), dtype=np.float32).copy()
        return arr.reshape(ne[::-1]) if len(ne) > 1 else arr

def load_bf16_bin(path, n):
    raw = np.fromfile(path, dtype=np.uint16)[:n]
    raw32 = raw.astype(np.uint32) << 16
    return np.frombuffer(raw32.tobytes(), dtype=np.float32).copy()

def rms_norm(x, weight, eps=RMS_NORM_EPS):
    rms = np.sqrt(np.mean(x**2) + eps)
    return (x / rms) * weight

def rms_norm_head(x, weight, eps=RMS_NORM_EPS):
    """Per-head RMS norm. x: [n_heads, head_dim], weight: [head_dim]"""
    result = np.zeros_like(x)
    for h in range(x.shape[0]):
        rms = np.sqrt(np.mean(x[h]**2) + eps)
        result[h] = (x[h] / rms) * weight
    return result

def apply_rope(x, pos, n_heads, head_dim, rope_dim, freq_base):
    """Apply RoPE to x: [n_heads, head_dim] at position pos."""
    result = x.copy()
    half = rope_dim // 2
    for h in range(n_heads):
        for p in range(half):
            freq = 1.0 / (freq_base ** (2.0 * p / rope_dim))
            theta = pos * freq
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            x0 = result[h, p]
            x1 = result[h, p + half]
            result[h, p] = x0 * cos_t - x1 * sin_t
            result[h, p + half] = x1 * cos_t + x0 * sin_t
    return result

def main():
    print("Loading GGUF...")
    f, tensors, data_start = read_gguf_tensors(GGUF_PATH)

    # Load hidden state after layer 2 (input to layer 3)
    hidden = load_bf16_bin("/tmp/hidden_after_layer2.bin", N_EMBD)
    print(f"Input hidden[:8]: {hidden[:8]}")

    # Layer 3 = attention layer (first attention, attn_idx=0)
    attn_norm = load_tensor(f, tensors, data_start, "blk.3.attn_norm.weight")
    normed = rms_norm(hidden, attn_norm)
    print(f"After norm[:8]: {normed[:8]}")

    # Q+Gate projection: [4096] -> [8192]
    wq = load_tensor(f, tensors, data_start, "blk.3.attn_q.weight")  # [8192, 4096]
    qg = torch.from_numpy(normed).float() @ torch.from_numpy(wq).float().T
    qg = qg.numpy()  # [8192]
    print(f"Q+Gate[:8]: {qg[:8]}")

    # Split Q and Gate: interleaved [h0_q(256), h0_gate(256), h1_q(256), h1_gate(256), ...]
    q_all = np.zeros(N_HEAD * HEAD_DIM, dtype=np.float32)
    gate_all = np.zeros(N_HEAD * HEAD_DIM, dtype=np.float32)
    for h in range(N_HEAD):
        q_all[h*HEAD_DIM:(h+1)*HEAD_DIM] = qg[h*HEAD_DIM*2:h*HEAD_DIM*2+HEAD_DIM]
        gate_all[h*HEAD_DIM:(h+1)*HEAD_DIM] = qg[h*HEAD_DIM*2+HEAD_DIM:h*HEAD_DIM*2+2*HEAD_DIM]
    print(f"Q[:8]: {q_all[:8]}")
    print(f"Gate[:8]: {gate_all[:8]}")

    # K projection: [4096] -> [1024]
    wk = load_tensor(f, tensors, data_start, "blk.3.attn_k.weight")  # [1024, 4096]
    k_all = (torch.from_numpy(normed).float() @ torch.from_numpy(wk).float().T).numpy()
    print(f"K[:8]: {k_all[:8]}")

    # V projection: [4096] -> [1024]
    wv = load_tensor(f, tensors, data_start, "blk.3.attn_v.weight")  # [1024, 4096]
    v_all = (torch.from_numpy(normed).float() @ torch.from_numpy(wv).float().T).numpy()
    print(f"V[:8]: {v_all[:8]}")

    # Q norm (per head)
    q_norm_w = load_tensor(f, tensors, data_start, "blk.3.attn_q_norm.weight")  # [256]
    q_heads = q_all.reshape(N_HEAD, HEAD_DIM)
    q_heads = rms_norm_head(q_heads, q_norm_w)
    print(f"Q after norm[:8]: {q_heads.flatten()[:8]}")

    # K norm (per head)
    k_norm_w = load_tensor(f, tensors, data_start, "blk.3.attn_k_norm.weight")  # [256]
    k_heads = k_all.reshape(N_HEAD_KV, HEAD_DIM)
    k_heads = rms_norm_head(k_heads, k_norm_w)
    print(f"K after norm[:8]: {k_heads.flatten()[:8]}")

    # RoPE at position 0
    q_heads = apply_rope(q_heads, 0, N_HEAD, HEAD_DIM, ROPE_DIM, ROPE_FREQ_BASE)
    k_heads = apply_rope(k_heads, 0, N_HEAD_KV, HEAD_DIM, ROPE_DIM, ROPE_FREQ_BASE)
    print(f"Q after RoPE[:8]: {q_heads.flatten()[:8]}")
    print(f"K after RoPE[:8]: {k_heads.flatten()[:8]}")

    # Attention (single token, kv_len=1)
    # For decode with only 1 token, attention is just: softmax(Q.K^T/scale) @ V
    # With 1 token, softmax of a single element is always 1.0
    # So output = V (for each head, using GQA mapping)
    attn_out = np.zeros(N_HEAD * HEAD_DIM, dtype=np.float32)
    for h in range(N_HEAD):
        kv_h = h // (N_HEAD // N_HEAD_KV)  # GQA mapping
        # With single token: score = Q.K^T * scale, softmax([score]) = [1.0]
        # output = V[kv_h]
        attn_out[h*HEAD_DIM:(h+1)*HEAD_DIM] = v_all[kv_h*HEAD_DIM:(kv_h+1)*HEAD_DIM]
    print(f"Attn out[:8] (=V for single token): {attn_out[:8]}")

    # Sigmoid gate
    sig_gate = 1.0 / (1.0 + np.exp(-gate_all))
    gated = attn_out * sig_gate
    print(f"After gate[:8]: {gated[:8]}")

    # Output projection: [4096] -> [4096]
    wo = load_tensor(f, tensors, data_start, "blk.3.attn_output.weight")  # [4096, 4096]
    projected = (torch.from_numpy(gated).float() @ torch.from_numpy(wo).float().T).numpy()
    print(f"After wo[:8]: {projected[:8]}")

    # Residual
    after_attn = hidden + projected
    print(f"After attn residual[:8]: {after_attn[:8]}")

    # Post-attention norm + FFN
    post_norm = load_tensor(f, tensors, data_start, "blk.3.post_attention_norm.weight")
    ffn_in = rms_norm(after_attn, post_norm)

    ffn_gate_w = torch.from_numpy(load_tensor(f, tensors, data_start, "blk.3.ffn_gate.weight")).float()
    ffn_up_w = torch.from_numpy(load_tensor(f, tensors, data_start, "blk.3.ffn_up.weight")).float()
    ffn_down_w = torch.from_numpy(load_tensor(f, tensors, data_start, "blk.3.ffn_down.weight")).float()
    ffn_in_t = torch.from_numpy(ffn_in).float()

    gate_v = ffn_in_t @ ffn_gate_w.T
    up_v = ffn_in_t @ ffn_up_w.T
    ffn_act = torch.nn.functional.silu(gate_v) * up_v
    ffn_out = (ffn_act @ ffn_down_w.T).numpy()

    final = after_attn + ffn_out
    print(f"\nPython final[:8]: {final[:8]}")

    # Compare with CUDA
    cuda_hidden = load_bf16_bin("/tmp/hidden_after_layer3.bin", N_EMBD)
    print(f"CUDA final[:8]:   {cuda_hidden[:8]}")

    diff = np.abs(final - cuda_hidden)
    print(f"\nMax diff: {np.max(diff):.6f}")
    print(f"Mean diff: {np.mean(diff):.6f}")

    worst = np.argsort(diff)[-5:]
    for idx in worst:
        print(f"  [{idx}] python={final[idx]:.6f}  cuda={cuda_hidden[idx]:.6f}  diff={diff[idx]:.6f}")

    f.close()

if __name__ == "__main__":
    main()
