#!/usr/bin/env python3
"""Verify attention layer 3 at position 4 (kv_len=5) against CUDA output."""
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
    if dtype == 30:
        raw = np.frombuffer(f.read(n_elems * 2), dtype=np.uint16).copy()
        raw32 = raw.astype(np.uint32) << 16
        f32 = np.frombuffer(raw32.tobytes(), dtype=np.float32).copy()
        return f32.reshape(ne[::-1]) if len(ne) > 1 else f32
    elif dtype == 0:
        arr = np.frombuffer(f.read(n_elems * 4), dtype=np.float32).copy()
        return arr.reshape(ne[::-1]) if len(ne) > 1 else arr

def load_bf16_bin(path, n):
    raw = np.fromfile(path, dtype=np.uint16)[:n]
    raw32 = raw.astype(np.uint32) << 16
    return np.frombuffer(raw32.tobytes(), dtype=np.float32).copy()

def rms_norm(x, weight, eps=RMS_NORM_EPS):
    return (x / np.sqrt(np.mean(x**2) + eps)) * weight

def rms_norm_head(x, weight, eps=RMS_NORM_EPS):
    result = np.zeros_like(x)
    for h in range(x.shape[0]):
        rms = np.sqrt(np.mean(x[h]**2) + eps)
        result[h] = (x[h] / rms) * weight
    return result

def apply_rope(x, pos, n_heads, head_dim, rope_dim, freq_base):
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

    # We need hidden states after layer 2 for all 5 positions to build KV cache
    # Then compute layer 3 (attention) at position 4

    # Load hidden states after layer 2 for all positions
    hiddens_after_l2 = []
    for pos in range(5):
        h = load_bf16_bin(f"/tmp/hidden_pos{pos}_layer2.bin", N_EMBD)
        hiddens_after_l2.append(h)
        print(f"  Loaded hidden pos {pos} after layer 2: L2={np.sqrt(np.sum(h**2)):.4f}")

    # Load weights
    attn_norm = load_tensor(f, tensors, data_start, "blk.3.attn_norm.weight")
    wq = load_tensor(f, tensors, data_start, "blk.3.attn_q.weight")
    wk = load_tensor(f, tensors, data_start, "blk.3.attn_k.weight")
    wv = load_tensor(f, tensors, data_start, "blk.3.attn_v.weight")
    q_norm_w = load_tensor(f, tensors, data_start, "blk.3.attn_q_norm.weight")
    k_norm_w = load_tensor(f, tensors, data_start, "blk.3.attn_k_norm.weight")
    wo = load_tensor(f, tensors, data_start, "blk.3.attn_output.weight")
    post_norm = load_tensor(f, tensors, data_start, "blk.3.post_attention_norm.weight")
    ffn_gate_w = torch.from_numpy(load_tensor(f, tensors, data_start, "blk.3.ffn_gate.weight")).float()
    ffn_up_w = torch.from_numpy(load_tensor(f, tensors, data_start, "blk.3.ffn_up.weight")).float()
    ffn_down_w = torch.from_numpy(load_tensor(f, tensors, data_start, "blk.3.ffn_down.weight")).float()

    wq_t = torch.from_numpy(wq).float()
    wk_t = torch.from_numpy(wk).float()
    wv_t = torch.from_numpy(wv).float()
    wo_t = torch.from_numpy(wo).float()

    # Build KV cache: compute K, V for all 5 positions
    k_cache = []  # list of [N_HEAD_KV, HEAD_DIM] arrays (after norm + rope)
    v_cache = []  # list of [N_HEAD_KV, HEAD_DIM] arrays

    for pos in range(5):
        hidden = hiddens_after_l2[pos]
        normed = rms_norm(hidden, attn_norm)

        # K projection
        k_all = (torch.from_numpy(normed).float() @ wk_t.T).numpy()
        k_heads = k_all.reshape(N_HEAD_KV, HEAD_DIM)
        k_heads = rms_norm_head(k_heads, k_norm_w)
        k_heads = apply_rope(k_heads, pos, N_HEAD_KV, HEAD_DIM, ROPE_DIM, ROPE_FREQ_BASE)
        k_cache.append(k_heads.copy())

        # V projection (no norm, no rope)
        v_all = (torch.from_numpy(normed).float() @ wv_t.T).numpy()
        v_heads = v_all.reshape(N_HEAD_KV, HEAD_DIM)
        v_cache.append(v_heads.copy())

    print(f"\nKV cache built for 5 positions")

    # Now compute attention for position 4
    pos = 4
    hidden = hiddens_after_l2[pos]
    normed = rms_norm(hidden, attn_norm)

    # Q + Gate projection
    qg = (torch.from_numpy(normed).float() @ wq_t.T).numpy()

    # Split Q and Gate
    q_all = np.zeros(N_HEAD * HEAD_DIM, dtype=np.float32)
    gate_all = np.zeros(N_HEAD * HEAD_DIM, dtype=np.float32)
    for h in range(N_HEAD):
        q_all[h*HEAD_DIM:(h+1)*HEAD_DIM] = qg[h*HEAD_DIM*2:h*HEAD_DIM*2+HEAD_DIM]
        gate_all[h*HEAD_DIM:(h+1)*HEAD_DIM] = qg[h*HEAD_DIM*2+HEAD_DIM:h*HEAD_DIM*2+2*HEAD_DIM]

    # Q norm
    q_heads = q_all.reshape(N_HEAD, HEAD_DIM)
    q_heads = rms_norm_head(q_heads, q_norm_w)

    # RoPE on Q at position 4
    q_heads = apply_rope(q_heads, pos, N_HEAD, HEAD_DIM, ROPE_DIM, ROPE_FREQ_BASE)

    # Compute attention scores: Q @ K^T * scale for each head
    kv_len = 5
    attn_out = np.zeros((N_HEAD, HEAD_DIM), dtype=np.float32)

    for h in range(N_HEAD):
        kv_h = h // (N_HEAD // N_HEAD_KV)  # GQA mapping

        # Compute scores
        scores = np.zeros(kv_len, dtype=np.float32)
        for kv_pos in range(kv_len):
            scores[kv_pos] = np.dot(q_heads[h], k_cache[kv_pos][kv_h]) * ATTN_SCALE

        # Softmax
        scores = scores - np.max(scores)
        scores = np.exp(scores)
        scores = scores / np.sum(scores)

        # Weighted sum of V
        for kv_pos in range(kv_len):
            attn_out[h] += scores[kv_pos] * v_cache[kv_pos][kv_h]

    print(f"Attention output[:8]: {attn_out.flatten()[:8]}")

    # Sigmoid gate
    sig_gate = 1.0 / (1.0 + np.exp(-gate_all))
    gated = attn_out.flatten() * sig_gate
    print(f"After gate[:8]: {gated[:8]}")

    # Output projection
    projected = (torch.from_numpy(gated).float() @ wo_t.T).numpy()

    # Residual
    after_attn = hidden + projected
    print(f"After attn residual[:8]: {after_attn[:8]}")

    # FFN
    ffn_in = rms_norm(after_attn, post_norm)
    ffn_in_t = torch.from_numpy(ffn_in).float()
    gate_v = ffn_in_t @ ffn_gate_w.T
    up_v = ffn_in_t @ ffn_up_w.T
    ffn_act = torch.nn.functional.silu(gate_v) * up_v
    ffn_out = (ffn_act @ ffn_down_w.T).numpy()
    final = after_attn + ffn_out

    print(f"\nPython final[:8]: {final[:8]}")

    # Compare
    cuda_hidden = load_bf16_bin("/tmp/hidden_pos4_layer3.bin", N_EMBD)
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
