#!/usr/bin/env python3
"""Verify full forward pass for layers 0-3, comparing CUDA dumps vs Python reference."""
import numpy as np, torch, struct

GGUF_PATH = "/home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-BF16.gguf"
N_EMBD = 4096; N_FF = 12288; RMS_NORM_EPS = 1e-6
SSM_D_INNER = 4096; SSM_D_STATE = 128; SSM_N_GROUP = 16; SSM_DT_RANK = 32
SSM_HEAD_V_DIM = 128; SSM_CONV_KERNEL = 4; SSM_CONV_CHANNELS = 8192
N_HEAD = 16; N_HEAD_KV = 4; HEAD_DIM = 256; ROPE_DIM = 64

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
            for _ in range(arr_len): skip_value(arr_type)
        else: f.read(sizes[vtype])
    for _ in range(n_kv):
        read_string(); vtype = struct.unpack('<I', f.read(4))[0]; skip_value(vtype)
    tensors = {}
    for _ in range(n_tensors):
        name = read_string()
        n_dims = struct.unpack('<I', f.read(4))[0]
        ne = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        dtype = struct.unpack('<I', f.read(4))[0]; offset = struct.unpack('<Q', f.read(8))[0]
        tensors[name] = (ne, dtype, offset)
    data_start = f.tell(); data_start = (data_start + 31) & ~31
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

def to_bf16(x):
    return torch.from_numpy(x).float().bfloat16().float().numpy()

def rms_norm(x, w, eps=RMS_NORM_EPS):
    return (x / np.sqrt(np.mean(x**2) + eps)) * w

def silu(x): return x / (1 + np.exp(-x))
def softplus(x): return np.where(x > 20, x, np.log1p(np.exp(x)))
def l2_norm(x, eps=RMS_NORM_EPS): return x / (np.sqrt(np.sum(x**2)) + eps)

def bf16_gemm(input_f32, weight_bf16_f32):
    """Simulate bf16 GEMM: bf16(input) @ bf16(weight)^T -> f32"""
    inp = torch.from_numpy(input_f32).float().bfloat16()
    w = torch.from_numpy(weight_bf16_f32).float().bfloat16()
    return (inp.float() @ w.float().T).numpy()

f, tensors, data_start = read_gguf_tensors(GGUF_PATH)
tok_embd = load_tensor(f, tensors, data_start, "token_embd.weight")
tokens = [760, 6511, 314, 9338, 369]  # "The capital of France is"

# SSM layer state
conv_states = [np.zeros((SSM_CONV_KERNEL - 1) * SSM_CONV_CHANNELS, dtype=np.float32) for _ in range(24)]
rec_states = [np.zeros((SSM_DT_RANK, SSM_D_STATE, SSM_HEAD_V_DIM), dtype=np.float32) for _ in range(24)]

def forward_ssm_layer(hidden, blk, ssm_idx, pos):
    norm_w = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_norm.weight")
    normed = rms_norm(hidden, norm_w)
    normed_bf16 = to_bf16(normed)

    qkv_w = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_qkv.weight")
    gate_w = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_gate.weight")
    alpha_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_alpha.weight")
    beta_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_beta.weight")
    dt_bias = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_dt.bias")
    ssm_a = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_a")
    conv_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_conv1d.weight")
    ssm_norm_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_norm.weight")
    ssm_out_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_out.weight")
    post_norm_w = load_tensor(f, tensors, data_start, f"blk.{blk}.post_attention_norm.weight")
    ffn_gate_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ffn_gate.weight")
    ffn_up_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ffn_up.weight")
    ffn_down_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ffn_down.weight")

    # QKV projection (bf16 GEMM)
    qkv = bf16_gemm(normed_bf16, qkv_w)
    z = bf16_gemm(normed_bf16, gate_w)
    alpha = bf16_gemm(normed_bf16, alpha_w)
    beta_raw = bf16_gemm(normed_bf16, beta_w)

    gate = softplus(alpha + dt_bias) * ssm_a
    beta = 1 / (1 + np.exp(-beta_raw))

    # Conv1d + SiLU
    conv_state = conv_states[ssm_idx]
    state_len = SSM_CONV_KERNEL - 1
    conv_out = np.zeros(SSM_CONV_CHANNELS, dtype=np.float32)
    for k in range(SSM_CONV_KERNEL):
        if k < state_len:
            vals = conv_state[k * SSM_CONV_CHANNELS:(k+1) * SSM_CONV_CHANNELS]
        else:
            vals = qkv[:SSM_CONV_CHANNELS]
        conv_out += vals * conv_w[:, k]
    conv_out = silu(conv_out)

    # Update conv state
    new_conv = np.zeros((SSM_CONV_KERNEL - 1) * SSM_CONV_CHANNELS, dtype=np.float32)
    for i in range(SSM_CONV_KERNEL - 1):
        src = 1 + i
        if src < state_len:
            new_conv[i*SSM_CONV_CHANNELS:(i+1)*SSM_CONV_CHANNELS] = conv_state[src*SSM_CONV_CHANNELS:(src+1)*SSM_CONV_CHANNELS]
        else:
            new_conv[i*SSM_CONV_CHANNELS:(i+1)*SSM_CONV_CHANNELS] = qkv[:SSM_CONV_CHANNELS]
    conv_states[ssm_idx] = new_conv

    # Q/K/V split and normalize
    qk_size = SSM_D_STATE * SSM_N_GROUP
    q = conv_out[:qk_size].copy(); k = conv_out[qk_size:2*qk_size].copy(); v = conv_out[2*qk_size:].copy()
    for h in range(SSM_N_GROUP):
        q[h*SSM_D_STATE:(h+1)*SSM_D_STATE] = l2_norm(q[h*SSM_D_STATE:(h+1)*SSM_D_STATE])
        k[h*SSM_D_STATE:(h+1)*SSM_D_STATE] = l2_norm(k[h*SSM_D_STATE:(h+1)*SSM_D_STATE])

    # Repeat heads
    q_rep = np.zeros(SSM_DT_RANK * SSM_D_STATE); k_rep = np.zeros(SSM_DT_RANK * SSM_D_STATE)
    for vh in range(SSM_DT_RANK):
        kh = vh * SSM_N_GROUP // SSM_DT_RANK
        q_rep[vh*SSM_D_STATE:(vh+1)*SSM_D_STATE] = q[kh*SSM_D_STATE:(kh+1)*SSM_D_STATE]
        k_rep[vh*SSM_D_STATE:(vh+1)*SSM_D_STATE] = k[kh*SSM_D_STATE:(kh+1)*SSM_D_STATE]

    # Delta-net
    scale = 1.0 / np.sqrt(SSM_D_STATE)
    delta_out = np.zeros(SSM_DT_RANK * SSM_HEAD_V_DIM)
    rec = rec_states[ssm_idx]
    for h in range(SSM_DT_RANK):
        g = np.exp(gate[h]); rec[h] *= g
        q_h = q_rep[h*SSM_D_STATE:(h+1)*SSM_D_STATE] * scale
        k_h = k_rep[h*SSM_D_STATE:(h+1)*SSM_D_STATE]
        v_h = v[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM]
        sk = rec[h].T @ k_h
        d = beta[h] * (v_h - sk)
        rec[h] += np.outer(k_h, d)
        delta_out[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM] = rec[h].T @ q_h
    rec_states[ssm_idx] = rec

    # Gated RMSNorm (f32 z)
    gated = np.zeros(SSM_D_INNER)
    for h in range(SSM_DT_RANK):
        hi = delta_out[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM]
        hz = z[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM]
        gated[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM] = rms_norm(hi, ssm_norm_w) * silu(hz)
    gated_bf16 = to_bf16(gated)

    # Output projection
    ssm_proj = bf16_gemm(gated_bf16, ssm_out_w)

    # Residual
    hidden = hidden + ssm_proj

    # FFN: post_norm -> gate*up -> silu(gate)*up -> down -> residual
    normed2 = rms_norm(hidden, post_norm_w)
    normed2_bf16 = to_bf16(normed2)
    ffn_g = bf16_gemm(normed2_bf16, ffn_gate_w)
    ffn_u = bf16_gemm(normed2_bf16, ffn_up_w)
    ffn_act = to_bf16(silu(ffn_g) * ffn_u)
    ffn_out = bf16_gemm(ffn_act, ffn_down_w)
    hidden = hidden + ffn_out

    return hidden

def forward_attn_layer(hidden, blk, attn_idx, pos, kv_cache):
    norm_w = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_norm.weight")
    wq = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_q.weight")
    wk = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_k.weight")
    wv = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_v.weight")
    wo = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_output.weight")
    q_norm_w = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_q_norm.weight")
    k_norm_w = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_k_norm.weight")
    post_norm_w = load_tensor(f, tensors, data_start, f"blk.{blk}.post_attention_norm.weight")
    ffn_gate_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ffn_gate.weight")
    ffn_up_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ffn_up.weight")
    ffn_down_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ffn_down.weight")

    normed = rms_norm(hidden, norm_w)
    normed_bf16 = to_bf16(normed)

    # Q projection: [4096] -> [8192] (Q + Gate packed)
    q_full = bf16_gemm(normed_bf16, wq)  # [8192]

    # Split Q and Gate: interleaved as [head, 2*head_dim] -> Q[head, head_dim], Gate[head, head_dim]
    q_full_4d = q_full.reshape(N_HEAD, HEAD_DIM * 2)
    q_raw = q_full_4d[:, :HEAD_DIM]    # [16, 256]
    q_gate = q_full_4d[:, HEAD_DIM:]   # [16, 256]

    # Q norm per head
    for h in range(N_HEAD):
        q_raw[h] = rms_norm(q_raw[h], q_norm_w)

    # K, V projections
    k_raw = bf16_gemm(normed_bf16, wk).reshape(N_HEAD_KV, HEAD_DIM)  # [4, 256]
    v_raw = bf16_gemm(normed_bf16, wv).reshape(N_HEAD_KV, HEAD_DIM)  # [4, 256]

    # K norm per head
    for h in range(N_HEAD_KV):
        k_raw[h] = rms_norm(k_raw[h], k_norm_w)

    # RoPE (simplified - only first 64 dims)
    # TODO: implement RoPE properly
    # For now, store in KV cache and compute attention

    # Add to KV cache
    kv_cache['k'].append(k_raw.copy())
    kv_cache['v'].append(v_raw.copy())

    print(f"  [Attn L{blk}] Q sum={q_raw.sum():.4f}, K sum={k_raw.sum():.4f}, V sum={v_raw.sum():.4f}")
    print(f"  [Attn L{blk}] Gate sum={q_gate.sum():.4f}")

    # This is a simplified version - we'd need full RoPE + attention to match
    # For now, just return the hidden state with a note
    return None  # Can't compute without proper RoPE

# Run forward pass
pos = 0
tok = tokens[pos]
hidden = tok_embd[tok].astype(np.float32)
print(f"Embedding: sum={hidden.sum():.4f}, L2={np.sqrt((hidden**2).sum()):.4f}")

cuda_hidden = np.fromfile(f'/tmp/hidden_f32_pos{pos}_layer0.bin', dtype=np.float32)

# Layer 0 (SSM)
hidden = forward_ssm_layer(hidden, 0, 0, pos)
print(f"\nLayer 0 (SSM): sum={hidden.sum():.4f}, L2={np.sqrt((hidden**2).sum()):.4f}")
print(f"  CUDA:        sum={cuda_hidden.sum():.4f}, L2={np.sqrt((cuda_hidden**2).sum()):.4f}")
diff = np.abs(hidden - cuda_hidden)
print(f"  max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")

# Layer 1 (SSM)
hidden_1 = forward_ssm_layer(hidden, 1, 1, pos)
cuda_hidden_1 = np.fromfile(f'/tmp/hidden_f32_pos{pos}_layer1.bin', dtype=np.float32)
print(f"\nLayer 1 (SSM): sum={hidden_1.sum():.4f}, L2={np.sqrt((hidden_1**2).sum()):.4f}")
print(f"  CUDA:        sum={cuda_hidden_1.sum():.4f}, L2={np.sqrt((cuda_hidden_1**2).sum()):.4f}")
diff1 = np.abs(hidden_1 - cuda_hidden_1)
print(f"  max_diff={diff1.max():.6f}, mean_diff={diff1.mean():.6f}")

# Layer 2 (SSM)
hidden_2 = forward_ssm_layer(hidden_1, 2, 2, pos)
cuda_hidden_2 = np.fromfile(f'/tmp/hidden_f32_pos{pos}_layer2.bin', dtype=np.float32)
print(f"\nLayer 2 (SSM): sum={hidden_2.sum():.4f}, L2={np.sqrt((hidden_2**2).sum()):.4f}")
print(f"  CUDA:        sum={cuda_hidden_2.sum():.4f}, L2={np.sqrt((cuda_hidden_2**2).sum()):.4f}")
diff2 = np.abs(hidden_2 - cuda_hidden_2)
print(f"  max_diff={diff2.max():.6f}, mean_diff={diff2.mean():.6f}")

print("\nDone!")
