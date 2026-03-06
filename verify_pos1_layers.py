#!/usr/bin/env python3
"""Quick verify: compute SSM layer 0 then layer 1 for position 1 (token "capital")
using state from position 0, and compare against CUDA dumps."""
import struct
import numpy as np
import torch

GGUF_PATH = "/home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-BF16.gguf"
N_EMBD = 4096
N_FF = 12288
RMS_NORM_EPS = 1e-6
SSM_D_INNER = 4096
SSM_D_STATE = 128
SSM_N_GROUP = 16
SSM_DT_RANK = 32
SSM_HEAD_V_DIM = 128
SSM_CONV_KERNEL = 4
SSM_CONV_CHANNELS = 8192

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
        return np.frombuffer(raw32.tobytes(), dtype=np.float32).copy().reshape(ne[::-1]) if len(ne) > 1 else np.frombuffer(raw32.tobytes(), dtype=np.float32).copy()
    elif dtype == 0:
        arr = np.frombuffer(f.read(n_elems * 4), dtype=np.float32).copy()
        return arr.reshape(ne[::-1]) if len(ne) > 1 else arr

def load_f32_bin(path, n):
    return np.fromfile(path, dtype=np.float32)[:n]

def rms_norm(x, weight, eps=RMS_NORM_EPS):
    return (x / np.sqrt(np.mean(x**2) + eps)) * weight

def silu(x): return x / (1 + np.exp(-x))
def softplus(x): return np.where(x > 20, x, np.log1p(np.exp(x)))
def l2_norm(x, eps=RMS_NORM_EPS): return x / (np.sqrt(np.sum(x**2)) + eps)

def process_ssm(f, tensors, data_start, blk, hidden, conv_state, rec_state):
    """Process one SSM layer, return (new_hidden, new_conv_state, new_rec_state)."""
    attn_norm = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_norm.weight")
    normed = rms_norm(hidden, attn_norm)
    nt = torch.from_numpy(normed).float()

    wqkv = torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.attn_qkv.weight")).float()
    qkv = (nt @ wqkv.T).numpy()

    wgate = torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.attn_gate.weight")).float()
    z = (nt @ wgate.T).numpy()

    ssm_alpha_w = torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_alpha.weight")).float()
    alpha = (nt @ ssm_alpha_w.T).numpy()

    ssm_beta_w = torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_beta.weight")).float()
    beta = 1 / (1 + np.exp(-(nt @ ssm_beta_w.T).numpy()))

    dt_bias = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_dt.bias")
    ssm_a = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_a")
    gate = softplus(alpha + dt_bias) * ssm_a

    conv_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_conv1d.weight")

    # Conv1d + SiLU
    conv_out = np.zeros(SSM_CONV_CHANNELS, dtype=np.float32)
    for ch in range(SSM_CONV_CHANNELS):
        s = 0.0
        for k in range(SSM_CONV_KERNEL):
            pos = k
            if pos < SSM_CONV_KERNEL - 1:
                val = conv_state[pos * SSM_CONV_CHANNELS + ch]
            else:
                val = qkv[ch]
            s += val * conv_w[ch, k]
        conv_out[ch] = s
    conv_out = silu(conv_out)

    # Update conv state
    new_conv = np.zeros((SSM_CONV_KERNEL - 1) * SSM_CONV_CHANNELS, dtype=np.float32)
    for i in range(SSM_CONV_KERNEL - 1):
        src = 1 + i
        for ch in range(SSM_CONV_CHANNELS):
            if src < SSM_CONV_KERNEL - 1:
                new_conv[i * SSM_CONV_CHANNELS + ch] = conv_state[src * SSM_CONV_CHANNELS + ch]
            else:
                new_conv[i * SSM_CONV_CHANNELS + ch] = qkv[ch]

    # Q/K/V split
    qk_size = SSM_D_STATE * SSM_N_GROUP
    q = conv_out[:qk_size]
    k = conv_out[qk_size:2*qk_size]
    v = conv_out[2*qk_size:]

    # L2 norm
    for h in range(SSM_N_GROUP):
        q[h*SSM_D_STATE:(h+1)*SSM_D_STATE] = l2_norm(q[h*SSM_D_STATE:(h+1)*SSM_D_STATE])
        k[h*SSM_D_STATE:(h+1)*SSM_D_STATE] = l2_norm(k[h*SSM_D_STATE:(h+1)*SSM_D_STATE])

    # Repeat
    q_rep = np.zeros(SSM_DT_RANK * SSM_D_STATE, dtype=np.float32)
    k_rep = np.zeros(SSM_DT_RANK * SSM_D_STATE, dtype=np.float32)
    for vh in range(SSM_DT_RANK):
        kh = vh * SSM_N_GROUP // SSM_DT_RANK
        q_rep[vh*SSM_D_STATE:(vh+1)*SSM_D_STATE] = q[kh*SSM_D_STATE:(kh+1)*SSM_D_STATE]
        k_rep[vh*SSM_D_STATE:(vh+1)*SSM_D_STATE] = k[kh*SSM_D_STATE:(kh+1)*SSM_D_STATE]

    # Delta-net
    scale = 1.0 / np.sqrt(SSM_D_STATE)
    delta_out = np.zeros(SSM_DT_RANK * SSM_HEAD_V_DIM, dtype=np.float32)
    new_rec = rec_state.copy()
    for h in range(SSM_DT_RANK):
        g = np.exp(gate[h])
        new_rec[h] *= g
        q_h = q_rep[h*SSM_D_STATE:(h+1)*SSM_D_STATE] * scale
        k_h = k_rep[h*SSM_D_STATE:(h+1)*SSM_D_STATE]
        v_h = v[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM]
        sk = new_rec[h].T @ k_h
        d = beta[h] * (v_h - sk)
        new_rec[h] += np.outer(k_h, d)
        delta_out[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM] = new_rec[h].T @ q_h

    # Gated RMSNorm
    ssm_norm = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_norm.weight")
    gated = np.zeros(SSM_D_INNER, dtype=np.float32)
    for h in range(SSM_DT_RANK):
        hi = delta_out[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM]
        hz = z[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM]
        gated[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM] = rms_norm(hi, ssm_norm) * silu(hz)

    # Output projection
    ssm_out_w = torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_out.weight")).float()
    proj = (torch.from_numpy(gated).float() @ ssm_out_w.T).numpy()
    after_attn = hidden + proj

    # FFN
    post_norm = load_tensor(f, tensors, data_start, f"blk.{blk}.post_attention_norm.weight")
    ffn_in = rms_norm(after_attn, post_norm)
    fft = torch.from_numpy(ffn_in).float()
    ffn_gate_w = torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.ffn_gate.weight")).float()
    ffn_up_w = torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.ffn_up.weight")).float()
    ffn_down_w = torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.ffn_down.weight")).float()
    gv = fft @ ffn_gate_w.T
    uv = fft @ ffn_up_w.T
    act = torch.nn.functional.silu(gv) * uv
    ffn_out = (act @ ffn_down_w.T).numpy()
    final = after_attn + ffn_out

    return final, new_conv, new_rec

# Process token 0 then token 1 through layers 0-2, comparing at each step
f, tensors, data_start = read_gguf_tensors(GGUF_PATH)
tok_embd = load_tensor(f, tensors, data_start, "token_embd.weight")
tokens = [760, 6511]  # "The", "capital"

# Initialize states
conv_states = [np.zeros((SSM_CONV_KERNEL - 1) * SSM_CONV_CHANNELS, dtype=np.float32) for _ in range(3)]
rec_states = [np.zeros((SSM_DT_RANK, SSM_HEAD_V_DIM, SSM_HEAD_V_DIM), dtype=np.float32) for _ in range(3)]

# We need to dump f32 hidden states from CUDA
# First run CUDA with DUMP_ALL_LAYERS for "The capital"
import subprocess, os

# Process both tokens through 3 SSM layers
for layer in range(3):
    for pos in range(2):
        hidden = tok_embd[tokens[pos]].copy() if layer == 0 else hiddens[pos]
        new_h, new_conv, new_rec = process_ssm(f, tensors, data_start, layer, hidden, conv_states[layer], rec_states[layer])
        conv_states[layer] = new_conv
        rec_states[layer] = new_rec
        if layer == 0 and pos == 0:
            hiddens = [None, None]
        if layer == 0:
            hiddens[pos] = new_h
        else:
            hiddens[pos] = new_h

    # After processing both tokens through this layer, compare position 1
    py_h = hiddens[1]
    py_l2 = np.sqrt(np.sum(py_h**2))
    print(f"Layer {layer}, pos 1: py_L2={py_l2:.4f}  py[:5]={py_h[:5]}")

print("\nNow comparing against CUDA dumps (need to run CUDA with DUMP_ALL_LAYERS first)...")
# Check if f32 dumps exist
for layer in range(3):
    fpath = f"/tmp/hidden_pos1_layer{layer}_f32.bin"
    if os.path.exists(fpath):
        cuda_h = np.fromfile(fpath, dtype=np.float32)[:N_EMBD]
        py_h = hiddens_by_layer[layer][1]
        diff = np.abs(py_h - cuda_h)
        print(f"  Layer {layer}: max_diff={np.max(diff):.6f}")
    else:
        print(f"  Layer {layer}: no f32 dump found at {fpath}")
