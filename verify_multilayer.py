#!/usr/bin/env python3
"""Verify first 4 layers for token 0 using full Python reference."""
import struct, numpy as np, torch, os

GGUF_PATH = "/home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-BF16.gguf"
N_EMBD = 4096; N_FF = 12288; RMS_NORM_EPS = 1e-6
SSM_D_INNER = 4096; SSM_D_STATE = 128; SSM_N_GROUP = 16; SSM_DT_RANK = 32
SSM_HEAD_V_DIM = 128; SSM_CONV_KERNEL = 4; SSM_CONV_CHANNELS = 8192

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

def rms_norm(x, w, eps=RMS_NORM_EPS):
    return (x / np.sqrt(np.mean(x**2) + eps)) * w

def to_bf16(x):
    return torch.from_numpy(np.asarray(x).astype(np.float32)).bfloat16().float().numpy()

def silu(x): return x / (1 + np.exp(-x))
def softplus(x): return np.where(x > 20, x, np.log1p(np.exp(x)))
def l2_norm(x, eps=RMS_NORM_EPS): return x / (np.sqrt(np.sum(x**2)) + eps)

def bf16_gemm(input_bf16, weight, out_bf16=True):
    """Simulate bf16 GEMM matching ggml: bf16 input @ bf16 weight.T -> bf16 -> f32"""
    inp = torch.from_numpy(input_bf16).bfloat16()
    w = torch.from_numpy(weight).bfloat16()
    result = (inp.float() @ w.float().T)
    if out_bf16:
        return result.bfloat16().float().numpy()
    else:
        return result.numpy()

def process_ssm_layer(f, tensors, data_start, blk, hidden_f32, conv_state, rec_state):
    """Process one SSM layer. hidden is f32, returns f32."""
    norm_w = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_norm.weight")

    # RMSNorm: f32 in, bf16 out (matching CUDA)
    normed = rms_norm(hidden_f32, norm_w)
    normed_bf16 = to_bf16(normed)

    # QKV projection: bf16 in -> f32 out (gemm_bf16_f32out)
    qkv_w = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_qkv.weight")
    nt = torch.from_numpy(normed_bf16).bfloat16()
    qkv = (nt.float() @ torch.from_numpy(qkv_w).bfloat16().float().T).bfloat16().float().numpy()

    # Gate Z: bf16 in -> bf16 out (gemm_bf16)
    gate_w = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_gate.weight")
    z = (nt.float() @ torch.from_numpy(gate_w).bfloat16().float().T).bfloat16().float().numpy()

    # Alpha: bf16 in -> f32 out (gemm_bf16_f32out)
    alpha_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_alpha.weight")
    alpha = (nt.float() @ torch.from_numpy(alpha_w).bfloat16().float().T).bfloat16().float().numpy()

    # Beta: bf16 in -> f32 out (gemm_bf16_f32out)
    beta_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_beta.weight")
    beta_raw = (nt.float() @ torch.from_numpy(beta_w).bfloat16().float().T).bfloat16().float().numpy()

    dt_bias = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_dt.bias")
    ssm_a = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_a")
    gate_val = softplus(alpha + dt_bias) * ssm_a
    beta = 1 / (1 + np.exp(-beta_raw))

    # Conv1d + SiLU
    conv_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_conv1d.weight")
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

    # Q/K/V split + L2 norm
    qk_size = SSM_D_STATE * SSM_N_GROUP
    q = conv_out[:qk_size].copy()
    k = conv_out[qk_size:2*qk_size].copy()
    v = conv_out[2*qk_size:].copy()
    for h in range(SSM_N_GROUP):
        q[h*SSM_D_STATE:(h+1)*SSM_D_STATE] = l2_norm(q[h*SSM_D_STATE:(h+1)*SSM_D_STATE])
        k[h*SSM_D_STATE:(h+1)*SSM_D_STATE] = l2_norm(k[h*SSM_D_STATE:(h+1)*SSM_D_STATE])

    # Repeat heads
    q_rep = np.zeros(SSM_DT_RANK * SSM_D_STATE)
    k_rep = np.zeros(SSM_DT_RANK * SSM_D_STATE)
    for vh in range(SSM_DT_RANK):
        kh = vh * SSM_N_GROUP // SSM_DT_RANK
        q_rep[vh*SSM_D_STATE:(vh+1)*SSM_D_STATE] = q[kh*SSM_D_STATE:(kh+1)*SSM_D_STATE]
        k_rep[vh*SSM_D_STATE:(vh+1)*SSM_D_STATE] = k[kh*SSM_D_STATE:(kh+1)*SSM_D_STATE]

    # Delta-net
    scale = 1.0 / np.sqrt(SSM_D_STATE)
    delta_out = np.zeros(SSM_DT_RANK * SSM_HEAD_V_DIM)
    new_rec = rec_state.copy()
    for h in range(SSM_DT_RANK):
        g = np.exp(gate_val[h])
        new_rec[h] *= g
        q_h = q_rep[h*SSM_D_STATE:(h+1)*SSM_D_STATE] * scale
        k_h = k_rep[h*SSM_D_STATE:(h+1)*SSM_D_STATE]
        v_h = v[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM]
        sk = new_rec[h].T @ k_h
        d = beta[h] * (v_h - sk)
        new_rec[h] += np.outer(k_h, d)
        delta_out[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM] = new_rec[h].T @ q_h

    # Gated RMSNorm
    ssm_norm_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_norm.weight")
    gated = np.zeros(SSM_D_INNER)
    for h in range(SSM_DT_RANK):
        hi = delta_out[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM]
        hz = z[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM]
        gated[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM] = rms_norm(hi, ssm_norm_w) * silu(hz)

    # Output projection: bf16 -> f32
    gated_bf16 = to_bf16(gated)
    ssm_out_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_out.weight")
    gt = torch.from_numpy(gated_bf16).bfloat16()
    proj = (gt.float() @ torch.from_numpy(ssm_out_w).bfloat16().float().T).bfloat16().float().numpy()

    # SSM residual
    hidden_f32 = hidden_f32 + proj

    # Post-attention norm + FFN
    post_norm_w = load_tensor(f, tensors, data_start, f"blk.{blk}.post_attention_norm.weight")
    ffn_normed = rms_norm(hidden_f32, post_norm_w)
    ffn_normed_bf16 = to_bf16(ffn_normed)

    ffn_gate_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ffn_gate.weight")
    ffn_up_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ffn_up.weight")
    ffn_down_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ffn_down.weight")

    fnt = torch.from_numpy(ffn_normed_bf16).bfloat16()
    gate_out = (fnt.float() @ torch.from_numpy(ffn_gate_w).bfloat16().float().T).bfloat16().float().numpy()
    up_out = (fnt.float() @ torch.from_numpy(ffn_up_w).bfloat16().float().T).bfloat16().float().numpy()

    swiglu = silu(gate_out) * up_out
    swiglu_bf16 = to_bf16(swiglu)

    st = torch.from_numpy(swiglu_bf16).bfloat16()
    ffn_out = (st.float() @ torch.from_numpy(ffn_down_w).bfloat16().float().T).bfloat16().float().numpy()

    # FFN residual
    hidden_f32 = hidden_f32 + ffn_out

    return hidden_f32, new_conv, new_rec

# Main
f, tensors, data_start = read_gguf_tensors(GGUF_PATH)
tok_embd = load_tensor(f, tensors, data_start, "token_embd.weight")

tokens = [760, 6511, 314, 9338, 369]
tok = tokens[0]
hidden = tok_embd[tok].astype(np.float32)
print(f"Embedding sum: {np.sum(hidden):.6f}")

# Process layers 0, 1, 2 (SSM)
conv_states = [np.zeros((SSM_CONV_KERNEL - 1) * SSM_CONV_CHANNELS, dtype=np.float32) for _ in range(3)]
rec_states = [np.zeros((SSM_DT_RANK, SSM_D_STATE, SSM_HEAD_V_DIM), dtype=np.float32) for _ in range(3)]

for il in range(3):  # layers 0, 1, 2 are SSM
    hidden, conv_states[il], rec_states[il] = process_ssm_layer(
        f, tensors, data_start, il, hidden, conv_states[il], rec_states[il])

    cuda_h = np.fromfile(f"/tmp/hidden_f32_pos0_layer{il}.bin", dtype=np.float32)[:N_EMBD]
    diff = np.abs(hidden - cuda_h)
    print(f"Layer {il} (SSM): py_sum={np.sum(hidden):.6f} cuda_sum={np.sum(cuda_h):.6f} "
          f"max_diff={np.max(diff):.6f} mean_diff={np.mean(diff):.6f}")
