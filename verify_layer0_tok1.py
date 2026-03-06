#!/usr/bin/env python3
"""Verify layer 0 SSM for token 1 (depends on state from token 0)."""
import struct
import numpy as np
import torch

GGUF_PATH = "/home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-BF16.gguf"

N_EMBD = 4096
SSM_D_INNER = 4096
SSM_D_STATE = 128
SSM_N_GROUP = 16
SSM_DT_RANK = 32
SSM_HEAD_V_DIM = SSM_D_INNER // SSM_DT_RANK  # 128
SSM_CONV_KERNEL = 4
SSM_CONV_CHANNELS = SSM_D_INNER + 2 * SSM_N_GROUP * SSM_D_STATE  # 8192
N_FF = 12288
RMS_NORM_EPS = 1e-6

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

def silu(x): return x / (1 + np.exp(-x))
def softplus(x): return np.where(x > 20, x, np.log1p(np.exp(x)))
def l2_norm(x, eps=RMS_NORM_EPS): return x / (np.sqrt(np.sum(x**2)) + eps)

def process_ssm_layer0(f, tensors, data_start, hidden, conv_state, rec_state):
    """Process layer 0 SSM and return (output, new_conv_state, new_rec_state)."""
    attn_norm = load_tensor(f, tensors, data_start, "blk.0.attn_norm.weight")
    normed = rms_norm(hidden, attn_norm)

    wqkv = load_tensor(f, tensors, data_start, "blk.0.attn_qkv.weight")
    qkv_mixed = normed @ wqkv.T

    wgate = load_tensor(f, tensors, data_start, "blk.0.attn_gate.weight")
    z = normed @ wgate.T

    ssm_alpha_w = load_tensor(f, tensors, data_start, "blk.0.ssm_alpha.weight")
    alpha = normed @ ssm_alpha_w.T

    ssm_beta_w = load_tensor(f, tensors, data_start, "blk.0.ssm_beta.weight")
    beta = 1 / (1 + np.exp(-(normed @ ssm_beta_w.T)))

    ssm_dt_bias = load_tensor(f, tensors, data_start, "blk.0.ssm_dt.bias")
    ssm_a = load_tensor(f, tensors, data_start, "blk.0.ssm_a")
    gate = softplus(alpha + ssm_dt_bias) * ssm_a

    conv_weight = load_tensor(f, tensors, data_start, "blk.0.ssm_conv1d.weight")

    # Conv1d + SiLU
    # Input concat: [conv_state(3, 8192), qkv_mixed(1, 8192)]
    conv_out = np.zeros(SSM_CONV_CHANNELS, dtype=np.float32)
    for ch in range(SSM_CONV_CHANNELS):
        s = 0.0
        for k in range(SSM_CONV_KERNEL):
            pos = 0 + k  # token=0
            state_len = SSM_CONV_KERNEL - 1
            if pos < state_len:
                val = conv_state[pos * SSM_CONV_CHANNELS + ch]
            else:
                val = qkv_mixed[ch]  # input_pos = pos - state_len = k - 3
            s += val * conv_weight[ch, k]
        conv_out[ch] = s
    conv_out_silu = silu(conv_out)

    # Update conv state: last 3 positions of [state(3)|input(1)] = [state[1], state[2], input[0]]
    new_conv_state = np.zeros((SSM_CONV_KERNEL - 1) * SSM_CONV_CHANNELS, dtype=np.float32)
    total = (SSM_CONV_KERNEL - 1) + 1  # state_len + n_tokens = 4
    for i in range(SSM_CONV_KERNEL - 1):
        src_pos = 1 + i  # positions 1, 2, 3 of the concat
        for ch in range(SSM_CONV_CHANNELS):
            if src_pos < SSM_CONV_KERNEL - 1:
                new_conv_state[i * SSM_CONV_CHANNELS + ch] = conv_state[src_pos * SSM_CONV_CHANNELS + ch]
            else:
                new_conv_state[i * SSM_CONV_CHANNELS + ch] = qkv_mixed[ch]

    # QKV split
    qk_size = SSM_D_STATE * SSM_N_GROUP
    q_ssm = conv_out_silu[:qk_size]
    k_ssm = conv_out_silu[qk_size:2*qk_size]
    v_ssm = conv_out_silu[2*qk_size:]

    # L2 norm
    q_norm = np.zeros_like(q_ssm)
    k_norm = np.zeros_like(k_ssm)
    for h in range(SSM_N_GROUP):
        q_norm[h*SSM_D_STATE:(h+1)*SSM_D_STATE] = l2_norm(q_ssm[h*SSM_D_STATE:(h+1)*SSM_D_STATE])
        k_norm[h*SSM_D_STATE:(h+1)*SSM_D_STATE] = l2_norm(k_ssm[h*SSM_D_STATE:(h+1)*SSM_D_STATE])

    # Repeat
    q_rep = np.zeros(SSM_DT_RANK * SSM_D_STATE, dtype=np.float32)
    k_rep = np.zeros(SSM_DT_RANK * SSM_D_STATE, dtype=np.float32)
    for vh in range(SSM_DT_RANK):
        kh = vh * SSM_N_GROUP // SSM_DT_RANK
        q_rep[vh*SSM_D_STATE:(vh+1)*SSM_D_STATE] = q_norm[kh*SSM_D_STATE:(kh+1)*SSM_D_STATE]
        k_rep[vh*SSM_D_STATE:(vh+1)*SSM_D_STATE] = k_norm[kh*SSM_D_STATE:(kh+1)*SSM_D_STATE]

    # Delta-net
    scale = 1.0 / np.sqrt(SSM_D_STATE)
    delta_out = np.zeros(SSM_DT_RANK * SSM_HEAD_V_DIM, dtype=np.float32)
    new_state = rec_state.copy()

    for h in range(SSM_DT_RANK):
        g = np.exp(gate[h])
        new_state[h] *= g

        q_h = q_rep[h*SSM_D_STATE:(h+1)*SSM_D_STATE] * scale
        k_h = k_rep[h*SSM_D_STATE:(h+1)*SSM_D_STATE]
        v_h = v_ssm[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM]

        sk = np.zeros(SSM_HEAD_V_DIM, dtype=np.float32)
        for i in range(SSM_HEAD_V_DIM):
            for j in range(SSM_HEAD_V_DIM):
                sk[i] += new_state[h, j, i] * k_h[j]
        d = beta[h] * (v_h - sk)
        for j in range(SSM_HEAD_V_DIM):
            for i in range(SSM_HEAD_V_DIM):
                new_state[h, j, i] += k_h[j] * d[i]
        for i in range(SSM_HEAD_V_DIM):
            out_val = 0.0
            for j in range(SSM_HEAD_V_DIM):
                out_val += new_state[h, j, i] * q_h[j]
            delta_out[h*SSM_HEAD_V_DIM + i] = out_val

    # Gated RMSNorm
    ssm_norm_w = load_tensor(f, tensors, data_start, "blk.0.ssm_norm.weight")
    gated_out = np.zeros(SSM_D_INNER, dtype=np.float32)
    for h in range(SSM_DT_RANK):
        head_in = delta_out[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM]
        head_z = z[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM]
        normed_h = rms_norm(head_in, ssm_norm_w)
        silu_z = silu(head_z)
        gated_out[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM] = normed_h * silu_z

    # Output projection
    ssm_out_w = load_tensor(f, tensors, data_start, "blk.0.ssm_out.weight")
    ssm_projected = gated_out @ ssm_out_w.T
    hidden_after_attn = hidden + ssm_projected

    # FFN
    post_norm_w = load_tensor(f, tensors, data_start, "blk.0.post_attention_norm.weight")
    ffn_in = rms_norm(hidden_after_attn, post_norm_w)
    ffn_in_t = torch.from_numpy(ffn_in).float()
    ffn_gate_w = torch.from_numpy(load_tensor(f, tensors, data_start, "blk.0.ffn_gate.weight")).float()
    ffn_up_w = torch.from_numpy(load_tensor(f, tensors, data_start, "blk.0.ffn_up.weight")).float()
    ffn_down_w = torch.from_numpy(load_tensor(f, tensors, data_start, "blk.0.ffn_down.weight")).float()
    gate_v = ffn_in_t @ ffn_gate_w.T
    up_v = ffn_in_t @ ffn_up_w.T
    ffn_act = torch.nn.functional.silu(gate_v) * up_v
    ffn_out = (ffn_act @ ffn_down_w.T).numpy()
    hidden_final = hidden_after_attn + ffn_out

    return hidden_final, new_conv_state, new_state

def main():
    print("Loading GGUF...")
    f, tensors, data_start = read_gguf_tensors(GGUF_PATH)

    # Token 0: "The" (760)
    tok_embd = load_tensor(f, tensors, data_start, "token_embd.weight")
    hidden0 = tok_embd[760].copy()
    conv_state = np.zeros((SSM_CONV_KERNEL - 1) * SSM_CONV_CHANNELS, dtype=np.float32)
    rec_state = np.zeros((SSM_DT_RANK, SSM_HEAD_V_DIM, SSM_HEAD_V_DIM), dtype=np.float32)

    print("Processing token 0 (The)...")
    hidden0_out, conv_state, rec_state = process_ssm_layer0(f, tensors, data_start, hidden0, conv_state, rec_state)
    print(f"After layer 0, tok 0: {hidden0_out[:8]}")

    cuda0 = load_bf16_bin("/tmp/hidden_pos0_layer0.bin", N_EMBD)
    print(f"CUDA after layer 0, tok 0: {cuda0[:8]}")
    print(f"Token 0 max diff: {np.max(np.abs(hidden0_out - cuda0)):.6f}")

    # Token 1: "capital" (6511)
    hidden1 = tok_embd[6511].copy()
    print(f"\nProcessing token 1 (capital)...")
    hidden1_out, conv_state2, rec_state2 = process_ssm_layer0(f, tensors, data_start, hidden1, conv_state, rec_state)
    print(f"After layer 0, tok 1: {hidden1_out[:8]}")

    cuda1 = load_bf16_bin("/tmp/hidden_pos1_layer0.bin", N_EMBD)
    print(f"CUDA after layer 0, tok 1: {cuda1[:8]}")

    diff = np.abs(hidden1_out - cuda1)
    print(f"Token 1 max diff: {np.max(diff):.6f}")
    print(f"Token 1 mean diff: {np.mean(diff):.6f}")

    worst = np.argsort(diff)[-5:]
    for idx in worst:
        print(f"  [{idx}] python={hidden1_out[idx]:.6f}  cuda={cuda1[idx]:.6f}  diff={diff[idx]:.6f}")

    f.close()

if __name__ == "__main__":
    main()
