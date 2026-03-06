#!/usr/bin/env python3
"""Full 32-layer forward pass in Python for last token (position 4).
Compare against CUDA hidden states at each layer boundary."""
import struct
import numpy as np
import torch
import sys

GGUF_PATH = "/home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-BF16.gguf"

N_EMBD = 4096
N_HEAD = 16
N_HEAD_KV = 4
HEAD_DIM = 256
N_FF = 12288
RMS_NORM_EPS = 1e-6
ROPE_DIM = 64
ROPE_FREQ_BASE = 10000000.0
ATTN_SCALE = 1.0 / 16.0

SSM_D_INNER = 4096
SSM_D_STATE = 128
SSM_N_GROUP = 16
SSM_DT_RANK = 32
SSM_HEAD_V_DIM = SSM_D_INNER // SSM_DT_RANK  # 128
SSM_CONV_KERNEL = 4
SSM_CONV_CHANNELS = SSM_D_INNER + 2 * SSM_N_GROUP * SSM_D_STATE  # 8192

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

def silu(x): return x / (1 + np.exp(-x))
def softplus(x): return np.where(x > 20, x, np.log1p(np.exp(x)))
def l2_norm(x, eps=RMS_NORM_EPS): return x / (np.sqrt(np.sum(x**2)) + eps)

def is_attention(layer_idx):
    return (layer_idx + 1) % 4 == 0

def process_ffn(f, tensors, data_start, blk, hidden_after_attn):
    """Common FFN processing."""
    post_norm_w = load_tensor(f, tensors, data_start, f"blk.{blk}.post_attention_norm.weight")
    ffn_in = rms_norm(hidden_after_attn, post_norm_w)
    ffn_in_t = torch.from_numpy(ffn_in).float()
    ffn_gate_w = torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.ffn_gate.weight")).float()
    ffn_up_w = torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.ffn_up.weight")).float()
    ffn_down_w = torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.ffn_down.weight")).float()
    gate_v = ffn_in_t @ ffn_gate_w.T
    up_v = ffn_in_t @ ffn_up_w.T
    ffn_act = torch.nn.functional.silu(gate_v) * up_v
    ffn_out = (ffn_act @ ffn_down_w.T).numpy()
    return hidden_after_attn + ffn_out

def process_attn_layer(f, tensors, data_start, blk, hidden, pos, k_cache_all_pos, v_cache_all_pos):
    """Process one attention layer. Returns (new_hidden, k_for_cache, v_for_cache)."""
    attn_norm = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_norm.weight")
    wq = torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.attn_q.weight")).float()
    wk = torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.attn_k.weight")).float()
    wv = torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.attn_v.weight")).float()
    q_norm_w = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_q_norm.weight")
    k_norm_w = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_k_norm.weight")
    wo = torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.attn_output.weight")).float()

    normed = rms_norm(hidden, attn_norm)
    normed_t = torch.from_numpy(normed).float()

    # Q + Gate
    qg = (normed_t @ wq.T).numpy()
    q_all = np.zeros(N_HEAD * HEAD_DIM, dtype=np.float32)
    gate_all = np.zeros(N_HEAD * HEAD_DIM, dtype=np.float32)
    for h in range(N_HEAD):
        q_all[h*HEAD_DIM:(h+1)*HEAD_DIM] = qg[h*HEAD_DIM*2:h*HEAD_DIM*2+HEAD_DIM]
        gate_all[h*HEAD_DIM:(h+1)*HEAD_DIM] = qg[h*HEAD_DIM*2+HEAD_DIM:h*HEAD_DIM*2+2*HEAD_DIM]

    # K, V
    k_all = (normed_t @ wk.T).numpy()
    v_all = (normed_t @ wv.T).numpy()

    # Q, K norm
    q_heads = rms_norm_head(q_all.reshape(N_HEAD, HEAD_DIM), q_norm_w)
    k_heads = rms_norm_head(k_all.reshape(N_HEAD_KV, HEAD_DIM), k_norm_w)

    # RoPE
    q_heads = apply_rope(q_heads, pos, N_HEAD, HEAD_DIM, ROPE_DIM, ROPE_FREQ_BASE)
    k_heads = apply_rope(k_heads, pos, N_HEAD_KV, HEAD_DIM, ROPE_DIM, ROPE_FREQ_BASE)

    v_heads = v_all.reshape(N_HEAD_KV, HEAD_DIM)

    # Attention
    kv_len = len(k_cache_all_pos) + 1  # previous + current
    all_k = k_cache_all_pos + [k_heads]
    all_v = v_cache_all_pos + [v_heads]

    attn_out = np.zeros((N_HEAD, HEAD_DIM), dtype=np.float32)
    for h in range(N_HEAD):
        kv_h = h // (N_HEAD // N_HEAD_KV)
        scores = np.zeros(kv_len, dtype=np.float32)
        for kv_pos in range(kv_len):
            scores[kv_pos] = np.dot(q_heads[h], all_k[kv_pos][kv_h]) * ATTN_SCALE
        scores = scores - np.max(scores)
        scores = np.exp(scores)
        scores = scores / np.sum(scores)
        for kv_pos in range(kv_len):
            attn_out[h] += scores[kv_pos] * all_v[kv_pos][kv_h]

    # Gate
    sig_gate = 1.0 / (1.0 + np.exp(-gate_all))
    gated = attn_out.flatten() * sig_gate

    # Output projection
    projected = (torch.from_numpy(gated).float() @ wo.T).numpy()
    hidden_after_attn = hidden + projected

    # FFN
    final = process_ffn(f, tensors, data_start, blk, hidden_after_attn)

    return final, k_heads, v_heads

def process_ssm_layer(f, tensors, data_start, blk, hidden, conv_state, rec_state):
    """Process one SSM layer. Returns (new_hidden, new_conv_state, new_rec_state)."""
    attn_norm = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_norm.weight")
    normed = rms_norm(hidden, attn_norm)
    normed_t = torch.from_numpy(normed).float()

    wqkv = torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.attn_qkv.weight")).float()
    qkv_mixed = (normed_t @ wqkv.T).numpy()

    wgate = torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.attn_gate.weight")).float()
    z = (normed_t @ wgate.T).numpy()

    ssm_alpha_w = torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_alpha.weight")).float()
    alpha = (normed_t @ ssm_alpha_w.T).numpy()

    ssm_beta_w = torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_beta.weight")).float()
    beta = 1 / (1 + np.exp(-(normed_t @ ssm_beta_w.T).numpy()))

    ssm_dt_bias = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_dt.bias")
    ssm_a = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_a")
    gate = softplus(alpha + ssm_dt_bias) * ssm_a

    conv_weight = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_conv1d.weight")

    # Conv1d + SiLU
    conv_out = np.zeros(SSM_CONV_CHANNELS, dtype=np.float32)
    state_len = SSM_CONV_KERNEL - 1
    for ch in range(SSM_CONV_CHANNELS):
        s = 0.0
        for k in range(SSM_CONV_KERNEL):
            pos = 0 + k
            if pos < state_len:
                val = conv_state[pos * SSM_CONV_CHANNELS + ch]
            else:
                val = qkv_mixed[ch]
            s += val * conv_weight[ch, k]
        conv_out[ch] = s
    conv_out_silu = silu(conv_out)

    # Update conv state
    new_conv_state = np.zeros((SSM_CONV_KERNEL - 1) * SSM_CONV_CHANNELS, dtype=np.float32)
    for i in range(SSM_CONV_KERNEL - 1):
        src_pos = 1 + i
        for ch in range(SSM_CONV_CHANNELS):
            if src_pos < state_len:
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

        sk = new_state[h].T @ k_h  # [128]
        d = beta[h] * (v_h - sk)
        new_state[h] += np.outer(k_h, d)
        delta_out[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM] = new_state[h].T @ q_h

    # Gated RMSNorm
    ssm_norm_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_norm.weight")
    gated_out = np.zeros(SSM_D_INNER, dtype=np.float32)
    for h in range(SSM_DT_RANK):
        head_in = delta_out[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM]
        head_z = z[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM]
        normed_h = rms_norm(head_in, ssm_norm_w)
        silu_z = silu(head_z)
        gated_out[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM] = normed_h * silu_z

    # Output projection
    ssm_out_w = torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_out.weight")).float()
    ssm_projected = (torch.from_numpy(gated_out).float() @ ssm_out_w.T).numpy()
    hidden_after_attn = hidden + ssm_projected

    # FFN
    final = process_ffn(f, tensors, data_start, blk, hidden_after_attn)

    return final, new_conv_state, new_state

def main():
    # Process only position 4 (last token) through all 32 layers
    # Use CUDA hidden states for earlier positions' KV cache building

    target_pos = 4
    n_layers = int(sys.argv[1]) if len(sys.argv) > 1 else 32

    print(f"Loading GGUF... (processing {n_layers} layers)")
    f, tensors, data_start = read_gguf_tensors(GGUF_PATH)

    tok_embd = load_tensor(f, tensors, data_start, "token_embd.weight")
    tokens = [760, 6511, 314, 9338, 369]  # "The capital of France is"

    # For attention layers, we need KV cache from all positions.
    # We'll process ALL positions through ALL layers to get correct KV caches.
    # This is expensive but necessary for correctness.

    # Initialize states for all positions
    hiddens = [tok_embd[t].copy() for t in tokens]

    # SSM states: [24 layers, ...]
    conv_states = [np.zeros((SSM_CONV_KERNEL - 1) * SSM_CONV_CHANNELS, dtype=np.float32) for _ in range(24)]
    rec_states = [np.zeros((SSM_DT_RANK, SSM_HEAD_V_DIM, SSM_HEAD_V_DIM), dtype=np.float32) for _ in range(24)]

    # KV caches for attention layers: [8 layers, list of (k, v) per position]
    attn_k_caches = [[] for _ in range(8)]
    attn_v_caches = [[] for _ in range(8)]

    for il in range(n_layers):
        ltype = "ATN" if is_attention(il) else "SSM"

        if is_attention(il):
            attn_idx = il // 4
        else:
            ssm_idx = il - (il // 4 + 1) if il > 2 else il
            # Calculate SSM index properly
            ssm_count = 0
            for j in range(il):
                if not is_attention(j):
                    ssm_count += 1
            ssm_idx = ssm_count

        new_hiddens = []
        for pos in range(5):
            if is_attention(il):
                attn_idx = il // 4
                # Use KV cache from previous positions at this layer
                k_prev = attn_k_caches[attn_idx][:pos] if pos > 0 else []
                v_prev = attn_v_caches[attn_idx][:pos] if pos > 0 else []

                new_h, k_new, v_new = process_attn_layer(
                    f, tensors, data_start, il, hiddens[pos], pos,
                    k_prev, v_prev)

                if pos == 0:
                    attn_k_caches[attn_idx] = [k_new]
                    attn_v_caches[attn_idx] = [v_new]
                else:
                    if len(attn_k_caches[attn_idx]) <= pos:
                        attn_k_caches[attn_idx].append(k_new)
                        attn_v_caches[attn_idx].append(v_new)
                    else:
                        attn_k_caches[attn_idx][pos] = k_new
                        attn_v_caches[attn_idx][pos] = v_new
            else:
                ssm_count = 0
                for j in range(il):
                    if not is_attention(j):
                        ssm_count += 1
                ssm_idx = ssm_count

                new_h, new_conv, new_rec = process_ssm_layer(
                    f, tensors, data_start, il, hiddens[pos],
                    conv_states[ssm_idx], rec_states[ssm_idx])
                conv_states[ssm_idx] = new_conv
                rec_states[ssm_idx] = new_rec

            new_hiddens.append(new_h)

        hiddens = new_hiddens

        # Compare position 4 against CUDA
        cuda_h = load_bf16_bin(f"/tmp/hidden_pos4_layer{il}.bin", N_EMBD)
        py_h = hiddens[4]
        diff = np.abs(py_h - cuda_h)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        status = "OK" if max_diff < 1.0 else "DIVERGED" if max_diff > 5.0 else "DRIFTING"
        print(f"  L{il:2d} ({ltype}): max_diff={max_diff:.4f}  mean_diff={mean_diff:.6f}  [{status}]  py_L2={np.sqrt(np.sum(py_h**2)):.2f}  cuda_L2={np.sqrt(np.sum(cuda_h**2)):.2f}")

        if max_diff > 5.0:
            worst = np.argsort(diff)[-3:]
            for idx in worst:
                print(f"         [{idx}] py={py_h[idx]:.4f}  cuda={cuda_h[idx]:.4f}")

    f.close()

if __name__ == "__main__":
    main()
