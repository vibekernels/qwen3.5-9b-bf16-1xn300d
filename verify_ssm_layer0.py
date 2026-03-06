#!/usr/bin/env python3
"""Verify SSM layer 0 for token 0 and token 1, comparing against CUDA dumps."""
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

def rms_norm(x, w, eps=RMS_NORM_EPS): return (x / np.sqrt(np.mean(x**2) + eps)) * w
def silu(x): return x / (1 + np.exp(-x))
def softplus(x): return np.where(x > 20, x, np.log1p(np.exp(x)))
def l2_norm(x, eps=RMS_NORM_EPS): return x / (np.sqrt(np.sum(x**2)) + eps)

def process_ssm_detailed(f, tensors, data_start, blk, hidden, conv_state, rec_state):
    """Process one SSM layer with detailed output for debugging."""
    norm_w = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_norm.weight")
    normed = rms_norm(hidden, norm_w)

    # Use torch for matmuls to match cuBLAS precision
    nt = torch.from_numpy(normed).float()

    # QKV projection (f32 output in ggml)
    qkv = (nt @ torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.attn_qkv.weight")).float().T).numpy()
    z = (nt @ torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.attn_gate.weight")).float().T).numpy()
    alpha = (nt @ torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_alpha.weight")).float().T).numpy()
    beta_raw = (nt @ torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_beta.weight")).float().T).numpy()
    beta = 1 / (1 + np.exp(-beta_raw))

    dt_bias = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_dt.bias")
    ssm_a = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_a")
    gate = softplus(alpha + dt_bias) * ssm_a

    conv_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_conv1d.weight")

    # Conv1d + SiLU
    state_len = SSM_CONV_KERNEL - 1
    conv_out = np.zeros(SSM_CONV_CHANNELS, dtype=np.float32)
    for k in range(SSM_CONV_KERNEL):
        if k < state_len:
            vals = conv_state[k * SSM_CONV_CHANNELS:(k+1) * SSM_CONV_CHANNELS]
        else:
            vals = qkv
        conv_out += vals * conv_w[:, k]
    conv_out_pre_silu = conv_out.copy()
    conv_out = silu(conv_out)

    # Update conv state
    new_conv = np.zeros((SSM_CONV_KERNEL - 1) * SSM_CONV_CHANNELS, dtype=np.float32)
    for i in range(SSM_CONV_KERNEL - 1):
        src = 1 + i
        if src < state_len:
            new_conv[i*SSM_CONV_CHANNELS:(i+1)*SSM_CONV_CHANNELS] = conv_state[src*SSM_CONV_CHANNELS:(src+1)*SSM_CONV_CHANNELS]
        else:
            new_conv[i*SSM_CONV_CHANNELS:(i+1)*SSM_CONV_CHANNELS] = qkv

    # Q/K/V split, L2 norm, repeat
    qk_size = SSM_D_STATE * SSM_N_GROUP
    q = conv_out[:qk_size].copy(); k = conv_out[qk_size:2*qk_size].copy(); v = conv_out[2*qk_size:].copy()
    for h in range(SSM_N_GROUP):
        q[h*SSM_D_STATE:(h+1)*SSM_D_STATE] = l2_norm(q[h*SSM_D_STATE:(h+1)*SSM_D_STATE])
        k[h*SSM_D_STATE:(h+1)*SSM_D_STATE] = l2_norm(k[h*SSM_D_STATE:(h+1)*SSM_D_STATE])

    q_rep = np.zeros(SSM_DT_RANK * SSM_D_STATE); k_rep = np.zeros(SSM_DT_RANK * SSM_D_STATE)
    for vh in range(SSM_DT_RANK):
        kh = vh * SSM_N_GROUP // SSM_DT_RANK
        q_rep[vh*SSM_D_STATE:(vh+1)*SSM_D_STATE] = q[kh*SSM_D_STATE:(kh+1)*SSM_D_STATE]
        k_rep[vh*SSM_D_STATE:(vh+1)*SSM_D_STATE] = k[kh*SSM_D_STATE:(kh+1)*SSM_D_STATE]

    # Delta-net
    scale = 1.0 / np.sqrt(SSM_D_STATE)
    delta_out = np.zeros(SSM_DT_RANK * SSM_HEAD_V_DIM)
    new_rec = rec_state.copy()
    for h in range(SSM_DT_RANK):
        g = np.exp(gate[h]); new_rec[h] *= g
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

    # Output projection
    proj = (torch.from_numpy(gated).float() @ torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_out.weight")).float().T).numpy()
    after_attn = hidden + proj

    # FFN
    post_norm = load_tensor(f, tensors, data_start, f"blk.{blk}.post_attention_norm.weight")
    ffn_in = rms_norm(after_attn, post_norm)
    fft = torch.from_numpy(ffn_in).float()
    gv = fft @ torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.ffn_gate.weight")).float().T
    uv = fft @ torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.ffn_up.weight")).float().T
    ffn_out = (torch.nn.functional.silu(gv) * uv @ torch.from_numpy(load_tensor(f, tensors, data_start, f"blk.{blk}.ffn_down.weight")).float().T).numpy()
    final = after_attn + ffn_out

    return final, new_conv, new_rec, {
        'normed': normed, 'qkv': qkv, 'z': z, 'alpha': alpha, 'beta': beta,
        'gate': gate, 'conv_out': conv_out, 'q': q, 'k': k, 'v': v,
        'delta_out': delta_out, 'gated': gated, 'proj': proj, 'after_attn': after_attn
    }

# Main
f, tensors, data_start = read_gguf_tensors(GGUF_PATH)
tok_embd = load_tensor(f, tensors, data_start, "token_embd.weight")
tokens = [760, 6511, 314, 9338, 369]  # "The capital of France is"

# Process each token through layer 0 only
conv_state = np.zeros((SSM_CONV_KERNEL - 1) * SSM_CONV_CHANNELS, dtype=np.float32)
rec_state = np.zeros((SSM_DT_RANK, SSM_HEAD_V_DIM, SSM_HEAD_V_DIM), dtype=np.float32)

for pos, tok in enumerate(tokens):
    hidden = tok_embd[tok].astype(np.float32)
    new_h, conv_state, rec_state, details = process_ssm_detailed(f, tensors, data_start, 0, hidden, conv_state, rec_state)

    # Compare with CUDA dump
    cuda_path = f"/tmp/hidden_f32_pos{pos}_layer0.bin"
    if os.path.exists(cuda_path):
        cuda_h = np.fromfile(cuda_path, dtype=np.float32)[:N_EMBD]
        diff = np.abs(new_h - cuda_h)
        print(f"\nPos {pos} (tok={tok}), Layer 0:")
        print(f"  Python L2={np.sqrt(np.sum(new_h**2)):.4f}  CUDA L2={np.sqrt(np.sum(cuda_h**2)):.4f}")
        print(f"  max_diff={np.max(diff):.6f}  mean_diff={np.mean(diff):.6f}")

        # Show worst elements
        worst = np.argsort(diff)[-3:]
        for idx in worst:
            print(f"  [{idx}] py={new_h[idx]:.6f}  cuda={cuda_h[idx]:.6f}  diff={diff[idx]:.6f}")

        # Also compare intermediate values
        print(f"  qkv L2: py={np.sqrt(np.sum(details['qkv']**2)):.4f}")
        print(f"  conv_out L2: {np.sqrt(np.sum(details['conv_out']**2)):.4f}")
        print(f"  delta_out L2: {np.sqrt(np.sum(details['delta_out']**2)):.4f}")
        print(f"  proj L2: {np.sqrt(np.sum(details['proj']**2)):.4f}")
    else:
        print(f"\nPos {pos} (tok={tok}): No CUDA dump at {cuda_path}")
        print(f"  Python L2={np.sqrt(np.sum(new_h**2)):.4f}")
