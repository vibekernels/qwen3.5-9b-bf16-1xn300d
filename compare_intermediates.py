#!/usr/bin/env python3
"""Compare SSM layer 0 intermediates between CUDA dumps and Python reference."""
import struct, numpy as np, torch, os, sys

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

def compare(name, py_val, cuda_path, n=None):
    """Compare Python value with CUDA dump."""
    if n is None: n = len(py_val)
    if os.path.exists(cuda_path):
        cuda_val = np.fromfile(cuda_path, dtype=np.float32)[:n]
        diff = np.abs(py_val.flatten()[:n] - cuda_val)
        print(f"  {name:20s}: max_diff={np.max(diff):.6f} mean_diff={np.mean(diff):.6f} "
              f"py_L2={np.sqrt(np.sum(py_val.flatten()[:n]**2)):.4f} cuda_L2={np.sqrt(np.sum(cuda_val**2)):.4f}")
        if np.max(diff) > 0.1:
            worst = np.argsort(diff)[-3:]
            for idx in worst:
                print(f"    [{idx}] py={py_val.flatten()[idx]:.6f} cuda={cuda_val[idx]:.6f} diff={diff[idx]:.6f}")
        return cuda_val
    else:
        print(f"  {name:20s}: NO CUDA DUMP at {cuda_path}")
        return None

# Main
f, tensors, data_start = read_gguf_tensors(GGUF_PATH)
tok_embd = load_tensor(f, tensors, data_start, "token_embd.weight")
tokens = [760, 6511, 314, 9338, 369]

conv_state = np.zeros((SSM_CONV_KERNEL - 1) * SSM_CONV_CHANNELS, dtype=np.float32)
rec_state = np.zeros((SSM_DT_RANK, SSM_D_STATE, SSM_HEAD_V_DIM), dtype=np.float32)

blk = 0
norm_w = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_norm.weight")
qkv_w = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_qkv.weight")
gate_w = load_tensor(f, tensors, data_start, f"blk.{blk}.attn_gate.weight")
alpha_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_alpha.weight")
beta_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_beta.weight")
dt_bias = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_dt.bias")
ssm_a = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_a")
conv_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_conv1d.weight")
ssm_norm_w = load_tensor(f, tensors, data_start, f"blk.{blk}.ssm_norm.weight")

# To match CUDA bf16 GEMM precision, truncate normed to bf16 before matmul
def to_bf16(x):
    """Truncate f32 to bf16 precision (matching CUDA bf16 GEMM input)."""
    t = torch.from_numpy(x).float()
    return t.bfloat16().float().numpy()

for pos in range(2):  # Only compare positions 0 and 1
    tok = tokens[pos]
    hidden = tok_embd[tok].astype(np.float32)
    print(f"\n=== Position {pos} (tok={tok}) ===")

    compare("hidden_in", hidden, f"/tmp/ssm_hidden_in_pos{pos}_layer0.bin", N_EMBD)

    # RMSNorm
    normed = rms_norm(hidden, norm_w)
    # CUDA outputs bf16, so truncate for comparison
    normed_bf16 = to_bf16(normed)
    compare("normed", normed_bf16, f"/tmp/ssm_normed_pos{pos}_layer0.bin", N_EMBD)

    # QKV projection (using bf16 inputs to match CUDA)
    nt = torch.from_numpy(normed_bf16).bfloat16()  # bf16 input to GEMM
    qkv = (nt.float() @ torch.from_numpy(qkv_w).float().T).numpy()
    compare("qkv_proj", qkv, f"/tmp/ssm_qkv_proj_pos{pos}_layer0.bin", SSM_CONV_CHANNELS)

    z = (nt.float() @ torch.from_numpy(gate_w).float().T).numpy()
    compare("z_gate", to_bf16(z), f"/tmp/ssm_z_gate_pos{pos}_layer0.bin", SSM_D_INNER)

    alpha = (nt.float() @ torch.from_numpy(alpha_w).float().T).numpy()
    compare("alpha", alpha, f"/tmp/ssm_alpha_pos{pos}_layer0.bin", SSM_DT_RANK)

    beta_raw = (nt.float() @ torch.from_numpy(beta_w).float().T).numpy()
    compare("beta_raw", beta_raw, f"/tmp/ssm_beta_raw_pos{pos}_layer0.bin", SSM_DT_RANK)

    gate = softplus(alpha + dt_bias) * ssm_a
    compare("gate", gate, f"/tmp/ssm_gate_pos{pos}_layer0.bin", SSM_DT_RANK)

    beta = 1 / (1 + np.exp(-beta_raw))
    compare("beta", beta, f"/tmp/ssm_beta_pos{pos}_layer0.bin", SSM_DT_RANK)

    # Conv state before
    compare("conv_state_before", conv_state, f"/tmp/ssm_conv_state_before_pos{pos}_layer0.bin", (SSM_CONV_KERNEL-1)*SSM_CONV_CHANNELS)

    # Conv1d + SiLU
    state_len = SSM_CONV_KERNEL - 1
    conv_out = np.zeros(SSM_CONV_CHANNELS, dtype=np.float32)
    for k in range(SSM_CONV_KERNEL):
        if k < state_len:
            vals = conv_state[k * SSM_CONV_CHANNELS:(k+1) * SSM_CONV_CHANNELS]
        else:
            vals = qkv[:SSM_CONV_CHANNELS]
        conv_out += vals * conv_w[:, k]
    conv_out = silu(conv_out)
    compare("conv_out", conv_out, f"/tmp/ssm_conv_out_pos{pos}_layer0.bin", SSM_CONV_CHANNELS)

    # Update conv state
    new_conv = np.zeros((SSM_CONV_KERNEL - 1) * SSM_CONV_CHANNELS, dtype=np.float32)
    for i in range(SSM_CONV_KERNEL - 1):
        src = 1 + i
        if src < state_len:
            new_conv[i*SSM_CONV_CHANNELS:(i+1)*SSM_CONV_CHANNELS] = conv_state[src*SSM_CONV_CHANNELS:(src+1)*SSM_CONV_CHANNELS]
        else:
            new_conv[i*SSM_CONV_CHANNELS:(i+1)*SSM_CONV_CHANNELS] = qkv[:SSM_CONV_CHANNELS]
    conv_state = new_conv

    # Q/K/V split
    qk_size = SSM_D_STATE * SSM_N_GROUP
    q = conv_out[:qk_size].copy(); k = conv_out[qk_size:2*qk_size].copy(); v = conv_out[2*qk_size:].copy()
    for h in range(SSM_N_GROUP):
        q[h*SSM_D_STATE:(h+1)*SSM_D_STATE] = l2_norm(q[h*SSM_D_STATE:(h+1)*SSM_D_STATE])
        k[h*SSM_D_STATE:(h+1)*SSM_D_STATE] = l2_norm(k[h*SSM_D_STATE:(h+1)*SSM_D_STATE])
    compare("q_norm", q, f"/tmp/ssm_q_norm_pos{pos}_layer0.bin", qk_size)
    compare("k_norm", k, f"/tmp/ssm_k_norm_pos{pos}_layer0.bin", qk_size)
    compare("v", v, f"/tmp/ssm_v_pos{pos}_layer0.bin", SSM_D_INNER)

    # Repeat heads
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
    rec_state = new_rec
    compare("delta_out", delta_out, f"/tmp/ssm_delta_out_pos{pos}_layer0.bin", SSM_DT_RANK * SSM_HEAD_V_DIM)

    # Gated RMSNorm
    gated = np.zeros(SSM_D_INNER)
    for h in range(SSM_DT_RANK):
        hi = delta_out[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM]
        hz = z[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM]
        gated[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM] = rms_norm(hi, ssm_norm_w) * silu(hz)
    compare("gated_out", to_bf16(gated), f"/tmp/ssm_gated_out_pos{pos}_layer0.bin", SSM_D_INNER)

print("\nDone!")
