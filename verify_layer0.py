#!/usr/bin/env python3
"""Verify layer 0 (SSM) computation against our CUDA implementation."""
import struct
import numpy as np
import torch

GGUF_PATH = "/home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-BF16.gguf"

# Model config
N_EMBD = 4096
N_FF = 12288
SSM_D_INNER = 4096
SSM_D_STATE = 128
SSM_N_GROUP = 16
SSM_DT_RANK = 32
SSM_HEAD_V_DIM = SSM_D_INNER // SSM_DT_RANK  # 128
SSM_CONV_KERNEL = 4
SSM_CONV_CHANNELS = SSM_D_INNER + 2 * SSM_N_GROUP * SSM_D_STATE  # 8192
RMS_NORM_EPS = 1e-6

def read_gguf_tensors(path):
    """Read all tensor info and data from GGUF file."""
    f = open(path, 'rb')
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]

    def read_string():
        l = struct.unpack('<Q', f.read(8))[0]
        return f.read(l).decode('utf-8')
    def skip_value(vtype):
        sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 8:0, 10:8, 11:8, 12:8}
        if vtype == 8: read_string()
        elif vtype == 9:
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            for i in range(arr_len): skip_value(arr_type)
        else: f.read(sizes[vtype])

    for i in range(n_kv):
        key = read_string()
        vtype = struct.unpack('<I', f.read(4))[0]
        skip_value(vtype)

    tensors = {}
    for i in range(n_tensors):
        name = read_string()
        n_dims = struct.unpack('<I', f.read(4))[0]
        ne = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        dtype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        tensors[name] = (ne, dtype, offset)

    data_start = f.tell()
    data_start = (data_start + 31) & ~31
    return f, tensors, data_start

def load_tensor(f, tensors, data_start, name):
    """Load a tensor from the GGUF file as float32 numpy array."""
    ne, dtype, offset = tensors[name]
    n_elems = 1
    for d in ne:
        n_elems *= d

    f.seek(data_start + offset)
    if dtype == 30:  # BF16
        raw = f.read(n_elems * 2)
        arr = np.frombuffer(raw, dtype=np.uint16)
        # Convert bf16 to f32
        f32 = np.zeros(n_elems, dtype=np.float32)
        for i in range(n_elems):
            bits = int(arr[i]) << 16
            f32[i] = struct.unpack('<f', struct.pack('<I', bits))[0]
        return f32.reshape(ne[::-1]) if len(ne) > 1 else f32  # gguf is row-major with ne[0] as inner
    elif dtype == 0:  # F32
        raw = f.read(n_elems * 4)
        arr = np.frombuffer(raw, dtype=np.float32).copy()
        return arr.reshape(ne[::-1]) if len(ne) > 1 else arr
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

def rms_norm(x, weight, eps=RMS_NORM_EPS):
    """RMS normalization."""
    rms = np.sqrt(np.mean(x**2) + eps)
    return (x / rms) * weight

def silu(x):
    return x / (1 + np.exp(-x))

def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(x)))

def l2_norm(x, eps=RMS_NORM_EPS):
    """L2 normalize a vector."""
    return x / (np.sqrt(np.sum(x**2)) + eps)

def main():
    print("Loading GGUF tensors...")
    f, tensors, data_start = read_gguf_tensors(GGUF_PATH)

    # Load embedding for token 760
    tok_embd = load_tensor(f, tensors, data_start, "token_embd.weight")
    # tok_embd shape: [n_vocab, n_embd] = [248320, 4096]
    print(f"tok_embd shape: {tok_embd.shape}")

    hidden = tok_embd[760].copy()  # [4096]
    print(f"Embedding[760][:8]: {hidden[:8]}")

    # Load CUDA result
    cuda_embed = np.fromfile("/tmp/hidden_after_embed.bin", dtype=np.uint16)
    cuda_embed_f32 = np.zeros(N_EMBD, dtype=np.float32)
    for i in range(N_EMBD):
        bits = int(cuda_embed[i]) << 16
        cuda_embed_f32[i] = struct.unpack('<f', struct.pack('<I', bits))[0]
    print(f"CUDA embed[:8]:     {cuda_embed_f32[:8]}")
    print(f"Embed max diff: {np.max(np.abs(hidden - cuda_embed_f32)):.6f}")

    # === Layer 0: SSM (delta-net) ===
    print("\n=== Layer 0 (SSM) ===")

    # 1. RMSNorm
    attn_norm = load_tensor(f, tensors, data_start, "blk.0.attn_norm.weight")
    print(f"attn_norm[:8]: {attn_norm[:8]}")
    normed = rms_norm(hidden, attn_norm)
    print(f"After norm[:8]: {normed[:8]}")

    # 2. QKV mixed projection: [4096] -> [8192]
    wqkv = load_tensor(f, tensors, data_start, "blk.0.attn_qkv.weight")
    print(f"wqkv shape: {wqkv.shape}")  # [8192, 4096] in row-major (ne=[4096, 8192])
    qkv_mixed = normed @ wqkv.T  # [4096] @ [4096, 8192] = [8192]
    print(f"qkv_mixed[:8]: {qkv_mixed[:8]}")

    # 3. Gate Z projection: [4096] -> [4096]
    wgate = load_tensor(f, tensors, data_start, "blk.0.attn_gate.weight")
    z = normed @ wgate.T  # [4096]
    print(f"z[:8]: {z[:8]}")

    # 4. Alpha: [4096] -> [32]
    ssm_alpha = load_tensor(f, tensors, data_start, "blk.0.ssm_alpha.weight")
    print(f"ssm_alpha shape: {ssm_alpha.shape}")  # [32, 4096]
    alpha = normed @ ssm_alpha.T  # [32]
    print(f"alpha[:8]: {alpha[:8]}")

    # 5. Beta: [4096] -> [32], then sigmoid
    ssm_beta = load_tensor(f, tensors, data_start, "blk.0.ssm_beta.weight")
    beta_raw = normed @ ssm_beta.T  # [32]
    beta = 1 / (1 + np.exp(-beta_raw))
    print(f"beta[:8]: {beta[:8]}")

    # 6. Gate computation
    ssm_dt_bias = load_tensor(f, tensors, data_start, "blk.0.ssm_dt.bias")
    ssm_a = load_tensor(f, tensors, data_start, "blk.0.ssm_a")
    gate = softplus(alpha + ssm_dt_bias) * ssm_a
    print(f"gate[:8]: {gate[:8]}")
    print(f"exp(gate)[:8]: {np.exp(gate[:8])}")

    # 7. Conv1d + SiLU (first token, conv_state is all zeros)
    conv_weight = load_tensor(f, tensors, data_start, "blk.0.ssm_conv1d.weight")
    print(f"conv_weight shape: {conv_weight.shape}")  # [8192, 4] in our row-major (ne=[4, 8192])

    # For first token with zero state: only the last kernel position matters
    # concat = [zeros(3, 8192), input(1, 8192)]
    # For token 0: conv picks up position 3 (the current input)
    # conv_out[ch] = sum_k input_concat[token+k, ch] * weight[ch, k]
    # With token=0: positions 0,1,2 are state (zeros), position 3 is input
    conv_out = np.zeros(SSM_CONV_CHANNELS, dtype=np.float32)
    for ch in range(SSM_CONV_CHANNELS):
        # Only k=3 contributes (k=0,1,2 multiply zero state)
        conv_out[ch] = qkv_mixed[ch] * conv_weight[ch, 3]
    conv_out_silu = silu(conv_out)
    print(f"conv_out_silu[:8]: {conv_out_silu[:8]}")

    # 8. Split into Q, K, V
    qk_size = SSM_D_STATE * SSM_N_GROUP  # 128 * 16 = 2048
    q_ssm = conv_out_silu[:qk_size]       # [2048]
    k_ssm = conv_out_silu[qk_size:2*qk_size]  # [2048]
    v_ssm = conv_out_silu[2*qk_size:]     # [4096]
    print(f"q_ssm[:8]: {q_ssm[:8]}")
    print(f"k_ssm[:8]: {k_ssm[:8]}")
    print(f"v_ssm[:8]: {v_ssm[:8]}")

    # 9. L2 normalize Q and K (per head of size 128)
    q_norm = np.zeros_like(q_ssm)
    k_norm = np.zeros_like(k_ssm)
    for h in range(SSM_N_GROUP):
        q_norm[h*SSM_D_STATE:(h+1)*SSM_D_STATE] = l2_norm(q_ssm[h*SSM_D_STATE:(h+1)*SSM_D_STATE])
        k_norm[h*SSM_D_STATE:(h+1)*SSM_D_STATE] = l2_norm(k_ssm[h*SSM_D_STATE:(h+1)*SSM_D_STATE])
    print(f"q_norm[:8]: {q_norm[:8]}")

    # 10. Repeat Q, K from 16 to 32 heads
    q_rep = np.zeros(SSM_DT_RANK * SSM_D_STATE, dtype=np.float32)
    k_rep = np.zeros(SSM_DT_RANK * SSM_D_STATE, dtype=np.float32)
    for vh in range(SSM_DT_RANK):
        kh = vh * SSM_N_GROUP // SSM_DT_RANK
        q_rep[vh*SSM_D_STATE:(vh+1)*SSM_D_STATE] = q_norm[kh*SSM_D_STATE:(kh+1)*SSM_D_STATE]
        k_rep[vh*SSM_D_STATE:(vh+1)*SSM_D_STATE] = k_norm[kh*SSM_D_STATE:(kh+1)*SSM_D_STATE]

    # 11. Delta-net autoregressive (state starts at zero)
    state = np.zeros((SSM_DT_RANK, SSM_HEAD_V_DIM, SSM_HEAD_V_DIM), dtype=np.float32)
    scale = 1.0 / np.sqrt(SSM_D_STATE)
    delta_out = np.zeros(SSM_DT_RANK * SSM_HEAD_V_DIM, dtype=np.float32)

    for h in range(SSM_DT_RANK):
        g = np.exp(gate[h])
        state[h] *= g  # decay (no-op since state is zero)

        q_h = q_rep[h*SSM_D_STATE:(h+1)*SSM_D_STATE] * scale
        k_h = k_rep[h*SSM_D_STATE:(h+1)*SSM_D_STATE]
        v_h = v_ssm[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM]

        # sk = state^T @ k (state is zero, so sk = 0)
        sk = np.zeros(SSM_HEAD_V_DIM, dtype=np.float32)
        for i in range(SSM_HEAD_V_DIM):
            for j in range(SSM_HEAD_V_DIM):
                sk[i] += state[h, j, i] * k_h[j]

        # delta = beta * (v - sk)
        d = beta[h] * (v_h - sk)

        # State update: state[j][i] += k[j] * d[i]
        for j in range(SSM_HEAD_V_DIM):
            for i in range(SSM_HEAD_V_DIM):
                state[h, j, i] += k_h[j] * d[i]

        # Output: out[i] = sum_j state[j][i] * q[j]
        for i in range(SSM_HEAD_V_DIM):
            out_val = 0.0
            for j in range(SSM_HEAD_V_DIM):
                out_val += state[h, j, i] * q_h[j]
            delta_out[h*SSM_HEAD_V_DIM + i] = out_val

    print(f"delta_out[:8]: {delta_out[:8]}")

    # 12. Gated RMSNorm: rmsnorm(delta_out) * silu(z)
    ssm_norm_w = load_tensor(f, tensors, data_start, "blk.0.ssm_norm.weight")
    gated_out = np.zeros(SSM_D_INNER, dtype=np.float32)
    for h in range(SSM_DT_RANK):
        head_in = delta_out[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM]
        head_z = z[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM]
        normed_head = rms_norm(head_in, ssm_norm_w)
        silu_z = silu(head_z)
        gated_out[h*SSM_HEAD_V_DIM:(h+1)*SSM_HEAD_V_DIM] = normed_head * silu_z
    print(f"gated_out[:8]: {gated_out[:8]}")

    # 13. Output projection: [4096] -> [4096]
    ssm_out_w = load_tensor(f, tensors, data_start, "blk.0.ssm_out.weight")
    ssm_projected = gated_out @ ssm_out_w.T
    print(f"ssm_projected[:8]: {ssm_projected[:8]}")

    # 14. Residual connection
    hidden_after_attn = hidden + ssm_projected
    print(f"After residual[:8]: {hidden_after_attn[:8]}")

    # 15. Post-attention norm + FFN (use torch for speed)
    post_norm_w = load_tensor(f, tensors, data_start, "blk.0.post_attention_norm.weight")
    ffn_input = rms_norm(hidden_after_attn, post_norm_w)
    print(f"ffn_input[:8]: {ffn_input[:8]}")

    # Use torch for large matmuls
    ffn_input_t = torch.from_numpy(ffn_input).float()

    print("Loading FFN weights...")
    ffn_gate_w = torch.from_numpy(load_tensor(f, tensors, data_start, "blk.0.ffn_gate.weight")).float()
    ffn_up_w = torch.from_numpy(load_tensor(f, tensors, data_start, "blk.0.ffn_up.weight")).float()
    ffn_down_w = torch.from_numpy(load_tensor(f, tensors, data_start, "blk.0.ffn_down.weight")).float()

    gate_val = ffn_input_t @ ffn_gate_w.T  # [12288]
    up_val = ffn_input_t @ ffn_up_w.T      # [12288]
    ffn_act = torch.nn.functional.silu(gate_val) * up_val
    ffn_out = (ffn_act @ ffn_down_w.T).numpy()

    hidden_final = hidden_after_attn + ffn_out
    print(f"\nFinal hidden[:8]: {hidden_final[:8]}")

    # Load CUDA result
    cuda_hidden = np.fromfile("/tmp/hidden_after_layer0.bin", dtype=np.uint16)
    cuda_hidden_f32 = np.zeros(N_EMBD, dtype=np.float32)
    for i in range(N_EMBD):
        bits = int(cuda_hidden[i]) << 16
        cuda_hidden_f32[i] = struct.unpack('<f', struct.pack('<I', bits))[0]
    print(f"CUDA hidden[:8]:   {cuda_hidden_f32[:8]}")

    diff = np.abs(hidden_final - cuda_hidden_f32)
    print(f"\nMax diff: {np.max(diff):.6f}")
    print(f"Mean diff: {np.mean(diff):.6f}")
    print(f"Max relative diff: {np.max(diff / (np.abs(hidden_final) + 1e-8)):.6f}")

    # Find where the biggest differences are
    worst = np.argsort(diff)[-10:]
    print(f"\nWorst 10 positions:")
    for idx in worst:
        print(f"  [{idx}] python={hidden_final[idx]:.6f}  cuda={cuda_hidden_f32[idx]:.6f}  diff={diff[idx]:.6f}")

    f.close()

if __name__ == "__main__":
    main()
