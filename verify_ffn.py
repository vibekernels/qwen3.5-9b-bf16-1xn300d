#!/usr/bin/env python3
"""Verify FFN layer 0 for position 0 against CUDA dumps."""
import struct, numpy as np, torch, os

GGUF_PATH = "/home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-BF16.gguf"
N_EMBD = 4096; N_FF = 12288; RMS_NORM_EPS = 1e-6

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
    t = torch.from_numpy(x).float()
    return t.bfloat16().float().numpy()

# Load weights
f, tensors, data_start = read_gguf_tensors(GGUF_PATH)

# Load after_ssm_residual from CUDA dump
hidden = np.fromfile("/tmp/ssm_after_ssm_residual_pos0_layer0.bin", dtype=np.float32)[:N_EMBD]
print(f"after_ssm_residual sum: {np.sum(hidden):.6f}")

# Load post_attention_norm weight
post_norm_w = load_tensor(f, tensors, data_start, "blk.0.post_attention_norm.weight")
print(f"post_norm_w shape: {post_norm_w.shape}, sum: {np.sum(post_norm_w):.6f}")

# RMSNorm
normed = rms_norm(hidden, post_norm_w)
normed_bf16 = to_bf16(normed)
print(f"normed sum: {np.sum(normed):.6f}, bf16 sum: {np.sum(normed_bf16):.6f}")

# Load FFN weights
ffn_gate_w = load_tensor(f, tensors, data_start, "blk.0.ffn_gate.weight")
ffn_up_w = load_tensor(f, tensors, data_start, "blk.0.ffn_up.weight")
ffn_down_w = load_tensor(f, tensors, data_start, "blk.0.ffn_down.weight")
print(f"ffn_gate shape: {ffn_gate_w.shape}")
print(f"ffn_up shape: {ffn_up_w.shape}")
print(f"ffn_down shape: {ffn_down_w.shape}")

# FFN computation (matching ggml bf16 GEMM)
nt = torch.from_numpy(normed_bf16).bfloat16()
gate = (nt.float() @ torch.from_numpy(ffn_gate_w).float().T).numpy()
up = (nt.float() @ torch.from_numpy(ffn_up_w).float().T).numpy()
print(f"gate sum: {np.sum(gate):.6f}, first8: {gate.flatten()[:8]}")
print(f"up sum: {np.sum(up):.6f}, first8: {up.flatten()[:8]}")

# SwiGLU
silu_gate = gate / (1 + np.exp(-gate))
swiglu = silu_gate * up
print(f"swiglu sum: {np.sum(swiglu):.6f}")

# Down projection
swiglu_bf16 = to_bf16(swiglu.flatten())
st = torch.from_numpy(swiglu_bf16).bfloat16()
ffn_out = (st.float() @ torch.from_numpy(ffn_down_w).float().T).numpy()
print(f"ffn_out sum: {np.sum(ffn_out):.6f}, first8: {ffn_out.flatten()[:8]}")

# Compare with CUDA FFN out
cuda_ffn = np.fromfile("/tmp/ssm_ffn_out_pos0_layer0.bin", dtype=np.float32)[:N_EMBD]
print(f"\nCUDA ffn_out sum: {np.sum(cuda_ffn):.6f}, first8: {cuda_ffn[:8]}")
diff = np.abs(ffn_out.flatten()[:N_EMBD] - cuda_ffn)
print(f"max_diff: {np.max(diff):.6f}, mean_diff: {np.mean(diff):.6f}")

# Also show what the final hidden state should be
final = hidden + ffn_out.flatten()[:N_EMBD]
print(f"\nPython final sum: {np.sum(final):.6f}")
print(f"llama.cpp post_ffn-0 sum: 1.213050")
