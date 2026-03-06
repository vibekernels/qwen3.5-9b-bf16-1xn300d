#!/usr/bin/env python3
"""Check hidden state norms across layers for each position."""
import numpy as np
import os

N_EMBD = 4096

def load_bf16_bin(path, n):
    raw = np.fromfile(path, dtype=np.uint16)[:n]
    raw32 = raw.astype(np.uint32) << 16
    return np.frombuffer(raw32.tobytes(), dtype=np.float32).copy()

for pos in range(5):
    print(f"\n=== Position {pos} ===")
    prev = None
    for layer in range(32):
        path = f"/tmp/hidden_pos{pos}_layer{layer}.bin"
        if not os.path.exists(path):
            continue
        h = load_bf16_bin(path, N_EMBD)
        l2 = np.sqrt(np.sum(h**2))
        ltype = "ATN" if (layer + 1) % 4 == 0 else "SSM"
        delta_str = ""
        if prev is not None:
            diff = np.max(np.abs(h - prev))
            delta_str = f"  max_change={diff:.4f}"
        print(f"  L{layer:2d} ({ltype}): L2={l2:10.4f}  mean={np.mean(h):+.6f}  max={np.max(np.abs(h)):8.4f}{delta_str}")
        prev = h
