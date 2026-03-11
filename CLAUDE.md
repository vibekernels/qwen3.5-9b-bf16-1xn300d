# Project goals

The point of this project is to write a Tenstorrent accelerator kernel. ALL computation should run on the Tenstorrent N300 device (two Wormhole chips). Do not fall back to pure CPU compute — use the device for matmuls and as many operations as possible. If quantization is needed, do not go below INT8.

# Project structure

- `src/engine.cpp` — inference engine: forward pass, `generate()`, `load_model_and_tokenizer()`
- `src/engine.h` — public API: `generate()`, `load_model_and_tokenizer()`, `reset_state()`, `shutdown()`, `get_tokenizer()`
- `src/gguf_loader.{h,cpp}` — GGUF weight loading into device DRAM MeshBuffers
- `src/model_config.h` — model hyperparameters and tile dimensions
- `src/tokenizer.{h,cpp}` — BPE tokenizer (GPT-2 byte-level)
- `src/download.{h,cpp}` — HuggingFace model download
- `src/kernels/compute/` — Tensix compute kernels (gemv, rmsnorm, swiglu, etc.)
- `src/kernels/dataflow/` — data movement kernels (readers/writers)
- `src/tests/` — test suite (test_inference.cpp, test_forward.cpp, benchmarks)
- `third_party/` — third-party headers (json.hpp) and tt-metal SDK (git submodule)
- `third_party/tt-metal/` — TT-Metalium SDK (git submodule)

## First-time setup

```sh
make setup             # init tt-metal submodule + build SDK (~13 min)
```

## Build & test

```sh
make -j$(nproc)        # build everything
make test              # run integration tests (~60s)
make quicktest         # fast smoke test: "The capital of France is" → Paris
make clean             # remove build artifacts
```

Environment variables:
- `TT_METAL_HOME` — tt-metal source tree (default: `third_party/tt-metal`)
- `TT_METAL_BUILD` — tt-metal build dir (default: `$(TT_METAL_HOME)/build_Release`)
- `MODEL_PATH` — path to .gguf model file or HuggingFace tag (default: `vibekernels/Qwen3.5-9B-GGUF:BFP8B-tiled`)

## Test suite details

`make test` runs `test_inference` with 8 tests (ctx=128, ~60s total):

- **test_basic_generation** — "1+1=" → expects "2" in output
- **test_long_prompt** — ~50-token repeated prompt, verifies generation works
- **test_prompt_exceeding_context** — prompt exceeding ctx=128, verifies graceful truncation
- **test_emoji_in_output** — asks for emoji, validates UTF-8 output
- **test_tok_per_sec** — 32-token decode, checks ≥5 tok/s (set `MIN_TOK_PER_SEC` to override)
- **test_prefill_tok_per_sec** — ~50-token prefill benchmark, checks ≥5 prefill tok/s
- **test_greedy_determinism** — two greedy runs produce identical output
- **test_stop_on_eos** — chat prompt stops on EOS token

Override test parameters via env vars:
```sh
TEST_CTX_SIZE=256 make test          # larger context (slower)
LONG_PROMPT_REPEAT=20 make test      # longer prompt test
DECODE_COUNT=64 make test            # more decode tokens for tok/s test
MIN_TOK_PER_SEC=8 make test          # stricter performance threshold
```

## Manual inference

```sh
# Quick test:
make quicktest
# Manual run (auto-downloads model from HuggingFace on first use):
TT_METAL_RUNTIME_ROOT=$(pwd)/third_party/tt-metal \
  ./build/test_forward "unsloth/Qwen3.5-9B-GGUF:BF16" "Your prompt here" 128
# Or with a local model:
TT_METAL_RUNTIME_ROOT=$(pwd)/third_party/tt-metal \
  ./build/test_forward /path/to/Qwen3.5-9B-BF16.gguf "Your prompt here" 128
```

## Measuring decode speed

The `test_tok_per_sec` test estimates decode tok/s by subtracting estimated prefill
time from total generate() wall time. This estimate can be inaccurate — use
`test_basic_generation` output instead for a reliable decode measurement:

```
test_basic_generation...
  output (16 tok, 1922.6 ms, 8.3 tok/s): ...
```

That `8.3 tok/s` is total wall time (prefill + decode) ÷ tokens. With a very short
prompt (4 tokens, ~300ms prefill), the number closely reflects true single-token
decode speed. For a purer decode-only measurement, use a short prompt and many
decode tokens:

```sh
TT_METAL_RUNTIME_ROOT=$(pwd)/third_party/tt-metal QUIET=1 \
  ./build/test_forward "$MODEL_PATH" "1+1=" 64 --raw 2>/dev/null
```

Then compute: `(total_time - prefill_time) / decode_tokens`. The `[prefill: X ms]`
line gives actual prefill time. Do NOT hardcode a ms/tok estimate for prefill
subtraction — batched prefill (~33ms/tok) is very different from single-token
(~120ms/tok), and short prompts don't batch well (~73ms/tok for <32 tokens).

Current baselines (as of 2026-03-10):
- **Decode**: ~8.4 tok/s (119ms/tok)
- **Prefill**: ~30 tok/s (33ms/tok, batched)

## Debugging hangs

Hangs are the most common failure mode when working with tt-metal. Key lessons:

**Diagnosing hangs with GDB:**
```sh
# Attach to a hung process:
gdb -p $(pgrep test_inference)
# Get all thread backtraces:
thread apply all bt
# Look for threads stuck in: atomic wait, mutex lock, completion_queue_wait_front
```
GDB (`sudo apt-get install -y gdb`) is the most effective tool for hang diagnosis.
Thread backtraces immediately reveal whether a hang is in SDK code, custom thread
pools, or device completion waits.

**tt-metal SDK hang pitfalls:**
- `FDMeshCommandQueue::reader_thread_pool_mutex_` is `inline static` — a **global
  mutex shared across ALL command queue instances**. Concurrent `EnqueueReadMeshBuffer`
  calls from different submeshes (mesh0 + mesh1) will deadlock. **Always serialize
  reads**: read chip 0 first, then chip 1.
- `TT_METAL_OPERATION_TIMEOUT_SECONDS` enables `yield()` in SDK spin loops but the
  timeout itself never fires — the SDK's progress counter keeps advancing even when
  the specific completion hasn't arrived. It only detects hard device hangs.
- Set `NO_TRACES=1` to disable trace replay during debugging (simplifies dispatch path).

**C++20 `std::atomic::wait/notify` (futex) pitfalls:**
- Classic lost-wakeup bug: `done.wait(done.load())` can pass a **new** value if the
  atomic changed between the outer check and the inner load. The wait then blocks
  forever because current == expected. Fix: load once, reuse for both check and wait:
  ```cpp
  int val = done.load(std::memory_order_acquire);
  while (val < target) {
      done.wait(val, std::memory_order_acquire);
      val = done.load(std::memory_order_acquire);
  }
  ```

**Hang diagnostics in engine.cpp:**
- `get_hang_info(layer, op)` returns the last layer/operation before a blocking call
- `g_hang_layer` and `g_hang_op` atomics are updated before each blocking device read
- test_inference.cpp reports these on timeout for quick root-cause identification

## Reference model (llama.cpp)

For comparison against llama.cpp:

```sh
llama-completion -hf unsloth/Qwen3.5-9B-GGUF:BF16 -p "Your prompt here" -n 128 -ngl 99
```
