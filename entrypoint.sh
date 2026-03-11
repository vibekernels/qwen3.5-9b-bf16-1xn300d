#!/bin/bash
set -e

PORT="${PORT:-8888}"
MODEL="${MODEL_PATH:-unsloth/Qwen3.5-9B-GGUF:BF16}"
CTX_SIZE="${CTX_SIZE:-4096}"

echo "Starting qwen-server on port $PORT (model: $MODEL, ctx: $CTX_SIZE)"

exec /app/build/qwen-server \
    -m "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --ctx-size "$CTX_SIZE"
