#!/bin/bash
set -e

PORT="${PORT:-8888}"
MODEL="${MODEL_PATH:-unsloth/Qwen3.5-9B-GGUF:BF16}"
CTX_SIZE="${CTX_SIZE:-4096}"

# Start SSH server if a public key is provided
if [ -n "$PUBLIC_KEY" ]; then
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 644 /root/.ssh/authorized_keys
    /usr/sbin/sshd
    echo "SSH server started on port 22"
fi

echo "Starting qwen-server on port $PORT (model: $MODEL, ctx: $CTX_SIZE)"

exec /app/build/qwen-server \
    -m "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --ctx-size "$CTX_SIZE"
