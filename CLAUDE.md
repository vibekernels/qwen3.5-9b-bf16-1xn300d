# llama.cpp

## Test inference

Run a one-shot completion (non-interactive) with:

```sh
llama-completion -hf unsloth/Qwen3.5-9B-GGUF:BF16 -p "Your prompt here" -n 128 -ngl 99
```

- `llama-completion` — non-interactive, exits after generation (use this instead of `llama-cli` which drops into an interactive chat loop)
- `-hf` — download and run a model directly from Hugging Face
- `-n 128` — max tokens to generate
- `-ngl 99` — offload all layers to GPU
