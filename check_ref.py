#!/usr/bin/env python3
import json, requests

prompts = ["The", "The capital", "The capital of", "The capital of France", "The capital of France is"]
for prompt in prompts:
    r = requests.post("http://localhost:18199/completion", json={
        "prompt": prompt, "n_predict": 1, "temperature": 0, "n_probs": 5
    })
    d = r.json()
    top = []
    for p in d.get("completion_probabilities", []):
        for t in p.get("top_logprobs", []):
            top.append(f"{t['id']}({t['logprob']:+.2f})")
    print(f'"{prompt}" -> {repr(d["content"])} top: {" ".join(top)}')
