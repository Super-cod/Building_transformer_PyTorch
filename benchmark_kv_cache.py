"""
KV-Cache vs Full-Context Recomputation Benchmark on NVIDIA RTX 3050 GPU
Measures token generation throughput, latency scaling, and complexity reduction.
"""

import time
import torch
import numpy as np
from transformer_core import ModernTransformerDecoder

def benchmark_generation(seq_len_gen: int = 128, prompt_len: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=================================================================")
    print(f"  TRANSFORMER INFERENCE & KV-CACHE BENCHMARK (Device: {device})   ")
    print("=================================================================")

    vocab_size = 4096
    d_model = 512
    n_layers = 6
    n_heads = 8
    d_ff = 2048

    model = ModernTransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=2048
    ).to(device).eval()

    prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device)

    # 1. Benchmark Full-Context Recomputation O(N^2)
    torch.cuda.synchronize()
    curr_tokens = prompt.clone()
    t0 = time.perf_counter()

    with torch.no_grad():
        for step in range(seq_len_gen):
            logits, _ = model(curr_tokens)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

    torch.cuda.synchronize()
    time_full_sec = time.perf_counter() - t0
    tok_per_sec_full = seq_len_gen / time_full_sec
    ms_per_tok_full = (time_full_sec / seq_len_gen) * 1000.0

    # 2. Benchmark KV-Cache Generation O(1) step
    torch.cuda.synchronize()
    curr_tokens_kv = prompt.clone()
    t0 = time.perf_counter()

    with torch.no_grad():
        # Prefill prompt
        logits, kv_caches = model(curr_tokens_kv)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        curr_tokens_kv = torch.cat([curr_tokens_kv, next_token], dim=1)

        # Autoregressive decode loop
        for step in range(1, seq_len_gen):
            pos = prompt_len + step - 1
            logits, kv_caches = model(next_token, start_pos=pos, kv_caches=kv_caches)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    torch.cuda.synchronize()
    time_kv_sec = time.perf_counter() - t0
    tok_per_sec_kv = seq_len_gen / time_kv_sec
    ms_per_tok_kv = (time_kv_sec / seq_len_gen) * 1000.0

    speedup = time_full_sec / time_kv_sec

    print(f"Prompt Length: {prompt_len} tokens | Generated Length: {seq_len_gen} tokens")
    print(f"Model: {n_layers} layers | d_model={d_model} | n_heads={n_heads} | d_ff={d_ff}\n")
    print(f"1. Full Context Recomputation : {time_full_sec:6.3f} s | {ms_per_tok_full:6.2f} ms/token | {tok_per_sec_full:7.1f} tokens/sec")
    print(f"2. Stateful KV-Cache Decoding : {time_kv_sec:6.3f} s | {ms_per_tok_kv:6.2f} ms/token | {tok_per_sec_kv:7.1f} tokens/sec")
    print(f"  -> KV-Cache Speedup         : {speedup:6.2f}x faster throughput")
    print(f"  -> Computational Complexity : Reduced from O(N^2) quadratic recomputation to O(N) linear time,")
    print(f"     reusing past Key/Value projections in GPU HBM/SRAM cache.\n")
    print("=================================================================")

if __name__ == "__main__":
    benchmark_generation(seq_len_gen=64, prompt_len=32)
