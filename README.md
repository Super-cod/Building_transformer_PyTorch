# Modern Transformer Decoder Architecture & KV-Cache Inference Engine (`Building_transformer_PyTorch`)

A decoder-only (GPT-style) Transformer in PyTorch, built as a learning project. It has two parts:

- **`transformer_core.py` (2026):** a pre-LayerNorm decoder with causal multi-head attention, sinusoidal positional encoding, a weight-tied LM head and an optional **KV-cache** for incremental decoding, plus tests and a decoding benchmark.
- **`Decoder.py`, `train.py`, `attention.py`, ... (Aug 2025):** my earlier word-level model trained with PyTorch Lightning on the DailyDialog text (`train.py`). It is simpler than the core model (single attention head) and is kept as the original learning exercise.

The architecture follows the standard Transformer decoder; it is not a novel design.

---

## Architectural Specification

```
Input Tokens (B, S)
       │
       ▼
Token Embedding + Positional Encoding (B, S, d_model)
       │
┌──────▼─────────────────────────────────────────────────┐
│ Transformer Decoder Block (Repeated N times)           │
│                                                        │
│   Input x                                              │
│      │                                                 │
│      ├───► LayerNorm(x)                                │
│      │        │                                        │
│      │        ▼                                        │
│      │     Multi-Head Self-Attention + Causal Mask     │
│      │        │                                        │
│      ▼        ▼                                        │
│     ( + ) ◄───┘  [Residual Connection 1]               │
│      │                                                 │
│      ├───► LayerNorm(x)                                │
│      │        │                                        │
│      │        ▼                                        │
│      │     Position-wise Feed-Forward Network (GELU)   │
│      │        │                                        │
│      ▼        ▼                                        │
│     ( + ) ◄───┘  [Residual Connection 2]               │
└──────┬─────────────────────────────────────────────────┘
       │
       ▼
Final LayerNorm (B, S, d_model)
       │
       ▼
Linear LM Head (Weight-Tied to Token Embedding) (B, S, vocab_size)
```

---

## Mathematical Foundations

### 1. Multi-Head Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
where:
* $Q = X W_Q$, $K = X W_K$, $V = X W_V$
* $M$ is the causal attention mask:
  $$M_{i, j} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$
* Normalization over keys occurs strictly along the last dimension ($\text{dim}=-1$).

### 2. Sinusoidal Positional Encoding
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$

### 3. Key-Value (KV) caching
Without a cache, generating each new token re-runs the whole model over all `t` tokens produced so far, so total work over `N` generated tokens grows as O(N^2) in tokens processed. With a cache, the Key and Value tensors of earlier tokens are stored and only the newest token is pushed through the network:
$$K_{	ext{new}} = [K_{	ext{cached}} \,\|\, k_{	ext{current}}], \quad V_{	ext{new}} = [V_{	ext{cached}} \,\|\, v_{	ext{current}}]$$
The projections and feed-forward work per step become constant, so those costs grow linearly with `N`. The attention step itself still reads all `t` cached keys, so its cost per step grows with `t`. The cache trades memory for compute: it stores `2 x layers x t x d_model` values per sequence.

---

## KV-cache benchmark (RTX 3050 Laptop GPU)

`benchmark_kv_cache.py` generates 64 tokens (greedy decoding) from a 32-token prompt with a randomly initialised model (6 layers, `d_model=512`, 8 heads, `d_ff=2048`, vocabulary 4096, batch size 1, FP32). Both modes get a warm-up pass first, then one timed run each. Results from 5 runs:

| Mode | Throughput | Latency per token |
| :--- | :---: | :---: |
| Full-context recomputation | 168-198 tokens/s | 5.1-5.9 ms |
| KV-cache decoding | 220-235 tokens/s | 4.3-4.6 ms |

**Speedup: 1.14x-1.30x.** The gain is modest at this size: with a batch of one and short sequences the loop is dominated by per-step kernel-launch and Python overhead rather than arithmetic, so the cache helps more as sequences get longer. An earlier version of this benchmark had no warm-up and showed a larger speedup (up to about 2x) because one-off CUDA start-up cost landed in whichever mode ran first; the warm-up pass removes that bias. The timings use random weights, so they say nothing about output quality.

---

## Automated Verification Suite

From the repository root (needs `torch`; the benchmark uses a CUDA GPU if available):

```bash
python tests/test_transformer_architecture.py
python benchmark_kv_cache.py
```

or run both with `.un_tests.ps1` in PowerShell. The legacy training scripts additionally need `lightning`, `transformers` and the bundled text datasets.

### What the tests check
- Positional encoding values match the sinusoidal formula.
- Output shape is `(batch, seq, vocab)` through all layers.
- Causal masking: changing future tokens leaves earlier logits unchanged (difference 0).
- KV-cache equivalence: token-by-token decoding with the cache matches full-sequence logits (max difference below 1e-5 in my runs, test tolerance 1e-4).

---

## Limitations
- The benchmark measures speed on random weights; no model in `transformer_core.py` is trained.
- The legacy Lightning model is word-level and small; its saved checkpoints are not evaluated here.
- Single GPU, FP32, batch size 1; no multi-query/grouped attention, no FlashAttention, no batching of the cache.

## License

MIT License. Copyright (c) 2026 Swayam.
