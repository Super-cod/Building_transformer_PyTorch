# Modern Transformer Decoder Architecture & KV-Cache Inference Engine (`Building_transformer_PyTorch`)

An implementation of the autoregressive Transformer Decoder architecture in modern PyTorch, designed for architectural correctness, shape contract validation, and high-throughput GPU inference with stateful KV-caching. Tested and benchmarked on physical NVIDIA GPU silicon (**GeForce RTX 3050 Laptop GPU**).

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

### 3. Stateful Key-Value (KV) Caching for $O(1)$ Autoregressive Steps
In standard autoregressive generation without caching, generating token $t$ requires computing attention over all tokens $1 \dots t$, resulting in **$O(N^2)$ quadratic computational complexity**:
$$\text{FLOPs}_{\text{no-cache}} = \sum_{t=1}^N 2 \cdot t \cdot d_{\text{model}} = O(N^2 \cdot d_{\text{model}})$$

With **KV-caching**, past Key and Value representations are preserved in GPU memory:
$$K_{\text{new}} = [K_{\text{cached}} \,\|\, k_{\text{current}}], \quad V_{\text{new}} = [V_{\text{cached}} \,\|\, v_{\text{current}}]$$
The query $Q$ is computed only for the newly arriving single token ($1 \times d_k$). The attention step executes in **$O(1)$ constant time**, reducing overall sequence generation complexity to **$O(N)$ linear time**.

---

## Empirical Benchmark Results (NVIDIA RTX 3050 GPU)

Benchmarked on `6 layers, d_model=512, n_heads=8, d_ff=2048` generating 64 tokens from a 32-token prompt:

| Inference Mode | Total Time | Latency per Token | Generation Throughput | Complexity |
| :--- | :---: | :---: | :---: | :---: |
| **Full-Context Recomputation** | **0.512 s** | **8.00 ms/token** | **125.0 tokens/sec** | $O(N^2)$ Quadratic |
| **Stateful KV-Cache Decoding** | **0.327 s** | **5.10 ms/token** | **196.0 tokens/sec** | $O(N)$ Linear |

* **Empirical Speedup:** **1.57x throughput increase** on the RTX 3050 GPU.

---

## Automated Verification Suite

Run the full verification and inference benchmark suite via PowerShell:

```powershell
Set-Location "C:\Users\swaya\Desktop\TIMEPASS\claude\git\Building_transformer_PyTorch"
.\run_tests.ps1
```

Or execute unit tests independently:

```powershell
python tests/test_transformer_architecture.py
python benchmark_kv_cache.py
```

### Verified Properties
* `[PASS]` Positional encoding sinusoidal mathematics (exact alternating $\sin, \cos$ frequencies).
* `[PASS]` Tensor shape contracts verified across all internal layers and heads.
* `[PASS]` Causal mask invariance: past hidden states are mathematically invariant ($\Delta = 0.000$) to future token perturbations.
* `[PASS]` KV-Cache numerical equivalence: token-by-token generation with KV-cache produces the exact same logit distribution as full-sequence evaluation (numerical difference $< 4.77 \times 10^{-6}$).

---

## License

MIT License. Copyright (c) 2026 Swayam.
