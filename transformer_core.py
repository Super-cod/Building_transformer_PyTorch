"""
Modern Transformer Decoder Architecture with Multi-Head Attention and KV-Cache
Designed for high-throughput inference and rigorous mathematical validation.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryOrSinusoidalPositionalEmbedding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding (Vaswani et al., 2017)
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0)) # Shape: (1, max_seq_len, d_model)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, start_pos:start_pos + seq_len, :]

class MultiHeadAttentionWithKVCache(nn.Module):
    """
    Multi-Head Scaled Dot-Product Attention with stateful KV-Cache for O(1) incremental decoding.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch_size, n_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Update and concatenate KV-cache
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        new_kv_cache = (k, v)

        # Scaled Dot-Product Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            # Mask should broadcast over n_heads: (batch, 1, q_len, k_len)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted value aggregation: (batch, n_heads, seq_len, head_dim)
        out = torch.matmul(attn_weights, v)

        # Transpose back and project: (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out), new_kv_cache

class PositionwiseFeedForward(nn.Module):
    """Two-layer MLP with GELU non-linearity: FFN(x) = GELU(x * W1 + b1) * W2 + b2"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.act(self.w_1(x))))

class TransformerDecoderBlock(nn.Module):
    """Pre-LayerNorm Transformer Decoder Block"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttentionWithKVCache(d_model, n_heads, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        norm_x = self.ln_1(x)
        attn_out, new_kv = self.attn(norm_x, mask=mask, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln_2(x))
        return x, new_kv

class ModernTransformerDecoder(nn.Module):
    """
    Complete Autoregressive Transformer Decoder
    Supports both full-sequence training and token-by-token KV-cached generation.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 1024,
        dropout: float = 0.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = RotaryOrSinusoidalPositionalEmbedding(d_model, max_seq_len)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (Press & Wolf, 2017)
        self.lm_head.weight = self.tok_embeddings.weight

    def forward(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
        kv_caches: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        batch_size, seq_len = token_ids.shape
        x = self.tok_embeddings(token_ids) * math.sqrt(self.d_model)
        x = self.pos_embeddings(x, start_pos=start_pos)
        x = self.drop(x)

        # Generate causal mask if seq_len > 1
        if seq_len > 1:
            total_k_len = seq_len + start_pos
            # Causal mask: each query can attend to keys up to its own position
            mask = torch.tril(torch.ones(seq_len, total_k_len, device=token_ids.device, dtype=torch.bool))
            mask = mask.unsqueeze(0).unsqueeze(1) # Broadcast shape: (1, 1, seq_len, total_k_len)
        else:
            mask = None

        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, mask=mask, kv_cache=cache)
            new_caches.append(new_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_caches
