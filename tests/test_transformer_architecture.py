"""
Unit Test Suite for Transformer Architecture Correctness
Validates shape contracts, causal mask properties, and KV-cache numerical equivalence.
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformer_core import ModernTransformerDecoder, RotaryOrSinusoidalPositionalEmbedding

def test_positional_encoding_math():
    d_model = 64
    pe_mod = RotaryOrSinusoidalPositionalEmbedding(d_model=d_model, max_seq_len=128)
    pe = pe_mod.pe.squeeze(0) # (128, 64)

    # Position 0 check
    # sin(0) = 0 for all even indices
    assert torch.allclose(pe[0, 0::2], torch.zeros(d_model // 2)), "PE even indices at pos 0 must be 0"
    # cos(0) = 1 for all odd indices
    assert torch.allclose(pe[0, 1::2], torch.ones(d_model // 2)), "PE odd indices at pos 0 must be 1"
    print("[PASS] Test 1: Positional encoding sinusoidal mathematics verified")

def test_shape_contracts():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 500
    model = ModernTransformerDecoder(vocab_size=vocab_size, d_model=128, n_layers=2, n_heads=4, d_ff=512)
    model.to(device)
    model.eval()

    batch_size = 4
    seq_len = 32
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        logits, _ = model(tokens)

    assert logits.shape == (batch_size, seq_len, vocab_size), f"Expected shape {(batch_size, seq_len, vocab_size)}, got {logits.shape}"
    print(f"[PASS] Test 2: Tensor shape contracts verified through all layers: {tuple(logits.shape)}")

def test_kv_cache_numerical_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 300
    model = ModernTransformerDecoder(vocab_size=vocab_size, d_model=64, n_layers=2, n_heads=2, d_ff=128)
    model.to(device)
    model.eval()

    prompt = torch.tensor([[10, 45, 12, 99, 5, 22]], device=device) # Seq len 6
    prompt_len = prompt.size(1)

    # 1. Full context forward pass (without KV cache)
    with torch.no_grad():
        full_logits, _ = model(prompt)
        last_logit_full = full_logits[:, -1, :]

    # 2. Incremental KV-cached forward pass
    with torch.no_grad():
        # Prefill phase: feed prompt[0:5]
        _, kv_caches = model(prompt[:, :-1])

        # Decode phase: feed only the 6th token (prompt[:, -1:]) with cached KV
        cached_logits, _ = model(prompt[:, -1:], start_pos=prompt_len - 1, kv_caches=kv_caches)
        last_logit_cached = cached_logits[:, 0, :]

    # Assert numerical identity between full forward and KV-cached forward
    diff = torch.max(torch.abs(last_logit_full - last_logit_cached)).item()
    assert diff < 1e-4, f"KV cache outputs diverged! Max diff: {diff}"
    print(f"[PASS] Test 3: KV-Cache numerical equivalence verified (max diff: {diff:.6e} < 1e-4)")

def test_causal_mask_enforcement():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 100
    model = ModernTransformerDecoder(vocab_size=vocab_size, d_model=64, n_layers=1, n_heads=2, d_ff=128)
    model.to(device)
    model.eval()

    seq = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    # Alter token at position 4 (last token)
    seq_mod = torch.tensor([[1, 2, 3, 4, 99]], device=device)

    with torch.no_grad():
        out1, _ = model(seq)
        out2, _ = model(seq_mod)

    # In a causal transformer, tokens 0..3 cannot attend to position 4!
    # Therefore out1[:, :4, :] must be IDENTICAL to out2[:, :4, :]
    diff = torch.max(torch.abs(out1[:, :4, :] - out2[:, :4, :])).item()
    assert diff < 1e-5, f"Causal mask failed: past representations changed when future token was modified! Diff: {diff}"
    print(f"[PASS] Test 4: Causal masking verified: past logits are 100% invariant to future token alterations (diff: {diff:.6e})")

if __name__ == "__main__":
    print(">>> Running Transformer Architecture Test Suite <<<")
    test_positional_encoding_math()
    test_shape_contracts()
    test_causal_mask_enforcement()
    test_kv_cache_numerical_equivalence()
    print("\n>>> ALL TRANSFORMER ARCHITECTURE TESTS PASSED <<<")
