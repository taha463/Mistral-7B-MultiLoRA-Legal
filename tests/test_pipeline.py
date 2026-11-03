"""
Unit and integration tests for Mistral-7B Multi-LoRA & Evidence Arbitration pipeline.
"""
import os
import sys

# Ensure repository root is always on sys.path regardless of how pytest is invoked
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import pytest

from ean_legal import (
    EvidenceGate,
    ean_loss,
    norm,
    clean,
    token_f1,
    score,
)


def test_text_normalization():
    raw = "  In Re: John Doe, 123 F.3d 456 (2020).  \n"
    expected = "in re john doe 123 f3d 456 2020"
    assert norm(raw) == expected


def test_token_f1_exact_match():
    pred = "Affirmed in part and reversed in part"
    ref = "Affirmed in part and reversed in part"
    sc = score(pred, ref)
    assert sc["em"] == 1.0
    assert sc["f1"] == 1.0
    assert sc["rougeL"] == 1.0


def test_token_f1_partial_overlap():
    pred = "affirmed in part"
    ref = "affirmed in part and reversed in part"
    f1 = token_f1(pred, ref)
    assert 0.0 < f1 < 1.0


def test_evidence_gate_forward_dimensions_and_sum():
    n_heads = 3
    d_model = 128
    d_evid = 64
    d = 32
    batch_tokens = 6

    gate = EvidenceGate(n=n_heads, d_model=d_model, d_evid=d_evid, d=d)
    h = torch.randn(n_heads, batch_tokens, d_model)
    e = torch.randn(batch_tokens, d_evid)

    log_alpha = gate(h, e, noise_std=0.0)
    alpha = log_alpha.exp()

    # Shape must be [n_heads, batch_tokens]
    assert alpha.shape == (n_heads, batch_tokens)
    # Probabilities across heads must sum to 1 for every token
    assert torch.allclose(alpha.sum(0), torch.ones(batch_tokens), atol=1e-5)


def test_ean_loss_computation():
    n_heads = 3
    n_tokens = 8

    logp = -torch.rand(n_heads, n_tokens)
    logits = torch.randn(n_heads, n_tokens)
    log_alpha = F.log_softmax(logits, dim=0)

    loss, nll_mix, head_nlls, mean_alpha = ean_loss(logp, log_alpha)

    assert not torch.isnan(loss)
    assert loss.item() > 0
    assert head_nlls.shape == (n_heads,)
    assert mean_alpha.shape == (n_heads,)
    assert torch.isclose(mean_alpha.sum(), torch.tensor(1.0), atol=1e-5)

