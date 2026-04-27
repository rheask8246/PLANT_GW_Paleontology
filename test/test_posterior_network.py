"""
Smoke tests for set encoder + conditional flow posterior (CPU).
"""
from __future__ import annotations

import os
import sys

import torch

# Run from project root: python test/test_posterior_network.py
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def test_lite_log_prob_and_sample() -> None:
    from models.posterior_network_lite import LitePosteriorNet, THETA_DIM

    batch, n_eve, d_in, k = 4, 32, 8, THETA_DIM
    model = LitePosteriorNet()
    model.eval()
    ev = torch.randn(batch, n_eve, d_in)
    m = torch.ones(batch, n_eve)
    theta = torch.randn(batch, k)
    lp = model.log_prob(theta, ev, m)
    assert lp.shape == (batch,)
    s = model.sample(ev, m, num_samples=1000)
    assert s.shape == (batch, 1000, k), s.shape
    n = sum(p.numel() for p in model.parameters())
    print(f"LitePosteriorNet parameters: {n:,}")


def test_full_param_count() -> None:
    from models.posterior_network_full import FullPosteriorNet, THETA_DIM

    m = FullPosteriorNet()
    n = sum(p.numel() for p in m.parameters())
    print(f"FullPosteriorNet parameters: {n:,}")
    # Expect multi-million; architecture-dependent lower bound
    assert n > 1_000_000, f"expected >1M params, got {n}"


def test_realnvp_roundtrip() -> None:
    from models.posterior_network_lite import ConditionalRealNVP

    B, H, D = 8, 64, 9
    f = ConditionalRealNVP(D, H, n_coupling=4, hidden_dim=32)
    x = torch.randn(B, D)
    c = torch.randn(B, H)
    z, _ = f.forward_to_base(x, c)
    x2 = f.inverse_from_base(z, c)
    err = (x - x2).abs().max().item()
    assert err < 1e-3, err


if __name__ == "__main__":
    test_lite_log_prob_and_sample()
    test_full_param_count()
    test_realnvp_roundtrip()
    print("ok")
