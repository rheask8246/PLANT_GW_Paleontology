"""
Ensemble combiners: K identical members -> same as single-member log p and samples.
"""
from __future__ import annotations

import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def test_log_mean_of_log_probs_matches_single() -> None:
    from models.ensemble_posterior import log_mean_of_log_probs
    from models.posterior_network_lite import LitePosteriorNet, THETA_DIM

    torch.manual_seed(0)
    m0 = LitePosteriorNet()
    m0.eval()
    # Two copies with same weights
    m1 = LitePosteriorNet()
    m1.load_state_dict(m0.state_dict())
    m1.eval()

    b, n_e, d_in, k = 2, 8, 8, THETA_DIM
    ev = torch.randn(b, n_e, d_in)
    mask = torch.ones(b, n_e)
    theta = torch.randn(b, k)
    lp0 = m0.log_prob(theta, ev, mask)
    lm = log_mean_of_log_probs([m0, m1], theta, ev, mask)
    err = (lp0 - lm).abs().max().item()
    assert err < 1e-5, err


def test_mixture_sample_shape() -> None:
    from models.ensemble_posterior import mixture_sample
    from models.posterior_network_lite import LitePosteriorNet, THETA_DIM

    torch.manual_seed(1)
    m0 = LitePosteriorNet()
    m0.eval()
    m1 = LitePosteriorNet()
    m1.load_state_dict(m0.state_dict())
    m1.eval()

    b, n_e, d_in, k = 1, 4, 8, THETA_DIM
    ev = torch.randn(b, n_e, d_in)
    mask = torch.ones(b, n_e)
    s = mixture_sample([m0, m1], ev, mask, num_samples=7)
    assert s.shape == (1, 7, k), s.shape


if __name__ == "__main__":
    test_log_mean_of_log_probs_matches_single()
    test_mixture_sample_shape()
    print("ok")
