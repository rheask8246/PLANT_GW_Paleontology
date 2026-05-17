"""
Full-capacity `PosteriorNet` for GPU / ACCESS Expanse training on large SSPC grids.

Inherits the same `PosteriorNet` class as `posterior_network_lite.py`; this module
only fixes wider defaults. API matches the lite build (log_prob, sample, encode).
"""
from __future__ import annotations

from .posterior_network_lite import THETA_DIM, PosteriorNet


class FullPosteriorNet(PosteriorNet):
    """
    Wider / deeper defaults for long-run GPU training on large SSPC grids
    (parameter count is typically ~20M+ with 512-dim hidden and 6 layers; use a
    large GPU and tune --batch-size / --n-max-events on the cluster).
    """

    def __init__(self) -> None:
        super().__init__(
            event_input_dim=6,
            theta_dim=THETA_DIM,
            hidden_dim=512,
            ffn_dim=2048,
            num_encoder_layers=6,
            num_heads=8,
            n_coupling=10,
            flow_hidden_dim=256,
            dropout=0.1,
        )


FullPosteriorNetwork = FullPosteriorNet
PosteriorNetworkFull = FullPosteriorNet
